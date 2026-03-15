"""
analyze_stock_suggestions.py

End-to-end analysis of your Stock-Suggestion log (stock-info.txt):
- Parse entries into rows with indicators (RR, W%R, StochRSI Delta, div_v3, tier, distance-to-LRC, EMA info, etc.)
- Fetch daily OHLC for tickers via yfinance
- Align each signal date to the nearest trading day close
- Compute forward returns (1/3/5/10 trading days)
- Score indicators and combinations (win rate, avg return, median return, expectancy proxy)
- Output ranked tables to CSV

USAGE:
  pip install pandas numpy yfinance pandas_market_calendars scikit-learn
  python analyze_stock_suggestions.py --log stock-info.txt --out_dir out

NOTES:
- This script assumes the close price "c" in the log corresponds to the prior trading day close
  when the post time is very early (e.g., 00:xx to 06:xx). That matches how many alert bots work.
- If your log timestamps behave differently, adjust SIGNAL_DATE_RULE in config section.

AUTHOR: You
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import re
import sys
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional dependencies
try:
    import yfinance as yf
except Exception as e:
    yf = None

try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None

# ----------------------------
# Config
# ----------------------------

TIER_PRIORITY = {
    # higher = more "final" / stronger setups
    "Tier1": 5,
    "BuyZone": 4,
    "LRC_touch": 3,
    "Tier3B": 2,
    "Tier3A": 1,
    "Tier2": 0,
    None: -1,
}

# If the log post-time is very early, treat the signal as prior day's close.
# Adjust if your bot runs at a different time relative to market close.
EARLY_POST_HOUR_CUTOFF = 6  # inclusive

FORWARD_HORIZONS = [1, 3, 5, 10]  # trading days ahead

# Rule grid for combination testing (you can tweak ranges)
GRID = {
    "tier_allow": [
        ["Tier1"],
        ["Tier1", "BuyZone"],
        ["Tier1", "BuyZone", "LRC_touch"],
        ["Tier1", "BuyZone", "LRC_touch", "Tier3B"],
        ["Tier1", "Tier2", "Tier3A", "Tier3B", "LRC_touch", "BuyZone"],
    ],
    "div_min": [None, 2, 3, 4, 5],
    "conf_min": [None, 30, 40, 50, 60, 70],
    "willr_max": [None, -80, -85, -90, -92, -95],
    "stoch_abs_max": [None, 5, 10, 15, 20],
    "rr_min": [None, 3, 5, 8, 10],
    "away_max": [None, 2, 3, 5, 7],  # (x% away) from LRC, only present in some tiers
}

# ----------------------------
# Parsing
# ----------------------------

@dataclass
class ParsedRow:
    sig_date: dt.date
    post_date: dt.date
    post_time: dt.time
    tier: Optional[str]
    ticker: str
    close: float
    div_v3: Optional[int]
    conf: Optional[int]
    willr: Optional[float]
    stoch_delta: Optional[float]
    lrc_lo: Optional[float]
    sug_buy: Optional[float]
    rr: Optional[float]
    ema50: Optional[float]
    ema200: Optional[float]
    away_pct: Optional[float]


def _safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def parse_log_file(path: str) -> pd.DataFrame:
    """
    Parse your Stock-Suggestion log into a DataFrame.
    """
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    rows: List[ParsedRow] = []
    current_post_date: Optional[dt.date] = None
    current_post_time: Optional[dt.time] = None
    current_tier: Optional[str] = None

    # Detect "Entries YYYY-MM-DD HH:MM:" which appears to be an "entry batch time"
    entries_header_re = re.compile(r"Entries\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2}):")
    # Tier lines (loose matching)
    tier_map = [
        ("Tier 1", "Tier1"),
        ("Tier 2", "Tier2"),
        ("Tier 3A", "Tier3A"),
        ("Tier 3B", "Tier3B"),
    ]

    # Data line starts like: "- AAPL [1D] c 250.12 | ..."
    data_line_re = re.compile(r"^-\s+([A-Z0-9\.\-]+)\s+\[1D\]\s+c\s+([0-9\.]+)\s+\|\s+(.*)$")

    # Metrics patterns
    div_re = re.compile(r"div_v3\s+(\d+)")
    conf_re = re.compile(r"conf\s+(\d+)")
    willr_re = re.compile(r"W%R\s+(-?[0-9\.]+)")
    stoch_re = re.compile(r"StochRSI Delta\s+(-?[0-9\.]+)")
    lrc_lo_re = re.compile(r"LRC_lo\s+([0-9\.]+)")
    sug_buy_re = re.compile(r"sug_buy\s+([0-9\.]+)")
    rr_re = re.compile(r"R/R\s+([0-9\.]+)")
    ema50_re = re.compile(r"EMA50\s+([0-9\.]+)")
    ema200_re = re.compile(r"EMA200\s+([0-9\.]+)")
    away_re = re.compile(r"\(([-0-9\.]+)%\s+away\)")

    for raw_line in text.splitlines():
        line = raw_line.strip()

        mh = entries_header_re.search(line)
        if mh:
            current_post_date = dt.date.fromisoformat(mh.group(1))
            current_post_time = dt.time(int(mh.group(2)), int(mh.group(3)))
            current_tier = None
            continue

        if current_post_date is None or current_post_time is None:
            continue

        # Tier detection
        if "LRC touch OK" in line:
            current_tier = "LRC_touch"
        elif "In Buy Zone" in line:
            current_tier = "BuyZone"
        else:
            for key, val in tier_map:
                if key in line:
                    current_tier = val
                    break

        md = data_line_re.match(line)
        if not md:
            continue

        ticker = md.group(1)
        close = float(md.group(2))
        rest = md.group(3)

        div_v3 = _safe_int(div_re.search(rest).group(1)) if div_re.search(rest) else None
        conf = _safe_int(conf_re.search(rest).group(1)) if conf_re.search(rest) else None
        willr = _safe_float(willr_re.search(rest).group(1)) if willr_re.search(rest) else None
        stoch_delta = _safe_float(stoch_re.search(rest).group(1)) if stoch_re.search(rest) else None
        lrc_lo = _safe_float(lrc_lo_re.search(rest).group(1)) if lrc_lo_re.search(rest) else None
        sug_buy = _safe_float(sug_buy_re.search(rest).group(1)) if sug_buy_re.search(rest) else None
        rr = _safe_float(rr_re.search(rest).group(1)) if rr_re.search(rest) else None
        ema50 = _safe_float(ema50_re.search(rest).group(1)) if ema50_re.search(rest) else None
        ema200 = _safe_float(ema200_re.search(rest).group(1)) if ema200_re.search(rest) else None
        away_pct = _safe_float(away_re.search(rest).group(1)) if away_re.search(rest) else None

        # Derive signal date from post date/time
        sig_date = current_post_date
        if current_post_time.hour <= EARLY_POST_HOUR_CUTOFF:
            sig_date = sig_date - dt.timedelta(days=1)

        rows.append(
            ParsedRow(
                sig_date=sig_date,
                post_date=current_post_date,
                post_time=current_post_time,
                tier=current_tier,
                ticker=ticker,
                close=close,
                div_v3=div_v3,
                conf=conf,
                willr=willr,
                stoch_delta=stoch_delta,
                lrc_lo=lrc_lo,
                sug_buy=sug_buy,
                rr=rr,
                ema50=ema50,
                ema200=ema200,
                away_pct=away_pct,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        raise RuntimeError("Parsed 0 rows from log. Check your input file format.")

    # Normalize tickers for yfinance (e.g., BRK.B -> BRK-B)
    # Your file doesn't show such tickers, but this keeps it robust.
    df["ticker_yf"] = df["ticker"].str.replace(".", "-", regex=False)

    # Deduplicate: same ticker can appear multiple times on same signal date in different tiers.
    # Keep the highest-priority tier row per (sig_date, ticker).
    df["tier_pri"] = df["tier"].map(TIER_PRIORITY).fillna(-1).astype(int)
    df = (
        df.sort_values(["sig_date", "ticker", "tier_pri"], ascending=[True, True, False])
          .drop_duplicates(["sig_date", "ticker"], keep="first")
          .reset_index(drop=True)
    )

    # Feature engineering helpers
    df["stoch_abs"] = df["stoch_delta"].abs()
    df["has_div"] = df["div_v3"].notna()
    df["has_rr"] = df["rr"].notna()
    df["has_away"] = df["away_pct"].notna()

    return df


# ----------------------------
# Market data
# ----------------------------

def get_trading_calendar() -> Optional["mcal.MarketCalendar"]:
    if mcal is None:
        return None
    try:
        return mcal.get_calendar("NYSE")
    except Exception:
        return None


def align_to_trading_day(date: dt.date, cal) -> pd.Timestamp:
    """
    Given a date, find the most recent trading day on or before that date (NYSE calendar).
    If calendar is unavailable, return pandas Timestamp of the date.
    """
    ts = pd.Timestamp(date)
    if cal is None:
        return ts
    start = ts - pd.Timedelta(days=7)
    end = ts + pd.Timedelta(days=1)
    sched = cal.schedule(start_date=start, end_date=end)
    if sched.empty:
        return ts
    trading_days = sched.index.tz_localize(None)
    trading_days = trading_days.sort_values()
    trading_days = trading_days[trading_days <= ts]
    if len(trading_days) == 0:
        return trading_days.min()
    return trading_days.max()


def fetch_ohlc_yfinance(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Fetch daily OHLCV for a list of tickers from yfinance.
    Returns a DataFrame with columns:
      ['Date','Ticker','Open','High','Low','Close','Adj Close','Volume']
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    # yfinance end is exclusive-ish, give buffer
    start_str = (start - dt.timedelta(days=5)).isoformat()
    end_str = (end + dt.timedelta(days=5)).isoformat()

    data = yf.download(
        tickers=tickers,
        start=start_str,
        end=end_str,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned empty data. Check connectivity or ticker symbols.")

    # yfinance returns MultiIndex columns if multiple tickers
    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            sub = data[t].copy()
            sub = sub.dropna(how="all")
            if sub.empty:
                continue
            sub = sub.reset_index().rename(columns={"index": "Date"})
            sub["Ticker"] = t
            rows.append(sub)
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    else:
        # Single ticker case
        out = data.reset_index()
        out["Ticker"] = tickers[0]

    if out.empty:
        raise RuntimeError("No usable OHLC data after parsing yfinance output.")

    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    # Standardize col names
    out = out.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume",
    })

    return out[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]


# ----------------------------
# Outcome labeling and scoring
# ----------------------------

def attach_forward_returns(df_signals: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Attach forward close-to-close returns for each signal at horizons in FORWARD_HORIZONS.
    """
    cal = get_trading_calendar()

    # Align signal date to a trading day (most recent trading day <= sig_date)
    df = df_signals.copy()
    df["sig_ts"] = df["sig_date"].apply(lambda d: align_to_trading_day(d, cal))

    # Build per-ticker trading day series
    o = ohlc.copy()
    o = o.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Create an index map: for each ticker, map date->row_idx and list dates
    by_ticker = {}
    for t, sub in o.groupby("Ticker"):
        dates = sub["Date"].tolist()
        closes = sub["Close"].to_numpy()
        date_to_i = {d: i for i, d in enumerate(dates)}
        by_ticker[t] = (sub.reset_index(drop=True), dates, closes, date_to_i)

    # Determine entry close on signal day and forward closes
    entry_close = []
    forw = {h: [] for h in FORWARD_HORIZONS}

    for _, r in df.iterrows():
        t = r["ticker_yf"]
        sig_ts = pd.Timestamp(r["sig_ts"])
        if t not in by_ticker:
            entry_close.append(np.nan)
            for h in FORWARD_HORIZONS:
                forw[h].append(np.nan)
            continue

        sub, dates, closes, d2i = by_ticker[t]

        # Find exact date match first, otherwise use the nearest previous trading day in the OHLC data
        if sig_ts in d2i:
            i0 = d2i[sig_ts]
        else:
            # nearest previous date present in sub
            # binary search
            ds = pd.Series(dates)
            prior = ds[ds <= sig_ts]
            if prior.empty:
                i0 = None
            else:
                i0 = d2i[prior.iloc[-1]]

        if i0 is None:
            entry_close.append(np.nan)
            for h in FORWARD_HORIZONS:
                forw[h].append(np.nan)
            continue

        c0 = closes[i0]
        entry_close.append(float(c0))

        for h in FORWARD_HORIZONS:
            i1 = i0 + h
            if i1 >= len(closes):
                forw[h].append(np.nan)
            else:
                c1 = closes[i1]
                ret = (c1 / c0) - 1.0
                forw[h].append(float(ret))

    df["entry_close_mkt"] = entry_close

    for h in FORWARD_HORIZONS:
        df[f"ret_{h}d"] = forw[h]
        df[f"win_{h}d"] = df[f"ret_{h}d"] > 0

    # Quality checks: compare log close vs market close
    df["log_vs_mkt_close_diff_pct"] = (df["close"] / df["entry_close_mkt"] - 1.0) * 100.0

    return df


def summarize_metric_effects(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Score individual metrics using simple binning.
    Returns a DataFrame of results.
    """
    col_ret = f"ret_{horizon}d"
    col_win = f"win_{horizon}d"

    base = df[df[col_ret].notna()].copy()
    if base.empty:
        return pd.DataFrame()

    results = []

    def add_group(name: str, mask: pd.Series):
        sub = base[mask]
        if len(sub) < 25:
            return
        results.append({
            "feature": name,
            "n": len(sub),
            "win_rate": float(sub[col_win].mean()),
            "avg_ret": float(sub[col_ret].mean()),
            "median_ret": float(sub[col_ret].median()),
        })

    # Tiers
    for tier in sorted(base["tier"].dropna().unique()):
        add_group(f"tier={tier}", base["tier"] == tier)

    # div_v3 thresholds
    for k in [2, 3, 4, 5]:
        add_group(f"div_v3>={k}", base["div_v3"].fillna(-1) >= k)

    # conf thresholds
    for k in [30, 40, 50, 60, 70]:
        add_group(f"conf>={k}", base["conf"].fillna(-1) >= k)

    # W%R thresholds (more negative means more oversold)
    for thr in [-80, -85, -90, -92, -95]:
        add_group(f"willr<={thr}", base["willr"].notna() & (base["willr"] <= thr))

    # StochRSI delta absolute thresholds
    for thr in [5, 10, 15, 20]:
        add_group(f"|stoch_delta|<={thr}", base["stoch_abs"].notna() & (base["stoch_abs"] <= thr))

    # R/R thresholds
    for thr in [3, 5, 8, 10]:
        add_group(f"rr>={thr}", base["rr"].notna() & (base["rr"] >= thr))

    # Away thresholds (distance to LRC)
    for thr in [2, 3, 5, 7]:
        add_group(f"away_pct<={thr}", base["away_pct"].notna() & (base["away_pct"] <= thr))

    out = pd.DataFrame(results)
    if out.empty:
        return out
    out["expectancy_proxy"] = out["win_rate"] * out["avg_ret"]
    out = out.sort_values(["expectancy_proxy", "win_rate", "n"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def score_rule_grid(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Evaluate a grid of combination rules and rank them by expectancy-like metrics.
    """
    col_ret = f"ret_{horizon}d"
    col_win = f"win_{horizon}d"

    base = df[df[col_ret].notna()].copy()
    if base.empty:
        return pd.DataFrame()

    rules_out = []

    def apply_rule(
        tier_allow: List[str],
        div_min: Optional[int],
        conf_min: Optional[int],
        willr_max: Optional[float],
        stoch_abs_max: Optional[float],
        rr_min: Optional[float],
        away_max: Optional[float],
    ) -> pd.Series:
        m = base["tier"].isin(tier_allow)

        if div_min is not None:
            m &= base["div_v3"].fillna(-1) >= div_min

        if conf_min is not None:
            m &= base["conf"].fillna(-1) >= conf_min

        if willr_max is not None:
            m &= base["willr"].notna() & (base["willr"] <= willr_max)

        if stoch_abs_max is not None:
            m &= base["stoch_abs"].notna() & (base["stoch_abs"] <= stoch_abs_max)

        if rr_min is not None:
            m &= base["rr"].notna() & (base["rr"] >= rr_min)

        if away_max is not None:
            m &= base["away_pct"].notna() & (base["away_pct"] <= away_max)

        return m

    # Brute-force grid. Keep it manageable.
    for tier_allow in GRID["tier_allow"]:
        for div_min in GRID["div_min"]:
            for conf_min in GRID["conf_min"]:
                for willr_max in GRID["willr_max"]:
                    for stoch_abs_max in GRID["stoch_abs_max"]:
                        for rr_min in GRID["rr_min"]:
                            for away_max in GRID["away_max"]:
                                mask = apply_rule(
                                    tier_allow=tier_allow,
                                    div_min=div_min,
                                    conf_min=conf_min,
                                    willr_max=willr_max,
                                    stoch_abs_max=stoch_abs_max,
                                    rr_min=rr_min,
                                    away_max=away_max,
                                )
                                sub = base[mask]
                                if len(sub) < 40:
                                    continue

                                win_rate = float(sub[col_win].mean())
                                avg_ret = float(sub[col_ret].mean())
                                med_ret = float(sub[col_ret].median())

                                rules_out.append({
                                    "tier_allow": ",".join(tier_allow),
                                    "div_min": div_min,
                                    "conf_min": conf_min,
                                    "willr_max": willr_max,
                                    "stoch_abs_max": stoch_abs_max,
                                    "rr_min": rr_min,
                                    "away_max": away_max,
                                    "n": len(sub),
                                    "win_rate": win_rate,
                                    "avg_ret": avg_ret,
                                    "median_ret": med_ret,
                                    "expectancy_proxy": win_rate * avg_ret,
                                })

    out = pd.DataFrame(rules_out)
    if out.empty:
        return out

    # Rank with a slight penalty for tiny samples
    out["score"] = out["expectancy_proxy"] * np.log1p(out["n"])
    out = out.sort_values(["score", "win_rate", "n"], ascending=[False, False, False]).reset_index(drop=True)
    return out


# ----------------------------
# Main
# ----------------------------

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to stock-info.txt")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    ap.add_argument("--max_tickers", type=int, default=500, help="Safety limit")
    args = ap.parse_args()

    df = parse_log_file(args.log)

    ensure_out_dir(args.out_dir)

    # Determine market data window
    start = df["sig_date"].min()
    end = df["sig_date"].max() + dt.timedelta(days=20)

    tickers = sorted(df["ticker_yf"].unique().tolist())
    tickers = tickers[: args.max_tickers]

    if yf is None:
        print("ERROR: yfinance not installed. Run: pip install yfinance", file=sys.stderr)
        return 2

    print(f"Parsed {len(df)} unique (date,ticker) signals across {len(tickers)} tickers.")
    print(f"Fetching OHLC from {start} to {end}...")

    ohlc = fetch_ohlc_yfinance(tickers, start, end)
    ohlc.to_csv(os.path.join(args.out_dir, "ohlc_raw.csv"), index=False)

    df2 = attach_forward_returns(df, ohlc)
    df2.to_csv(os.path.join(args.out_dir, "signals_with_returns.csv"), index=False)

    # Sanity check: how often log close differs from market close
    qc = df2["log_vs_mkt_close_diff_pct"].dropna()
    if len(qc) > 0:
        print("Log-vs-market close diff (pct):")
        print(qc.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())

    # Metric effects per horizon
    for h in FORWARD_HORIZONS:
        metric_summary = summarize_metric_effects(df2, h)
        metric_path = os.path.join(args.out_dir, f"metric_effects_{h}d.csv")
        metric_summary.to_csv(metric_path, index=False)
        print(f"Wrote {metric_path} with {len(metric_summary)} rows.")

    # Rule grid per horizon
    for h in FORWARD_HORIZONS:
        rules = score_rule_grid(df2, h)
        rules_path = os.path.join(args.out_dir, f"combo_rules_{h}d.csv")
        rules.to_csv(rules_path, index=False)
        print(f"Wrote {rules_path} with {len(rules)} rows.")

        # Print top 10 to console
        if not rules.empty:
            print(f"\nTop 10 combo rules for {h}d horizon:")
            cols = ["n", "win_rate", "avg_ret", "median_ret", "tier_allow", "div_min", "conf_min",
                    "willr_max", "stoch_abs_max", "rr_min", "away_max"]
            print(rules[cols].head(10).to_string(index=False))

    print("\nDone. Use the combo_rules_*.csv files to choose your best gating logic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())