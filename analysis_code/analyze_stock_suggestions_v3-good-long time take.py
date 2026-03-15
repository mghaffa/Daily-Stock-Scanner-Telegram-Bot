# =========================
# FINAL FULL VERSION
# =========================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_stock_suggestions_v2.py

Purpose
- Parse your scanner Telegram/log output (stock-info.txt) AND/OR your scanner CSV output
  (scanner_dual_tf_vp_dip7.csv) into a single signals table.
- Fetch daily OHLC via yfinance OR load from a cached OHLC CSV.
- Compute forward returns (1/3/5/10 trading days).
- Rank which single-metric thresholds and metric combinations had the best results.

Important fixes
- Uses adjusted pricing consistently for forward returns by default (Adj Close if present).
- Adds "snap-to-close" alignment: if your log close does not match the aligned market close,
  the code searches within ±N trading days for a date where market close matches your log close.
  This eliminates huge split-driven mismatches (e.g., 900% diffs).

Install
  pip install pandas numpy yfinance pandas_market_calendars scikit-learn

Run examples
  python analyze_stock_suggestions_v2.py --log stock-info.txt --out_dir out_v2
  python analyze_stock_suggestions_v2.py --csv scanner_dual_tf_vp_dip7.csv --out_dir out_v2
  python analyze_stock_suggestions_v2.py --log stock-info.txt --csv scanner_dual_tf_vp_dip7.csv --out_dir out_v2

Use cached OHLC (recommended when Yahoo blocks / throttles you)
  python analyze_stock_suggestions_v2.py --log stock-info.txt --ohlc_csv .\\out-v1\\ohlc_raw.csv --out_dir out_v2

Outputs
- out/signals_with_returns.csv
- out/metric_effects_{horizon}d.csv
- out/combo_rules_{horizon}d.csv

Notes
- This is analysis tooling only. Not financial advice.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None


# ----------------------------
# Config
# ----------------------------

FORWARD_HORIZONS = [1, 3, 5, 10]

TIER_PRIORITY = {
    "Tier1": 50,
    "Tier2": 40,
    "Tier3A": 30,
    "Tier3B": 20,
    "BuyZone": 15,
    "LRC_touch_setup": 12,      # "LRC touch OK (setup only)"
    "LRC_touch_confirmed": 14,  # "LRC touch OK + other gates"
    "BUY": 60,                  # if line is in BUY block
    "WATCH": 10,                # if line is in watch block
    None: -1,
}

# If your "Entries" timestamp is in early hours, your logged close is usually prior day.
EARLY_POST_HOUR_CUTOFF = 6  # inclusive

# Combination testing grid (tweak freely)
GRID = {
    "tier_allow": [
        ["Tier1"],
        ["Tier1", "Tier2"],
        ["Tier1", "Tier2", "LRC_touch_confirmed"],
        ["Tier1", "Tier2", "Tier3A"],
        ["Tier1", "Tier2", "Tier3A", "Tier3B"],
        ["Tier1", "Tier2", "Tier3A", "Tier3B", "BuyZone", "LRC_touch_setup", "LRC_touch_confirmed", "BUY", "WATCH"],
    ],
    "div_min": [None, 1, 2, 3, 4, 5],
    "conf_min": [None, 30, 40, 50, 60, 70, 80],
    "willr_max": [None, -80, -85, -90, -92, -95],
    "stoch_abs_max": [None, 5, 10, 15, 20],
    "rr_min": [None, 1.5, 3, 5, 8, 10],
    "dist_lrc_max": [None, 0.02, 0.04, 0.06, 0.08, 0.10],  # pct distance to LRC lower
    "ema_gap_min": [None, 0.01, 0.025, 0.05],              # (ema50-close)/ema50
}

MIN_SAMPLE_SINGLE = 25
MIN_SAMPLE_COMBO = 40

# Snap-to-close parameters
SNAP_WINDOW_TRADING_DAYS = 3     # search +/- this many trading days around the aligned date index
SNAP_TOL_PCT = 0.02              # accept a match if market close is within 2% of log close
SNAP_ALWAYS = False              # if True, always snap even when already close


# ----------------------------
# Helpers
# ----------------------------

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
        return int(float(x))
    except Exception:
        return None


def _norm_ticker_yf(t: str) -> str:
    return (t or "").strip().upper().replace(".", "-")


def _get_calendar():
    if mcal is None:
        return None
    try:
        return mcal.get_calendar("NYSE")
    except Exception:
        return None


def _align_to_trading_day(date_: dt.date, cal) -> pd.Timestamp:
    ts = pd.Timestamp(date_)
    if cal is None:
        return ts
    start = ts - pd.Timedelta(days=10)
    end = ts + pd.Timedelta(days=1)
    sched = cal.schedule(start_date=start, end_date=end)
    if sched.empty:
        return ts
    trading_days = sched.index.tz_localize(None).sort_values()
    trading_days = trading_days[trading_days <= ts]
    if trading_days.empty:
        return trading_days.min()
    return trading_days.max()


def _dedupe_best(df: pd.DataFrame, keys: List[str], pri_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = (
        df.sort_values(keys + [pri_col], ascending=[True] * len(keys) + [False])
          .drop_duplicates(keys, keep="first")
          .reset_index(drop=True)
    )
    return out


def _pct_diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return (a / b) - 1.0


# ----------------------------
# Parsing: text log (Telegram / console)
# ----------------------------

@dataclass
class ParsedSignal:
    sig_date: dt.date
    post_date: dt.date
    post_time: dt.time
    ticker: str
    tf: str

    tier: Optional[str] = None
    status: Optional[str] = None  # BUY/WATCH if known

    close_log: Optional[float] = None

    div_v3: Optional[int] = None
    div_v3_names_raw: Optional[str] = None
    conf: Optional[int] = None
    willr: Optional[float] = None
    stoch_delta: Optional[float] = None

    rr: Optional[float] = None
    sug_buy: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None

    lrc_lower: Optional[float] = None
    lrc_mid: Optional[float] = None
    lrc_upper: Optional[float] = None

    ema50: Optional[float] = None
    ema200: Optional[float] = None

    away_pct_logged: Optional[float] = None


def parse_log_text(path: str) -> pd.DataFrame:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    entries_header_re = re.compile(r"Entries\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})")

    tier1_re = re.compile(r"Tier\s*1", re.IGNORECASE)
    tier2_re = re.compile(r"Tier\s*2", re.IGNORECASE)
    tier3a_re = re.compile(r"Tier\s*3A", re.IGNORECASE)
    tier3b_re = re.compile(r"Tier\s*3B", re.IGNORECASE)
    buyzone_re = re.compile(r"In\s+Buy\s+Zone", re.IGNORECASE)

    lrc_setup_re = re.compile(r"LRC\s+touch\s+OK\s+\(setup\s+only", re.IGNORECASE)
    lrc_confirmed_re = re.compile(r"LRC\s+touch\s+OK\s*\+\s*other\s+gates", re.IGNORECASE)

    buy_block_re = re.compile(r"BUY\s*\(confirmed\)", re.IGNORECASE)
    watch_block_re = re.compile(r"In\s+Buy\s+Zone\s*\(watching\)", re.IGNORECASE)

    line_re = re.compile(r"^\s*-\s+([A-Z0-9\.\-]+)\s+\[([0-9A-Za-z]+)\]\s+c\s+([0-9]+(?:\.[0-9]+)?)\s*\|\s*(.*)$")

    away_re = re.compile(r"\(\s*([-+]?[0-9]+(?:\.[0-9]+)?)%\s+away\s*\)", re.IGNORECASE)

    div_re = re.compile(r"div_v3\s+(\d+)", re.IGNORECASE)
    div_names_re = re.compile(r"div_v3\s+\d+\s+(\[[^\]]*\])")

    conf_re = re.compile(r"conf\s+(\d+)", re.IGNORECASE)
    willr_re = re.compile(r"W%R\s+(-?[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    stoch_re = re.compile(r"StochRSI\s+Delta\s+(-?[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

    rr_re = re.compile(r"R/R\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    sug_buy_re = re.compile(r"sug_buy\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    stop_re = re.compile(r"\bstop\s+([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
    tgt_re = re.compile(r"\btgt\s+([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)

    lrc_lo_re = re.compile(r"LRC_lo\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    lrc_lower_re = re.compile(r"lrc_lower\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    lrc_mid_re = re.compile(r"lrc_mid\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    lrc_up_re = re.compile(r"lrc_upper\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

    ema50_re = re.compile(r"EMA50\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    ema200_re = re.compile(r"EMA200\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

    current_post_date: Optional[dt.date] = None
    current_post_time: Optional[dt.time] = None
    current_tier: Optional[str] = None
    current_status: Optional[str] = None
    signals: List[ParsedSignal] = []

    for raw in text.splitlines():
        line = raw.strip()

        mh = entries_header_re.search(line)
        if mh:
            current_post_date = dt.date.fromisoformat(mh.group(1))
            current_post_time = dt.time(int(mh.group(2)), int(mh.group(3)))
            current_tier = None
            current_status = None
            continue

        if current_post_date is None or current_post_time is None:
            continue

        if tier1_re.search(line):
            current_tier = "Tier1"
        elif tier2_re.search(line):
            current_tier = "Tier2"
        elif tier3a_re.search(line):
            current_tier = "Tier3A"
        elif tier3b_re.search(line):
            current_tier = "Tier3B"
        elif buyzone_re.search(line):
            current_tier = "BuyZone"
        elif lrc_setup_re.search(line):
            current_tier = "LRC_touch_setup"
        elif lrc_confirmed_re.search(line):
            current_tier = "LRC_touch_confirmed"

        if buy_block_re.search(line):
            current_status = "BUY"
        elif watch_block_re.search(line):
            current_status = "WATCH"

        md = line_re.match(line)
        if not md:
            continue

        ticker = md.group(1).strip().upper()
        tf = md.group(2).strip().upper()
        close_log = float(md.group(3))
        rest = md.group(4)

        away = _safe_float(away_re.search(rest).group(1)) if away_re.search(rest) else None
        if away is not None:
            away = away / 100.0

        div_v3 = _safe_int(div_re.search(rest).group(1)) if div_re.search(rest) else None
        div_names_raw = div_names_re.search(rest).group(1) if div_names_re.search(rest) else None
        conf = _safe_int(conf_re.search(rest).group(1)) if conf_re.search(rest) else None
        willr = _safe_float(willr_re.search(rest).group(1)) if willr_re.search(rest) else None
        stoch_delta = _safe_float(stoch_re.search(rest).group(1)) if stoch_re.search(rest) else None

        rr = _safe_float(rr_re.search(rest).group(1)) if rr_re.search(rest) else None
        sug_buy = _safe_float(sug_buy_re.search(rest).group(1)) if sug_buy_re.search(rest) else None
        stop = _safe_float(stop_re.search(rest).group(1)) if stop_re.search(rest) else None
        target = _safe_float(tgt_re.search(rest).group(1)) if tgt_re.search(rest) else None

        lrc_lower = None
        if lrc_lo_re.search(rest):
            lrc_lower = _safe_float(lrc_lo_re.search(rest).group(1))
        elif lrc_lower_re.search(rest):
            lrc_lower = _safe_float(lrc_lower_re.search(rest).group(1))

        lrc_mid = _safe_float(lrc_mid_re.search(rest).group(1)) if lrc_mid_re.search(rest) else None
        lrc_upper = _safe_float(lrc_up_re.search(rest).group(1)) if lrc_up_re.search(rest) else None

        ema50 = _safe_float(ema50_re.search(rest).group(1)) if ema50_re.search(rest) else None
        ema200 = _safe_float(ema200_re.search(rest).group(1)) if ema200_re.search(rest) else None

        sig_date = current_post_date
        if current_post_time.hour <= EARLY_POST_HOUR_CUTOFF:
            sig_date = sig_date - dt.timedelta(days=1)

        signals.append(
            ParsedSignal(
                sig_date=sig_date,
                post_date=current_post_date,
                post_time=current_post_time,
                ticker=ticker,
                tf=tf,
                tier=current_tier,
                status=current_status,
                close_log=close_log,
                div_v3=div_v3,
                div_v3_names_raw=div_names_raw,
                conf=conf,
                willr=willr,
                stoch_delta=stoch_delta,
                rr=rr,
                sug_buy=sug_buy,
                stop=stop,
                target=target,
                lrc_lower=lrc_lower,
                lrc_mid=lrc_mid,
                lrc_upper=lrc_upper,
                ema50=ema50,
                ema200=ema200,
                away_pct_logged=away,
            )
        )

    df = pd.DataFrame([s.__dict__ for s in signals])
    if df.empty:
        return df

    df["ticker_yf"] = df["ticker"].map(_norm_ticker_yf)
    df["tier_pri"] = df["tier"].map(TIER_PRIORITY).fillna(-1).astype(int)
    df["status_pri"] = df["status"].map(TIER_PRIORITY).fillna(-1).astype(int)
    df["pri"] = df["tier_pri"] + df["status_pri"]

    df = _dedupe_best(df, ["sig_date", "ticker", "tf"], "pri")
    return df


# ----------------------------
# Parsing: CSV output from your scanner (recommended)
# ----------------------------

def parse_scanner_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    col_map = {
        "ticker": "ticker",
        "tf": "tf",
        "date": "sig_date_str",
        "status": "status",
        "close": "close_log",
        "conf": "conf",
        "rr": "rr",
        "sug_buy": "sug_buy",
        "stop": "stop",
        "target": "target",
        "ema50": "ema50",
        "ema200": "ema200",
        "willr": "willr",
        "stochrsi_delta": "stoch_delta",
        "div_v3_cnt": "div_v3",
        "div_v3_names": "div_v3_names_raw",
        "lrc_lower": "lrc_lower",
        "lrc_mid": "lrc_mid",
        "lrc_upper": "lrc_upper",
        "primary_ok": "primary_ok",
        "strict_ok": "strict_ok",
        "trigger_ok": "trigger_ok",
        "lrc_touch_ok": "lrc_touch_ok",
        "regime_ok": "regime_ok",
        "avwap4h_ok": "avwap4h_ok",
        "vol_ok": "vol_ok",
        "sweet_ok": "sweet_ok",
        "momo_ok": "momo_ok",
        "trend_ok": "trend_ok",
        "liq_ok": "liq_ok",
        "rr_ok": "rr_ok",
    }

    keep = {}
    for k, v in col_map.items():
        if k in df.columns:
            keep[k] = v

    df = df.rename(columns=keep)

    if "ticker" not in df.columns or "sig_date_str" not in df.columns:
        raise RuntimeError("CSV missing required columns: ticker and date (sig_date_str).")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["tf"] = df["tf"].astype(str).str.upper() if "tf" in df.columns else "1D"
    df["sig_date"] = pd.to_datetime(df["sig_date_str"], errors="coerce").dt.date
    df = df.dropna(subset=["sig_date"]).copy()

    df["post_date"] = df["sig_date"]
    df["post_time"] = dt.time(0, 0)

    df["tier"] = None
    if "lrc_touch_ok" in df.columns and "strict_ok" in df.columns:
        df.loc[(df["lrc_touch_ok"] == True) & (df["strict_ok"] == True), "tier"] = "LRC_touch_confirmed"
        df.loc[(df["lrc_touch_ok"] == True) & (df["strict_ok"] != True), "tier"] = "LRC_touch_setup"

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.upper().replace({"NAN": ""})

    df["ticker_yf"] = df["ticker"].map(_norm_ticker_yf)

    df["tier_pri"] = df["tier"].map(TIER_PRIORITY).fillna(-1).astype(int)
    df["status_pri"] = df.get("status", "").map(TIER_PRIORITY).fillna(-1).astype(int) if "status" in df.columns else -1
    df["pri"] = df["tier_pri"] + df["status_pri"]

    df = _dedupe_best(df, ["sig_date", "ticker", "tf"], "pri")
    return df


# ----------------------------
# Combine inputs
# ----------------------------

def load_signals(log_path: Optional[str], csv_path: Optional[str]) -> pd.DataFrame:
    parts = []
    if log_path:
        tdf = parse_log_text(log_path)
        if not tdf.empty:
            parts.append(tdf)
    if csv_path:
        cdf = parse_scanner_csv(csv_path)
        if not cdf.empty:
            parts.append(cdf)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True, sort=False)

    if "pri" not in df.columns:
        df["tier_pri"] = df.get("tier", None).map(TIER_PRIORITY).fillna(-1).astype(int)
        df["status_pri"] = df.get("status", "").map(TIER_PRIORITY).fillna(-1).astype(int)
        df["pri"] = df["tier_pri"] + df["status_pri"]

    df = _dedupe_best(df, ["sig_date", "ticker", "tf"], "pri")

    df["stoch_abs"] = pd.to_numeric(df.get("stoch_delta", np.nan), errors="coerce").astype(float).abs()

    df["dist_to_lrc_pct"] = np.nan
    if "lrc_lower" in df.columns:
        ll = pd.to_numeric(df["lrc_lower"], errors="coerce")
        c = pd.to_numeric(df.get("close_log", np.nan), errors="coerce")
        df.loc[ll.notna() & (ll != 0) & c.notna(), "dist_to_lrc_pct"] = ((c - ll).abs() / ll)

    df["ema_gap_pct"] = np.nan
    if "ema50" in df.columns:
        e50 = pd.to_numeric(df["ema50"], errors="coerce")
        c = pd.to_numeric(df.get("close_log", np.nan), errors="coerce")
        df.loc[e50.notna() & (e50 != 0) & c.notna(), "ema_gap_pct"] = ((e50 - c) / e50)

    if "away_pct_logged" in df.columns:
        df["away_pct_logged"] = pd.to_numeric(df["away_pct_logged"], errors="coerce")

    for col in ["close_log", "div_v3", "conf", "willr", "stoch_delta", "rr", "sug_buy", "stop", "target",
                "lrc_lower", "lrc_mid", "lrc_upper", "ema50", "ema200"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ----------------------------
# Market data fetch
# ----------------------------

def fetch_ohlc_yfinance(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    start_str = (start - dt.timedelta(days=7)).isoformat()
    end_str = (end + dt.timedelta(days=7)).isoformat()

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
        raise RuntimeError("yfinance returned empty data. Check internet access and tickers.")

    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            sub = data[t].dropna(how="all").copy()
            if sub.empty:
                continue
            sub = sub.reset_index()
            sub["Ticker"] = t
            rows.append(sub)
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    else:
        out = data.reset_index()
        out["Ticker"] = tickers[0]

    if out.empty:
        raise RuntimeError("No usable OHLC after parsing yfinance output.")

    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)

    for need in ["Open", "High", "Low", "Close"]:
        if need not in out.columns:
            raise RuntimeError(f"Missing {need} in yfinance data for at least one ticker.")

    # Some yfinance modes may omit Adj Close; keep it if present, else create it
    if "Adj Close" not in out.columns:
        out["Adj Close"] = out["Close"]

    if "Volume" not in out.columns:
        out["Volume"] = np.nan

    return out[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()


def _find_best_match_index(
    closes: np.ndarray,
    i0: int,
    close_log: float,
    window: int,
    tol_pct: float
) -> Optional[int]:
    """
    Search around i0 within +/-window indices for a close that matches close_log within tol_pct.
    Pick the closest match by pct diff, breaking ties by proximity to i0.
    """
    if i0 is None or not np.isfinite(close_log) or close_log <= 0:
        return None

    lo = max(0, i0 - window)
    hi = min(len(closes) - 1, i0 + window)

    best_i = None
    best_abs_diff = None
    best_dist = None

    for i in range(lo, hi + 1):
        c = closes[i]
        if not np.isfinite(c) or c <= 0:
            continue
        diff = abs((close_log / c) - 1.0)
        if diff <= tol_pct:
            dist = abs(i - i0)
            if best_abs_diff is None or diff < best_abs_diff or (diff == best_abs_diff and dist < best_dist):
                best_abs_diff = diff
                best_i = i
                best_dist = dist

    return best_i


def attach_forward_returns(df_signals: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    cal = _get_calendar()
    df = df_signals.copy()

    df["sig_ts"] = df["sig_date"].apply(lambda d: _align_to_trading_day(d, cal))

    o = ohlc.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    by_ticker: Dict[str, Tuple[pd.DataFrame, List[pd.Timestamp], np.ndarray, Dict[pd.Timestamp, int]]] = {}
    for t, sub in o.groupby("Ticker"):
        sub = sub.reset_index(drop=True)
        dates = sub["Date"].tolist()
        price_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
        closes = pd.to_numeric(sub[price_col], errors="coerce").to_numpy(dtype=float)
        date_to_i = {d: i for i, d in enumerate(dates)}
        by_ticker[t] = (sub, dates, closes, date_to_i)

    entry_close = []
    snapped = []  # whether we snapped to a nearby day

    forw = {h: [] for h in FORWARD_HORIZONS}

    for _, r in df.iterrows():
        t = r["ticker_yf"]
        sig_ts = pd.Timestamp(r["sig_ts"])
        close_log = float(r["close_log"]) if "close_log" in r and pd.notna(r["close_log"]) else np.nan

        if t not in by_ticker:
            entry_close.append(np.nan)
            snapped.append(False)
            for h in FORWARD_HORIZONS:
                forw[h].append(np.nan)
            continue

        sub, dates, closes, d2i = by_ticker[t]

        # Find base index i0 (aligned to nearest trading day <= sig_ts)
        if sig_ts in d2i:
            i0 = d2i[sig_ts]
        else:
            ds = pd.Series(dates)
            prior = ds[ds <= sig_ts]
            if prior.empty:
                i0 = None
            else:
                i0 = d2i[prior.iloc[-1]]

        if i0 is None:
            entry_close.append(np.nan)
            snapped.append(False)
            for h in FORWARD_HORIZONS:
                forw[h].append(np.nan)
            continue

        # Snap-to-close if needed
        i_use = i0
        did_snap = False
        if np.isfinite(close_log) and close_log > 0 and np.isfinite(closes[i0]) and closes[i0] > 0:
            base_diff = abs((close_log / closes[i0]) - 1.0)
            if SNAP_ALWAYS or base_diff > SNAP_TOL_PCT:
                best_i = _find_best_match_index(
                    closes=closes,
                    i0=i0,
                    close_log=close_log,
                    window=SNAP_WINDOW_TRADING_DAYS,
                    tol_pct=SNAP_TOL_PCT,
                )
                if best_i is not None:
                    i_use = best_i
                    did_snap = (i_use != i0)

        snapped.append(did_snap)

        c0 = closes[i_use]
        entry_close.append(float(c0) if np.isfinite(c0) else np.nan)

        for h in FORWARD_HORIZONS:
            i1 = i_use + h
            if i1 >= len(closes):
                forw[h].append(np.nan)
            else:
                c1 = closes[i1]
                if not np.isfinite(c0) or not np.isfinite(c1) or c0 == 0:
                    forw[h].append(np.nan)
                else:
                    forw[h].append(float((c1 / c0) - 1.0))

    df["entry_close_mkt"] = entry_close
    df["snapped_to_close"] = snapped

    for h in FORWARD_HORIZONS:
        df[f"ret_{h}d"] = forw[h]
        df[f"win_{h}d"] = df[f"ret_{h}d"] > 0

    if "close_log" in df.columns:
        df["log_vs_mkt_close_diff_pct"] = (df["close_log"] / df["entry_close_mkt"] - 1.0) * 100.0
    else:
        df["log_vs_mkt_close_diff_pct"] = np.nan

    return df


# ----------------------------
# Scoring: single metrics
# ----------------------------

def summarize_metric_effects(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    col_ret = f"ret_{horizon}d"
    col_win = f"win_{horizon}d"

    base = df[df[col_ret].notna()].copy()
    if base.empty:
        return pd.DataFrame()

    results = []

    def add_group(name: str, mask: pd.Series):
        sub = base[mask]
        if len(sub) < MIN_SAMPLE_SINGLE:
            return
        results.append({
            "feature": name,
            "n": int(len(sub)),
            "win_rate": float(sub[col_win].mean()),
            "avg_ret": float(sub[col_ret].mean()),
            "median_ret": float(sub[col_ret].median()),
        })

    if "tier" in base.columns:
        for t in sorted([x for x in base["tier"].dropna().unique()]):
            add_group(f"tier={t}", base["tier"] == t)

    if "status" in base.columns:
        for s in sorted([x for x in base["status"].dropna().unique()]):
            if str(s).strip() == "":
                continue
            add_group(f"status={s}", base["status"] == s)

    if "div_v3" in base.columns:
        for k in [1, 2, 3, 4, 5]:
            add_group(f"div_v3>={k}", base["div_v3"].fillna(-1) >= k)

    if "conf" in base.columns:
        for k in [30, 40, 50, 60, 70, 80]:
            add_group(f"conf>={k}", base["conf"].fillna(-1) >= k)

    if "willr" in base.columns:
        for thr in [-80, -85, -90, -92, -95]:
            add_group(f"willr<={thr}", base["willr"].notna() & (base["willr"] <= thr))

    if "stoch_abs" in base.columns:
        for thr in [5, 10, 15, 20]:
            add_group(f"|stoch_delta|<={thr}", base["stoch_abs"].notna() & (base["stoch_abs"] <= thr))

    if "rr" in base.columns:
        for thr in [1.5, 3, 5, 8, 10]:
            add_group(f"rr>={thr}", base["rr"].notna() & (base["rr"] >= thr))

    if "dist_to_lrc_pct" in base.columns:
        for thr in [0.02, 0.04, 0.06, 0.08, 0.10]:
            add_group(f"dist_to_lrc<={thr:.0%}", base["dist_to_lrc_pct"].notna() & (base["dist_to_lrc_pct"] <= thr))

    if "ema_gap_pct" in base.columns:
        for thr in [0.01, 0.025, 0.05]:
            add_group(f"ema_gap>={thr:.1%}", base["ema_gap_pct"].notna() & (base["ema_gap_pct"] >= thr))

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out["expectancy_proxy"] = out["win_rate"] * out["avg_ret"]
    out = out.sort_values(["expectancy_proxy", "win_rate", "n"], ascending=[False, False, False]).reset_index(drop=True)
    return out


# ----------------------------
# Scoring: combination rules
# ----------------------------

def score_rule_grid(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
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
        dist_lrc_max: Optional[float],
        ema_gap_min: Optional[float],
    ) -> pd.Series:
        m = pd.Series(True, index=base.index)

        if "tier" in base.columns:
            m &= base["tier"].isin(tier_allow)

        if div_min is not None and "div_v3" in base.columns:
            m &= base["div_v3"].fillna(-1) >= div_min

        if conf_min is not None and "conf" in base.columns:
            m &= base["conf"].fillna(-1) >= conf_min

        if willr_max is not None and "willr" in base.columns:
            m &= base["willr"].notna() & (base["willr"] <= willr_max)

        if stoch_abs_max is not None and "stoch_abs" in base.columns:
            m &= base["stoch_abs"].notna() & (base["stoch_abs"] <= stoch_abs_max)

        if rr_min is not None and "rr" in base.columns:
            m &= base["rr"].notna() & (base["rr"] >= rr_min)

        if dist_lrc_max is not None and "dist_to_lrc_pct" in base.columns:
            m &= base["dist_to_lrc_pct"].notna() & (base["dist_to_lrc_pct"] <= dist_lrc_max)

        if ema_gap_min is not None and "ema_gap_pct" in base.columns:
            m &= base["ema_gap_pct"].notna() & (base["ema_gap_pct"] >= ema_gap_min)

        return m

    for tier_allow in GRID["tier_allow"]:
        for div_min in GRID["div_min"]:
            for conf_min in GRID["conf_min"]:
                for willr_max in GRID["willr_max"]:
                    for stoch_abs_max in GRID["stoch_abs_max"]:
                        for rr_min in GRID["rr_min"]:
                            for dist_lrc_max in GRID["dist_lrc_max"]:
                                for ema_gap_min in GRID["ema_gap_min"]:
                                    mask = apply_rule(
                                        tier_allow=tier_allow,
                                        div_min=div_min,
                                        conf_min=conf_min,
                                        willr_max=willr_max,
                                        stoch_abs_max=stoch_abs_max,
                                        rr_min=rr_min,
                                        dist_lrc_max=dist_lrc_max,
                                        ema_gap_min=ema_gap_min,
                                    )
                                    sub = base[mask]
                                    if len(sub) < MIN_SAMPLE_COMBO:
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
                                        "dist_lrc_max": dist_lrc_max,
                                        "ema_gap_min": ema_gap_min,
                                        "n": int(len(sub)),
                                        "win_rate": win_rate,
                                        "avg_ret": avg_ret,
                                        "median_ret": med_ret,
                                        "expectancy_proxy": win_rate * avg_ret,
                                    })

    out = pd.DataFrame(rules_out)
    if out.empty:
        return out

    out["score"] = out["expectancy_proxy"] * np.log1p(out["n"])
    out = out.sort_values(["score", "win_rate", "n"], ascending=[False, False, False]).reset_index(drop=True)
    return out


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="", help="Path to stock-info.txt (Telegram/log output)")
    ap.add_argument("--csv", default="", help="Path to scanner_dual_tf_vp_dip7.csv (recommended)")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    ap.add_argument("--max_tickers", type=int, default=500, help="Safety limit")
    ap.add_argument("--ohlc_csv", default="", help="Optional: path to pre-downloaded OHLC CSV (from v1 ohlc_raw.csv) to skip yfinance")
    args = ap.parse_args()

    log_path = args.log.strip() or None
    csv_path = args.csv.strip() or None

    df = load_signals(log_path, csv_path)
    if df.empty:
        print("ERROR: No signals loaded. Provide --log and/or --csv.", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    start = df["sig_date"].min()
    end = df["sig_date"].max() + dt.timedelta(days=30)

    tickers = sorted(df["ticker_yf"].dropna().unique().tolist())[: args.max_tickers]
    print(f"Loaded {len(df)} unique signals across {len(tickers)} tickers.")
    print(f"Market window: {start} -> {end}")

    ohlc_csv = args.ohlc_csv.strip()
    if ohlc_csv:
        print(f"Loading OHLC from {ohlc_csv} (skipping yfinance)...")
        ohlc = pd.read_csv(ohlc_csv)
        if "Date" not in ohlc.columns or "Ticker" not in ohlc.columns:
            raise RuntimeError("OHLC CSV missing required columns: Date, Ticker")
        if "Adj Close" not in ohlc.columns and "Close" not in ohlc.columns:
            raise RuntimeError("OHLC CSV must include Close or Adj Close")
        if "Adj Close" not in ohlc.columns:
            ohlc["Adj Close"] = ohlc["Close"]
        if "Close" not in ohlc.columns:
            ohlc["Close"] = ohlc["Adj Close"]
        if "Volume" not in ohlc.columns:
            ohlc["Volume"] = np.nan
        ohlc["Date"] = pd.to_datetime(ohlc["Date"]).dt.tz_localize(None)
    else:
        if yf is None:
            print("ERROR: yfinance is not installed. Run: pip install yfinance", file=sys.stderr)
            return 2
        print("Fetching daily OHLC via yfinance...")
        ohlc = fetch_ohlc_yfinance(tickers, start, end)
        ohlc.to_csv(os.path.join(args.out_dir, "ohlc_raw.csv"), index=False)

    df2 = attach_forward_returns(df, ohlc)
    df2.to_csv(os.path.join(args.out_dir, "signals_with_returns.csv"), index=False)

    qc = df2["log_vs_mkt_close_diff_pct"].dropna()
    if len(qc) > 0:
        desc = qc.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
        print("\nLog close vs market close diff (pct) stats:")
        print(desc.to_string())

        if "snapped_to_close" in df2.columns:
            print(f"\nSnapped rows: {int(df2['snapped_to_close'].fillna(False).sum())} / {len(df2)}")

    for h in FORWARD_HORIZONS:
        ms = summarize_metric_effects(df2, h)
        ms_path = os.path.join(args.out_dir, f"metric_effects_{h}d.csv")
        ms.to_csv(ms_path, index=False)
        print(f"Wrote {ms_path} ({len(ms)} rows).")

    for h in FORWARD_HORIZONS:
        rules = score_rule_grid(df2, h)
        rules_path = os.path.join(args.out_dir, f"combo_rules_{h}d.csv")
        rules.to_csv(rules_path, index=False)
        print(f"Wrote {rules_path} ({len(rules)} rows).")

        if not rules.empty:
            print(f"\nTop 10 combo rules for {h}d horizon:")
            cols = [
                "n", "win_rate", "avg_ret", "median_ret",
                "tier_allow", "div_min", "conf_min",
                "willr_max", "stoch_abs_max", "rr_min",
                "dist_lrc_max", "ema_gap_min",
            ]
            print(rules[cols].head(10).to_string(index=False))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())