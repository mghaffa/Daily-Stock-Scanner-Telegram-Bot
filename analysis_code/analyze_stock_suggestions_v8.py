#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from dataclasses import dataclass
from itertools import product
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


DEFAULT_FORWARD_HORIZONS = [1, 3, 5, 10, 20, 40, 60]
DEFAULT_LONG_TERM_HORIZONS = [20, 40, 60]
HORIZON_LABELS = {1: "1d", 3: "3d", 5: "5d", 10: "10d", 20: "1m_20d", 40: "2m_40d", 60: "3m_60d"}
EARLY_POST_HOUR_CUTOFF = 6

TIER_PRIORITY = {
    "Tier1": 50,
    "Tier2": 40,
    "Tier3A": 30,
    "Tier3B": 20,
    "BuyZone": 15,
    "LRC_touch": 10,
    None: -1,
}

MIN_SAMPLE_SINGLE = 25
MIN_SAMPLE_COMBO = 40

SNAP_WINDOW_TRADING_DAYS = 3
SNAP_TOL_PCT = 0.02

# Common stock split / reverse-split factors to test
SPLIT_FACTORS = [
    1.0,
    2.0, 0.5,
    3.0, 1.0 / 3.0,
    4.0, 0.25,
    5.0, 0.2,
    10.0, 0.1,
    20.0, 0.05,
]

FAST_GRID = {
    "tier_allow": [
        ["Tier1"],
        ["Tier1", "BuyZone"],
        ["Tier1", "BuyZone", "LRC_touch"],
        ["Tier1", "Tier2", "Tier3A", "Tier3B", "LRC_touch", "BuyZone"],
    ],
    "div_min": [None, 2, 3, 4],
    "conf_min": [None, 30, 40, 50],
    "willr_max": [None, -85, -90, -92, -95],
    "stoch_abs_max": [None, 10, 15],
    "rr_min": [None, 3, 5, 8],
    "away_max": [None, 2, 3, 5],
    "dist_lrc_max": [None, 0.04, 0.06, 0.10],
    "ema_gap_min": [None, 0.01, 0.025, 0.05],
}

FULL_GRID = {
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
    "away_max": [None, 2, 3, 5, 7],
    "dist_lrc_max": [None, 0.02, 0.04, 0.06, 0.08, 0.10],
    "ema_gap_min": [None, 0.01, 0.025, 0.05],
}


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
        return int(float(x))
    except Exception:
        return None


def _norm_ticker_yf(t: str) -> str:
    return (t or "").strip().upper().replace(".", "-")


def get_trading_calendar():
    if mcal is None:
        return None
    try:
        return mcal.get_calendar("NYSE")
    except Exception:
        return None


def align_to_trading_day(date_: dt.date, cal) -> pd.Timestamp:
    ts = pd.Timestamp(date_)
    if cal is None:
        return ts
    start = ts - pd.Timedelta(days=7)
    end = ts + pd.Timedelta(days=1)
    sched = cal.schedule(start_date=start, end_date=end)
    if sched.empty:
        return ts
    trading_days = sched.index.tz_localize(None).sort_values()
    trading_days = trading_days[trading_days <= ts]
    if len(trading_days) == 0:
        return trading_days.min()
    return trading_days.max()


def parse_log_file(path: str) -> pd.DataFrame:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    rows: List[ParsedRow] = []
    current_post_date: Optional[dt.date] = None
    current_post_time: Optional[dt.time] = None
    current_tier: Optional[str] = None

    entries_header_re = re.compile(r"Entries\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2}):")
    tier_map = [
        ("Tier 1", "Tier1"),
        ("Tier 2", "Tier2"),
        ("Tier 3A", "Tier3A"),
        ("Tier 3B", "Tier3B"),
    ]
    data_line_re = re.compile(r"^-\s+([A-Z0-9\.\-]+)\s+\[1D\]\s+c\s+([0-9\.]+)\s+\|\s+(.*)$")

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
        raise RuntimeError("Parsed 0 rows from log. Check input format.")

    df["ticker_yf"] = df["ticker"].map(_norm_ticker_yf)
    df["tier_pri"] = df["tier"].map(TIER_PRIORITY).fillna(-1).astype(int)
    df = (
        df.sort_values(["sig_date", "ticker", "tier_pri"], ascending=[True, True, False])
          .drop_duplicates(["sig_date", "ticker"], keep="first")
          .reset_index(drop=True)
    )

    df["stoch_abs"] = pd.to_numeric(df["stoch_delta"], errors="coerce").abs()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["lrc_lo"] = pd.to_numeric(df["lrc_lo"], errors="coerce")
    df["ema50"] = pd.to_numeric(df["ema50"], errors="coerce")
    df["ema200"] = pd.to_numeric(df["ema200"], errors="coerce")
    df["away_pct"] = pd.to_numeric(df["away_pct"], errors="coerce")

    df["dist_to_lrc_pct"] = np.nan
    mask_lrc = df["lrc_lo"].notna() & df["close"].notna() & (df["lrc_lo"] != 0)
    df.loc[mask_lrc, "dist_to_lrc_pct"] = ((df.loc[mask_lrc, "close"] - df.loc[mask_lrc, "lrc_lo"]).abs() /
                                             df.loc[mask_lrc, "lrc_lo"])

    df["ema_gap_pct"] = np.nan
    mask_ema = df["ema50"].notna() & df["close"].notna() & (df["ema50"] != 0)
    df.loc[mask_ema, "ema_gap_pct"] = ((df.loc[mask_ema, "ema50"] - df.loc[mask_ema, "close"]) /
                                         df.loc[mask_ema, "ema50"])
    return df


def fetch_ohlc_yfinance(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed.")

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
        raise RuntimeError("yfinance returned empty data.")

    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            sub = data[t].dropna(how="all").copy()
            if sub.empty:
                continue
            sub = sub.reset_index().rename(columns={"index": "Date"})
            sub["Ticker"] = t
            rows.append(sub)
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    else:
        out = data.reset_index()
        out["Ticker"] = tickers[0]

    if out.empty:
        raise RuntimeError("No usable OHLC after parsing yfinance output.")

    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)

    if "Adj Close" not in out.columns:
        out["Adj Close"] = out["Close"]

    return out[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()


def _find_best_match_index(
    closes: np.ndarray,
    i0: int,
    close_log: float
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Returns:
      (best_index, scale_factor_used, best_abs_diff)

    scale_factor_used means:
      adjusted_log_close = close_log * scale_factor_used

    Example:
      if close_log=210 and market close is ~21, best scale factor will be 0.1
    """
    if i0 is None or not np.isfinite(close_log) or close_log <= 0:
        return None, None, None

    lo = max(0, i0 - SNAP_WINDOW_TRADING_DAYS)
    hi = min(len(closes) - 1, i0 + SNAP_WINDOW_TRADING_DAYS)

    best_i = None
    best_scale = None
    best_abs_diff = None
    best_dist = None

    for scale in SPLIT_FACTORS:
        scaled_log = close_log * scale
        if not np.isfinite(scaled_log) or scaled_log <= 0:
            continue

        for i in range(lo, hi + 1):
            c = closes[i]
            if not np.isfinite(c) or c <= 0:
                continue

            diff = abs((scaled_log / c) - 1.0)
            if diff <= SNAP_TOL_PCT:
                dist = abs(i - i0)
                if (
                    best_abs_diff is None
                    or diff < best_abs_diff
                    or (diff == best_abs_diff and dist < best_dist)
                ):
                    best_abs_diff = diff
                    best_i = i
                    best_scale = scale
                    best_dist = dist

    return best_i, best_scale, best_abs_diff


def attach_forward_returns(df_signals: pd.DataFrame, ohlc: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    cal = get_trading_calendar()
    df = df_signals.copy()
    df["sig_ts"] = df["sig_date"].apply(lambda d: align_to_trading_day(d, cal))

    o = ohlc.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    by_ticker = {}
    for t, sub in o.groupby("Ticker"):
        sub = sub.reset_index(drop=True)
        dates = sub["Date"].tolist()
        closes = pd.to_numeric(sub["Adj Close"], errors="coerce").to_numpy(dtype=float)
        date_to_i = {d: i for i, d in enumerate(dates)}
        by_ticker[t] = (dates, closes, date_to_i)

    entry_close = []
    snapped = []
    split_scaled = []
    split_scale_factor = []
    adjusted_log_close = []
    forw = {h: [] for h in horizons}

    for _, r in df.iterrows():
        t = r["ticker_yf"]
        sig_ts = pd.Timestamp(r["sig_ts"])
        close_log = float(r["close"]) if pd.notna(r["close"]) else np.nan

        if t not in by_ticker:
            entry_close.append(np.nan)
            snapped.append(False)
            split_scaled.append(False)
            split_scale_factor.append(np.nan)
            adjusted_log_close.append(np.nan)
            for h in horizons:
                forw[h].append(np.nan)
            continue

        dates, closes, d2i = by_ticker[t]

        if sig_ts in d2i:
            i0 = d2i[sig_ts]
        else:
            ds = pd.Series(dates)
            prior = ds[ds <= sig_ts]
            i0 = d2i[prior.iloc[-1]] if not prior.empty else None

        if i0 is None:
            entry_close.append(np.nan)
            snapped.append(False)
            split_scaled.append(False)
            split_scale_factor.append(np.nan)
            adjusted_log_close.append(np.nan)
            for h in horizons:
                forw[h].append(np.nan)
            continue

        i_use = i0
        did_snap = False
        did_split_scale = False
        used_scale = 1.0
        used_log_close = close_log

        if np.isfinite(close_log) and np.isfinite(closes[i0]) and closes[i0] > 0:
            base_diff = abs((close_log / closes[i0]) - 1.0)

            if base_diff > SNAP_TOL_PCT:
                best_i, best_scale, best_diff = _find_best_match_index(closes, i0, close_log)
                if best_i is not None and best_scale is not None:
                    i_use = best_i
                    used_scale = float(best_scale)
                    used_log_close = close_log * used_scale
                    did_snap = i_use != i0
                    did_split_scale = abs(used_scale - 1.0) > 1e-12

        snapped.append(did_snap)
        split_scaled.append(did_split_scale)
        split_scale_factor.append(used_scale)
        adjusted_log_close.append(used_log_close)

        c0 = closes[i_use]
        entry_close.append(float(c0) if np.isfinite(c0) else np.nan)

        for h in horizons:
            i1 = i_use + h
            if i1 >= len(closes) or not np.isfinite(c0) or c0 == 0 or not np.isfinite(closes[i1]):
                forw[h].append(np.nan)
            else:
                forw[h].append(float((closes[i1] / c0) - 1.0))

    df["entry_close_mkt"] = entry_close
    df["snapped_to_close"] = snapped
    df["split_scaled"] = split_scaled
    df["split_scale_factor"] = split_scale_factor
    df["adjusted_log_close_for_match"] = adjusted_log_close

    for h in horizons:
        df[f"ret_{h}d"] = forw[h]
        df[f"win_{h}d"] = df[f"ret_{h}d"] > 0

    df["log_vs_mkt_close_diff_pct"] = (df["close"] / df["entry_close_mkt"] - 1.0) * 100.0
    df["adjusted_log_vs_mkt_close_diff_pct"] = (
        df["adjusted_log_close_for_match"] / df["entry_close_mkt"] - 1.0
    ) * 100.0

    return df


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

    for tier in sorted(base["tier"].dropna().unique()):
        add_group(f"tier={tier}", base["tier"] == tier)

    for k in [2, 3, 4, 5]:
        add_group(f"div_v3>={k}", base["div_v3"].fillna(-1) >= k)

    for k in [30, 40, 50, 60, 70]:
        add_group(f"conf>={k}", base["conf"].fillna(-1) >= k)

    for thr in [-80, -85, -90, -92, -95]:
        add_group(f"willr<={thr}", base["willr"].notna() & (base["willr"] <= thr))

    for thr in [5, 10, 15, 20]:
        add_group(f"|stoch_delta|<={thr}", base["stoch_abs"].notna() & (base["stoch_abs"] <= thr))

    for thr in [3, 5, 8, 10]:
        add_group(f"rr>={thr}", base["rr"].notna() & (base["rr"] >= thr))

    for thr in [2, 3, 5, 7]:
        add_group(f"away_pct<={thr}", base["away_pct"].notna() & (base["away_pct"] <= thr))

    for thr in [0.02, 0.04, 0.06, 0.08, 0.10]:
        add_group(f"dist_to_lrc_pct<={thr:.0%}", base["dist_to_lrc_pct"].notna() & (base["dist_to_lrc_pct"] <= thr))

    for thr in [0.01, 0.025, 0.05]:
        add_group(f"ema_gap_pct>={thr:.1%}", base["ema_gap_pct"].notna() & (base["ema_gap_pct"] >= thr))

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out["expectancy_proxy"] = out["win_rate"] * out["avg_ret"]
    out = out.sort_values(["expectancy_proxy", "win_rate", "n"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def score_rule_grid(df: pd.DataFrame, horizon: int, combo_mode: str) -> pd.DataFrame:
    col_ret = f"ret_{horizon}d"
    col_win = f"win_{horizon}d"
    base = df[df[col_ret].notna()].copy()
    if base.empty:
        return pd.DataFrame()

    grid = FAST_GRID if combo_mode == "fast" else FULL_GRID
    rules_out = []

    for tier_allow, div_min, conf_min, willr_max, stoch_abs_max, rr_min, away_max, dist_lrc_max, ema_gap_min in product(
        grid["tier_allow"],
        grid["div_min"],
        grid["conf_min"],
        grid["willr_max"],
        grid["stoch_abs_max"],
        grid["rr_min"],
        grid["away_max"],
        grid["dist_lrc_max"],
        grid["ema_gap_min"],
    ):
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
        if dist_lrc_max is not None:
            m &= base["dist_to_lrc_pct"].notna() & (base["dist_to_lrc_pct"] <= dist_lrc_max)
        if ema_gap_min is not None:
            m &= base["ema_gap_pct"].notna() & (base["ema_gap_pct"] >= ema_gap_min)

        sub = base[m]
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
            "away_max": away_max,
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



def build_horizon_summary_table(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    rows = []
    for h in horizons:
        ret_col = f"ret_{h}d"
        win_col = f"win_{h}d"
        if ret_col not in df.columns:
            continue
        sub = df[df[ret_col].notna()].copy()
        if sub.empty:
            continue
        rows.append({
            "horizon_days": h,
            "horizon_label": HORIZON_LABELS.get(h, f"{h}d"),
            "n": int(len(sub)),
            "win_rate": float(sub[win_col].mean()),
            "avg_ret": float(sub[ret_col].mean()),
            "median_ret": float(sub[ret_col].median()),
            "p25_ret": float(sub[ret_col].quantile(0.25)),
            "p75_ret": float(sub[ret_col].quantile(0.75)),
            "best_ret": float(sub[ret_col].max()),
            "worst_ret": float(sub[ret_col].min()),
        })
    return pd.DataFrame(rows)




def build_horizon_coverage_table(df: pd.DataFrame, ohlc: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    max_ohlc_by_ticker = ohlc.groupby("Ticker")["Date"].max().to_dict()
    rows = []
    for h in horizons:
        ret_col = f"ret_{h}d"
        label = HORIZON_LABELS.get(h, f"{h}d")
        eligible = 0
        missing = 0
        for _, r in df.iterrows():
            t = r.get("ticker_yf")
            sig_ts = pd.Timestamp(r.get("sig_ts")) if pd.notna(r.get("sig_ts")) else pd.NaT
            max_dt = max_ohlc_by_ticker.get(t)
            if max_dt is None or pd.isna(sig_ts):
                missing += 1
                continue
            needed = add_trading_days(sig_ts, h, get_trading_calendar())
            if needed <= max_dt:
                eligible += 1
            else:
                missing += 1
        realized = int(df[ret_col].notna().sum()) if ret_col in df.columns else 0
        rows.append({
            "horizon_days": h,
            "horizon_label": label,
            "eligible_signals_by_date": eligible,
            "realized_returns": realized,
            "missing_returns": missing,
        })
    return pd.DataFrame(rows)


def build_long_term_combo_ranking(df: pd.DataFrame, combo_mode: str, long_horizons: List[int]) -> pd.DataFrame:
    rows = []
    for h in long_horizons:
        ret_col = f"ret_{h}d"
        win_col = f"win_{h}d"
        if ret_col not in df.columns:
            continue
        rules = score_rule_grid(df, h, combo_mode)
        if rules.empty:
            continue
        # Recompute richer expectancy fields directly from filtered subsets
        base = df[df[ret_col].notna()].copy()
        if base.empty:
            continue
        top = rules.head(min(500, len(rules))).copy()
        for _, rr in top.iterrows():
            tier_allow = [x for x in str(rr["tier_allow"]).split(",") if x]
            m = base["tier"].isin(tier_allow)
            for col, op in [
                ("div_min", ">="), ("conf_min", ">="), ("willr_max", "<="),
                ("stoch_abs_max", "<="), ("rr_min", ">="), ("away_max", "<="),
                ("dist_lrc_max", "<="), ("ema_gap_min", ">=")
            ]:
                val = rr.get(col)
                if pd.isna(val):
                    continue
                if col == "div_min":
                    m &= base["div_v3"].fillna(-1) >= val
                elif col == "conf_min":
                    m &= base["conf"].fillna(-1) >= val
                elif col == "willr_max":
                    m &= base["willr"].notna() & (base["willr"] <= val)
                elif col == "stoch_abs_max":
                    m &= base["stoch_abs"].notna() & (base["stoch_abs"] <= val)
                elif col == "rr_min":
                    m &= base["rr"].notna() & (base["rr"] >= val)
                elif col == "away_max":
                    m &= base["away_pct"].notna() & (base["away_pct"] <= val)
                elif col == "dist_lrc_max":
                    m &= base["dist_to_lrc_pct"].notna() & (base["dist_to_lrc_pct"] <= val)
                elif col == "ema_gap_min":
                    m &= base["ema_gap_pct"].notna() & (base["ema_gap_pct"] >= val)
            sub = base[m]
            if len(sub) < MIN_SAMPLE_COMBO:
                continue
            rows.append({
                "focus_horizon_days": h,
                "focus_horizon_label": HORIZON_LABELS.get(h, f"{h}d"),
                "n": int(len(sub)),
                "win_rate": float(sub[win_col].mean()),
                "avg_ret": float(sub[ret_col].mean()),
                "median_ret": float(sub[ret_col].median()),
                "p25_ret": float(sub[ret_col].quantile(0.25)),
                "p75_ret": float(sub[ret_col].quantile(0.75)),
                "expectancy_proxy": float(sub[win_col].mean() * sub[ret_col].mean()),
                "score": float((sub[win_col].mean() * sub[ret_col].mean()) * np.log1p(len(sub))),
                "tier_allow": rr["tier_allow"],
                "div_min": rr["div_min"],
                "conf_min": rr["conf_min"],
                "willr_max": rr["willr_max"],
                "stoch_abs_max": rr["stoch_abs_max"],
                "rr_min": rr["rr_min"],
                "away_max": rr["away_max"],
                "dist_lrc_max": rr["dist_lrc_max"],
                "ema_gap_min": rr["ema_gap_min"],
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["focus_horizon_days", "score", "win_rate", "n"], ascending=[True, False, False, False]).reset_index(drop=True)
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to stock-info.txt")
    ap.add_argument("--out_dir", default="out_fixed", help="Output directory")
    ap.add_argument("--max_tickers", type=int, default=500, help="Safety limit")
    ap.add_argument("--ohlc_csv", default="", help="Optional cached OHLC CSV")
    ap.add_argument("--combo_mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--skip_combos", action="store_true")
    ap.add_argument("--horizons", default="1,3,5,10,20,40,60", help="Comma-separated trading-day horizons, for example 1,3,5,10,20,40,60")
    ap.add_argument("--long_term_only", action="store_true", help="Use only long-term horizons 20,40,60 trading days")
    args = ap.parse_args()

    if args.long_term_only:
        horizons = DEFAULT_LONG_TERM_HORIZONS.copy()
    else:
        horizons = []
        for token in str(args.horizons).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token)
            except ValueError:
                raise SystemExit(f"Invalid horizon value: {token}")
            if value <= 0:
                raise SystemExit(f"Horizon must be positive: {value}")
            horizons.append(value)
        if not horizons:
            raise SystemExit("No valid horizons provided.")
        horizons = sorted(set(horizons))

    df = parse_log_file(args.log)
    os.makedirs(args.out_dir, exist_ok=True)

    start = df["sig_date"].min()
    max_horizon = max(horizons)
    end = df["sig_date"].max() + dt.timedelta(days=max(20, int(max_horizon * 2.2)))
    tickers = sorted(df["ticker_yf"].unique().tolist())[:args.max_tickers]

    print(f"Parsed {len(df)} unique (date,ticker) signals across {len(tickers)} tickers.")
    print(f"Fetching OHLC from {start} to {end}...")

    if args.ohlc_csv:
        ohlc = pd.read_csv(args.ohlc_csv)
        ohlc["Date"] = pd.to_datetime(ohlc["Date"]).dt.tz_localize(None)
        if "Adj Close" not in ohlc.columns and "Close" in ohlc.columns:
            ohlc["Adj Close"] = ohlc["Close"]
    else:
        ohlc = fetch_ohlc_yfinance(tickers, start, end)
        ohlc.to_csv(os.path.join(args.out_dir, "ohlc_raw.csv"), index=False)

    df2 = attach_forward_returns(df, ohlc, horizons)
    df2.to_csv(os.path.join(args.out_dir, "signals_with_returns.csv"), index=False)

    qc_raw = df2["log_vs_mkt_close_diff_pct"].dropna()
    qc_adj = df2["adjusted_log_vs_mkt_close_diff_pct"].dropna()

    if len(qc_raw) > 0:
        print("Raw log-vs-market close diff (pct):")
        print(qc_raw.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())

    if len(qc_adj) > 0:
        print("\nAdjusted log-vs-market close diff (pct):")
        print(qc_adj.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())

    print(f"\nSnapped rows: {int(df2['snapped_to_close'].fillna(False).sum())} / {len(df2)}")
    print(f"Split-scaled rows: {int(df2['split_scaled'].fillna(False).sum())} / {len(df2)}")

    for h in horizons:
        metric_summary = summarize_metric_effects(df2, h)
        metric_path = os.path.join(args.out_dir, f"metric_effects_{HORIZON_LABELS.get(h, f"{h}d")}.csv")
        metric_summary.to_csv(metric_path, index=False)
        print(f"Wrote {metric_path} with {len(metric_summary)} rows.")

    if args.skip_combos:
        print("\nSkipped combo search.")
        print("Done.")
        return 0

    for h in horizons:
        rules = score_rule_grid(df2, h, args.combo_mode)
        rules_path = os.path.join(args.out_dir, f"combo_rules_{HORIZON_LABELS.get(h, f"{h}d")}.csv")
        rules.to_csv(rules_path, index=False)
        print(f"Wrote {rules_path} with {len(rules)} rows.")

        if not rules.empty:
            print(f"\nTop 10 combo rules for {h}d horizon:")
            cols = ["n", "win_rate", "avg_ret", "median_ret", "tier_allow", "div_min", "conf_min",
                    "willr_max", "stoch_abs_max", "rr_min", "away_max", "dist_lrc_max", "ema_gap_min"]
            print(rules[cols].head(10).to_string(index=False))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())