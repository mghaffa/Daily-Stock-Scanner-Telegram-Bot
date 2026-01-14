#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scanner_dual_tf_vp_spyder_telegram_loop_dip7.py

Adds:
- Linear Regression Channel (LRC) on 1D (regression + residual stdev band)
- Strong BUY trigger: price touches LOWER LRC band (with ATR tolerance)

Keeps:
- Your existing core gates (trend/liquidity/rr + supporting gates + strict regime/AVWAP/trigger)

EDU only, not financial advice.
"""

from __future__ import annotations
import os, sys, json, time, re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict
from datetime import datetime, time as dtime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===================== USER TOGGLES (env-driven) ===================== #
def _env_bool(name, default):
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1","true","yes","on","y")

def _env_float(name, default):
    try: return float(os.getenv(name, str(default)))
    except: return float(default)

def _env_int(name, default):
    try: return int(float(os.getenv(name, str(default))))
    except: return int(default)

IDE_ENABLE_1D = True
IDE_ENABLE_4H = _env_bool("IDE_ENABLE_4H", True)
IDE_SORT_TF_FIRST = "1D"
IDE_PRINT_TITLE = True

IDE_LOOP = False
IDE_INTERVAL_MIN = 30

# Market-hours gate
MARKET_ONLY  = _env_bool("MARKET_ONLY", True)
MARKET_TZ    = "America/New_York"
MARKET_START = dtime(9, 30)
MARKET_END   = dtime(16, 0)

# Messaging
WEBHOOK_URL            = os.getenv("WEBHOOK_URL", "").strip()
TELEGRAM_BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_CHAT_IDS      = os.getenv("TELEGRAM_CHAT_IDS", "").strip()
TELEGRAM_AUTO_DISCOVER = True

# Banner / prefix handling
NOTIFY_PREFIX = os.getenv("NOTIFY_PREFIX", "")
PROFILE_EFFECTIVE = os.getenv("PROFILE_EFFECTIVE", "").strip().lower()
FORCE_PREFIX_FROM_PROFILE = _env_bool("FORCE_PREFIX_FROM_PROFILE", False)

NOTIFY_PREFIX = os.getenv("NOTIFY_PREFIX", "")
ALWAYS_NOTIFY          = _env_bool("ALWAYS_NOTIFY", False)
TELEGRAM_PING_ON_START = _env_bool("TELEGRAM_PING_ON_START", False)
TELEGRAM_PING_ON_END   = _env_bool("TELEGRAM_PING_ON_END", False)

def _computed_prefix_from_profile(profile: str) -> str:
    mode_map = {
        "balanced (core)": "Balanced",
        "conservative (strict)": "Conservative",
        "aggressive (loose)": "Aggressive",
    }
    mode = mode_map.get((profile or "balanced (core)").lower(), "Balanced")
    return f"[ Enhanced Version V4_ {mode} Mode ] "

if FORCE_PREFIX_FROM_PROFILE or not NOTIFY_PREFIX:
    NOTIFY_PREFIX = _computed_prefix_from_profile(PROFILE_EFFECTIVE)

# Universe
RAW_TICKERS = ["nvda","amd","orcl","avgo","pltr","net","amzn","googl","msft","klac","ibm","aapl",
               "tqqq","intc","bulz","cost","tsla","meta","now","nflx","hims","ntra","ddog","tsm",
               "mu","crm","tem","rklb","crwd","uvxy","unh","jpm","abt","bynd","race","sofi","dell","upst","sofi",
               "gld", "gldm","shny","msci","ccj","shop","ionq","regti","qbts","qtum","qubt","laes","arqq","holo"]
ALIAS = {"google":"GOOGL"}

# --- Preset-driven risk/filters ---
MIN_DOLLAR_VOL = _env_float("MIN_20D_DOLLAR_VOL", os.getenv("MIN_DOLLAR_VOL", "10000000"))

REGIME_TICKER = os.getenv("REGIME_SYMBOL", os.getenv("REGIME_TICKER", "SPY"))
REGIME_STRICT = _env_bool("REGIME_DOWNGRADE_TO_WATCH", os.getenv("REGIME_STRICT", "true"))

# Sweet-spot (inner band) around EMA50
SWEET_ATR_LOW  = _env_float("BUY_LOWER_BAND_ATR_MULT", os.getenv("SWEET_ATR_LOW",  "0.5"))
SWEET_ATR_HIGH = _env_float("BUY_UPPER_BAND_ATR_MULT", os.getenv("SWEET_ATR_HIGH", "0.25"))
SOFT_SWEET_PAD_ATR = _env_float("SOFT_SWEET_PAD_ATR", 0.25)

VOL_UP_LOOKBACK = _env_int("VOL_UP_LOOKBACK", 10)
VOL_UP_MULT     = _env_float("VOL_UP_MULT", 1.2)

# 4H AVWAP gate and its tolerance (in 4H ATRs)
USE_AVWAP_4W_4H = _env_bool("ENFORCE_4H_ANCHORED_VWAP", os.getenv("USE_AVWAP_4W_4H", "true"))
AVWAP_TOL_ATR   = _env_float("AVWAP_TOL_ATR", 0.25)

# Easier price trigger toggle
PRICE_TRIGGER_EASED = _env_bool("PRICE_TRIGGER_EASED", True)

# R/R and divergence
RR_MIN  = _env_float("MIN_RR", os.getenv("RR_MIN", "1.5"))
DIV_MIN = _env_int("DIV_MIN", 2)  # bonus only now

# New: how many supporting gates must pass (sweet, volume, momentum)
SUPPORT_MIN_PASS = _env_int("SUPPORT_MIN_PASS", 2)

DEBUG_DETAIL = _env_bool("DEBUG_DETAIL", False)

# Periods
DAILY_PERIOD="max"; DAILY_INTERVAL="1d"
INTRA_PERIOD="90d"; INTRA_INTERVAL="60m"

LB=5; RB=5

# ===================== NEW: LINEAR REGRESSION CHANNEL (LRC) ===================== #
LRC_ENABLE         = _env_bool("LRC_ENABLE", True)
LRC_LEN            = _env_int("LRC_LEN", 100)
LRC_DEVLEN         = _env_float("LRC_DEVLEN", 2.0)
LRC_TOUCH_USE_LOW  = _env_bool("LRC_TOUCH_USE_LOW", True)
LRC_TOUCH_TOL_ATR  = _env_float("LRC_TOUCH_TOL_ATR", 0.10)  # tolerance in ATR units

TITLE_TEXT = (
    "Scanner (dip7): Trend + Sweet-spot + Volume + Momentum + R/R + optional 4H AVWAP\n"
    f"- Primary: Trend + Liquidity + R/R. Supporting (need ≥{SUPPORT_MIN_PASS}): Sweet, Volume, Momentum.\n"
    f"- Sweet inner band: [EMA50-{SWEET_ATR_LOW:.2f}*ATR, EMA50+{SWEET_ATR_HIGH:.2f}*ATR]; soft pad ±{SOFT_SWEET_PAD_ATR:.2f}*ATR.\n"
    f"- LRC: len={LRC_LEN}, dev={LRC_DEVLEN}, touch_use_low={LRC_TOUCH_USE_LOW}, tol={LRC_TOUCH_TOL_ATR} ATR.\n"
)

print("[cfg] IDE_ENABLE_4H=", IDE_ENABLE_4H)
print("[cfg] REGIME_TICKER=", REGIME_TICKER, "REGIME_STRICT=", REGIME_STRICT)
print("[cfg] MIN_DOLLAR_VOL=", MIN_DOLLAR_VOL)
print("[cfg] SWEET_ATR_LOW/HIGH=", SWEET_ATR_LOW, SWEET_ATR_HIGH, "SOFT_PAD_ATR=", SOFT_SWEET_PAD_ATR)
print("[cfg] VOL_UP_LOOKBACK/MULT=", VOL_UP_LOOKBACK, VOL_UP_MULT)
print("[cfg] USE_AVWAP_4W_4H=", USE_AVWAP_4W_4H, "AVWAP_TOL_ATR=", AVWAP_TOL_ATR)
print("[cfg] RR_MIN=", RR_MIN, "DIV_MIN=", DIV_MIN, "SUPPORT_MIN_PASS=", SUPPORT_MIN_PASS)
print("[cfg] PRICE_TRIGGER_EASED=", PRICE_TRIGGER_EASED)
print("[cfg] LRC_ENABLE=", LRC_ENABLE, "LRC_LEN=", LRC_LEN, "LRC_DEVLEN=", LRC_DEVLEN,
      "LRC_TOUCH_USE_LOW=", LRC_TOUCH_USE_LOW, "LRC_TOUCH_TOL_ATR=", LRC_TOUCH_TOL_ATR)
print("[cfg] DEBUG_DETAIL=", DEBUG_DETAIL)
print("[cfg] ALWAYS_NOTIFY=", ALWAYS_NOTIFY, "NOTIFY_PREFIX=", NOTIFY_PREFIX)

# ===================== UTIL / FETCH ===================== #
def _squeeze_col(x):
    if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
        return x.iloc[:,0]
    return x

def _ensure_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        sliced = None
        for level in range(df.columns.nlevels):
            try:
                sliced = df.xs(ticker, axis=1, level=level)
                break
            except Exception:
                continue
        df = sliced if sliced is not None else df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [''.join([str(x) for x in tup if str(x)!='']) for x in df.columns.to_flat_index()]
    for col in list(df.columns):
        try: df[col] = _squeeze_col(df[col])
        except Exception: pass
    rename = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename)
    need = ["Open","High","Low","Close","Volume"]
    alt = {c.lower(): c for c in df.columns}
    fixed = {}
    for r in need:
        if r in df.columns:
            fixed[r] = r
        elif r.lower() in alt:
            fixed[r] = alt[r.lower()]
        else:
            cand = [c for c in df.columns if r.lower() in c.lower()]
            if not cand: raise ValueError(f"Missing column {r}")
            fixed[r] = cand[0]
    df = df.rename(columns={fixed[k]:k for k in fixed})
    return df[["Open","High","Low","Close","Volume"]]

def _retry_download(*, ticker, period, interval, **kwargs):
    for _ in range(3):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False, group_by="column", **kwargs)
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(1.0)
    return pd.DataFrame()

def fetch_daily(ticker: str) -> pd.DataFrame:
    df = _retry_download(ticker=ticker, period=DAILY_PERIOD, interval=DAILY_INTERVAL)
    if df.empty or len(df)<250: raise ValueError("insufficient daily data")
    return _ensure_ohlcv(df, ticker)

def fetch_4h(ticker: str) -> pd.DataFrame:
    df = _retry_download(ticker=ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL, prepost=False)
    if df.empty or len(df)<100: raise ValueError("insufficient intraday data")
    df = _ensure_ohlcv(df, ticker)
    if getattr(df.index, "tz", None) is not None:
        df = df.tz_localize(None)
    return df.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna(how="any")

# ===================== INDICATORS ===================== #
def ema(s: pd.Series, n: int): s=_squeeze_col(s); return s.ewm(span=n, adjust=False).mean()
def sma(s: pd.Series, n: int): s=_squeeze_col(s); return s.rolling(n, min_periods=n).mean()

def true_range(h,l,c):
    h=_squeeze_col(h); l=_squeeze_col(l); c=_squeeze_col(c); pc=c.shift(1)
    return pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)

def atr(h,l,c,n=14): return true_range(h,l,c).ewm(alpha=1/n, adjust=False).mean()

def williams_r(h,l,c,n=14):
    h=_squeeze_col(h); l=_squeeze_col(l); c=_squeeze_col(c)
    hh=h.rolling(n).max(); ll=l.rolling(n).min()
    return -100*((hh-c)/(hh-ll))

def rsi(c,n=14):
    c=_squeeze_col(c); d=c.diff()
    up=d.clip(lower=0.0); dn=-d.clip(upper=0.0)
    ag=up.ewm(alpha=1/n, adjust=False).mean(); al=dn.ewm(alpha=1/n, adjust=False).mean()
    rs=ag/al.replace(0,np.nan); return 100-(100/(1+rs))

def aroon_up_down(h,l,n=14):
    h=_squeeze_col(h); l=_squeeze_col(l)
    up  = 100 * (h.rolling(n).apply(lambda x: (n-1)-np.argmax(x[::-1]), raw=True)) / n
    dn  = 100 * (l.rolling(n).apply(lambda x: (n-1)-np.argmin(x[::-1]), raw=True)) / n
    return up, dn

def adx_plus_minus(h,l,c,n=14):
    h=_squeeze_col(h); l=_squeeze_col(l); c=_squeeze_col(c)
    up=h.diff(); dn=-l.diff()
    plus_dm=np.where((up>dn)&(up>0),up,0.0); minus_dm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=true_range(h,l,c); atr_=tr.ewm(alpha=1/n, adjust=False).mean().replace(0,np.nan)
    plus_di=100*pd.Series(plus_dm,index=h.index).ewm(alpha=1/n, adjust=False).mean()/atr_
    minus_di=100*pd.Series(minus_dm,index=l.index).ewm(alpha=1/n, adjust=False).mean()/atr_
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di)).replace([np.inf,-np.inf],np.nan)
    adx=dx.ewm(alpha=1/n, adjust=False).mean()
    return adx, plus_di, minus_di

def macd(c, fast=12, slow=26, signal=9):
    c=_squeeze_col(c); f=ema(c,fast); s=ema(c,slow)
    line=f-s; sig=line.ewm(span=signal, adjust=False).mean(); hist=line-sig
    return line, sig, hist

def ema_lr_slope_up(series: pd.Series, window: int = 10) -> bool:
    s = _squeeze_col(series).dropna()
    if s.size < window: return False
    x = np.arange(window, dtype=float)
    y = s.tail(window).values
    m, _ = np.polyfit(x, y, 1)
    return bool(m > 0)

def anchored_vwap_last(df: pd.DataFrame, days: int = 28) -> float:
    if df.empty: return np.nan
    end = df.index[-1]
    anchor = end - timedelta(days=days)
    w = df.loc[df.index >= anchor]
    if w.empty: return np.nan
    tp = (w["High"] + w["Low"] + w["Close"]) / 3.0
    v  = w["Volume"].replace(0, np.nan)
    return float((tp*v).sum() / v.sum())

def compute(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    o["EMA20"] = ema(o["Close"], 20)
    o["EMA50"] = ema(o["Close"], 50)
    o["EMA200"] = ema(o["Close"], 200)
    o["ATR14"] = atr(o["High"], o["Low"], o["Close"], 14)
    o["WILLR"] = williams_r(o["High"], o["Low"], o["Close"], 14)
    o["RSI14"] = rsi(o["Close"], 14)
    adx, pdi, ndi = adx_plus_minus(o["High"], o["Low"], o["Close"], 14)
    o["ADX"], o["+DI"], o["-DI"] = adx, pdi, ndi
    up, dn = aroon_up_down(o["High"], o["Low"], 14)
    o["AROON_UP"], o["AROON_DOWN"] = up, dn
    o["VOL20"] = o["Volume"].rolling(20).mean()
    o["ATR_PCT"] = (o["ATR14"] / o["Close"]) * 100
    return o.dropna()

# ---------- Divergences (current simple) ---------- #
def pivot_points(series: pd.Series, lb: int, rb: int):
    s=_squeeze_col(series)
    highs, lows = [], []
    vals=s.values; n=len(vals)
    for i in range(lb, n-rb):
        w=vals[i-lb:i+rb+1]; c=vals[i]
        if np.isfinite(c):
            if c==np.nanmax(w): highs.append(i)
            if c==np.nanmin(w): lows.append(i)
    return highs, lows

def bullish_div(price: pd.Series, indi: pd.Series, lb: int, rb: int):
    p=_squeeze_col(price); q=_squeeze_col(indi)
    _, lows = pivot_points(p, lb, rb)
    if len(lows) < 2: return False
    i1, i2 = lows[-2], lows[-1]
    return (p.iloc[i2] < p.iloc[i1]) and (q.iloc[i2] > q.iloc[i1])

def divergence_score(df: pd.DataFrame, lb: int, rb: int):
    inds = {"RSI": df["RSI14"], "MACD": macd(df["Close"])[2], "WILLR": df["WILLR"]}
    price = df["Close"]; names=[]
    for k, s in inds.items():
        try:
            if bullish_div(price, s, lb, rb): names.append(k)
        except Exception: pass
    return len(names), names

# ===================== NEW: Linear Regression Channel ===================== #
def linear_regression_channel(series: pd.Series, length: int, dev_mult: float) -> Tuple[float,float,float,float,float]:
    """
    Returns (mid_last, upper_last, lower_last, slope, stdev_resid)
    using least-squares fit on last `length` points.
    """
    s = _squeeze_col(series).dropna()
    if s.size < length:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    y = s.tail(length).to_numpy(dtype=float)
    x = np.arange(length, dtype=float)  # 0..len-1
    slope, intercept = np.polyfit(x, y, 1)  # y = slope*x + intercept

    y_hat = slope * x + intercept
    resid = y - y_hat
    stdev = float(np.sqrt(np.mean(resid**2)))  # matches Pine's sqrt(dev/len) style

    mid_last = float(y_hat[-1])
    upper_last = float(mid_last + dev_mult * stdev)
    lower_last = float(mid_last - dev_mult * stdev)
    return mid_last, upper_last, lower_last, float(slope), stdev

def lrc_touch_ok(o: pd.DataFrame) -> Tuple[bool, str, float, float, float, float]:
    """
    Touch means price hits the LOWER channel line (optionally via Low) with tolerance.
    Returns: (ok, msg, mid, upper, lower, slope)
    """
    if (not LRC_ENABLE) or (len(o) < max(220, LRC_LEN+5)):
        return True, "LRC disabled/skip", np.nan, np.nan, np.nan, np.nan

    mid, upper, lower, slope, stdev = linear_regression_channel(o["Close"], LRC_LEN, LRC_DEVLEN)
    if not np.isfinite(lower):
        return False, "LRC n/a", mid, upper, lower, slope

    r = o.iloc[-1]
    tol = float(LRC_TOUCH_TOL_ATR * r.ATR14) if np.isfinite(r.ATR14) else 0.0

    probe = float(r.Low) if LRC_TOUCH_USE_LOW else float(r.Close)
    touched = (probe <= (lower + tol))

    msg = f"LRC touch {'OK' if touched else 'no'} (probe={probe:.2f} <= lower={lower:.2f}+tol={tol:.2f})"
    return bool(touched), msg, mid, upper, lower, slope

# ---------- Volume Profile ---------- #
def volume_profile(close: pd.Series, volume: pd.Series, lookback=180, bins=120):
    close=_squeeze_col(close).tail(lookback); volume=_squeeze_col(volume).reindex(close.index)
    pmin=float(close.min()); pmax=float(close.max())
    if not (np.isfinite(pmin) and np.isfinite(pmax)) or pmax<=pmin:
        return None, None, None, None, None
    edges=np.linspace(pmin, pmax, bins+1)
    centers=(edges[:-1]+edges[1:])/2.0
    idx=np.clip(np.digitize(close.values, edges)-1, 0, bins-1)
    vol_by_bin=np.bincount(idx, weights=volume.values, minlength=bins)
    poc_idx=int(np.argmax(vol_by_bin)); poc=float(centers[poc_idx])
    thresh_hi=0.7*vol_by_bin.max(); thresh_lo=0.3*vol_by_bin.max()
    cur=float(close.iloc[-1])
    hvn_below=lvn_below=None
    hv_inds=np.where(vol_by_bin>=thresh_hi)[0]
    if hv_inds.size:
        below=hv_inds[centers[hv_inds] <= cur]
        if below.size: hvn_below=float(centers[below[-1]])
    lv_inds=np.where(vol_by_bin<=thresh_lo)[0]
    if lv_inds.size:
        below=lv_inds[centers[lv_inds] <= cur]
        if below.size: lvn_below=float(centers[below[-1]])
    return poc, hvn_below, lvn_below, centers, vol_by_bin

# ===================== RULES ===================== #
def long_trend_ok(o: pd.DataFrame) -> Tuple[bool, List[str]]:
    r=o.iloc[-1]
    ema50_up = ema_lr_slope_up(o["EMA50"], 10)
    ema200_flat_up = o["EMA200"].iloc[-1] >= o["EMA200"].iloc[-5]
    above_200 = r.Close > r.EMA200
    ok = ema50_up and ema200_flat_up and above_200
    reasons=[]
    reasons.append("EMA50 LR-slope>0" if ema50_up else "EMA50 slope DOWN")
    reasons.append("EMA200 not falling" if ema200_flat_up else "EMA200 falling")
    reasons.append(">EMA200" if above_200 else "<=EMA200")
    return ok, reasons

def volume_accum_ok(o: pd.DataFrame, lookback=10, mult=1.2) -> Tuple[bool, str]:
    w=o.tail(lookback+1)
    if len(w)<lookback+1: return False, "vol check: not enough bars"
    up_mask = (w["Close"].diff() > 0)
    heavy = w["Volume"] >= (mult * w["VOL20"].iloc[-1])
    up_vol = (w["Volume"].where(up_mask, 0)).tail(lookback).sum()
    dn_vol = (w["Volume"].where(~up_mask, 0)).tail(lookback).sum()
    dom_ok = (up_vol >= 0.9 * max(dn_vol, 1e-9))
    ok = bool((up_mask & heavy).tail(lookback).any() or dom_ok)
    return ok, ("vol accum OK" if ok else f"no up-day ≥{mult:.1f}×20d and up-vol<{0.9:.0%} down-vol")

def sweet_spot_ok(o: pd.DataFrame) -> Tuple[bool, str, float, float]:
    r=o.iloc[-1]
    inner_lo = r.EMA50 - SWEET_ATR_LOW  * r.ATR14
    inner_hi = r.EMA50 + SWEET_ATR_HIGH * r.ATR14
    outer_lo = inner_lo - SOFT_SWEET_PAD_ATR * r.ATR14
    outer_hi = inner_hi + SOFT_SWEET_PAD_ATR * r.ATR14
    if inner_lo <= r.Close <= inner_hi:
        return True, f"sweet-band(inner) {inner_lo:.2f}..{inner_hi:.2f}", inner_lo, inner_hi
    ok = (outer_lo <= r.Close <= outer_hi)
    return ok, f"sweet-band(soft) {outer_lo:.2f}..{outer_hi:.2f}", inner_lo, inner_hi

def momentum_turn_ok(o: pd.DataFrame) -> Tuple[bool, List[str], int]:
    r=o.iloc[-1]
    wr_cross = (o["WILLR"].iloc[-2] <= -80) and (r.WILLR > -80)
    pdi_now, ndi_now = r["+DI"], r["-DI"]
    di_cross = (pdi_now > ndi_now) and (o["+DI"].iloc[-2] <= o["-DI"].iloc[-2])
    aup, adn = r.AROON_UP, r.AROON_DOWN
    aroon_ok = ((aup > 70 and adn < 30) or (o["AROON_UP"].iloc[-2] <= o["AROON_DOWN"].iloc[-2] and aup > adn))
    rsi_up = (o["RSI14"].iloc[-1] >= 45) and (o["RSI14"].iloc[-1] > o["RSI14"].iloc[-2])
    _, _, macd_hist = macd(o["Close"])
    macd_up = macd_hist.iloc[-1] > macd_hist.iloc[-2]
    votes = [wr_cross, di_cross, aroon_ok, rsi_up, macd_up]
    ok = (sum(bool(v) for v in votes) >= 2)
    reasons=[]
    reasons.append("W%R cross -80" if wr_cross else "W%R not crossed")
    reasons.append("+DI cross up" if di_cross else "+DI not > -DI")
    reasons.append("Aroon dom/flip" if aroon_ok else "Aroon weak")
    reasons.append("RSI rising≥45" if rsi_up else "RSI not rising/low")
    reasons.append("MACD hist ↑" if macd_up else "MACD hist ↓/flat")
    return ok, reasons, int(sum(bool(v) for v in votes))

def divergence_ok(o: pd.DataFrame, min_count=2) -> Tuple[bool, str, int, List[str]]:
    cnt, names = divergence_score(o, LB, RB)
    ok = cnt >= min_count
    return ok, f"div {cnt}/{min_count}", cnt, names

def risk_reward(o: pd.DataFrame, hvn_below: float|None, lvn_below: float|None) -> Tuple[bool, str, float, float, float]:
    r=o.iloc[-1]
    stop = min((lvn_below if lvn_below is not None else r.EMA50 - r.ATR14), r.Low)
    risk = max(r.Close - stop, 1e-6)
    swing_high = float(o["Close"].rolling(20).max().iloc[-2]) if len(o)>=22 else r.Close + 2*r.ATR14
    tgt   = max(swing_high, r.EMA50 + 2*r.ATR14)
    rr = (tgt - r.Close) / risk
    ok = rr >= RR_MIN
    return ok, f"R/R {rr:.2f} (≥{RR_MIN})", float(stop), float(tgt), rr

def liquidity_ok(o: pd.DataFrame) -> Tuple[bool, str, float]:
    r=o.iloc[-1]
    dollar = float(r.VOL20 * r.Close)
    ok = dollar >= MIN_DOLLAR_VOL
    return ok, f"$vol20={dollar/1e6:.1f}M", dollar

def regime_ok() -> Tuple[bool, List[str]]:
    try:
        d = fetch_daily(REGIME_TICKER)
        o = compute(d)
        r = o.iloc[-1]
        cond1 = r.Close > r.EMA200
        cond2 = ema_lr_slope_up(o["EMA50"], 10)
        ok = cond1 and cond2
        rs=[]
        rs.append("regime >200EMA" if cond1 else "regime <=200EMA")
        rs.append("regime EMA50 slope up" if cond2 else "regime EMA50 slope down")
        return ok, rs
    except Exception as e:
        return True, [f"regime check skipped: {e}"]

def suggest_levels(o: pd.DataFrame, lookback=180, bins=120):
    r=o.iloc[-1]
    poc, hvn_below, lvn_below, centers, vols = volume_profile(o["Close"], o["Volume"], lookback, bins)
    lo = r.EMA50 - SWEET_ATR_LOW  * r.ATR14
    hi = r.EMA50 + SWEET_ATR_HIGH * r.ATR14

    if hvn_below is None:
        hvn_below = (poc if (poc is not None and poc <= float(r.Close)) else float(r.EMA50))
    if lvn_below is None:
        lvn_below = float(r.EMA50 - r.ATR14)

    base = max(lo, float(hvn_below))
    sug_buy  = round(float(np.clip(base, lo, hi + SOFT_SWEET_PAD_ATR*r.ATR14)), 2)
    sug_stop = round(float(min(float(lvn_below), float(r.Low))), 2)
    return sug_buy, sug_stop, (None if poc is None else float(poc)), float(hvn_below), float(lvn_below), (lo, hi)

# ===================== I/O & TELEGRAM ===================== #
def telegram_get_updates(token: str) -> Optional[dict]:
    try:
        return requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=8).json()
    except Exception:
        return None

def telegram_discover_chat_id(token: str) -> str:
    data = telegram_get_updates(token)
    if not data or not data.get("ok"): return ""
    for upd in reversed(data.get("result", [])):
        for key in ("message","channel_post","edited_message","my_chat_member"):
            if key in upd and "chat" in upd[key]:
                cid = upd[key]["chat"].get("id")
                if cid is not None: return str(cid)
    return ""

def _parse_ids(s: str):
    if not s: return []
    return [p for p in re.split(r"[,\s]+", s) if p]

def notify_all(text: str):
    if NOTIFY_PREFIX:
        text = f"{NOTIFY_PREFIX}{text}"

    if WEBHOOK_URL:
        try: requests.post(WEBHOOK_URL, json={"text": text}, timeout=7)
        except Exception: pass
    if not TELEGRAM_BOT_TOKEN: return
    ids = []
    ids += _parse_ids(TELEGRAM_CHAT_IDS)
    if TELEGRAM_CHAT_ID: ids.append(TELEGRAM_CHAT_ID)
    if not ids and TELEGRAM_AUTO_DISCOVER:
        auto = telegram_discover_chat_id(TELEGRAM_BOT_TOKEN)
        if auto: ids = [auto]
    for cid in ids:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data={"chat_id": cid, "text": text}, timeout=12, verify=False
            )
        except Exception:
            pass

# ===================== CORE ===================== #
@dataclass
class Row:
    ticker: str; tf: str; date: str; status: str; close: float; sug_buy: float; sug_stop: float
    stop: float; target: float; rr: float; ema50: float; ema200: float; adx: float
    aroon_up: float; aroon_dn: float; hvn_below: float|None; lvn_below: float|None; poc: float|None
    div_cnt: int
    # NEW: LRC fields
    lrc_mid: float|None; lrc_upper: float|None; lrc_lower: float|None; lrc_slope: float|None
    reasons: str; conf: int; error: str
    # debug flags
    trend_ok: Optional[bool] = None; liq_ok: Optional[bool] = None; vol_ok: Optional[bool] = None
    sweet_ok: Optional[bool] = None; momo_ok: Optional[bool] = None; div_ok: Optional[bool] = None
    rr_ok: Optional[bool] = None; regime_ok: Optional[bool] = None; avwap4h_ok: Optional[bool] = None
    trigger_ok: Optional[bool] = None
    lrc_touch_ok: Optional[bool] = None
    fail_primary: Optional[str] = None
    fail_secondary: Optional[str] = None

def _parse_universe() -> List[str]:
    env = (os.getenv("TICKERS","") or "").strip()
    if env:
        toks = [t.strip().upper() for t in re.split(r"[,\s;|]+", env) if t.strip()]
        return sorted(set(toks))
    return sorted({ALIAS.get(t.lower(), t.upper()) for t in RAW_TICKERS})

def confidence_score(flags: Dict[str,bool]) -> int:
    order = ["trend","volume","sweet","momentum","lrc_touch","div","liquidity","rr","regime","avwap4h","price_trigger"]
    return int(min(100, sum(10 for k in order if flags.get(k, False))))

def _bullish_candle(o: pd.DataFrame) -> bool:
    r, p = o.iloc[-1], o.iloc[-2]
    engulf = (r.Close > r.Open) and (p.Close < p.Open) and (r.Close >= p.Open) and (r.Open <= p.Close)
    body = abs(r.Close - r.Open)
    range_ = r.High - r.Low
    lower_wick = (r.Open - r.Low) if (r.Close >= r.Open) else (r.Close - r.Low)
    hammer = (r.Close > r.Open) and (range_ > 0) and (lower_wick / max(range_, 1e-9) >= 0.4) and (body / max(range_,1e-9) <= 0.6)
    return bool(engulf or hammer)

def price_trigger_ok(o: pd.DataFrame) -> Tuple[bool, str]:
    r, p = o.iloc[-1], o.iloc[-2]
    trig1 = r.Close > p.High
    trig2 = r.Close > r.EMA20
    trig3 = _bullish_candle(o)
    if PRICE_TRIGGER_EASED:
        ok = trig1 or trig2 or trig3
        why = "trigger ok (prevHigh/EMA20/bullishCandle)" if ok else "trigger pending"
    else:
        ok = trig1
        why = "trigger ok (prevHigh)" if ok else "trigger pending"
    return ok, why

def process_one(ticker: str, tf: str, df: pd.DataFrame, vp_lookback=180) -> Row:
    o = compute(df)
    if o.empty or len(o) < 210: raise ValueError("indicator frame short")

    ok_trend, t_reasons = long_trend_ok(o)
    ok_liq, liq_msg, dollar = liquidity_ok(o)
    ok_vol, vol_msg = volume_accum_ok(o, VOL_UP_LOOKBACK, VOL_UP_MULT)
    ok_sweet, sweet_msg, band_lo, band_hi = sweet_spot_ok(o)

    ok_momo, m_reasons, momo_votes = momentum_turn_ok(o)
    ok_div, div_msg, div_cnt, div_names = divergence_ok(o, DIV_MIN)

    # NEW: LRC touch gate
    ok_lrc_touch, lrc_msg, lrc_mid, lrc_upper, lrc_lower, lrc_slope = lrc_touch_ok(o)

    sug_buy, sug_stop, poc, hvn_b, lvn_b, (inner_lo, inner_hi) = suggest_levels(o, vp_lookback, 120)
    ok_rr, rr_msg, stop, target, rr = risk_reward(o, hvn_b, lvn_b)

    ok_regime, regime_msgs = regime_ok()

    avwap4h_ok = True
    if IDE_ENABLE_4H and USE_AVWAP_4W_4H:
        try:
            h4 = compute(fetch_4h(ticker))
            av = anchored_vwap_last(h4, 28)
            if np.isfinite(av):
                atr4 = float(h4["ATR14"].iloc[-1])
                close4 = float(h4["Close"].iloc[-1])
                avwap4h_ok = (close4 >= av - AVWAP_TOL_ATR * atr4)
            else:
                avwap4h_ok = True
        except Exception:
            avwap4h_ok = True

    trig_ok, trig_msg = price_trigger_ok(o)

    flags = {
        "trend": ok_trend, "liquidity": ok_liq, "volume": ok_vol, "sweet": ok_sweet,
        "momentum": ok_momo, "div": ok_div, "rr": ok_rr, "regime": ok_regime,
        "avwap4h": avwap4h_ok, "price_trigger": trig_ok,
        "lrc_touch": ok_lrc_touch
    }
    conf = confidence_score(flags)

    # Original gate: must-have + supporting >= N
    must_names = ["trend", "liquidity", "rr"]
    support_names = ["sweet", "volume", "momentum"]
    support_pass = sum(1 for n in support_names if flags.get(n, False))
    must_ok = all(flags[n] for n in must_names)
    primary_ok = must_ok and (support_pass >= SUPPORT_MIN_PASS)

    # Status logic:
    # WATCH stays based on primary_ok (as before)
    # BUY requires LRC-touch + strict gates (if REGIME_STRICT) or primary + trigger (if not strict)
    if REGIME_STRICT:
        strict_ok = primary_ok and flags["regime"] and flags["avwap4h"] and flags["price_trigger"]
        status = "BUY" if (strict_ok and flags["lrc_touch"]) else ("WATCH" if primary_ok else "")
    else:
        # still require trigger, and now require LRC touch for BUY
        status = "BUY" if (primary_ok and flags["price_trigger"] and flags["lrc_touch"]) else ("WATCH" if primary_ok else "")

    reasons = []
    reasons += t_reasons
    reasons += [liq_msg, vol_msg, sweet_msg]
    reasons += m_reasons + [f"momo votes={momo_votes}", div_msg, rr_msg]
    reasons += regime_msgs
    if IDE_ENABLE_4H and USE_AVWAP_4W_4H: reasons.append("4H AVWAP ok" if avwap4h_ok else "4H AVWAP fail")
    reasons.append(trig_msg)
    reasons.append(lrc_msg)
    reasons.append(f"supports_passed={support_pass}/{SUPPORT_MIN_PASS}")

    r = o.iloc[-1]
    return Row(
        ticker=ticker, tf=tf, date=str(r.name.date()), status=status,
        close=round(float(r.Close),2), sug_buy=sug_buy, sug_stop=sug_stop,
        stop=round(float(stop),2), target=round(float(target),2), rr=round(float(rr),2),
        ema50=round(float(r.EMA50),2), ema200=round(float(r.EMA200),2),
        adx=round(float(o.ADX.dropna().iloc[-1]),1) if o.ADX.dropna().size else np.nan,
        aroon_up=int(o.AROON_UP.dropna().iloc[-1]) if o.AROON_UP.dropna().size else np.nan,
        aroon_dn=int(o.AROON_DOWN.dropna().iloc[-1]) if o.AROON_DOWN.dropna().size else np.nan,
        hvn_below=(None if hvn_b is None else round(float(hvn_b),2)),
        lvn_below=(None if lvn_b is None else round(float(lvn_b),2)),
        poc=(None if poc is None else round(float(poc),2)),
        div_cnt=div_cnt,
        lrc_mid=(None if not np.isfinite(lrc_mid) else round(float(lrc_mid),2)),
        lrc_upper=(None if not np.isfinite(lrc_upper) else round(float(lrc_upper),2)),
        lrc_lower=(None if not np.isfinite(lrc_lower) else round(float(lrc_lower),2)),
        lrc_slope=(None if not np.isfinite(lrc_slope) else round(float(lrc_slope),6)),
        reasons=" | ".join(reasons), conf=conf, error="",
        trend_ok=flags["trend"], liq_ok=flags["liquidity"], vol_ok=flags["volume"], sweet_ok=flags["sweet"],
        momo_ok=flags["momentum"], div_ok=flags["div"], rr_ok=flags["rr"], regime_ok=flags["regime"],
        avwap4h_ok=flags["avwap4h"], trigger_ok=flags["price_trigger"],
        lrc_touch_ok=flags["lrc_touch"],
        fail_primary=(", ".join([n for n in must_names if not flags.get(n, False)]) or ""),
        fail_secondary=(", ".join([n for n in support_names if not flags.get(n, False)]) or "")
    )

def build_dataframe() -> pd.DataFrame:
    rows: List[Row] = []
    tickers = _parse_universe()
    for t in tickers:
        try:
            ddf = fetch_daily(t)
            rows.append(process_one(t, "1D", ddf, 180))
        except Exception as e:
            rows.append(Row(t, "1D", "", "", np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                            np.nan,np.nan,np.nan,None,None,None,0,
                            None,None,None,None,
                            "",0, str(e)))
    df = pd.DataFrame([r.__dict__ for r in rows])
    df["tf"] = pd.Categorical(df["tf"], categories=["1D"], ordered=True)
    return df.sort_values(["tf","ticker"]).reset_index(drop=True)

# -------- de-dup state for alerts -------- #
CACHE_DIR = os.getenv("CACHE_DIR", ".scanner_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
STATE_FILE = os.path.join(CACHE_DIR, "alerts_dip7_seen.json")

def load_seen() -> Set[str]:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f); return set(data if isinstance(data,list) else [])
    except Exception: return set()

def save_seen(s: Set[str]):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(s)), f, ensure_ascii=False, indent=2)
    except Exception: pass

def key_buy(row) -> str: return f"BUY|{row['ticker']}|{row['tf']}|{row['date']}"
def key_watch(row) -> str: return f"WATCH|{row['ticker']}|{row['tf']}|{row['date']}"

# ===================== LOOP ===================== #
def is_market_open_now() -> bool:
    if not MARKET_ONLY: return True
    now = datetime.now(ZoneInfo(MARKET_TZ))
    if now.weekday() >= 5: return False
    return MARKET_START <= now.time() <= MARKET_END

def run_once(first_run=False):
    if first_run and IDE_PRINT_TITLE:
        print("="*88); print(TITLE_TEXT); print("="*88)

    if first_run and TELEGRAM_PING_ON_START:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        notify_all(f"dip7 scanner is live at {ts} (UTC).")

    df = build_dataframe()

    base_cols = ["ticker","tf","date","status","conf","close",
                 "lrc_lower","lrc_mid","lrc_upper",
                 "sug_buy","sug_stop","stop","target","rr",
                 "ema50","ema200","adx","aroon_up","aroon_dn","poc","hvn_below","lvn_below",
                 "div_cnt","reasons","error"]
    dbg_cols  = ["trend_ok","liq_ok","vol_ok","sweet_ok","momo_ok","div_ok","rr_ok","regime_ok","avwap4h_ok","trigger_ok","lrc_touch_ok",
                 "fail_primary","fail_secondary"]
    cols = base_cols + (dbg_cols if DEBUG_DETAIL else [])
    df = df[[c for c in cols if c in df.columns]]

    df_view = df.copy()
    for c in ("poc","hvn_below","lvn_below","lrc_lower","lrc_mid","lrc_upper"):
        if c in df_view.columns:
            df_view[c] = df_view[c].astype(object).where(pd.notna(df_view[c]), "")

    print(df_view.to_string(index=False))

    out = "scanner_dual_tf_vp_dip7.csv"
    df.to_csv(out, index=False)
    print("\nSaved:", out)

    if DEBUG_DETAIL:
        gates = [c for c in dbg_cols if c in df.columns and not c.startswith("fail_")]
        print("\n[debug] gate pass counts:")
        for g in gates:
            print(f"  {g:12s} : {int(df[g].fillna(False).sum())} / {len(df)}")
        print()

    seen = load_seen()
    buys = df[(df["status"]=="BUY") & df["error"].eq("")]
    watch = df[(df["status"]=="WATCH") & df["error"].eq("")]

    new_buys, new_watch = [], []
    for _, r in buys.iterrows():
        k = key_buy(r)
        if k not in seen:
            new_buys.append(r); seen.add(k)
    for _, r in watch.iterrows():
        k = key_watch(r)
        if k not in seen:
            new_watch.append(r); seen.add(k)

    if new_buys or new_watch:
        lines = [f"Entries {datetime.now().strftime('%Y-%m-%d %H:%M')}:", ""]
        if new_buys:
            lines += ["BUY (confirmed):", "——————"]
            for r in sorted(new_buys, key=lambda x: (-x['conf'], x['ticker'])):
                lines.append(
                    f"- {r['ticker']} [{r['tf']}] c {r['close']:.2f} | "
                    f"LRC_lo {r.get('lrc_lower','')} | stop {r['stop']:.2f} | tgt {r['target']:.2f} | "
                    f"R/R {r['rr']:.2f} | conf {r['conf']}"
                )
        if new_watch:
            if new_buys: lines += [""]
            lines += ["In Buy Zone (watching):", "——————"]
            for r in sorted(new_watch, key=lambda x: (-x['conf'], x['ticker'])):
                lines.append(
                    f"- {r['ticker']} [{r['tf']}] c {r['close']:.2f} ≤ sug_buy {r['sug_buy']:.2f} | "
                    f"LRC_lo {r.get('lrc_lower','')} | conf {r['conf']}"
                )
        notify_all("\n".join(lines))
        save_seen(seen)

    if ALWAYS_NOTIFY:
        summary = [
            f"Summary {datetime.now(ZoneInfo(MARKET_TZ)).strftime('%Y-%m-%d %H:%M %Z')}",
            f"Universe: {len(_parse_universe())}",
            "BUY now: " + (", ".join(sorted(buys['ticker'].tolist())) or "—"),
            "Watch: " + (", ".join(sorted(watch['ticker'].tolist())) or "—"),
        ]
        notify_all("\n".join(summary))

    if (not first_run) and TELEGRAM_PING_ON_END:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        notify_all(f"dip7 iteration completed at {ts} (UTC).")

def main_loop():
    if IDE_LOOP:
        first=True
        while True:
            if is_market_open_now(): run_once(first_run=first); first=False
            else: print("[info] market closed; sleeping...]")
            time.sleep(max(60, int(IDE_INTERVAL_MIN*60)))
    else:
        run_once(first_run=True)

def market_open_now() -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    return (now.weekday() < 5) and (dtime(9,30) <= now.time() <= dtime(16,0))

if __name__ == "__main__":
    if MARKET_ONLY and not market_open_now():
        print("[done] Market closed — no scanning (set MARKET_ONLY=false or use the workflow toggle).", flush=True)
        sys.exit(0)
    main_loop()
