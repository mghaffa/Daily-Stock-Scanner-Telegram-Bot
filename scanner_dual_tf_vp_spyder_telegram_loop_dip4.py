#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scanner_dual_tf_vp_spyder_telegram_loop_dip4.py

Purpose:
- Find *timely* long entries on quality pullbacks, not just "under sug_buy".
- Dual-timeframe capable (1D core; optional 4H entry gate).
- Telegram alerts (multi-recipient) + CSV output + de-dup state.

Key upgrades vs dip3:
  1) Trend quality: EMA200 non-declining + EMA50 linear-regression slope > 0
  2) Regime filter: SPY must be constructive or signals are downgraded
  3) Volume accumulation: recent up-day with ≥ 1.2x 20d average volume
  4) Anchored VWAP (4w) guard for 4H entry (optional)
  5) “Sweet-spot” band near EMA50 (EMA50 -0.5*ATR ... +0.25*ATR)
  6) Reward/Risk screen vs recent swing-high target (≥ 1.5R)
  7) Liquidity screen: 20d dollar volume ≥ $10M (default)
  8) Confidence score + cleaner, non-duplicative Telegram summaries

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

# ===================== USER TOGGLES ===================== #
IDE_ENABLE_1D = True
IDE_ENABLE_4H = True          # use 4H for entry timing (optional)
IDE_SORT_TF_FIRST = "1D"
IDE_PRINT_TITLE = True

IDE_LOOP = False              # one-and-done by default
IDE_INTERVAL_MIN = 30

# Market-hours gate
MARKET_ONLY = os.getenv("MARKET_ONLY", "true").lower() == "true"
MARKET_TZ = "America/New_York"
MARKET_START = dtime(9, 30)
MARKET_END   = dtime(16, 0)

# Messaging
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_CHAT_IDS  = os.getenv("TELEGRAM_CHAT_IDS", "").strip()
TELEGRAM_AUTO_DISCOVER = True
ALWAYS_NOTIFY = os.getenv("ALWAYS_NOTIFY", "false").lower() == "true"
TELEGRAM_PING_ON_START = os.getenv("TELEGRAM_PING_ON_START", "false").lower() == "true"
TELEGRAM_PING_ON_END   = os.getenv("TELEGRAM_PING_ON_END", "false").lower() == "true"

# Universe
RAW_TICKERS = ["nvda","amd","orcl","avgo","pltr","net","amzn","googl","msft","klac","ibm","aapl",
               "tqqq","intc","bulz","cost","tsla","meta","now","nflx","hims","ntra","ddog","tsm",
               "mu","crm","tem","rklb","crwd","uvxy","unh","jpm","abt","bynd"]
ALIAS = {"google":"GOOGL"}  # small typo map

# Risk/Regime/Liquidity controls
MIN_DOLLAR_VOL = float(os.getenv("MIN_DOLLAR_VOL", "10000000"))  # ≥ $10M 20d avg
REGIME_TICKER  = os.getenv("REGIME_TICKER", "SPY")
REGIME_MIN = float(os.getenv("REGIME_MIN", "0.0"))               # reserved
REGIME_STRICT = os.getenv("REGIME_STRICT", "true").lower()=="true"

# Sweet-spot / Confirmation
SWEET_ATR_LOW  = float(os.getenv("SWEET_ATR_LOW",  "0.5"))   # EMA50 - 0.5*ATR
SWEET_ATR_HIGH = float(os.getenv("SWEET_ATR_HIGH", "0.25"))  # EMA50 + 0.25*ATR
DIV_MIN = int(os.getenv("DIV_MIN", "2"))                     # min bullish div count
VOL_UP_LOOKBACK = int(os.getenv("VOL_UP_LOOKBACK","10"))     # up-day window
VOL_UP_MULT     = float(os.getenv("VOL_UP_MULT","1.2"))      # ≥1.2x 20d avg volume
USE_AVWAP_4W_4H = os.getenv("USE_AVWAP_4W_4H","true").lower()=="true"
RR_MIN          = float(os.getenv("RR_MIN", "1.5"))

# Periods
DAILY_PERIOD="2y"; DAILY_INTERVAL="1d"
INTRA_PERIOD="60d"; INTRA_INTERVAL="60m"

LB=5; RB=5  # pivots for divergence

TITLE_TEXT = (
    "Scanner (dip4): Trend + Sweet-spot + Volume + R/R + optional 4H AVWAP gate\n"
    "- Uses EMA50 LR-slope>0, EMA200 not falling, volume accumulation, and R/R.\n"
    "- Sweet-spot is near EMA50: [EMA50-0.5*ATR, EMA50+0.25*ATR]."
)

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
    # normalize names
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

def fetch_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=DAILY_PERIOD, interval=DAILY_INTERVAL, auto_adjust=True, progress=False, group_by="column")
    if df.empty or len(df)<250: raise ValueError("insufficient daily data")
    return _ensure_ohlcv(df, ticker)

def fetch_4h(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL, auto_adjust=True, prepost=False, progress=False, group_by="column")
    if df.empty or len(df)<100: raise ValueError("insufficient intraday data")
    df = _ensure_ohlcv(df, ticker)
    # resample to exact 4H
    if getattr(df.index, "tz", None) is not None:
        df = df.tz_localize(None)
    return df.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna(how="any")

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
    hh=h.rolling(n).max(); ll=l.rolling(n).min()
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

# ---------- Divergences ---------- #
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
    inds = {
        "RSI": df["RSI14"], "MACD": macd(df["Close"])[0],
        "WILLR": df["WILLR"]
    }
    price = df["Close"]; names=[]
    for k, s in inds.items():
        try:
            if bullish_div(price, s, lb, rb): names.append(k)
        except Exception: pass
    return len(names), names

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
    ema200_flat_up = o["EMA200"].iloc[-1] >= o["EMA200"].iloc[-5]  # not falling
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
    up = (w["Close"].diff() > 0)
    heavy = w["Volume"] >= (mult * w["VOL20"].iloc[-1])
    ok = bool((up & heavy).tail(lookback).any())
    return ok, ("vol accum OK" if ok else f"no up-day ≥{mult:.1f}×20d in {lookback}")

def sweet_spot_ok(o: pd.DataFrame) -> Tuple[bool, str]:
    r=o.iloc[-1]
    lo = r.EMA50 - SWEET_ATR_LOW * r.ATR14
    hi = r.EMA50 + SWEET_ATR_HIGH* r.ATR14
    ok = (r.Close >= lo) and (r.Close <= hi)
    return ok, f"sweet-band {lo:.2f}..{hi:.2f}"

def momentum_turn_ok(o: pd.DataFrame) -> Tuple[bool, List[str]]:
    r=o.iloc[-1]
    wr_cross = (o["WILLR"].iloc[-2] <= -80) and (r.WILLR > -80)
    adx, pdi, ndi = r.ADX, r["+DI"], r["-DI"]
    di_cross = (pdi > ndi) and (o["+DI"].iloc[-2] <= o["-DI"].iloc[-2])
    aup, adn = r.AROON_UP, r.AROON_DOWN
    aroon_ok = ((aup > 70 and adn < 30) or (o["AROON_UP"].iloc[-2] <= o["AROON_DOWN"].iloc[-2] and aup > adn))
    ok = wr_cross and di_cross and aroon_ok
    reasons=[]
    reasons.append("W%R cross -80" if wr_cross else "W%R not crossed")
    reasons.append("+DI cross up" if di_cross else "+DI not > -DI")
    reasons.append("Aroon dom/flip" if aroon_ok else "Aroon weak")
    return ok, reasons

def divergence_ok(o: pd.DataFrame, min_count=2) -> Tuple[bool, str, int, List[str]]:
    cnt, names = divergence_score(o, LB, RB)
    ok = cnt >= min_count
    return ok, f"div {cnt}/{min_count}", cnt, names

def risk_reward(o: pd.DataFrame, hvn_below: float|None, lvn_below: float|None) -> Tuple[bool, str, float, float, float]:
    r=o.iloc[-1]
    # Stop under LVN (if exists) or EMA50-ATR
    stop = min((lvn_below if lvn_below is not None else r.EMA50 - r.ATR14), r.Low)  # conservative
    risk = max(r.Close - stop, 1e-6)
    # Target = recent swing-high (20) or EMA50+2*ATR
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
        return True, [f"regime check skipped: {e}"]  # fail-open

# -------- VP suggested levels -------- #
def suggest_levels(o: pd.DataFrame, lookback=180, bins=120):
    r=o.iloc[-1]
    poc, hvn_below, lvn_below, centers, vols = volume_profile(o["Close"], o["Volume"], lookback, bins)
    lo = r.EMA50 - SWEET_ATR_LOW * r.ATR14
    hi = r.EMA50 + SWEET_ATR_HIGH* r.ATR14
    # Choose a suggested buy inside sweet spot and ≥ HVN_below if present
    base = max(lo, hvn_below if hvn_below is not None else lo)
    sug_buy = round(float(np.clip(base, lo, hi)), 2)
    sug_stop = round(float(min((lvn_below if lvn_below is not None else r.EMA50 - r.ATR14), r.Low)), 2)
    return sug_buy, sug_stop, (None if poc is None else float(poc)), hvn_below, lvn_below, (lo, hi)

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
    div_cnt: int; reasons: str; conf: int; error: str

def _parse_universe() -> List[str]:
    env = (os.getenv("TICKERS","") or "").strip()
    if env:
        toks = [t.strip().upper() for t in re.split(r"[,\s;|]+", env) if t.strip()]
        return sorted(set(toks))
    return sorted({ALIAS.get(t.lower(), t.upper()) for t in RAW_TICKERS})

def confidence_score(flags: Dict[str,bool]) -> int:
    # 10 points each; cap to 100
    order = ["trend","volume","sweet","momentum","div","liquidity","rr","regime","avwap4h","price_trigger"]
    return int(min(100, sum(10 for k in order if flags.get(k, False))))

def process_one(ticker: str, tf: str, df: pd.DataFrame, vp_lookback=180) -> Row:
    o = compute(df)
    if o.empty or len(o) < 210: raise ValueError("indicator frame short")

    # Trend + liquidity + volume + sweet spot
    ok_trend, t_reasons = long_trend_ok(o)
    ok_liq, liq_msg, dollar = liquidity_ok(o)
    ok_vol, vol_msg = volume_accum_ok(o, VOL_UP_LOOKBACK, VOL_UP_MULT)
    ok_sweet, sweet_msg = sweet_spot_ok(o)

    # Momentum & divergence
    ok_momo, m_reasons = momentum_turn_ok(o)
    ok_div, div_msg, div_cnt, div_names = divergence_ok(o, DIV_MIN)

    # VP levels + risk/reward
    sug_buy, sug_stop, poc, hvn_b, lvn_b, (band_lo, band_hi) = suggest_levels(o, vp_lookback, 120)
    ok_rr, rr_msg, stop, target, rr = risk_reward(o, hvn_b, lvn_b)

    # Price action trigger (above prev high)
    r = o.iloc[-1]; r1 = o.iloc[-2]
    price_trigger = (r.Close > r1.High)

    # Regime
    ok_regime, regime_msgs = regime_ok()

    # Optional 4H anchored VWAP gate
    avwap4h_ok = True
    if IDE_ENABLE_4H and USE_AVWAP_4W_4H:
        try:
            h4 = compute(fetch_4h(ticker))
            av = anchored_vwap_last(h4, 28)
            avwap4h_ok = (h4.iloc[-1].Close >= av) if np.isfinite(av) else True
        except Exception:
            avwap4h_ok = True  # fail-open
    # Flags summary
    flags = {
        "trend": ok_trend, "liquidity": ok_liq, "volume": ok_vol, "sweet": ok_sweet,
        "momentum": ok_momo, "div": ok_div, "rr": ok_rr, "regime": ok_regime,
        "avwap4h": avwap4h_ok, "price_trigger": price_trigger
    }
    conf = confidence_score(flags)

    # Status decision
    # BUY = all primary gates pass: trend+liquidity+volume+sweet+momentum+div+rr (+regime & 4H gate if strict)
    primary_ok = ok_trend and ok_liq and ok_vol and ok_sweet and ok_momo and ok_div and ok_rr
    strict_ok  = primary_ok and ok_regime and avwap4h_ok and price_trigger
    if REGIME_STRICT:
        status = "BUY" if strict_ok else ("WATCH" if primary_ok else "")
    else:
        status = "BUY" if primary_ok and price_trigger else ("WATCH" if primary_ok else "")

    reasons = []
    reasons += t_reasons
    reasons += [liq_msg, vol_msg, sweet_msg]
    reasons += m_reasons + [div_msg, rr_msg]
    reasons += regime_msgs
    if IDE_ENABLE_4H and USE_AVWAP_4W_4H: reasons.append("4H AVWAP ok" if avwap4h_ok else "4H AVWAP fail")
    reasons.append("trigger ok" if price_trigger else "trigger pending")

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
        div_cnt=div_cnt, reasons=" | ".join(reasons), conf=conf, error=""
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
                            np.nan,np.nan,np.nan,None,None,None,0,"",0, str(e)))
        if IDE_ENABLE_4H:
            try:
                h4 = fetch_4h(t)
                rows.append(process_one(t, "4H", h4, 200))
            except Exception as e:
                rows.append(Row(t, "4H", "", "", np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                                np.nan,np.nan,np.nan,None,None,None,0,"",0, str(e)))
    df = pd.DataFrame([r.__dict__ for r in rows])
    order = [IDE_SORT_TF_FIRST.upper()] + [x for x in ["1D","4H"] if x != IDE_SORT_TF_FIRST.upper()]
    df["tf"] = pd.Categorical(df["tf"], categories=order, ordered=True)
    return df.sort_values(["tf","ticker"]).reset_index(drop=True)

# -------- de-dup state for alerts -------- #
STATE_FILE = os.path.join(os.getcwd(), "alerts_dip4_seen.json")
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
        notify_all(f"dip4 scanner is live at {ts} (UTC).")

    df = build_dataframe()

    cols = ["ticker","tf","date","status","conf","close","sug_buy","sug_stop","stop","target","rr",
            "ema50","ema200","adx","aroon_up","aroon_dn","poc","hvn_below","lvn_below",
            "div_cnt","reasons","error"]
    df = df[[c for c in cols if c in df.columns]]

    print(df.to_string(index=False))
    out = "scanner_dual_tf_vp_dip4.csv"
    df.to_csv(out, index=False)
    print("\nSaved:", out)

    # -------- Alerts (de-dup) --------
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
                lines.append(f"- {r['ticker']} [{r['tf']}] c {r['close']:.2f} | stop {r['stop']:.2f} | tgt {r['target']:.2f} | R/R {r['rr']:.2f} | conf {r['conf']}")
        if new_watch:
            if new_buys: lines += [""]
            lines += ["In Buy Zone (watching):", "——————"]
            for r in sorted(new_watch, key=lambda x: (-x['conf'], x['ticker'])):
                lines.append(f"- {r['ticker']} [{r['tf']}] c {r['close']:.2f} ≤ sug_buy {r['sug_buy']:.2f} | stop {r['sug_stop']:.2f} | conf {r['conf']}")
        notify_all("\n".join(lines))
        save_seen(seen)

    if ALWAYS_NOTIFY and not (new_buys or new_watch):
        summary = [
            f"Summary {datetime.now(ZoneInfo(MARKET_TZ)).strftime('%Y-%m-%d %H:%M %Z')}",
            f"Universe: {len(_parse_universe())}",
            "BUY now: " + (", ".join(sorted(buys['ticker'].tolist())) or "—"),
            "Watch: " + (", ".join(sorted(watch['ticker'].tolist())) or "—"),
        ]
        notify_all("\n".join(summary))

    if (not first_run) and TELEGRAM_PING_ON_END:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        notify_all(f"dip4 iteration completed at {ts} (UTC).")

def main_loop():
    if IDE_LOOP:
        first=True
        while True:
            if is_market_open_now(): run_once(first_run=first); first=False
            else: print("[info] market closed; sleeping...")
            time.sleep(max(60, int(IDE_INTERVAL_MIN*60)))
    else:
        run_once(first_run=True)

# ------------------------------ Entry ------------------------------ #
def market_open_now() -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    return (now.weekday() < 5) and (dtime(9,30) <= now.time() <= dtime(16,0))

if __name__ == "__main__":
    if MARKET_ONLY and not market_open_now():
        print("[done] Market closed — no scanning (set MARKET_ONLY=false or use the workflow toggle).", flush=True)
        sys.exit(0)
    main_loop()
