#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scanner_dual_tf_vp_spyder_telegram_loop_dip3.py

- 1D/4H toggles + dual-timeframe gating
- Dip alerts + BUY alerts (de-duped)
- Telegram alerts (multi-recipient) + optional test pings
- Market-hours gate (US/Eastern) – overridable by env MARKET_ONLY=false
- CSV + JSON state artifacts
"""

from __future__ import annotations
import os, sys, json, time, re
from dataclasses import dataclass
from typing import List, Optional, Set
from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import certifi
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===================== CONFIG ===================== #
IDE_ENABLE_1D = True
IDE_ENABLE_4H = False
IDE_SORT_TF_FIRST = "1D"
IDE_PRINT_TITLE = True

IDE_LOOP = False
IDE_INTERVAL_MIN = 30  # if IDE_LOOP=True

# Market-hours gate (default true; workflow can override via env)
MARKET_ONLY = os.getenv("MARKET_ONLY", "true").lower() == "true"
MARKET_TZ = "America/New_York"
MARKET_START = dtime(9, 30)
MARKET_END   = dtime(16, 0)

IDE_SEND_TEST_MESSAGE_ONCE = False  # in-IDE only

# --- Telegram / webhooks (use secrets; no hard-coded tokens) --- #
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_CHAT_IDS  = os.getenv("TELEGRAM_CHAT_IDS", "").strip()  # multi: "-480...,11122..." or "-480... 11122..."
TELEGRAM_AUTO_DISCOVER = True

ALWAYS_NOTIFY = os.getenv("ALWAYS_NOTIFY", "false").lower() == "true"
TELEGRAM_PING_ON_START = os.getenv("TELEGRAM_PING_ON_START", "false").lower() == "true"
TELEGRAM_PING_ON_END   = os.getenv("TELEGRAM_PING_ON_END", "false").lower() == "true"

# --------------------------- Scanner Config --------------------------- #
RAW_TICKERS = ["nvda","amd","orcl","avgo","pltr","net","amzn","googl","msft","klac","ibm","aapl","tqqq","intc","bulz","cost","tsla","meta","now","nflx","hims","ntra","ddog","tsm","mu","crm","tem","rklb","crwd","uvxy","intc","unh","jpm","abt","bynd","race"]
TICKER_MAP = {"google":"GOOGL"}

# Allow override via env TICKERS (comma/space separated)
_env_tickers = (os.getenv("TICKERS", "") or "").strip()
if _env_tickers:
    TICKERS = sorted({t.upper() for t in re.split(r"[,\s]+", _env_tickers) if t.strip()})
else:
    TICKERS = sorted({TICKER_MAP.get(t.lower(), t.upper()) for t in RAW_TICKERS})

DAILY_PERIOD="2y"; DAILY_INTERVAL="1d"
INTRA_PERIOD="60d"; INTRA_INTERVAL="60m"
LB=5; RB=5
DIV_MIN=2

# ---------------------- Dip-alert Settings ---------------------- #
IDE_ALERT_ON_DIP = True
DIP_TOL_PCT = 0.3
DIP_MIN_ABOVE_STOP = 0.0
DIP_REQUIRE_BELOW = True

VP_LOOKBACK_D = 180
VP_LOOKBACK_4H = 200
VP_BINS = 120

TITLE_TEXT = (
    "Volume Profile landmarks used by the scanner:\n"
    "- POC: point of control in lookback window.\n"
    "- HVN_below: last high-volume node below price (≥70% of max).\n"
    "- LVN_below: last low-volume node below price (≤30% of max)."
)

# ---------------------- Utils ----------------------- #
def _squeeze_col(x):
    if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
        return x.iloc[:, 0]
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
            df.columns = [''.join([str(x) for x in tup if str(x)!='']) for tup in df.columns.to_flat_index()]
    for col in list(df.columns):
        try: df[col] = _squeeze_col(df[col])
        except Exception: pass
    required = ["Open","High","Low","Close","Volume"]
    alt = {c.lower(): c for c in df.columns}
    fixed = {}
    for r in required:
        if r in df.columns:
            fixed[r] = r
        elif r.lower() in alt:
            fixed[r] = alt[r.lower()]
        else:
            cand = [c for c in df.columns if r.lower() in str(c).lower()]
            if cand: fixed[r] = cand[0]
            else: raise ValueError(f"Missing required column '{r}'. Columns: {list(df.columns)}")
    if fixed:
        df = df.rename(columns={fixed[k]:k for k in fixed})
    return df[["Open","High","Low","Close","Volume"]]

# ---------------------- Indicators ----------------------- #
def ema(s: pd.Series, length: int) -> pd.Series:
    s = _squeeze_col(s); return s.ewm(span=length, adjust=False).mean()

def sma(s: pd.Series, length: int) -> pd.Series:
    s = _squeeze_col(s); return s.rolling(length, min_periods=length).mean()

def williams_r(h, l, c, length=14) -> pd.Series:
    h=_squeeze_col(h); l=_squeeze_col(l); c=_squeeze_col(c)
    hh = h.rolling(length).max(); ll = l.rolling(length).min()
    return -100 * ((hh - c) / (hh - ll))

def true_range(h, l, c):
    h=_squeeze_col(h); l=_squeeze_col(l); c=_squeeze_col(c)
    pc = c.shift(1)
    return pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def atr(h, l, c, length=14):
    return true_range(h, l, c).ewm(alpha=1/length, adjust=False).mean()

def aroon_up_down(h, l, length=14):
    # simple robust impl:
    hh = h.rolling(length).max(); ll = l.rolling(length).min()
    up  = 100 * (h.rolling(length).apply(lambda x: (length-1)-np.argmax(x[::-1]), raw=True)) / length
    dn  = 100 * (l.rolling(length).apply(lambda x: (length-1)-np.argmin(x[::-1]), raw=True)) / length
    return up, dn

def adx(h, l, c, length=14):
    h=_squeeze_col(h); l=_squeeze_col(l); c=_squeeze_col(c)
    up = h.diff(); dn = -l.diff()
    plus_dm = np.where((up>dn) & (up>0), up, 0.0)
    minus_dm = np.where((dn>up) & (dn>0), dn, 0.0)
    tr = true_range(h, l, c)
    atr_ = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1/length, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=l.index).ewm(alpha=1/length, adjust=False).mean() / atr_
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf,-np.inf], np.nan)
    return dx.ewm(alpha=1/length, adjust=False).mean()

def rsi(close, length=14):
    close=_squeeze_col(close)
    d=close.diff(); gain=d.clip(lower=0); loss=-d.clip(upper=0)
    ag=gain.ewm(alpha=1/length, adjust=False).mean()
    al=loss.ewm(alpha=1/length, adjust=False).mean()
    rs=ag/al.replace(0,np.nan); return 100 - (100/(1+rs))

def macd(close, fast=12, slow=26, signal=9):
    close=_squeeze_col(close); fast_=ema(close,fast); slow_=ema(close,slow)
    line=fast_-slow_; sig=line.ewm(span=signal, adjust=False).mean(); hist=line-sig
    return line, sig, hist

def mom(close, length=10): return _squeeze_col(close).diff(length)

def cci(h,l,c,length=10):
    h=_squeeze_col(h); l=_squeeze_col(l); c=_squeeze_col(c)
    tp=(h+l+c)/3.0; sma_tp=sma(tp,length); mad=(tp-sma_tp).abs().rolling(length).mean()
    return (tp-sma_tp)/(0.015*mad)

def obv(close, vol):
    close=_squeeze_col(close); vol=_squeeze_col(vol)
    delta=close.diff(); sign=pd.Series(np.where(delta>0,1,np.where(delta<0,-1,0)), index=close.index)
    return (sign*vol).cumsum()

def stoch_k(h,l,c,length=14,smooth=3):
    h=_squeeze_col(h); l=_squeeze_col(l); c=_squeeze_col(c)
    ll=l.rolling(length).min(); hh=h.rolling(length).max()
    k=100*(c-ll)/(hh-ll); return k.rolling(smooth).mean()

def vwma(close, vol, length):
    close=_squeeze_col(close); vol=_squeeze_col(vol)
    return (close*vol).rolling(length).sum()/vol.rolling(length).sum()

def cmf(h,l,c,vol,length=21):
    h=_squeeze_col(h); l=_squeeze_col(l); c=_squeeze_col(c); vol=_squeeze_col(vol)
    mfm=((c-l)-(h-c))/(h-l).replace(0,np.nan); mfv=mfm*vol
    return mfv.rolling(length).sum()/vol.rolling(length).sum()

def mfi(h,l,c,vol,length=14):
    h=_squeeze_col(h); l=_squeeze_col(l); c=_squeeze_col(c); vol=_squeeze_col(vol)
    tp=(h+l+c)/3.0; rmf=tp*vol
    pos=rmf.where(tp>tp.shift(1),0.0); neg=rmf.where(tp<tp.shift(1),0.0)
    pos_sum=pos.rolling(length).sum(); neg_sum=neg.rolling(length).sum().replace(0,np.nan)
    mr=pos_sum/neg_sum; return 100-(100/(1+mr))

# ----------------------- Volume Profile ----------------------- #
def volume_profile(close: pd.Series, volume: pd.Series, lookback: int=180, bins: int=120):
    close=_squeeze_col(close).tail(lookback); volume=_squeeze_col(volume).reindex(close.index)
    pmin=float(close.min()); pmax=float(close.max())
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax<=pmin:
        return None, None, None, None, None
    edges=np.linspace(pmin, pmax, bins+1)
    centers=(edges[:-1]+edges[1:])/2.0
    idx=np.clip(np.digitize(close.values, edges)-1, 0, bins-1)
    vol_by_bin=np.bincount(idx, weights=volume.values, minlength=bins)
    poc_idx=int(np.argmax(vol_by_bin)); poc=float(centers[poc_idx])
    thresh_hi=0.7*vol_by_bin.max(); thresh_lo=0.3*vol_by_bin.max()
    current=float(close.iloc[-1])

    hvn_candidates=np.where(vol_by_bin>=thresh_hi)[0]
    hvn_below=None
    if hvn_candidates.size>0:
        below=hvn_candidates[centers[hvn_candidates] <= current]
        if below.size>0: hvn_below=float(centers[below[-1]])
    lvn_candidates=np.where(vol_by_bin<=thresh_lo)[0]
    lvn_below=None
    if lvn_candidates.size>0:
        below=lvn_candidates[centers[lvn_candidates] <= current]
        if below.size>0: lvn_below=float(centers[below[-1]])
    return poc, hvn_below, lvn_below, centers, vol_by_bin

# ----------------------- Indicators bundle ------------------- #
def compute(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    o["EMA50"]  = ema(o["Close"], 50)
    o["EMA200"] = ema(o["Close"], 200)
    o["WILLR"]  = williams_r(o["High"], o["Low"], o["Close"], 14)
    up, dn = aroon_up_down(o["High"], o["Low"], 14)
    o["AROON_UP"] = up; o["AROON_DOWN"] = dn
    o["ADX"]   = adx(o["High"], o["Low"], o["Close"], 14)
    o["ATR14"] = atr(o["High"], o["Low"], o["Close"], 14)

    # Divergence set
    o["RSI14"] = rsi(o["Close"], 14)
    macd_line, macd_sig, macd_hist = macd(o["Close"], 12, 26, 9)
    o["MACD"] = macd_line; o["MACD_HIST"] = macd_hist
    o["MOM10"] = mom(o["Close"], 10)
    o["CCI10"] = cci(o["High"], o["Low"], o["Close"], 10)
    o["OBV"]   = obv(o["Close"], o["Volume"])
    o["STOCHK"] = stoch_k(o["High"], o["Low"], o["Close"], 14, 3)
    DI = o["High"].diff() - (-o["Low"].diff())
    TR = true_range(o["High"], o["Low"], o["Close"])
    o["DIOSC"] = 100 * DI.rolling(14, min_periods=1).mean() / TR.rolling(14, min_periods=1).mean()
    o["VW_MACD"] = vwma(o["Close"], o["Volume"], 12) - vwma(o["Close"], o["Volume"], 26)
    o["CMF21"] = cmf(o["High"], o["Low"], o["Close"], o["Volume"], 21)
    o["MFI14"] = mfi(o["High"], o["Low"], o["Close"], 14)
    return o

# -------------------------- Divergences ------------------------ #
def pivot_points(series: pd.Series, lb: int, rb: int):
    series=_squeeze_col(series)
    highs, lows = [], []
    vals = series.values; n = len(series)
    for i in range(lb, n-rb):
        w = vals[i-lb:i+rb+1]; c = vals[i]
        if np.isfinite(c):
            if c == np.nanmax(w): highs.append(i)
            if c == np.nanmin(w): lows.append(i)
    return highs, lows

def bullish_div(price: pd.Series, indi: pd.Series, lb: int, rb: int):
    price=_squeeze_col(price); indi=_squeeze_col(indi)
    _, lows = pivot_points(price, lb, rb)
    if len(lows) < 2: return False, -1, -1
    i1, i2 = lows[-2], lows[-1]
    return (price.iloc[i2] < price.iloc[i1]) and (indi.iloc[i2] > indi.iloc[i1]), i2, i1

def divergence_score(df: pd.DataFrame, lb: int, rb: int):
    inds = {"RSI": df["RSI14"], "MACD": df["MACD"], "MACD_HIST": df["MACD_HIST"],
            "MOM": df["MOM10"], "CCI": df["CCI10"], "OBV": df["OBV"], "STOCH": df["STOCHK"],
            "DIOSC": df["DIOSC"], "VW_MACD": df["VW_MACD"], "CMF": df["CMF21"], "MFI": df["MFI14"]}
    price = df["Close"]; names = []
    for k, s in inds.items():
        try:
            bull, _, _ = bullish_div(price, s, LB, RB)
            if bull: names.append(k)
        except Exception:
            pass
    return len(names), names

# ------------------------- Strategy rules ---------------------- #
def long_signal(df: pd.DataFrame, require_div=True):
    needed = ["EMA50","EMA200","WILLR","ADX","ATR14","Close","High","Low","Volume"]
    df = df.dropna(subset=[c for c in needed if c in df.columns])

    if len(df) < 210:
        return False, ["Insufficient data"], None, None, (0,[])

    r  = df.iloc[-1]
    if len(df) < 2:
        return False, ["Frame too short"], None, None, (0,[])

    r1 = df.iloc[-2]
    if len(df) < 6:
        return False, ["Not enough bars for EMA slope check"], None, None, (0,[])

    trend = (r.Close > r.EMA200) and (r.EMA50 > df.EMA50.iloc[-6])
    if not trend: return False, ["Trend filter failed"], None, None, (0,[])

    pullback = (df.Low.tail(3).min() <= r.EMA50) and (df.WILLR.tail(3).min() < -80)
    if not pullback: return False, ["No qualifying pullback"], None, None, (0,[])

    wr_rev = (df.WILLR.iloc[-2] <= -80) and (r.WILLR > -80)
    aroon_ok = ((r["AROON_UP"] > 70 and r["AROON_DOWN"] < 30) or
                (df["AROON_UP"].iloc[-2] <= df["AROON_DOWN"].iloc[-2] and r["AROON_UP"] > r["AROON_DOWN"]))
    adx_ok = (r.ADX > 20) and (r.ADX > df.ADX.iloc[-2])
    if not (wr_rev and aroon_ok and adx_ok):
        return False, ["Momentum turn not confirmed"], None, None, (0,[])

    trigger = (r.Close > r.EMA50) and (r.Close > r1.High)
    if not trigger: return False, ["Trigger not met"], None, None, (0,[])

    bull_cnt, bull_names = divergence_score(df, LB, RB)
    if require_div and bull_cnt < DIV_MIN:
        return False, [f"Bullish divergence {bull_cnt} < {DIV_MIN}"], None, None, (bull_cnt, bull_names)

    stop = min(df.Low.tail(5).min(), r.EMA50 - r.ATR14)
    risk_R = max(r.Close - stop, 1e-6)

    reasons = ["Trend OK", "Pullback W%R<-80", "W%R cross -80", "Aroon dominance/flip", "ADX rising >20"]
    if require_div: reasons.append(f"Divergence: {bull_cnt} {bull_names}")
    return True, reasons, float(stop), float(risk_R), (bull_cnt, bull_names)

# ---------------------- Suggested levels ----------------------- #
def suggest_levels(df: pd.DataFrame, vp_lookback: int, vp_bins: int):
    df = df.dropna(subset=["EMA50","EMA200","ATR14","Close"])
    if df.empty:
        return np.nan, np.nan, None, None, None
    r = df.iloc[-1]
    poc, hvn_below, lvn_below, centers, vol_by_bin = volume_profile(df["Close"], df["Volume"], vp_lookback, vp_bins)
    ema50=float(r.EMA50); ema200=float(r.EMA200); atr=float(r.ATR14); close=float(r.Close)

    if (close > ema200) and (ema50 > df.EMA50.iloc[-6] if len(df)>=6 else True):
        base = ema50
        if hvn_below is not None:
            base = max(base, hvn_below)
        suggested_buy = round(base, 2)
        stop = min(ema50 - 1.0*atr, lvn_below if lvn_below is not None else close - 2.0*atr)
    elif close > ema200:
        suggested_buy = round(max(ema50, hvn_below or ema200), 2)
        stop = min(ema200 - 1.0*atr, lvn_below if lvn_below is not None else ema200 - 1.5*atr)
    else:
        suggested_buy = round(ema200 + 0.005*ema200, 2)
        stop = ema200 - 1.5*atr
    return suggested_buy, round(float(stop),2), float(poc) if poc else None, hvn_below, lvn_below

# ----------------------------- I/O & Notifications ----------------------------- #
@dataclass
class Row:
    ticker: str; tf: str; date: str; buy: str; close: float; ema50: float; ema200: float; adx: float
    aroon_up: float; aroon_dn: float; stop: str; sug_buy: float; sug_stop: float
    poc: float|None; hvn_below: float|None; lvn_below: float|None
    div_cnt: int; div_names: str; reasons: str; error: str

def fetch_daily(ticker: str):
    df = yf.download(ticker, period=DAILY_PERIOD, interval=DAILY_INTERVAL, auto_adjust=True, progress=False, group_by="column")
    if df.empty or len(df)<250: raise ValueError("insufficient daily data")
    return _ensure_ohlcv(df, ticker)

def fetch_4h(ticker: str):
    df = yf.download(ticker, period=INTRA_PERIOD, interval=INTRA_INTERVAL, auto_adjust=True, prepost=False, progress=False, group_by="column")
    if df.empty or len(df)<100: raise ValueError("insufficient intraday data")
    df = _ensure_ohlcv(df, ticker)
    df = df.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna(how="any")
    return df

def telegram_get_updates(token: str) -> Optional[dict]:
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        r = requests.get(url, timeout=8)
        return r.json()
    except Exception as e:
        print("[warn] telegram getUpdates:", e, file=sys.stderr)
        return None

def telegram_discover_chat_id(token: str) -> str:
    data = telegram_get_updates(token)
    if not data or not data.get("ok"):
        return ""
    for upd in reversed(data.get("result", [])):
        for key in ("message", "channel_post", "edited_message", "my_chat_member"):
            if key in upd and "chat" in upd[key]:
                chat = upd[key]["chat"]
                cid = chat.get("id")
                if cid is not None:
                    return str(cid)
    return ""

def _parse_ids(s: str):
    if not s: return []
    return [p for p in re.split(r"[,\s]+", s) if p]

def telegram_send(token: str, chat_id: str, text: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=12, verify=False)
        if r.status_code == 200:
            return True
        else:
            print("[warn] telegram send status:", r.status_code, r.text[:200], file=sys.stderr)
    except Exception as e:
        print("[warn] telegram send:", e, file=sys.stderr)
    return False

def notify_all_channels(text: str):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"text": text}, timeout=7)
        except Exception as e:
            print("[warn] webhook notify:", e, file=sys.stderr)

    token = TELEGRAM_BOT_TOKEN
    if not token:
        print("[warn] TELEGRAM_BOT_TOKEN not set; skipping Telegram send")
        return

    ids = []
    ids += _parse_ids(TELEGRAM_CHAT_IDS)
    if TELEGRAM_CHAT_ID:
        ids.append(TELEGRAM_CHAT_ID)

    if not ids and TELEGRAM_AUTO_DISCOVER:
        print("[info] TELEGRAM_CHAT_ID(S) not set; attempting auto-discovery via getUpdates...")
        auto = telegram_discover_chat_id(token)
        if auto:
            ids = [auto]

    if not ids:
        print("[warn] Telegram chat id missing. DM your bot first, or set TELEGRAM_CHAT_ID(S).")
        return

    for cid in ids:
        ok = telegram_send(token, cid, text)
        print(f"[debug] telegram -> {cid}: {ok}")

# -------------------------- Main pass --------------------------- #
def process_one(ticker: str, tf: str, df: pd.DataFrame, vp_lookback: int):
    o = compute(df)
    if o.empty or len(o) < 210:
        raise ValueError("indicator frame empty/short after compute")

    buy, reasons, stop, _, (dcnt, dnames) = long_signal(o, require_div=True)
    sug_buy, sug_stop, poc, hvn_below, lvn_below = suggest_levels(o, vp_lookback, VP_BINS)
    r = o.dropna(subset=["Close","EMA50","EMA200"]).iloc[-1]

    return Row(
        ticker=ticker, tf=tf, date=str(r.name.date()), buy=("YES" if buy else ""),
        close=round(float(r.Close),2), ema50=round(float(r.EMA50),2), ema200=round(float(r.EMA200),2),
        adx=round(float(o.ADX.dropna().iloc[-1]),1) if o.ADX.dropna().size else np.nan,
        aroon_up=int(o.AROON_UP.dropna().iloc[-1]) if o.AROON_UP.dropna().size else np.nan,
        aroon_dn=int(o.AROON_DOWN.dropna().iloc[-1]) if o.AROON_DOWN.dropna().size else np.nan,
        stop=("" if not buy else round(float(stop),2)),
        sug_buy=sug_buy, sug_stop=sug_stop,
        poc=(None if poc is None else round(float(poc),2)),
        hvn_below=(None if hvn_below is None else round(float(hvn_below),2)),
        lvn_below=(None if lvn_below is None else round(float(lvn_below),2)),
        div_cnt=dcnt, div_names=", ".join(dnames), reasons=" | ".join(reasons), error=""
    )

def build_dataframe() -> pd.DataFrame:
    rows: List[Row] = []
    for t in TICKERS:
        if IDE_ENABLE_1D:
            try:
                ddf = fetch_daily(t)
                rows.append(process_one(t, "1D", ddf, VP_LOOKBACK_D))
            except Exception as e:
                rows.append(Row(t, "1D", "", "", np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,"","", "", None, None, None,0,"","", str(e)))
        if IDE_ENABLE_4H:
            try:
                h4df = fetch_4h(t)
                rows.append(process_one(t, "4H", h4df, VP_LOOKBACK_4H))
            except Exception as e:
                rows.append(Row(t, "4H", "", "", np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,"","", "", None, None, None,0,"","", str(e)))
    df = pd.DataFrame([r.__dict__ for r in rows])
    tf_order = [IDE_SORT_TF_FIRST.upper()] + ([x for x in ["1D","4H"] if x != IDE_SORT_TF_FIRST.upper() and ((x=="1D" and IDE_ENABLE_1D) or (x=="4H" and IDE_ENABLE_4H))])
    df["tf"] = pd.Categorical(df["tf"], categories=tf_order, ordered=True)
    df = df.sort_values(["tf","ticker"]).reset_index(drop=True)
    return df

# ----------------------- Dedup state for alerts ------------------------ #
STATE_FILE = os.path.join(os.getcwd(), "alerts_seen.json")
def load_seen() -> Set[str]:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data if isinstance(data, list) else [])
    except Exception:
        return set()

def save_seen(s: Set[str]):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(s)), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[warn] could not save state:", e, file=sys.stderr)

def buy_key(row) -> str:
    return f"{row['ticker']}|{row['tf']}|{row['date']}"

# ---------------------------- Loop & Dips ---------------------------- #
def is_market_open_now() -> bool:
    if not MARKET_ONLY:
        return True
    now = datetime.now(ZoneInfo(MARKET_TZ))
    if now.weekday() >= 5:
        return False
    return MARKET_START <= now.time() <= MARKET_END

def _compute_dips(df):
    if not IDE_ALERT_ON_DIP:
        return df.iloc[0:0].copy()
    tol = 1.0 + (DIP_TOL_PCT / 100.0)
    dips = df[
        (df.get("buy", "") != "YES") &
        df["sug_buy"].notna() & df["sug_stop"].notna() &
        ( (df["close"] < df["sug_buy"]) if DIP_REQUIRE_BELOW
          else (df["close"] <= df["sug_buy"] * tol) ) &
        (df["close"] >= df["sug_stop"] * (1.0 + (DIP_MIN_ABOVE_STOP/100.0)))
    ].copy()
    return dips

def _send_dip_alerts(dips_df):
    # If literally nothing is in dip now, stay quiet to avoid spam.
    if dips_df.empty:
        return

    seen = load_seen()
    old_rows = []   # previously alerted & still dipping
    new_rows = []   # first time seen this run/day

    # Classify rows as old/new using the de-dup state
    for _, r in dips_df.iterrows():
        key = "DIP|" + buy_key(r)  # e.g., DIP|AAPL|1D|2025-10-14
        if key in seen:
            old_rows.append(r)
        else:
            new_rows.append(r)
            seen.add(key)

    # Nicely formatted line for each row
    def _fmt(row):
        close    = row.get("close",    float("nan"))
        sug_buy  = row.get("sug_buy",  float("nan"))
        sug_stop = row.get("sug_stop", float("nan"))
        return (f"- {row['ticker']} [{row['tf']}] "
                f"close {close:.2f} ≤ sug_buy {sug_buy:.2f} "
                f"(stop {sug_stop:.2f})")

    # Build the message with two always-present sections
    lines = [f"Stock status V1 {datetime.now().strftime('%Y-%m-%d %H:%M')}:",
             "",
             "Still below sug_buy (previously alerted):",
             "======="]

    if old_rows:
        # sort for stable output
        for r in sorted(old_rows, key=lambda x: (x['tf'], x['ticker'])):
            lines.append(_fmt(r))
    else:
        lines.append("— none —")

    lines += ["",
              "New",
              "======="]

    if new_rows:
        for r in sorted(new_rows, key=lambda x: (x['tf'], x['ticker'])):
            lines.append(_fmt(r))
    else:
        lines.append("— none —")

    # Send the message
    notify_all_channels("\n".join(lines))

    # Persist only when something new appeared
    if new_rows:
        save_seen(seen)


def run_once(first_run=False):
    if IDE_PRINT_TITLE and first_run:
        print("="*88); print(TITLE_TEXT); print("="*88)

    if first_run and TELEGRAM_PING_ON_START:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        notify_all_channels(f"Scanner is live at {ts} (UTC).")

    df = build_dataframe()

    desired = [
        "ticker","tf","date","sug_buy","close","buy","sug_stop","stop","poc",
        "ema50","ema200","adx","aroon_up","aroon_dn","hvn_below","lvn_below",
        "div_cnt","div_names","reasons","error"
    ]
    df = df[[c for c in desired if c in df.columns]]

    print(df.to_string(index=False))

    out = "scanner_dual_tf_vp.csv"
    df.to_csv(out, index=False)
    print("\nSaved:", out)

    # Dips + new BUYs
    dips_df = _compute_dips(df)
    _send_dip_alerts(dips_df)

    seen = load_seen()
    buys = df[(df["buy"]=="YES")]
    new_rows = []
    for _, r in buys.iterrows():
        key = buy_key(r)
        if key not in seen:
            new_rows.append(r); seen.add(key)
    if new_rows:
        lines = ["BUY signals:", "", TITLE_TEXT, ""]
        for r in new_rows:
            poc = r.get("poc", None); hvn = r.get("hvn_below", None); lvn = r.get("lvn_below", None)
            poc_s = ("POC " + str(poc)) if pd.notna(poc) else "POC n/a"
            hvn_s = ("HVN " + str(hvn)) if pd.notna(hvn) else "HVN n/a"
            lvn_s = ("LVN " + str(lvn)) if pd.notna(lvn) else "LVN n/a"
            lines.append(f"- {r['ticker']} [{r['tf']}] @ {r['close']} | Stop {r['stop']} | Div({r['div_cnt']}): {r['div_names']} | {poc_s}, {hvn_s}, {lvn_s}")
        notify_all_channels("\n".join(lines))
        save_seen(seen)

    # Optional summary even if nothing new
    if ALWAYS_NOTIFY:
        summary = [
            f"Summary {datetime.now(ZoneInfo(MARKET_TZ)).strftime('%Y-%m-%d %H:%M %Z')}",
            f"Universe size: {len(TICKERS)}",
            f"BUY tickers now: {', '.join(sorted(buys['ticker'].tolist())) or '—'}",
            f"Dips beneath sug_buy: {', '.join(sorted(dips_df['ticker'].tolist())) if not dips_df.empty else '—'}",
        ]
        notify_all_channels("\n".join(summary))

    if (not first_run) and TELEGRAM_PING_ON_END:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        notify_all_channels(f"Scanner iteration completed at {ts} (UTC).")

def main_loop():
    first = True
    while True if IDE_LOOP else False:
        if is_market_open_now():
            run_once(first_run=first); first = False
        else:
            print("[info] market closed; sleeping...")
        time.sleep(max(60, int(IDE_INTERVAL_MIN*60)))
    if not IDE_LOOP:
        run_once(first_run=True)

# ------------------------------ Entry ------------------------------ #
def market_open_now() -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    return (now.weekday() < 5) and (dtime(9,30) <= now.time() <= dtime(16,0))

if __name__ == "__main__":
    # Respect market hours unless overridden by env MARKET_ONLY=false
    if MARKET_ONLY and not market_open_now():
        print("[done] Market closed — no scanning (set MARKET_ONLY=false or use the workflow toggle).", flush=True)
        sys.exit(0)
    main_loop()









