# streamlit_app.py — 0DTE Cockpit (SPY/QQQ) — TAAPI-first Charts
# - Charts: TAAPI.io candles by default (fallback to yfinance)
# - Bias engine (VWAP, ORB, breadth, futures)
# - Confidence Meter (RSI, MACD hist, EMA9>20)
# - Candle Watch (1m & 5m)
# - Perfect-setup toast alerts
# - Expected Move band (SPX/VIX-based)
# - Time-of-day warnings
# - Max daily loss guardrail
# - Single chart section (VWAP + ORB + EM band)
# - Trade logger & CSV
#
# ENV (Streamlit Cloud secrets):
#   TAAPI_SECRET="your_taapi_key_here"
#   (optional) TAAPI_URL="https://api.taapi.io/bulk"

import os, time, math, datetime as dt
import numpy as np, pandas as pd
import streamlit as st, yfinance as yf, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objs as go
from zoneinfo import ZoneInfo

st.set_page_config(page_title="0DTE Cockpit — SPY & QQQ", layout="wide")

# ====== Secrets & Config ======
def _get_secret(name, default=""):
    # flat key
    if name in st.secrets:
        return st.secrets.get(name, default)
    # nested block support, e.g. [taapi] secret="..."
    if "taapi" in st.secrets and isinstance(st.secrets["taapi"], dict):
        low = name.lower()
        if low in st.secrets["taapi"]:
            return st.secrets["taapi"][low]
    # env fallback (local dev)
    return os.getenv(name, default)

TAAPI_SECRET = _get_secret("TAAPI_SECRET")
if not TAAPI_SECRET:
    st.error("Missing **TAAPI_SECRET**. Add it in your app’s *Settings → Secrets* (or env var) and rerun.")
    st.stop()

TAAPI_URL = _get_secret("TAAPI_URL", "https://api.taapi.io/bulk")

INTERVAL = "1m"                  # TAAPI stocks/ETFs: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
DEFAULT_REFRESH_SECS = 5
TRADE_LOG = "0dte_trade_log.csv"
SECTORS = ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY","XLC","XLRE"]
ORB_MINUTES = 15

CORE_INDIS = [
    {"indicator": "price"},
    {"indicator": "rsi", "period": 14},
    {"indicator": "macd"},
    {"indicator": "ema", "period": 9},
    {"indicator": "ema", "period": 20},
    {"indicator": "ema", "period": 50},
    {"indicator": "ema", "period": 200},
    {"indicator": "vwap"},
    {"indicator": "supertrend"},
    {"indicator": "stochrsi"},
]

# ====== Robust HTTP for TAAPI ======
def make_session():
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False, raise_on_redirect=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
    return s

SESSION = make_session()

# ====== TAAPI helpers (STOCKS/ETFs: no exchange prefix) ======
def make_stock_construct(symbol: str, interval: str, indicators: list[dict]) -> dict:
    """
    TAAPI 'stocks' style construct: use SYMBOL:INTERVAL (no exchange).
    Example: 'SPY:1m'
    """
    return {"construct": f"{symbol.upper()}:{interval}", "indicators": indicators}

def taapi_bulk(constructs, timeout_sec=45):
    """POST /bulk with one or more constructs. Return a dict[id -> result]."""
    try:
        payload = {"secret": TAAPI_SECRET, "construct": constructs if len(constructs) != 1 else constructs[0]}
        r = SESSION.post(TAAPI_URL, json=payload, timeout=(5, timeout_sec))
        r.raise_for_status()
        j = r.json()
        data = j.get("data", [])
        # Map TAAPI items by their 'id' so we can fuzzy-match later
        out = {}
        for it in data:
            _id = it.get("id")
            res = it.get("result", {})
            if _id:
                out[_id] = res
        # if TAAPI returned a different shape, at least return something debuggable
        if not out and isinstance(j, dict):
            # fallback: map whole response
            out["__raw__"] = j
        return out
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = r.text[:800]
        except Exception:
            pass
        st.error(f"TAAPI bulk call failed: HTTP {getattr(r,'status_code', '???')} — {e}\n\nResponse: `{body}`")
        return {}
    except requests.exceptions.RequestException as e:
        st.error(f"Network/timeout error calling TAAPI: {e}")
        return {}
    except Exception as e:
        st.error(f"Unexpected TAAPI error: {e}")
        return {}

def get_taapi_candles(symbol, interval="1m", limit=300, timeout_sec=45):
    """Fetch OHLCV candles via TAAPI 'candles' indicator for STOCKS/ETFs."""
    try:
        construct = {
            "construct": f"{symbol.upper()}:{interval}",
            "indicators": [{"indicator": "candles", "backtrack": int(limit)}],
        }
        payload = {"secret": TAAPI_SECRET, "construct": construct}
        r = SESSION.post(TAAPI_URL, json=payload, timeout=(5, timeout_sec))
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()
        item = data[0].get("result", {}) if isinstance(data[0], dict) else {}
        values = item.get("values") or item.get("result") or item.get("data")
        if not isinstance(values, list) or not values:
            return pd.DataFrame()
        records = []
        for c in values:
            ts = c.get("timestamp") or c.get("time") or c.get("t")
            o = c.get("open") or c.get("o")
            h = c.get("high") or c.get("h")
            l = c.get("low")  or c.get("l")
            cl= c.get("close")or c.get("c")
            v = c.get("volume") or c.get("v") or 0.0
            if ts is None or o is None or h is None or l is None or cl is None:
                continue
            ts = int(ts)
            idx = pd.to_datetime(ts, unit="ms" if ts > 10_000_000_000 else "s")
            records.append([idx, float(o), float(h), float(l), float(cl), float(v)])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records, columns=["Datetime","Open","High","Low","Close","Volume"])\
               .set_index("Datetime").sort_index()
        return df
    except Exception:
        return pd.DataFrame()

# ====== General helpers ======
def first_numeric(d):
    if isinstance(d, dict):
        for k in ("value","valueMACD","valueMACDSignal","valueMACDHist"):
            if k in d and isinstance(d[k], (int,float)): return float(d[k])
        for v in d.values():
            if isinstance(v,(int,float)): return float(v)
            if isinstance(v,dict):
                for vv in v.values():
                    if isinstance(vv,(int,float)): return float(vv)
    return float("nan")

def get_value_by_id(results: dict, id_contains: str, default=float("nan")):
    # Fuzzy match TAAPI item ids
    for id_str, result in results.items():
        if id_contains in id_str:
            num = first_numeric(result)
            if isinstance(num,(int,float)) and not math.isnan(num):
                return float(num)
    return default

def compute_vwap_from_df(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, np.nan)

def get_intraday_1m_yf(ticker, period="1d"):
    try:
        df = yf.download(ticker, period=period, interval="1m", auto_adjust=False, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.rename(columns=str.title).dropna()
    except Exception:
        return pd.DataFrame()

def first5_momentum(df):
    if df is None or df.empty: return None
    df5 = df.resample("5min").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
    if len(df5)==0: return None
    c = df5.iloc[0]
    rng = max(c["High"]-c["Low"], 1e-6)
    if c["Close"] > c["Open"] and c["Close"] >= (c["High"] - 0.2*rng): return True
    if c["Close"] < c["Open"] and c["Close"] <= (c["Low"] + 0.2*rng): return False
    return None

def get_futures_quote(sym):
    try:
        day = yf.download(sym, period="2d", interval="1d", auto_adjust=False, progress=False)
        last_close = float(day["Close"].iloc[-1])
        prev_close = float(day["Close"].iloc[-2]) if len(day)>=2 else last_close
        intr = yf.download(sym, period="1d", interval="1m", auto_adjust=False, progress=False)
        last = float(intr["Close"].iloc[-1]) if not intr.empty else last_close
        chg = ((last - prev_close) / prev_close * 100.0) if prev_close else float("nan")
        return last, chg
    except Exception:
        return float("nan"), float("nan")

def sector_breadth_yf(tickers):
    try:
        data = yf.download(tickers, period="1d", interval="1m", auto_adjust=False, progress=False, group_by='ticker')
    except Exception:
        data = None
    rows = []
    green = 0; total = 0
    for t in tickers:
        try:
            df = data[t] if data is not None else get_intraday_1m_yf(t)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns=str.title).dropna()
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty:
            rows.append((t, float("nan"), float("nan"), False))
            continue
        last = float(df["Close"].iloc[-1])
        vwap = float(compute_vwap_from_df(df).iloc[-1])
        ok = not (math.isnan(last) or math.isnan(vwap))
        is_green = (ok and last > vwap)
        rows.append((t, last if ok else float("nan"), vwap if ok else float("nan"), is_green))
        if ok:
            total += 1; green += 1 if is_green else 0
    breadth = (green / total * 100.0) if total else float("nan")
    return rows, breadth

def round_magnets(price, step=5.0):
    if price is None or (isinstance(price, float) and math.isnan(price)): return []
    base = round(price / step) * step
    return [round(base - 2*step, 2), round(base - step, 2), round(base, 2), round(base + step, 2), round(base + 2*step, 2)]

def choose_strike(last, decision, step=1.0):
    mags = round_magnets(last, 5.0)
    if not mags or math.isnan(last): return None
    if "CALL" in decision:
        above = sorted([m for m in mags if m >= last])
        target = above[0] if above else mags[-1]
    elif "PUT" in decision:
        below = sorted([m for m in mags if m <= last], reverse=True)
        target = below[0] if below else mags[0]
    else:
        return None
    return round(target / step) * step

def compute_bias(spy_last, qqq_last, spy_vwap, qqq_vwap, breadth_pct, f5_spy, f5_qqq, es_green, nq_green):
    score = 0.0; parts = {}
    for tag, g in [("ES", es_green), ("NQ", nq_green)]:
        if g is not None: s = 1.0 if g else -1.0; score += s; parts[tag]=s
    for tag, g in [("SPY 1st5", f5_spy), ("QQQ 1st5", f5_qqq)]:
        if g is not None: s = 1.0 if g else -1.0; score += s; parts[tag]=s
    for tag, g in [("SPY>VWAP", (spy_last>spy_vwap) if not (math.isnan(spy_last) or math.isnan(spy_vwap)) else None),
                   ("QQQ>VWAP", (qqq_last>qqq_vwap) if not (math.isnan(qqq_last) or math.isnan(qqq_vwap)) else None)]:
        if g is not None: s = 1.0 if g else -1.0; score += s; parts[tag]=s
    if not math.isnan(breadth_pct):
        s = 1.0 if breadth_pct>=60 else (-1.0 if breadth_pct<=40 else 0.0)
        score += s; parts["Breadth"] = s
    decision = "CALL bias" if score>=2 else ("PUT bias" if score<=-2 else "NEUTRAL / WAIT")
    return score, decision, parts

def compute_opening_range(df, minutes=15):
    if df is None or df.empty: return None, None, float("nan"), float("nan")
    day = df.index[-1].date()
    todays = df[df.index.date == day]
    if todays.empty: return None, None, float("nan"), float("nan")
    start_ts = todays.index[0]
    end_ts = start_ts + pd.Timedelta(minutes=minutes)
    window = todays[(todays.index >= start_ts) & (todays.index <= end_ts)]
    if window.empty: return start_ts, end_ts, float("nan"), float("nan")
    or_high = float(window["High"].max()); or_low  = float(window["Low"].min())
    return start_ts, end_ts, or_high, or_low

# Confidence meter inputs from TAAPI
def taapi_confidence_bits(core: dict, prefix: str):
    rsi = get_value_by_id(core, prefix + "rsi_14_")
    macd_hist = float("nan")
    for k, v in core.items():
        if prefix + "macd_" in k and isinstance(v, dict):
            macd_hist = v.get("valueMACDHist", float("nan")); break
    ema9  = get_value_by_id(core, prefix + "ema_9_")
    ema20 = get_value_by_id(core, prefix + "ema_20_")
    bits = {"rsi_up": (rsi>55) if not math.isnan(rsi) else None,
            "macd_green": (macd_hist>0) if not math.isnan(macd_hist) else None,
            "ema_stack": (ema9>ema20) if not (math.isnan(ema9) or math.isnan(ema20)) else None}
    return bits

def confidence_label(bits, decision):
    score = 0; used = []
    for key, name in [("rsi_up","RSI>55"),("macd_green","MACD hist>0"),("ema_stack","EMA9>20")]:
        val = bits.get(key, None)
        if val is True: score += 1; used.append(name)
        elif val is False: score -= 1; used.append("-"+name)
    if "WAIT" in decision: return "Low", score, used
    if score >= 2: return "🔥 High", score, used
    if score == 1: return "Medium", score, used
    if score <= -1: return "Low", score, used
    return "Medium", score, used

def last_n_dirs(df, n=3):
    if df is None or df.empty: return []
    closes = df["Close"].tail(n).tolist(); opens = df["Open"].tail(n).tolist()
    out = []
    for o, c in zip(opens, closes):
        if c > o: out.append(True)
        elif c < o: out.append(False)
        else: out.append(None)
    return out

def candle_emojis(df, n=3):
    if df is None or df.empty: return "—"
    closes = df["Close"].tail(n).tolist(); opens = df["Open"].tail(n).tolist()
    out = []
    for o, c in zip(opens, closes):
        if c > o: out.append("🟩")
        elif c < o: out.append("🟥")
        else: out.append("➖")
    return " ".join(out)

def resample_5m(df):
    if df is None or df.empty: return pd.DataFrame()
    return df.resample("5min").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

# ====== Expected Move (VIX-based) ======
def expected_move_vix():
    """Approximate 1-day expected move using VIX: EM ≈ SPX * (VIX/100)/sqrt(252)."""
    try:
        spx = yf.download("^GSPC", period="2d", interval="1d", progress=False)
        vix = yf.download("^VIX",  period="2d", interval="1d", progress=False)
        spx_last = float(spx["Close"].iloc[-1]); vix_last = float(vix["Close"].iloc[-1])
        em = spx_last * (vix_last/100.0) / math.sqrt(252)
        return spx_last, vix_last, em
    except Exception:
        return float("nan"), float("nan"), float("nan")

# ====== Sidebar ======
with st.sidebar:
    st.title("⚙️ Settings")
    enable_refresh = st.checkbox("Enable auto-refresh", value=True)
    freeze_charts = st.checkbox("Freeze charts (keep zoom)", value=False)
    if freeze_charts: enable_refresh = False
    refresh = st.slider("Refresh every (sec)", 3, 30, DEFAULT_REFRESH_SECS, step=1)
    st.divider()
    st.caption("Risk / Budget")
    acct = st.number_input("Account size ($)", min_value=50.0, value=100.0, step=50.0)
    risk_pct = st.slider("Risk per trade (%)", 1, 10, 3, step=1)
    est_cost = st.number_input("Est. option cost ($/contract)", min_value=1.0, value=20.0, step=1.0)
    max_risk = acct * (risk_pct/100.0)
    max_contracts = max(1, int(max_risk // est_cost))
    st.metric("Max risk / contracts", f"${max_risk:.0f}", f"{max_contracts}x")
    st.caption("Default exit: +50–100% target / −50% stop.")
    st.divider()
    daily_limit = st.number_input("Max daily loss ($)", min_value=0.0, value=100.0, step=10.0)

# ====== Header ======
st.header("0DTE Cockpit — SPY & QQQ")

# ====== TAAPI indicators for confidence + core prices ======
# Use stock constructs: "SYMBOL:INTERVAL" (no exchange)
core = taapi_bulk([
    make_stock_construct("SPY", INTERVAL, CORE_INDIS),
    make_stock_construct("QQQ", INTERVAL, CORE_INDIS),
], timeout_sec=60)

# For stocks, TAAPI bulk ids look like "<SYMBOL>_<INTERVAL>_<indicator>_..."
SPY_PREFIX = f"SPY_{INTERVAL}_"
QQQ_PREFIX = f"QQQ_{INTERVAL}_"

spy_last = get_value_by_id(core, SPY_PREFIX + "price_")
qqq_last = get_value_by_id(core, QQQ_PREFIX + "price_")
spy_vwap = get_value_by_id(core, SPY_PREFIX + "vwap_")
qqq_vwap = get_value_by_id(core, QQQ_PREFIX + "vwap_")

# ====== Futures & Expected Move ======
colF1, colF2, colF3 = st.columns(3)
es_last, es_chg = get_futures_quote("ES=F")
nq_last, nq_chg = get_futures_quote("NQ=F")
spx_last, vix_last, em_spx = expected_move_vix()
with colF1: st.metric("ES=F last", f"{es_last:.2f}" if not math.isnan(es_last) else "—", f"{es_chg:.2f}%")
with colF2: st.metric("NQ=F last", f"{nq_last:.2f}" if not math.isnan(nq_last) else "—", f"{nq_chg:.2f}%")
with colF3: st.metric("SPX/VIX EM (±)", f"{em_spx:.1f}" if not math.isnan(em_spx) else "—", f"VIX {vix_last:.1f}" if not math.isnan(vix_last) else "")

# ====== Candles for charts — TAAPI-first with fallback to yfinance ======
spy_df = get_taapi_candles("SPY", interval=INTERVAL)
qqq_df = get_taapi_candles("QQQ", interval=INTERVAL)
if spy_df is None or spy_df.empty: spy_df = get_intraday_1m_yf("SPY")
if qqq_df is None or qqq_df.empty: qqq_df = get_intraday_1m_yf("QQQ")

# ====== First 5-min momentum, ORB, Breadth ======
f5_spy = first5_momentum(spy_df); f5_qqq = first5_momentum(qqq_df)
sSPY, eSPY, orhSPY, orlSPY = compute_opening_range(spy_df, minutes=ORB_MINUTES)
sQQQ, eQQQ, orhQQQ, orlQQQ = compute_opening_range(qqq_df, minutes=ORB_MINUTES)

rows, breadth_pct = sector_breadth_yf(SECTORS)
st.subheader("Sector breadth: price > VWAP")
if not math.isnan(breadth_pct):
    st.progress(int(min(max(breadth_pct,0),100)))
    st.caption(f"{breadth_pct:.0f}% sectors green • {sum([1 for r in rows if not math.isnan(r[1])]):d} loaded")
else:
    st.caption("Waiting for sector data...")
st.dataframe(pd.DataFrame(rows, columns=["Sector","Last","VWAP","Green"])
             .style.format({"Last":"{:.2f}","VWAP":"{:.2f}"}), use_container_width=True)

# ====== Bias & Confidence ======
es_green = (es_chg > 0) if not math.isnan(es_chg) else None
nq_green = (nq_chg > 0) if not math.isnan(nq_chg) else None
score, decision, parts = compute_bias(spy_last, qqq_last, spy_vwap, qqq_vwap, breadth_pct, f5_spy, f5_qqq, es_green, nq_green)
spy_bits = taapi_confidence_bits(core, SPY_PREFIX); qqq_bits = taapi_confidence_bits(core, QQQ_PREFIX)
conf_label_spy, conf_score_spy, _ = confidence_label(spy_bits, decision)
conf_label_qqq, conf_score_qqq, _ = confidence_label(qqq_bits, decision)

# ====== Candle Watch ======
def candle_emojis(df, n=3):
    if df is None or df.empty: return "—"
    closes = df["Close"].tail(n).tolist(); opens = df["Open"].tail(n).tolist()
    out = []
    for o, c in zip(opens, closes):
        if c > o: out.append("🟩")
        elif c < o: out.append("🟥")
        else: out.append("➖")
    return " ".join(out)

spy_1m_watch = candle_emojis(spy_df, n=3); qqq_1m_watch = candle_emojis(qqq_df, n=3)
spy_5m_watch = candle_emojis(resample_5m(spy_df), n=1); qqq_5m_watch = candle_emojis(resample_5m(qqq_df), n=1)

st.subheader("Bias, Confidence & Suggestions")
met1, met2, met3, met4 = st.columns(4)
with met1: st.metric("Bias score", f"{score:.1f}")
with met2: st.metric("Decision", decision)
with met3: st.metric("SPY confidence", conf_label_spy, f"{conf_score_spy:+d}")
with met4: st.metric("QQQ confidence", conf_label_qqq, f"{conf_score_qqq:+d}")
st.caption("Bias parts: " + " | ".join([f"{k}:{'+' if v>0 else '-' if v<0 else '0'}" for k,v in parts.items()]) if parts else "—")

def sugg_for(symbol, last):
    if last is None or math.isnan(last): return "—"
    strike = choose_strike(last, decision, step=1.0)
    if strike is None: return "—"
    dir_word = "CALL" if "CALL" in decision else ("PUT" if "PUT" in decision else "WAIT")
    if dir_word == "WAIT": return "WAIT for confirmation"
    return f"{symbol} **{dir_word}** near strike **{strike:.0f}** (magnet-based)"

cTop1, cTop2 = st.columns(2)
with cTop1:
    st.info(sugg_for("SPY", spy_last))
    st.write(f"🕯️ 1m: {spy_1m_watch} • 5m: {spy_5m_watch}")
with cTop2:
    st.info(sugg_for("QQQ", qqq_last))
    st.write(f"🕯️ 1m: {qqq_1m_watch} • 5m: {qqq_5m_watch}")

# ====== Time-of-day warnings (US/Eastern) ======
now_et = dt.datetime.now(ZoneInfo("America/New_York"))
rth_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
rth_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
if now_et < rth_open + dt.timedelta(minutes=5):
    st.warning("First 5 minutes of RTH — spreads/whipsaws elevated. Consider waiting for confirmation.")
elif now_et > rth_close - dt.timedelta(minutes=30):
    st.warning("Last 30 minutes of RTH — increased gamma pin/reversals. Manage risk tightly.")

# ====== Perfect-setup Alerts (3x 1m same dir + 5m agree + VWAP/OR confirm) ======
def is_perfect(decision, conf_label, df_1m, last, vwap, or_high, or_low):
    if "WAIT" in decision or conf_label != "🔥 High": return False, ""
    dir_up = "CALL" in decision
    dirs_1m = last_n_dirs(df_1m, n=3)
    if len(dirs_1m)<3 or any(d is None for d in dirs_1m) or not (dirs_1m[0]==dirs_1m[1]==dirs_1m[2]==dir_up):
        return False, ""
    last5 = last_n_dirs(resample_5m(df_1m), n=1)
    if not last5 or last5[-1] != dir_up: return False, ""
    if math.isnan(last) or math.isnan(vwap) or math.isnan(or_high) or math.isnan(or_low): return False, ""
    if dir_up and not (last > vwap and last > or_high): return False, ""
    if (not dir_up) and not (last < vwap and last < or_low): return False, ""
    target_strike = choose_strike(last, decision, step=1.0)
    magnets = round_magnets(last, 5.0); near = min(magnets, key=lambda m: abs(m-last)) if magnets else None
    price_note = f"Pullback toward {'ORH' if dir_up else 'ORL'} or magnet ~{near:.0f}" if near else "Pullback toward OR level"
    call_range = f"{int(target_strike-1)}–{int(target_strike+1)}" if target_strike is not None else "ATM ±1"
    msg = f"Perfect: **{decision}** • Entry: {price_note} • Strike: **{call_range}**"
    return True, msg

ok_spy, msg_spy = is_perfect(decision, conf_label_spy, spy_df, spy_last, spy_vwap, orhSPY, orlSPY)
if ok_spy: st.toast(f"SPY: {msg_spy}", icon="✅")
ok_qqq, msg_qqq = is_perfect(decision, conf_label_qqq, qqq_df, qqq_last, qqq_vwap, orhQQQ, orlQQQ)
if ok_qqq: st.toast(f"QQQ: {msg_qqq}", icon="✅")

# ====== ORB Break Alerts ======
def orb_break_alert(ticker, df, orh, orl):
    if df is None or df.empty or math.isnan(orh) or math.isnan(orl): return
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df)>=2 else last_close
    if last_close > orh and prev_close <= orh:
        st.toast(f"{ticker}: ORH break/close ↑", icon="📈")
    if last_close < orl and prev_close >= orl:
        st.toast(f"{ticker}: ORL break/close ↓", icon="📉")
orb_break_alert("SPY", spy_df, orhSPY, orlSPY)
orb_break_alert("QQQ", qqq_df, orhQQQ, orlQQQ)

# ====== Trade Logger ======
st.subheader("Trade Logger")
with st.form("trade_log_form", clear_on_submit=False):
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1: sym = st.selectbox("Symbol", ["SPY","QQQ"], index=0)
    with col2: side = st.selectbox("Direction", ["CALL","PUT"], index=0)
    with col3:
        base_last = spy_last if sym=="SPY" else qqq_last
        strike_default = float("nan") if (base_last is None or math.isnan(base_last)) else float(round(base_last))
        strike = st.number_input("Strike", min_value=0.0, value=strike_default, step=1.0)
    with col4: qty = st.number_input("Qty (contracts)", min_value=1, value=1, step=1)
    col5, col6, col7 = st.columns([1,1,2])
    with col5: entry = st.number_input("Entry ($/contract)", min_value=0.0, value=20.0, step=0.5)
    with col6: exitp = st.number_input("Exit ($/contract)", min_value=0.0, value=0.0, step=0.5)
    with col7: notes = st.text_input("Notes", value="")
    submitted = st.form_submit_button("Add trade")
    if submitted:
        pl_pct = ( (exitp - entry) / entry * 100.0 ) if entry>0 and exitp>0 else float("nan")
        row = {"time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
               "symbol": sym, "side": side, "strike": strike, "qty": qty,
               "entry": entry, "exit": exitp, "p_l_pct": pl_pct, "notes": notes}
        try:
            if os.path.exists(TRADE_LOG):
                df_log = pd.read_csv(TRADE_LOG); df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
            else:
                df_log = pd.DataFrame([row])
            df_log.to_csv(TRADE_LOG, index=False); st.success("Trade added to log.")
        except Exception as e:
            st.error(f"Failed to write log: {e}")

# Show last trades + download + daily P/L
todays_pl = 0.0
if os.path.exists(TRADE_LOG):
    tdf = pd.read_csv(TRADE_LOG)
    tdf["date"] = pd.to_datetime(tdf["time"]).dt.date
    tdf_today = tdf[tdf["date"] == pd.Timestamp.now().date()].copy()
    if not tdf_today.empty:
        tdf_today["pl_$"] = tdf_today.apply(lambda r: (r["p_l_pct"]/100.0) * r["entry"] * r["qty"] if pd.notnull(r["p_l_pct"]) else 0.0, axis=1)
        todays_pl = float(tdf_today["pl_$"].sum())
    tdf_tail = tdf.tail(20).copy().drop(columns=["date"], errors="ignore")
    st.dataframe(tdf_tail, use_container_width=True, hide_index=True)
    csv_bytes = tdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download full trade log (CSV)", data=csv_bytes, file_name="0dte_trade_log.csv", mime="text/csv")
else:
    st.caption("No trades logged yet.")

# ====== Max daily loss guardrail ======
daily_limit = st.session_state.get("daily_limit_value", None)
st.caption(f"Today's P/L (approx): ${todays_pl:.2f}")

# ====== Charts — VWAP + ORB + EM band (single section) ======
st.subheader(f"Charts — 1m with VWAP + Opening Range ({ORB_MINUTES}m) + Expected Move (SPX)")
colC1, colC2 = st.columns(2, gap="large")

def plot_with_orb_em(ticker, df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    vwap_series = compute_vwap_from_df(df)
    fig.add_trace(go.Scatter(x=df.index, y=vwap_series, mode="lines", name="VWAP"))
    s, e, orh, orl = compute_opening_range(df, minutes=ORB_MINUTES)
    if s is not None and e is not None and not (math.isnan(orh) or math.isnan(orl)):
        fig.add_hrect(y0=orl, y1=orh, x0=s, x1=e, opacity=0.15, line_width=0, fillcolor="LightSkyBlue")
        fig.add_hline(y=orh, line_dash="dot", opacity=0.5); fig.add_hline(y=orl, line_dash="dot", opacity=0.5)
    # Expected move band (translate from SPX EM to % move)
    try:
        last_px = float(df["Close"].iloc[-1])
        spx_last, vix_last, em_spx = expected_move_vix()
        if not math.isnan(em_spx) and spx_last>0:
            pct = em_spx / spx_last
            up = last_px * (1 + pct); dn = last_px * (1 - pct)
            fig.add_hline(y=up, line_dash="dash", opacity=0.2)
            fig.add_hline(y=dn, line_dash="dash", opacity=0.2)
    except Exception:
        pass
    fig.update_layout(title=f"{ticker} — 1m", height=420, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=50,b=10))
    return fig

with colC1:
    if spy_df.empty: st.warning("No SPY 1m data.")
    else: st.plotly_chart(plot_with_orb_em("SPY", spy_df), use_container_width=True, key="spy_chart")
with colC2:
    if qqq_df.empty: st.warning("No QQQ 1m data.")
    else: st.plotly_chart(plot_with_orb_em("QQQ", qqq_df), use_container_width=True, key="qqq_chart")

# ====== Auto-refresh ======
with st.sidebar:
    enable_refresh = st.checkbox("Enable auto-refresh", value=True, key="enable_refresh_2")
    refresh = st.slider("Refresh every (sec)", 3, 30, DEFAULT_REFRESH_SECS, step=1, key="refresh_2")
if enable_refresh:
    time.sleep(refresh)
    st.rerun()
