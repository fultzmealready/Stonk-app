# streamlit_app.py — 0DTE Cockpit (SPY/QQQ)
# - Exactly TWO charts (SPY, QQQ) with 24h view + RTH shading
# - TAAPI-first for indicators & candles (fallback to yfinance)
# - Bias engine, confidence, ORB, sector breadth
# - Simple trade logger in this file (since trade_log module not present)

import os
import time
import math
import datetime as dt
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

# ===== Import our local modules  =====
from tappi_client import taapi_bulk, make_stock_construct, get_taapi_candles
from data_sources import get_intraday_1m_yf, get_futures_quote, sector_breadth_yf
from indicators import (
    compute_bias, taapi_confidence_bits, confidence_label, compute_opening_range,
    compute_vwap_from_df, first5_momentum, last_n_dirs, candle_emojis, resample_5m,
    get_value_by_id, choose_strike
)
from plotting import plot_with_orb_em

st.set_page_config(page_title="0DTE Cockpit — SPY & QQQ", layout="wide")

# ====== Constants ======
INTERVAL = "1m"
DEFAULT_REFRESH_SECS = 5
SECTORS = ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY","XLC","XLRE"]
ORB_MINUTES = 15
TRADE_LOG = "0dte_trade_log.csv"

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

# ====== Sidebar ======
with st.sidebar:
    st.title("⚙️ Settings")
    enable_refresh = st.checkbox("Enable auto-refresh", value=True, key="enable_refresh")
    freeze_charts = st.checkbox("Freeze charts (keep zoom)", value=False, key="freeze_charts")
    if freeze_charts:
        enable_refresh = False
    refresh = st.slider("Refresh every (sec)", 3, 30, DEFAULT_REFRESH_SECS, step=1, key="refresh_secs")

    st.divider()
    st.caption("Risk / Budget")
    acct = st.number_input("Account size ($)", min_value=50.0, value=100.0, step=50.0, key="acct_size")
    risk_pct = st.slider("Risk per trade (%)", 1, 10, 3, step=1, key="risk_pct")
    est_cost = st.number_input("Est. option cost ($/contract)", min_value=1.0, value=20.0, step=1.0, key="est_cost")
    max_risk = acct * (risk_pct/100.0)
    max_contracts = max(1, int(max_risk // est_cost)) if est_cost > 0 else 1
    st.metric("Max risk / contracts", f"${max_risk:.0f}", f"{max_contracts}x")
    st.caption("Default exit: +50–100% target / −50% stop.")
    st.divider()
    daily_limit_input = st.number_input("Max daily loss ($)", min_value=0.0, value=100.0, step=10.0, key="daily_limit")

# ====== Header ======
st.header("0DTE Cockpit — SPY & QQQ")

# ====== Expected Move (VIX-based) helper ======
def expected_move_vix():
    """Approximate 1-day expected move using VIX: EM ≈ SPX * (VIX/100)/sqrt(252)."""
    import yfinance as yf
    try:
        spx = yf.download("^GSPC", period="2d", interval="1d", progress=False)
        vix = yf.download("^VIX",  period="2d", interval="1d", progress=False)
        spx_last = float(spx["Close"].iloc[-1]); vix_last = float(vix["Close"].iloc[-1])
        em = spx_last * (vix_last/100.0) / math.sqrt(252)
        return spx_last, vix_last, em
    except Exception:
        return float("nan"), float("nan"), float("nan")

# ====== TAAPI indicators for confidence + core prices ======
core = taapi_bulk([
    make_stock_construct("SPY", INTERVAL, CORE_INDIS),
    make_stock_construct("QQQ", INTERVAL, CORE_INDIS),
], timeout_sec=60)

# TAAPI stock IDs look like: "stocks_SPY_1m_<indicator>_..."
SPY_PREFIX = f"stocks_SPY_{INTERVAL}_"
QQQ_PREFIX = f"stocks_QQQ_{INTERVAL}_"

spy_last = get_value_by_id(core, SPY_PREFIX + "price_")
qqq_last = get_value_by_id(core, QQQ_PREFIX + "price_")
spy_vwap_taapi = get_value_by_id(core, SPY_PREFIX + "vwap_")
qqq_vwap_taapi = get_value_by_id(core, QQQ_PREFIX + "vwap_")

# ====== Futures & Expected Move ======
colF1, colF2, colF3 = st.columns(3)
es_last, es_chg = get_futures_quote("ES=F")
nq_last, nq_chg = get_futures_quote("NQ=F")
spx_last, vix_last, em_spx = expected_move_vix()
with colF1:
    st.metric("ES=F last", f"{es_last:.2f}" if not math.isnan(es_last) else "—", f"{es_chg:.2f}%")
with colF2:
    st.metric("NQ=F last", f"{nq_last:.2f}" if not math.isnan(nq_last) else "—", f"{nq_chg:.2f}%")
with colF3:
    st.metric("SPX/VIX EM (±)", f"{em_spx:.1f}" if not math.isnan(em_spx) else "—",
              f"VIX {vix_last:.1f}" if not math.isnan(vix_last) else "")

# ====== Candles for charts — TAAPI-first with fallback to yfinance (24h incl. pre/post) ======
spy_df = get_taapi_candles("SPY", interval=INTERVAL)
qqq_df = get_taapi_candles("QQQ", interval=INTERVAL)
if spy_df is None or spy_df.empty:
    spy_df = get_intraday_1m_yf("SPY", period="2d", include_ext_hours=True)
if qqq_df is None or qqq_df.empty:
    qqq_df = get_intraday_1m_yf("QQQ", period="2d", include_ext_hours=True)

# If we used TAAPI candles, compute VWAP from price/vol (more stable for chart overlay)
spy_vwap = spy_vwap_taapi if not math.isnan(spy_vwap_taapi) else (compute_vwap_from_df(spy_df).iloc[-1] if not spy_df.empty else float("nan"))
qqq_vwap = qqq_vwap_taapi if not math.isnan(qqq_vwap_taapi) else (compute_vwap_from_df(qqq_df).iloc[-1] if not qqq_df.empty else float("nan"))

# ====== First 5-min momentum, ORB, Breadth ======
f5_spy = first5_momentum(spy_df)
f5_qqq = first5_momentum(qqq_df)
sSPY, eSPY, orhSPY, orlSPY = compute_opening_range(spy_df, minutes=ORB_MINUTES)
sQQQ, eQQQ, orhQQQ, orlQQQ = compute_opening_range(qqq_df, minutes=ORB_MINUTES)

rows, breadth_pct = sector_breadth_yf(SECTORS)
st.subheader("Sector breadth: price > VWAP")
if not math.isnan(breadth_pct):
    st.progress(int(min(max(breadth_pct, 0), 100)))
    st.caption(f"{breadth_pct:.0f}% sectors green • {sum([1 for r in rows if not math.isnan(r[1])]):d} loaded")
else:
    st.caption("Waiting for sector data...")
st.dataframe(
    pd.DataFrame(rows, columns=["Sector", "Last", "VWAP", "Green"]).style.format({"Last": "{:.2f}", "VWAP": "{:.2f}"}),
    use_container_width=True
)

# ====== Bias & Confidence ======
es_green = (es_chg > 0) if not math.isnan(es_chg) else None
nq_green = (nq_chg > 0) if not math.isnan(nq_chg) else None
score, decision, parts = compute_bias(spy_last, qqq_last, spy_vwap, qqq_vwap, breadth_pct, f5_spy, f5_qqq, es_green, nq_green)
spy_bits = taapi_confidence_bits(core, SPY_PREFIX)
qqq_bits = taapi_confidence_bits(core, QQQ_PREFIX)
conf_label_spy, conf_score_spy, _ = confidence_label(spy_bits, decision)
conf_label_qqq, conf_score_qqq, _ = confidence_label(qqq_bits, decision)

# ====== Candle Watch (emoji summary only) ======
spy_1m_watch = candle_emojis(spy_df, n=3)
qqq_1m_watch = candle_emojis(qqq_df, n=3)
spy_5m_watch = candle_emojis(resample_5m(spy_df), n=1)
qqq_5m_watch = candle_emojis(resample_5m(qqq_df), n=1)

st.subheader("Bias, Confidence & Suggestions")
met1, met2, met3, met4 = st.columns(4)
with met1:
    st.metric("Bias score", f"{score:.1f}")
with met2:
    st.metric("Decision", decision)
with met3:
    st.metric("SPY confidence", conf_label_spy, f"{conf_score_spy:+d}")
with met4:
    st.metric("QQQ confidence", conf_label_qqq, f"{conf_score_qqq:+d}")
st.caption("Bias parts: " + " | ".join([f"{k}:{'+' if v>0 else '-' if v<0 else '0'}" for k, v in parts.items()]) if parts else "—")

def sugg_for(symbol, last):
    if last is None or math.isnan(last):
        return "—"
    strike = choose_strike(last, decision, step=1.0)
    if strike is None:
        return "—"
    dir_word = "CALL" if "CALL" in decision else ("PUT" if "PUT" in decision else "WAIT")
    if dir_word == "WAIT":
        return "WAIT for confirmation"
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

# ====== Charts — exactly TWO (SPY & QQQ), 24h with RTH shading ======
st.subheader(f"Charts — 1m (24h) with VWAP + Opening Range ({ORB_MINUTES}m)")
colC1, colC2 = st.columns(2, gap="large")
with colC1:
    if spy_df.empty:
        st.warning("No SPY 1m data.")
    else:
        st.plotly_chart(plot_with_orb_em("SPY", spy_df, orb_minutes=ORB_MINUTES),
                        use_container_width=True, key="spy_chart")
with colC2:
    if qqq_df.empty:
        st.warning("No QQQ 1m data.")
    else:
        st.plotly_chart(plot_with_orb_em("QQQ", qqq_df, orb_minutes=ORB_MINUTES),
                        use_container_width=True, key="qqq_chart")

# ====== Trade Logger (kept inline since trade_log.py not in repo) ======
st.subheader("Trade Logger")
with st.form("trade_log_form", clear_on_submit=False):
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        sym = st.selectbox("Symbol", ["SPY", "QQQ"], index=0)
    with col2:
        side = st.selectbox("Direction", ["CALL", "PUT"], index=0)
    with col3:
        base_last = spy_last if sym == "SPY" else qqq_last
        strike_default = float("nan") if (base_last is None or math.isnan(base_last)) else float(round(base_last))
        strike = st.number_input("Strike", min_value=0.0, value=strike_default, step=1.0)
    with col4:
        qty = st.number_input("Qty (contracts)", min_value=1, value=1, step=1)

    col5, col6, col7 = st.columns([1, 1, 2])
    with col5:
        entry = st.number_input("Entry ($/contract)", min_value=0.0, value=20.0, step=0.5)
    with col6:
        exitp = st.number_input("Exit ($/contract)", min_value=0.0, value=0.0, step=0.5)
    with col7:
        notes = st.text_input("Notes", value="")

    submitted = st.form_submit_button("Add trade")
    if submitted:
        pl_pct = ((exitp - entry) / entry * 100.0) if entry > 0 and exitp > 0 else float("nan")
        row = {
            "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": sym, "side": side, "strike": strike, "qty": qty,
            "entry": entry, "exit": exitp, "p_l_pct": pl_pct, "notes": notes
        }
        try:
            if os.path.exists(TRADE_LOG):
                df_log = pd.read_csv(TRADE_LOG)
                df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
            else:
                df_log = pd.DataFrame([row])
            df_log.to_csv(TRADE_LOG, index=False)
            st.success("Trade added to log.")
        except Exception as e:
            st.error(f"Failed to write log: {e}")

# Show last trades + download + daily P/L
todays_pl = 0.0
if os.path.exists(TRADE_LOG):
    tdf = pd.read_csv(TRADE_LOG)
    tdf["date"] = pd.to_datetime(tdf["time"]).dt.date
    tdf_today = tdf[tdf["date"] == pd.Timestamp.now().date()].copy()
    if not tdf_today.empty:
        def _pl(r):
            try:
                if pd.notnull(r["p_l_pct"]) and pd.notnull(r["entry"]) and pd.notnull(r["qty"]):
                    return (float(r["p_l_pct"]) / 100.0) * float(r["entry"]) * float(r["qty"])
            except Exception:
                pass
            return 0.0
        tdf_today["pl_$"] = tdf_today.apply(_pl, axis=1)
        todays_pl = float(tdf_today["pl_$"].sum())
    tdf_tail = tdf.tail(20).copy().drop(columns=["date"], errors="ignore")
    st.dataframe(tdf_tail, use_container_width=True, hide_index=True)
    csv_bytes = tdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download full trade log (CSV)",
                       data=csv_bytes, file_name="0dte_trade_log.csv", mime="text/csv")
else:
    st.caption("No trades logged yet.")
st.caption(f"Today's P/L (approx): ${todays_pl:.2f}")

# ====== Auto-refresh (disabled if charts frozen) ======
if enable_refresh:
    time.sleep(refresh)
    st.rerun()
