# Main app: thin UI orchestrator
import time, math, datetime as dt
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

from taapi_client import taapi_bulk, make_stock_construct, get_taapi_candles
from data_sources import get_intraday_1m_yf, get_futures_quote, sector_breadth_yf
from indicators import (
    compute_bias, taapi_confidence_bits, confidence_label, compute_opening_range,
    compute_vwap_from_df, first5_momentum, last_n_dirs, candle_emojis, resample_5m,
    get_value_by_id, choose_strike, round_magnets
)
from plotting import plot_with_orb_em
from trade_log import load_trade_log, save_trade_log, append_trade, compute_levels, compute_daily_pl, expected_columns

from guardrails import check_daily_limits
from components.sidebar import render_sidebar

st.set_page_config(page_title="0DTE Cockpit — SPY & QQQ", layout="wide")

# ====== Constants ======
INTERVAL = "1m"
DEFAULT_REFRESH_SECS = 5
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
TRADE_LOG = "0dte_trade_log.csv"
ET = ZoneInfo("America/New_York")

# ====== Sidebar & settings ======
settings = render_sidebar(DEFAULT_REFRESH_SECS)

# ====== Header ======
st.header("0DTE Cockpit — SPY & QQQ")

# ====== TAAPI indicators (bulk) ======
core = taapi_bulk([
    make_stock_construct("SPY", INTERVAL, CORE_INDIS),
    make_stock_construct("QQQ", INTERVAL, CORE_INDIS),
], timeout_sec=60)

SPY_PREFIX = f"stocks_SPY_{INTERVAL}_"
QQQ_PREFIX = f"stocks_QQQ_{INTERVAL}_"
spy_last = get_value_by_id(core, SPY_PREFIX + "price_")
qqq_last = get_value_by_id(core, QQQ_PREFIX + "price_")
spy_vwap = get_value_by_id(core, SPY_PREFIX + "vwap_")
qqq_vwap = get_value_by_id(core, QQQ_PREFIX + "vwap_")

# ====== Futures & Expected Move ======
colF1, colF2, colF3 = st.columns(3)
es_last, es_chg = get_futures_quote("ES=F")
nq_last, nq_chg = get_futures_quote("NQ=F")
def expected_move_vix():
    import math
    import yfinance as yf
    try:
        spx = yf.download("^GSPC", period="2d", interval="1d", progress=False)
        vix = yf.download("^VIX",  period="2d", interval="1d", progress=False)
        spx_last = float(spx["Close"].iloc[-1]); vix_last = float(vix["Close"].iloc[-1])
        em = spx_last * (vix_last/100.0) / math.sqrt(252)
        return spx_last, vix_last, em
    except Exception:
        return float("nan"), float("nan"), float("nan")
spx_last, vix_last, em_spx = expected_move_vix()
with colF1: st.metric("ES=F last", f"{es_last:.2f}" if not math.isnan(es_last) else "—", f"{es_chg:.2f}%")
with colF2: st.metric("NQ=F last", f"{nq_last:.2f}" if not math.isnan(nq_last) else "—", f"{nq_chg:.2f}%")
with colF3: st.metric("SPX/VIX EM (±)", f"{em_spx:.1f}" if not math.isnan(em_spx) else "—", f"VIX {vix_last:.1f}" if not math.isnan(vix_last) else "")

# ====== Candles (TAAPI-first, fallback yfinance 24h incl. pre/post) ======
spy_df = get_taapi_candles("SPY", interval=INTERVAL)
qqq_df = get_taapi_candles("QQQ", interval=INTERVAL)
if spy_df is None or spy_df.empty: spy_df = get_intraday_1m_yf("SPY", period="2d", include_ext_hours=True)
if qqq_df is None or qqq_df.empty: qqq_df = get_intraday_1m_yf("QQQ", period="2d", include_ext_hours=True)

# ====== First 5-min momentum, ORB, Sector breadth ======
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

# ====== Candle Watch (emoji summary only) ======
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

# ====== Charts — 24h with RTH shading ======
st.subheader(f"Charts — 1m (24h) with VWAP + Opening Range ({ORB_MINUTES}m)")
colC1, colC2 = st.columns(2, gap="large")
with colC1:
    if spy_df.empty: st.warning("No SPY 1m data.")
    else: st.plotly_chart(plot_with_orb_em("SPY", spy_df, orb_minutes=ORB_MINUTES), use_container_width=True, key="spy_chart")
with colC2:
    if qqq_df.empty: st.warning("No QQQ 1m data.")
    else: st.plotly_chart(plot_with_orb_em("QQQ", qqq_df, orb_minutes=ORB_MINUTES), use_container_width=True, key="qqq_chart")

# ====== Guardrails: load log, compute today's P/L, set thresholds ======
tl_df = load_trade_log(TRADE_LOG)

# Today's P/L in $
todays_pl = compute_daily_pl(tl_df)

# Account size from sidebar settings
acct_size = float(settings.get("acct", 0.0))

# Thresholds (+30% / −20%)
up_guard   = 0.30 * acct_size if acct_size > 0 else float("inf")
down_guard = -0.20 * acct_size if acct_size > 0 else float("-inf")

breach_up = acct_size > 0 and todays_pl >= up_guard
breach_down = acct_size > 0 and todays_pl <= down_guard
guardrails_hit = bool(breach_up or breach_down)

# Banner if tripped
if guardrails_hit:
    st.error("🛑 STOP TRADING — Daily guardrail hit "
             + ("(+30% target reached)" if breach_up else "(−20% loss limit)"))


# ====== Trade Logger ======
st.subheader("Trade Logger")

disable_form = guardrails_hit

with st.form("trade_log_form", clear_on_submit=False):
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        sym = st.selectbox("Symbol", ["SPY","QQQ"], index=0, disabled=disable_form)
    with col2:
        side = st.selectbox("Direction", ["CALL","PUT"], index=0, disabled=disable_form)
    with col3:
        # default strike near last
        base_last = spy_last if sym=="SPY" else qqq_last
        strike_default = float("nan") if (base_last is None or math.isnan(base_last)) else float(round(base_last))
        strike = st.number_input("Strike", min_value=0.0, value=strike_default, step=1.0, disabled=disable_form)
    with col4:
        qty = st.number_input("Qty (contracts)", min_value=1, value=1, step=1, disabled=disable_form)

    col5, col6, col7 = st.columns([1,1,2])
    with col5:
        entry = st.number_input("Entry ($/contract)", min_value=0.0, value=20.0, step=0.5, disabled=disable_form)
    with col6:
        exitp = st.number_input("Exit ($/contract)", min_value=0.0, value=0.0, step=0.5, disabled=disable_form)
    with col7:
        notes = st.text_input("Notes", value="", disabled=disable_form)

    # Live preview of targets/stop
    tgt100, tgt120, stop30 = compute_levels(entry if entry > 0 else float("nan"))
    cA, cB, cC = st.columns(3)
    cA.metric("+100% target", f"${tgt100:.2f}" if tgt100==tgt100 else "—")
    cB.metric("+120% raw",   f"${tgt120:.2f}" if tgt120==tgt120 else "—")
    cC.metric("−30% stop",   f"${stop30:.2f}" if stop30==stop30 else "—")

    submitted = st.form_submit_button("Add trade", disabled=disable_form)

    if submitted:
        time_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        base_df = tl_df if not tl_df.empty else pd.DataFrame(columns=expected_columns())
        new_df = append_trade(
            base_df,
            time_str=time_str, symbol=sym, side=side, strike=strike, qty=qty,
            entry=entry, exitp=exitp, notes=notes
        )
        save_trade_log(new_df, TRADE_LOG)
        st.success("Trade added to log.")

        # Toast the guard levels for quick reference
        st.toast(f"Targets: +100% ${tgt100:.2f}, +120% ${tgt120:.2f} • Stop: ${stop30:.2f}", icon="🎯")

        # Soft reminder of daily guardrails
        st.toast(f"Daily guardrails: STOP TRADING if P/L ≥ ${up_guard:.0f} or ≤ ${down_guard:.0f}", icon="⚠️")
        st.rerun()

        save_trade_log(new_df, TRADE_LOG)
        st.success("Trade added to log.")
        st.toast(f"Targets: +100% ${tgt100:.2f}, +120% ${tgt120:.2f} • Stop: ${stop30:.2f}", icon="🎯")
        st.toast(f"Daily guardrails: STOP TRADING if P/L ≥ ${up_guard:.0f} or ≤ ${down_guard:.0f}", icon="⚠️")

        # refresh local copies
        tl_df = new_df
        todays_pl = compute_daily_pl(tl_df)
        st.rerun()


# Load existing log and compute today's P/L
st.caption(f"Today's P/L (approx): ${todays_pl:.2f}")
if not tl_df.empty:
    show_cols = [c for c in tl_df.columns if c in expected_columns()]
    st.dataframe(tl_df.tail(20)[show_cols], use_container_width=True, hide_index=True)
    st.download_button("Download full trade log (CSV)",
                       data=tl_df.to_csv(index=False).encode("utf-8"),
                       file_name="0dte_trade_log.csv", mime="text/csv")
else:
    st.caption("No trades logged yet.")



# ====== Auto-refresh ======
stop_now = guardrails_hit 
if settings["enable_refresh"] and not stop_now:
    time.sleep(settings["refresh_secs"])
    st.rerun()
