# Main app: thin UI orchestrator
import time, math, datetime as dt
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import gspread
import uuid

from google.oauth2.service_account import Credentials
from zoneinfo import ZoneInfo
from taapi_client import taapi_bulk, make_stock_construct, get_taapi_candles
from data_sources import get_intraday_1m_yf, get_futures_quote, sector_breadth_yf
from indicators import (
    compute_bias, taapi_confidence_bits, confidence_label, compute_opening_range,
    compute_vwap_from_df, first5_momentum, last_n_dirs, candle_emojis, resample_5m,
    get_value_by_id, choose_strike, round_magnets
)
from plotting import plot_with_orb_em
from trade_log import compute_levels, compute_daily_pl, expected_columns
from sheets_store import load_trades, save_trades, append_trade_row, delete_by_ids, _open_sheet
from discipline import render_discipline_panel
from guardrails import check_daily_limits
from components.sidebar import render_sidebar
from expectancy import render_expectancy_panel, suggest_target_badge

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

st.cache_data(ttl=30, show_spinner=False)   # cache for 30s
def _cached_breadth(tickers):
    return sector_breadth_yf(tickers)

rows, breadth_pct = _cached_breadth(SECTORS)

st.subheader("Sector breadth: price > VWAP")
loaded = sum(1 for r in rows if not math.isnan(r[1]))
if not math.isnan(breadth_pct) and loaded:
    st.progress(int(min(max(breadth_pct, 0), 100)))
    st.caption(f"{breadth_pct:.0f}% sectors green • {loaded}/{len(SECTORS)} loaded")
else:
    st.caption(f"Waiting for sector data… {loaded}/{len(SECTORS)} loaded")

st.dataframe(
    pd.DataFrame(rows, columns=["Sector", "Last", "VWAP", "Green"]).style.format({"Last":"{:.2f}","VWAP":"{:.2f}"}),
    use_container_width=True
)

# internal helper in the module we wrote
ws, err = _open_sheet()
if ws is not None:
    st.success("Google Sheets: connected ✅")
    st.caption(f"Spreadsheet tab: {ws.title}")
else:
    st.error("Google Sheets: NOT connected ❌ (falling back to CSV)")
    if err:
        st.code(err)

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
# Requires: TRADE_LOG constant and 'settings' dict already set.

# make sure flag exists across reruns
if "guardrails_hit" not in st.session_state:
    st.session_state.guardrails_hit = False

# Load log + compute today's P/L
tl_df = load_trade_log(TRADE_LOG)
todays_pl = compute_daily_pl(tl_df)

# Account size from sidebar settings
acct_size = float(settings.get("acct", 0.0))

# Defaults for display
up_guard = float("nan"); down_guard = float("nan")

# Only enforce guardrails if account size is set
if acct_size > 0:
    up_guard   = 0.30 * acct_size     # +30%
    down_guard = -0.20 * acct_size    # -20%
    breach_up   = todays_pl >= up_guard
    breach_down = todays_pl <= down_guard
    st.session_state.guardrails_hit = bool(breach_up or breach_down)

    if st.session_state.guardrails_hit:
        st.error("🛑 STOP TRADING — Daily guardrail hit "
                 + ("(+30% target reached)" if breach_up else "(−20% loss limit)"))
else:
    # don’t trip guardrails without an account size; just inform
    st.info("Set your Account size in the sidebar to enable daily guardrails.")

# ====== Trade Logger ======
st.subheader("Trade Logger")

# Disable everything if guardrails tripped
disable_form = st.session_state.guardrails_hit if "guardrails_hit" in st.session_state else False

# --- LIVE inputs (outside any form -> recompute every keystroke) ---
col5, col6, col7 = st.columns([1, 1, 2])
with col5:
    entry = st.number_input(
        "Entry ($/contract)", min_value=0.0, value=20.0, step=0.5,
        key="entry_live", disabled=disable_form
    )
with col6:
    exitp = st.number_input(
        "Exit ($/contract)", min_value=0.0, value=0.0, step=0.5,
        key="exit_live", disabled=disable_form
    )
with col7:
    notes = st.text_input(
        "Notes", value="", key="notes_live", disabled=disable_form
    )

# Live preview of targets/stop
tgt100, tgt120, stop30 = compute_levels(entry if entry > 0 else float("nan"))
cA, cB, cC = st.columns(3)
cA.metric("+100% target", f"${tgt100:.2f}" if tgt100 == tgt100 else "—")
cB.metric("+120% raw",    f"${tgt120:.2f}" if tgt120 == tgt120 else "—")
cC.metric("−30% stop",    f"${stop30:.2f}" if stop30 == stop30 else "—")

# (Optional) Suggested target badge from expectancy.py
try:
    from expectancy import suggest_target_badge
    badge_label, badge_help = suggest_target_badge(win_rate_assumption=0.45)
    st.info(f"{badge_label} — {badge_help}")
except Exception:
    pass

# --- Submit form (unique key) ---
with st.form("trade_submit_form", clear_on_submit=False):
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        sym = st.selectbox("Symbol", ["SPY", "QQQ"], index=0, key="trade_symbol", disabled=disable_form)

    with col2:
        side = st.selectbox("Direction", ["CALL", "PUT"], index=0, key="trade_side", disabled=disable_form)

    with col3:
        # default strike near the currently selected underlying's last
        base_last = spy_last if sym == "SPY" else qqq_last
        strike_default = float("nan") if (base_last is None or math.isnan(base_last)) else float(round(base_last))
        strike = st.number_input("Strike", min_value=0.0, value=strike_default, step=1.0,
                                 key="trade_strike", disabled=disable_form)

    with col4:
        qty = st.number_input("Qty (contracts)", min_value=1, value=1, step=1,
                              key="trade_qty", disabled=disable_form)

    submitted = st.form_submit_button("Add trade", disabled=disable_form)

    if submitted:
        
        # append directly to Google Sheet
        append_trade_row(
            time_str=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol=sym,
            side=side,
            strike=strike,
            qty=qty,
            entry=entry,
            exitp=exitp,
            notes=notes,
        )
        st.success("Trade added to Google Sheet.")
        st.rerun()

# ==== Load from Google Sheets ====
df = load_trades()

todays_pl = 0.0
if df is not None and not df.empty:
    # --- compute today P/L (unchanged) ---
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.date
        tdf_today = df[df["date"] == pd.Timestamp.now().date()].copy()
        if not tdf_today.empty:
            tdf_today["pl_$"] = tdf_today.apply(
                lambda r: (float(r.get("p_l_pct", 0)) / 100.0) * float(r.get("entry", 0)) * int(r.get("qty", 0)),
                axis=1
            )
            todays_pl = float(tdf_today["pl_$"].sum())

    # --- deletion UI (uses 'id') ---
    if "id" not in df.columns:
        st.warning("No 'id' column found. Add ids on insert to enable per-row deletion.")
    else:
        view = df.copy()
        view.insert(0, "Select", False)

        edited = st.data_editor(
            view,
            use_container_width=True,
            hide_index=True,
            column_config={"Select": st.column_config.CheckboxColumn(required=False)},
            disabled=[c for c in view.columns if c != "Select"],
            key="trade_table_editor",
        )

        # Get the ids for rows where Select == True
        to_delete_ids = edited.loc[edited["Select"] == True, "id"].astype(str).tolist()

        left, right = st.columns([1, 3])
        with left:
            if st.button("🗑️ Delete selected"):
                if to_delete_ids:
                    new_df = delete_by_ids(to_delete_ids)   # <-- your helper
                    st.success(f"Deleted {len(to_delete_ids)} trade(s).")
                    st.rerun()
                else:
                    st.info("No rows selected.")

    # Download snapshot
    st.download_button(
        "Download full trade log (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="0dte_trade_log.csv",
        mime="text/csv"
    )
else:
    st.caption("No trades logged yet.")

st.caption(f"Today's P/L (approx): ${todays_pl:.2f}")

st.divider()
render_expectancy_panel(df, settings)

st.divider()
with st.expander("📈 Expectancy Roadmap (playbook-aligned)", expanded=True):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    start_equity = c1.number_input("Start equity ($)",  min_value=50.0, value=float(settings.get("account_size", 500.0)), step=50.0)
    risk_pct     = c2.number_input("Risk per trade (%)", min_value=1.0, value=float(settings.get("risk_pct", 25.0)), step=0.5)
    win_rate     = c3.slider("Win rate (%)", 10, 80, 45, step=1)
    avg_win_pct  = c4.number_input("Avg win (%)",  min_value=10.0, value=100.0, step=5.0, help="Playbook sweet spot: +100%")
    avg_loss_pct = c5.number_input("Avg loss (%)", min_value=5.0,  value=30.0,  step=1.0, help="Playbook stop: −30%")
    weeks        = c6.slider("Weeks", 4, 52, 24, step=1)

    # Trading cadence
    d1, d2, d3 = st.columns(3)
    trades_per_day = d1.number_input("Trades per day",   min_value=1, value=4, step=1)
    days_per_week  = d2.number_input("Days per week",    min_value=1, value=5, step=1)
    # Optional: RH slippage tag
    _ = d3.caption("Tip: +120% raw target ≈ +100% net on RH fills.")

    # Expectancy per trade (in % of risk)
    p = win_rate / 100.0
    E = p*avg_win_pct - (1-p)*avg_loss_pct                  # e.g., 0.45*100 - 0.55*30 = 15.5
    exp_ret_per_trade = (risk_pct/100.0) * (E/100.0)        # as a fraction of equity

    trades_per_week = int(trades_per_day * days_per_week)
    total_trades = trades_per_week * int(weeks)

    # Deterministic expected curve (compounding on expected value)
    # equity_T = equity_0 * (1 + exp_ret_per_trade)^T
    trades = np.arange(0, total_trades + 1, dtype=int)
    equity_by_trade = start_equity * (1.0 + exp_ret_per_trade) ** trades

    # Week summary
    week_idx = np.arange(0, total_trades + 1, trades_per_week)
    week_equity = equity_by_trade[week_idx]
    df_week = pd.DataFrame({"Week": np.arange(len(week_equity)), "Equity": week_equity})

    # Milestones
    milestones = [1_000, 10_000, 100_000, 1_000_000]
    hit = {}
    for m in milestones:
        hit_week = next((int(w) for w, eq in zip(df_week["Week"], df_week["Equity"]) if eq >= m), None)
        hit[m] = hit_week

    # Header metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expectancy per trade", f"{E:.1f}% of risk", help="p*AvgWin - (1-p)*AvgLoss")
    m2.metric("Exp. return / trade",  f"{exp_ret_per_trade*100:.2f}%")
    m3.metric("Trades / week",        f"{trades_per_week}")
    m4.metric("Total trades",         f"{total_trades}")

    # Chart
    fig = px.line(df_week, x="Week", y="Equity", title="Expected Equity (deterministic)")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=360)
    st.plotly_chart(fig, use_container_width=True)

    # Milestone table
    st.caption("Milestones (expected—not guarantees):")
    mt = pd.DataFrame(
        {"Target": ["$1k", "$10k", "$100k", "$1M"],
         "Week hit": [hit[1_000], hit[10_000], hit[100_000], hit[1_000_000]]}
    )
    st.dataframe(mt, hide_index=True, use_container_width=True)

    # Guidance callouts
    st.info("🎯 Suggested target: **+100%** (highest expectancy). **+120% raw** is the Robinhood-adjusted alternative.", icon="🎯")
    st.caption("Formulae: Expectancy = p·AvgWin − (1−p)·AvgLoss. Equity(T) ≈ E0·(1 + risk·Expectancy)^T")

st.divider()
render_discipline_panel(tl_df, settings, downshift_risk_to=12.5)
# ====== Auto-refresh ======
stop_now = st.session_state.guardrails_hit  # pause refresh if guardrails tripped
if settings["enable_refresh"] and not stop_now:
    time.sleep(settings["refresh_secs"])
    st.rerun()
