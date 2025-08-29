# discipline.py
from __future__ import annotations
import datetime as dt
import pandas as pd
import streamlit as st

PLAYBOOK_EXPECTANCY_TARGET = 0.15  # 15% per trade (of risked amount)

def _date_col(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df["time"], errors="coerce")
    return d.dt.date

def daily_pl_series(tl_df: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by date with daily P/L $ (approx)."""
    if tl_df is None or tl_df.empty:
        return pd.Series(dtype=float)
    df = tl_df.copy()
    df["date"] = _date_col(df)
    def _pl_row(r):
        try:
            if pd.notnull(r["p_l_pct"]) and pd.notnull(r["entry"]) and pd.notnull(r["qty"]):
                return (float(r["p_l_pct"]) / 100.0) * float(r["entry"]) * float(r["qty"])
        except Exception:
            pass
        return 0.0
    df["pl_$"] = df.apply(_pl_row, axis=1)
    return df.groupby("date")["pl_$"].sum().sort_index()

def two_red_days_in_row(tl_df: pd.DataFrame, as_of: dt.date | None = None) -> bool:
    """True if the last two trading days before `as_of` have negative P/L."""
    s = daily_pl_series(tl_df)
    if s.empty:
        return False
    as_of = as_of or dt.date.today()
    # take the two most recent days strictly before/as_of that have entries
    s2 = s[s.index <= as_of].tail(2)
    if len(s2) < 2:
        return False
    return (s2.iloc[-1] < 0) and (s2.iloc[-2] < 0)

def weekly_review_metrics(tl_df: pd.DataFrame, week_start: dt.date | None = None):
    """Compute week win rate, avg win %, avg loss %, expectancy (as fraction)."""
    if tl_df is None or tl_df.empty:
        return None, None, None, None
    df = tl_df.copy()
    df["date"] = _date_col(df)
    if week_start is None:
        today = dt.date.today()
        week_start = today - dt.timedelta(days=today.weekday())  # Monday
    week_end = week_start + dt.timedelta(days=6)
    dfw = df[(df["date"] >= week_start) & (df["date"] <= week_end)].copy()
    if dfw.empty:
        return 0.0, None, None, None
    wins = dfw[dfw["p_l_pct"] > 0]
    losses = dfw[dfw["p_l_pct"] < 0]
    win_rate = len(wins) / len(dfw) if len(dfw) else 0.0
    avg_win = (wins["p_l_pct"].mean() / 100.0) if not wins.empty else None
    avg_loss = (abs(losses["p_l_pct"].mean()) / 100.0) if not losses.empty else None
    if avg_win is None or avg_loss is None:
        E = None
    else:
        E = win_rate * avg_win - (1 - win_rate) * avg_loss
    return win_rate, avg_win, avg_loss, E

def render_discipline_panel(tl_df: pd.DataFrame, settings: dict, *, downshift_risk_to: float = 12.5) -> None:
    """
    UI:
    - If 2 red days → show banner “Size down tomorrow.”
    - Toggle to auto-apply risk% downshift for next session via session_state.
    - Weekly metrics vs. playbook.
    """
    today = dt.date.today()
    red_streak = two_red_days_in_row(tl_df, as_of=today)

    if red_streak:
        st.warning("⚠️ Two red days in a row — **Size down tomorrow.**")

    with st.expander("🧠 Discipline & Weekly Review", expanded=False):
        # Auto-downshift control
        if "risk_downshift_enabled" not in st.session_state:
            st.session_state.risk_downshift_enabled = True  # default on

        c1, c2 = st.columns([2,1])
        with c1:
            st.checkbox(
                f"Auto-downshift risk% to {downshift_risk_to:.1f}% after 2 red days",
                value=st.session_state.risk_downshift_enabled,
                key="risk_downshift_enabled"
            )
            if red_streak and st.session_state.risk_downshift_enabled:
                st.info(f"Next session risk% will default to **{downshift_risk_to:.1f}%**.")
                st.session_state.next_session_risk_pct = downshift_risk_to
            else:
                # clear any scheduled downshift
                st.session_state.next_session_risk_pct = None

        # Weekly review metrics
        win_rate, avg_win, avg_loss, E = weekly_review_metrics(tl_df)
        c3, c4, c5, c6 = st.columns(4)
        with c3:
            st.metric("Week Win Rate", f"{(win_rate or 0)*100:.1f}%")
        with c4:
            st.metric("Avg Win", f"{(avg_win or 0)*100:.1f}%")
        with c5:
            st.metric("Avg Loss", f"-{(avg_loss or 0)*100:.1f}%")
        with c6:
            if E is None:
                st.metric("Expectancy", "—")
            else:
                delta = (E - PLAYBOOK_EXPECTANCY_TARGET) * 100.0
                st.metric("Expectancy", f"{E*100:.1f}%", f"{delta:+.1f} pp vs 15% target")
