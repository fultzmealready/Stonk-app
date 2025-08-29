import streamlit as st

def render_sidebar(default_refresh_secs: int = 5) -> dict:
    with st.sidebar:
        st.title("⚙️ Settings")
        enable_refresh = st.checkbox("Enable auto-refresh", value=True, key="enable_refresh")
        freeze_charts = st.checkbox("Freeze charts (keep zoom)", value=False, key="freeze_charts")
        if freeze_charts: enable_refresh = False
        refresh = st.slider("Refresh every (sec)", 3, 30, default_refresh_secs, step=1, key="refresh_secs")
        st.divider()
        st.caption("Risk / Budget")
        acct = st.number_input("Account size ($)", min_value=50.0, value=100.0, step=50.0, key="acct_size")
        risk_pct = st.slider("Risk per trade (%)", 1, 10, 3, step=1, key="risk_pct")
        est_cost = st.number_input("Est. option cost ($/contract)", min_value=1.0, value=20.0, step=1.0, key="est_cost")
        max_risk = acct * (risk_pct/100.0)
        max_contracts = max(1, int(max_risk // est_cost)) if est_cost > 0 else 1
        st.metric("Max risk / contracts", f"${max_risk:.0f}", f"{max_contracts}x")
        st.caption("Default exit: +50–100% target / −50% stop.")
        default_risk = st.session_state.get("next_session_risk_pct", 25.0) or 25.0
        risk_pct = st.slider("Risk per trade (%)", 1, 50, int(default_risk), step=1)
        # if we consumed the override, clear it so it only applies once
        if "next_session_risk_pct" in st.session_state and st.session_state.next_session_risk_pct:
        # apply only on first render of the new session; you can add date logic if desired
        st.session_state.next_session_risk_pct = None
        st.divider()
        daily_limit = st.number_input("Max daily loss ($)", min_value=0.0, value=100.0, step=10.0, key="daily_limit")
    return {
        "enable_refresh": enable_refresh,
        "refresh_secs": refresh,
        "account_size": acct,
        "risk_pct": risk_pct,
        "est_cost": est_cost,
        "max_risk": max_risk,
        "max_contracts": max_contracts,
        "daily_limit": daily_limit,
    }
