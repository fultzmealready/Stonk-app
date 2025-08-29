import streamlit as st

def check_daily_limits(account_size: float, todays_pl: float, stop_up_pct: float = 30.0, stop_down_pct: float = -20.0) -> bool:
    """
    Returns True if trading should STOP for the day based on playbook.
    stop_up_pct: +30% -> stop (lock day)
    stop_down_pct: -20% -> stop
    """
    if account_size <= 0:
        return False
    pct = (todays_pl / account_size) * 100.0
    if pct >= stop_up_pct:
        st.success(f"Day target reached (+{pct:.1f}% ≥ +{stop_up_pct:.0f}%). Stop trading and lock gains. ✅")
        return True
    if pct <= stop_down_pct:
        st.error(f"Daily loss limit hit ({pct:.1f}% ≤ {stop_down_pct:.0f}%). Stop trading. ⛔")
        return True
    return False
