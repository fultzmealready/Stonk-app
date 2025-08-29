# expectancy.py
from __future__ import annotations
import math
import pandas as pd
import streamlit as st

PLAYBOOK_DEFAULT_WIN_RATE = 0.45
PLAYBOOK_DEFAULT_AVG_WIN_PCT = 1.00   # +100%
PLAYBOOK_DEFAULT_AVG_LOSS_PCT = 0.30  # -30% (as a positive magnitude)

def compute_realized_stats_from_log(tl_df: pd.DataFrame) -> tuple[float|None, float|None, float|None]:
    """
    Derive user's realized win rate, avg win %, avg loss % from the trade log.
    Returns (win_rate, avg_win_pct, avg_loss_pct) as fractions (e.g., 0.55, 0.92, 0.28).
    If not enough data, returns None for that metric.
    """
    if tl_df is None or tl_df.empty or "p_l_pct" not in tl_df.columns:
        return None, None, None

    df = tl_df.copy()
    df = df[pd.notna(df["p_l_pct"])]
    if df.empty:
        return None, None, None

    wins = df[df["p_l_pct"] > 0]
    losses = df[df["p_l_pct"] < 0]

    try:
        win_rate = len(wins) / len(df) if len(df) else None
    except ZeroDivisionError:
        win_rate = None

    avg_win = (wins["p_l_pct"].mean() / 100.0) if not wins.empty else None
    avg_loss = (abs(losses["p_l_pct"].mean()) / 100.0) if not losses.empty else None

    return win_rate, avg_win, avg_loss

def compute_expectancy_per_trade(win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> float:
    """
    Expectancy per trade as a fraction of the RISKED amount.
    E = p*avg_win - (1-p)*avg_loss
    All inputs are fractions (e.g., 0.45, 1.20, 0.30).
    """
    if any(v is None for v in (win_rate, avg_win_pct, avg_loss_pct)):
        return float("nan")
    return (win_rate * avg_win_pct) - ((1.0 - win_rate) * avg_loss_pct)

def expected_equity_curve(start_equity: float, risk_frac: float, expectancy: float, n_trades: int = 120) -> pd.Series:
    """
    Deterministic expected path using expectancy * risk per trade.
    equity_{i+1} = equity_i * (1 + risk_frac * expectancy)
    """
    eq = [float(start_equity)]
    g = 1.0 + float(risk_frac) * float(expectancy)
    for _ in range(n_trades):
        eq.append(eq[-1] * g)
    return pd.Series(eq)

def milestone_trades_needed(eq0: float, target: float, risk_frac: float, expectancy: float) -> int | None:
    """
    How many trades (expected path) to reach a target equity.
    Uses log growth formula; returns None if growth <= 0 or invalid.
    """
    if eq0 <= 0 or target <= eq0:
        return 0
    g = 1.0 + risk_frac * expectancy
    if g <= 1.0:
        return None
    try:
        return math.ceil(math.log(target / eq0, g))
    except Exception:
        return None

def suggest_target_badge(win_rate_assumption: float) -> tuple[str, str]:
    """
    Return (label, help_text) for suggested target.
    We always highlight +100% as 'sweet spot' and +120% as 'Robinhood adjusted'.
    """
    label = "🎯 Suggested target: +100%"
    help_text = "Sweet spot (highest expectancy). +120% is Robinhood-adjusted alternative."
    return label, help_text

def render_expectancy_panel(tl_df: pd.DataFrame, settings: dict) -> None:
    """
    Streamlit UI: collapsible 'Expectancy Roadmap' panel with:
    - sliders (win rate), optional overrides for avg win/loss
    - computed expectancy per trade
    - expected equity curve (deterministic)
    - milestone table ($1k, $10k, $100k, $1M)
    """
    with st.expander("📈 Expectancy Roadmap", expanded=False):
        # Sidebar-derived defaults
        start_equity = float(settings.get("acct", 100.0))
        risk_pct_ui = float(settings.get("risk_pct", 25))  # percent in sidebar
        risk_frac = max(0.0, min(1.0, risk_pct_ui / 100.0))

        # Pull realized stats to pre-fill (fall back to playbook)
        r_win, r_avg_win, r_avg_loss = compute_realized_stats_from_log(tl_df)
        default_wr = r_win if r_win is not None else PLAYBOOK_DEFAULT_WIN_RATE
        default_avg_win = r_avg_win if r_avg_win is not None else PLAYBOOK_DEFAULT_AVG_WIN_PCT
        default_avg_loss = r_avg_loss if r_avg_loss is not None else PLAYBOOK_DEFAULT_AVG_LOSS_PCT

        # Controls
        c1, c2, c3 = st.columns(3)
        with c1:
            win_rate = st.slider("Assumed Win Rate", 0.30, 0.70, float(round(default_wr, 2)), step=0.01)
        with c2:
            avg_win_pct = st.number_input("Avg Win (× of risked premium)", min_value=0.10, max_value=3.00,
                                          value=float(round(default_avg_win, 2)), step=0.05, format="%.2f")
        with c3:
            avg_loss_pct = st.number_input("Avg Loss (× of risked premium)", min_value=0.10, max_value=1.00,
                                           value=float(round(default_avg_loss, 2)), step=0.05, format="%.2f")

        # Expectancy and curve
        E = compute_expectancy_per_trade(win_rate, avg_win_pct, avg_loss_pct)
        st.markdown(f"**Expectancy / trade (of risked amount):** `{E*100:.2f}%`")

        n_trades = st.slider("Trades to project", 20, 240, 120, step=10)
        curve = expected_equity_curve(start_equity, risk_frac, E, n_trades=n_trades)
        df_curve = pd.DataFrame({"Trade #": range(len(curve)), "Equity ($)": curve})
        st.line_chart(df_curve.set_index("Trade #"))

        # Milestones
        targets = [1_000, 10_000, 100_000, 1_000_000]
        rows = []
        for t in targets:
            trades_needed = milestone_trades_needed(start_equity, t, risk_frac, E)
            when = f"{trades_needed} trades" if trades_needed is not None else "Not reachable (E≤0)"
            rows.append({"Target": f"${t:,}", "When (expected)": when})
        st.table(pd.DataFrame(rows))

        # Context / guidance
        st.caption(
            "Notes: This uses a deterministic expected path (no randomness). "
            "Expectancy is applied to the **risked fraction** of equity each trade. "
            "Playbook defaults: +100% target, −30% stop, win rate ≈45%."
        )
