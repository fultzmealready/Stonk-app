import os
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_LOG = "0dte_trade_log.csv"

def get_log_path() -> str:
    return DEFAULT_LOG

def expected_columns():
    return ["time","symbol","side","strike","qty","entry","exit","p_l_pct","notes"]

def load_trade_log(path: str | None = None) -> pd.DataFrame:
    path = path or get_log_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=expected_columns())
    try:
        df = pd.read_csv(path)
        for col in expected_columns():
            if col not in df.columns:
                df[col] = pd.NA
        return df
    except Exception:
        return pd.DataFrame(columns=expected_columns())

def save_trade_log(df: pd.DataFrame, path: str | None = None) -> None:
    path = path or get_log_path()
    df.to_csv(path, index=False)

def append_trade(df: pd.DataFrame, *, time_str: str, symbol: str, side: str, strike: float, qty: int, entry: float, exitp: float, notes: str) -> pd.DataFrame:
    pl_pct = ((exitp - entry) / entry * 100.0) if (entry and exitp) else float("nan")
    row = {
        "time": time_str, "symbol": symbol, "side": side, "strike": strike, "qty": qty,
        "entry": entry, "exit": exitp, "p_l_pct": pl_pct, "notes": notes
    }
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

def compute_daily_pl(df: pd.DataFrame, as_of: dt.date | None = None) -> float:
    if df is None or df.empty:
        return 0.0
    if "time" not in df.columns:
        return 0.0
    as_of = as_of or dt.date.today()
    _df = df.copy()
    try:
        _df["date"] = pd.to_datetime(_df["time"]).dt.date
    except Exception:
        return 0.0
    tdf_today = _df[_df["date"] == as_of]
    if tdf_today.empty:
        return 0.0
    def _pl(r):
        try:
            if pd.notnull(r["p_l_pct"]) and pd.notnull(r["entry"]) and pd.notnull(r["qty"]):
                return (float(r["p_l_pct"])/100.0) * float(r["entry"]) * float(r["qty"])
        except Exception:
            pass
        return 0.0
    return float(tdf_today.apply(_pl, axis=1).sum())

def render_trade_form(default_symbol="SPY", default_side="CALL", default_entry=20.0):
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1: sym = st.selectbox("Symbol", ["SPY","QQQ"], index=(0 if default_symbol=="SPY" else 1))
    with col2: side = st.selectbox("Direction", ["CALL","PUT"], index=(0 if default_side=="CALL" else 1))
    with col3:
        strike = st.number_input("Strike", min_value=0.0, value=float("nan"), step=1.0)
    with col4: qty = st.number_input("Qty (contracts)", min_value=1, value=1, step=1)
    col5, col6, col7 = st.columns([1,1,2])
    with col5: entry = st.number_input("Entry ($/contract)", min_value=0.0, value=default_entry, step=0.5)
    with col6: exitp = st.number_input("Exit ($/contract)", min_value=0.0, value=0.0, step=0.5)
    with col7: notes = st.text_input("Notes", value="")
    submitted = st.form_submit_button("Add trade")
    if submitted:
        time_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        row_df = append_trade(
            pd.DataFrame(columns=expected_columns()),
            time_str=time_str, symbol=sym, side=side, strike=strike, qty=qty, entry=entry, exitp=exitp, notes=notes
        )
        st.success("Trade added to log.")
        return row_df
    return None

def render_trade_log(df: pd.DataFrame, show_download: bool = True):
    if df is None or df.empty:
        st.caption("No trades logged yet.")
        return
    tdf_tail = df.tail(20).copy()
    st.dataframe(tdf_tail, use_container_width=True, hide_index=True)
    if show_download:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download full trade log (CSV)", data=csv_bytes, file_name="0dte_trade_log.csv", mime="text/csv")
