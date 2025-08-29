from __future__ import annotations
import os, datetime as dt
import pandas as pd
import streamlit as st

DEFAULT_LOG = "0dte_trade_log.csv"

def expected_columns() -> list[str]:
    return ["time","symbol","side","strike","qty","entry","exit","p_l_pct","notes",
            "tgt_100","tgt_120","stop_30","status"]

def get_log_path() -> str:
    return DEFAULT_LOG

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
    except Exception as e:
        st.warning(f"Could not read trade log: {e}")
        return pd.DataFrame(columns=expected_columns())

def save_trade_log(df: pd.DataFrame, path: str | None = None) -> None:
    (path or get_log_path())
    df.to_csv(path or get_log_path(), index=False)

def compute_levels(entry: float) -> tuple[float,float,float]:
    if not entry or pd.isna(entry):
        return float("nan"), float("nan"), float("nan")
    tgt_100 = entry * 2.00      # +100%
    tgt_120 = entry * 2.20      # +120% raw
    stop_30 = entry * 0.70      # −30%
    return tgt_100, tgt_120, stop_30

def append_trade(df: pd.DataFrame, *, time_str: str, symbol: str, side: str,
                 strike: float, qty: int, entry: float, exitp: float, notes: str) -> pd.DataFrame:
    pl_pct = ((exitp - entry) / entry * 100.0) if (entry and exitp) else float("nan")
    tgt_100, tgt_120, stop_30 = compute_levels(entry)
    row = {
        "time": time_str, "symbol": symbol, "side": side, "strike": strike, "qty": qty,
        "entry": entry, "exit": exitp, "p_l_pct": pl_pct, "notes": notes,
        "tgt_100": tgt_100, "tgt_120": tgt_120, "stop_30": stop_30,
        "status": "closed" if (exitp and exitp > 0) else "open",
    }
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

def compute_daily_pl(df: pd.DataFrame, as_of: dt.date | None = None) -> float:
    if df is None or df.empty or "time" not in df.columns:
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
                return (float(r["p_l_pct"]) / 100.0) * float(r["entry"]) * float(r["qty"])
        except Exception:
            pass
        return 0.0

    return float(tdf_today.apply(_pl, axis=1).sum())
