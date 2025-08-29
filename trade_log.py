from __future__ import annotations
import os, datetime as dt
import pandas as pd
import streamlit as st

DEFAULT_LOG = "0dte_trade_log.csv"

def expected_columns():
    # include an 'id' column so each row can be targeted safely
    return ["id","time","symbol","side","strike","qty","entry","exit","p_l_pct","notes"]

def get_log_path() -> str:
    return DEFAULT_LOG

def load_trade_log(path: str | None = None) -> pd.DataFrame:
    path = path or TRADE_LOG
    if not os.path.exists(path):
        return pd.DataFrame(columns=expected_columns())
    df = pd.read_csv(path)
    return ensure_id_column(df)

def save_trade_log(df: pd.DataFrame, path: str | None = None) -> None:
    path = path or TRADE_LOG
    df = df.copy()
    df = ensure_id_column(df)
    # keep columns ordered
    cols = [c for c in expected_columns() if c in df.columns] + [c for c in df.columns if c not in expected_columns()]
    df[cols].to_csv(path, index=False)

def ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill an 'id' column if it doesn't exist."""
    if "id" in df.columns:
        # coerce to string ids
        df = df.copy()
        df["id"] = df["id"].astype(str)
        return df
    out = df.copy()
    # make deterministic-but-unique-ish ids if 'time' exists; otherwise UUIDs
    if "time" in out.columns:
        ts = pd.to_datetime(out["time"], errors="coerce").fillna(pd.Timestamp.utcnow())
        base = (ts.view("int64") // 10**6).astype("int64")  # ms
        out["id"] = (base + np.arange(len(out))).astype(str)
    else:
        out["id"] = [uuid.uuid4().hex for _ in range(len(out))]
    return out

def append_trade(df: pd.DataFrame, *, time_str: str, symbol: str, side: str, strike: float,
                 qty: int, entry: float, exitp: float, notes: str) -> pd.DataFrame:
    df = ensure_id_column(df)
    row = {
        "id": uuid.uuid4().hex,  # new unique id
        "time": time_str,
        "symbol": symbol,
        "side": side,
        "strike": strike,
        "qty": qty,
        "entry": entry,
        "exit": exitp,
        "p_l_pct": ((exitp - entry)/entry*100.0) if entry and exitp else float("nan"),
        "notes": notes,
    }
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

def delete_trades_by_id(df: pd.DataFrame, ids_to_delete: list[str]) -> pd.DataFrame:
    df = ensure_id_column(df)
    ids = set(str(x) for x in ids_to_delete)
    return df[~df["id"].astype(str).isin(ids)].copy()

def delete_and_save(ids_to_delete: list[str], path: str | None = None) -> pd.DataFrame:
    path = path or TRADE_LOG
    df = load_trade_log(path)
    new_df = delete_trades_by_id(df, ids_to_delete)
    save_trade_log(new_df, path)
    return new_df

def compute_levels(entry: float) -> tuple[float,float,float]:
    if not entry or pd.isna(entry):
        return float("nan"), float("nan"), float("nan")
    tgt_100 = entry * 2.00      # +100%
    tgt_120 = entry * 2.20      # +120% raw
    stop_30 = entry * 0.70      # −30%
    return tgt_100, tgt_120, stop_30

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
