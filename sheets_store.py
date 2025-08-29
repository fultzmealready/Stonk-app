# sheets_store.py
# Read/write the trade log to Google Sheets using a service account.
# Falls back to local CSV if Sheets isn’t configured.

from __future__ import annotations

import os
import uuid
import pandas as pd
import streamlit as st

# ---------- Constants ----------
CSV_PATH = "0dte_trade_log.csv"
HEADERS = ["id","time","symbol","side","strike","qty","entry","exit","p_l_pct","notes"]

# ---------- gspread client helpers ----------
def _get_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        sa_info = st.secrets.get("gcp_service_account", None)
        if not sa_info:
            return None, "Missing gcp_service_account in secrets"

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        client = gspread.authorize(creds)
        return client, None
    except Exception as e:
        return None, f"Auth error: {e}"

def _ensure_headers(ws):
    try:
        current = ws.row_values(1)
        if current != HEADERS:
            ws.update("A1", [HEADERS])
    except Exception:
        pass

def _open_sheet():
    """
    Returns (worksheet, error_message). Creates tab/headers if needed.
    Requires st.secrets:
      - SHEETS_DOC_KEY (preferred) or SHEETS_DOC_NAME
      - SHEETS_TAB_NAME (optional; default 'trades')
    """
    client, err = _get_client()
    if err or client is None:
        return None, err or "No client"

    doc_key  = (st.secrets.get("SHEETS_DOC_KEY") or "").strip()
    doc_name = (st.secrets.get("SHEETS_DOC_NAME") or "").strip()
    tab_name = (st.secrets.get("SHEETS_TAB_NAME") or "trades").strip() or "trades"

    try:
        if doc_key:
            sh = client.open_by_key(doc_key)
        elif doc_name:
            sh = client.open(doc_name)
        else:
            return None, "Provide SHEETS_DOC_KEY or SHEETS_DOC_NAME in secrets."

        try:
            ws = sh.worksheet(tab_name)
        except Exception:
            ws = sh.add_worksheet(title=tab_name, rows=2000, cols=20)
            _ensure_headers(ws)

        _ensure_headers(ws)
        return ws, None
    except Exception as e:
        return None, f"Open sheet error: {e}"

# ---------- conversions ----------
def _rows_to_df(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=HEADERS)
    df = pd.DataFrame(rows, columns=HEADERS)
    # type cleanup
    for c in ["strike","qty","entry","exit","p_l_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _normalize_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=HEADERS)
    out = df.copy()
    for c in HEADERS:
        if c not in out.columns:
            out[c] = pd.NA
    # keep known order first, then any extras
    ordered = [c for c in HEADERS] + [c for c in out.columns if c not in HEADERS]
    return out[ordered]

# ---------- Public API ----------
def load_trades() -> pd.DataFrame:
    """
    Try Google Sheets; fallback to CSV.
    """
    ws, err = _open_sheet()
    if ws is None:
        # Fallback to CSV
        if os.path.exists(CSV_PATH):
            try:
                df = pd.read_csv(CSV_PATH)
                return _normalize_df(df)
            except Exception:
                return pd.DataFrame(columns=HEADERS)
        return pd.DataFrame(columns=HEADERS)

    try:
        values = ws.get_all_values()
        if len(values) <= 1:
            return pd.DataFrame(columns=HEADERS)
        # values[0] is header row; use our canonical HEADERS instead
        rows = values[1:]
        return _rows_to_df(rows)
    except Exception:
        # Sheets read failed; fallback
        if os.path.exists(CSV_PATH):
            try:
                df = pd.read_csv(CSV_PATH)
                return _normalize_df(df)
            except Exception:
                return pd.DataFrame(columns=HEADERS)
        return pd.DataFrame(columns=HEADERS)

def save_trades(df: pd.DataFrame) -> None:
    """
    Try writing to Google Sheets; also mirror to CSV locally.
    """
    df_norm = _normalize_df(df)

    # Always mirror to CSV (best-effort)
    try:
        df_norm.to_csv(CSV_PATH, index=False)
    except Exception:
        pass

    ws, err = _open_sheet()
    if ws is None:
        return  # no Sheets available; CSV already saved

    # Build 2D list with canonical headers
    out = df_norm.copy()
    values = [HEADERS] + out[HEADERS].astype(str).fillna("").values.tolist()
    try:
        ws.clear()
        ws.update("A1", values)
    except Exception:
        # If Sheets write fails, we’ve at least saved CSV
        pass

def append_trade_row(*, time_str: str, symbol: str, side: str, strike: float,
                     qty: int, entry: float, exitp: float, notes: str) -> pd.DataFrame:
    """
    Append a single row and persist.
    Returns the updated DataFrame.
    """
    df = load_trades()
    row = {
        "id": uuid.uuid4().hex,
        "time": time_str,
        "symbol": symbol,
        "side": side,
        "strike": strike,
        "qty": qty,
        "entry": entry,
        "exit": exitp,
        "p_l_pct": ((exitp - entry)/entry*100.0) if (pd.notna(entry) and entry and pd.notna(exitp) and exitp) else float("nan"),
        "notes": notes,
    }
    new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_trades(new_df)
    return new_df

def delete_by_ids(ids: list[str]) -> pd.DataFrame:
    df = load_trades()
    if df.empty or "id" not in df.columns or not ids:
        return df
    ids = [str(x) for x in ids]
    new_df = df[~df["id"].astype(str).isin(ids)].copy()
    save_trades(new_df)
    return new_df
