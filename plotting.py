import math, datetime as dt
import plotly.graph_objs as go
import pandas as pd
from zoneinfo import ZoneInfo
from indicators import compute_opening_range, compute_vwap_from_df

ET = ZoneInfo("America/New_York")

def _ensure_et(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure the index is tz-aware in ET."""
    if df.index.tz is None:
        # most sources are UTC; localize then convert
        df = df.tz_localize("UTC").tz_convert(ET)
    else:
        df = df.tz_convert(ET)
    return df

def shade_rth(fig: go.Figure, df: pd.DataFrame,
              fillcolor="LightGreen", opacity=0.10) -> None:
    """Shade each day's 09:30–16:00 ET session."""
    # df must already be ET
    for day in df.index.normalize().unique():
        rth_start = day + pd.Timedelta(hours=9, minutes=30)
        rth_end   = day + pd.Timedelta(hours=16)
        fig.add_vrect(
            x0=rth_start, x1=rth_end,
            fillcolor=fillcolor, opacity=opacity, layer="below", line_width=0,
            annotation_text="RTH", annotation_position="top left"
        )

def _last_eod_8pm(ts_max: pd.Timestamp) -> pd.Timestamp:
    """Return the most recent 20:00 ET at or before ts_max (also in ET)."""
    day_8pm = ts_max.normalize() + pd.Timedelta(hours=20)
    if ts_max >= day_8pm:
        return day_8pm
    return (ts_max.normalize() - pd.Timedelta(days=1)) + pd.Timedelta(hours=20)

def plot_with_orb_em(ticker: str, df: pd.DataFrame, orb_minutes: int = 15):
    # --- ensure ET first ---
    df = _ensure_et(df)

    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ))

    # VWAP
    vwap_series = vwap_series.groupby(vwap_series.index.date).ffill()
    fig.add_trace(go.Scatter(x=df.index, y=vwap_series, mode="lines", name="VWAP"))
    
    # ORB
    s, e, orh, orl = compute_opening_range(df, minutes=orb_minutes)
    if s is not None and e is not None and not (math.isnan(orh) or math.isnan(orl)):
        fig.add_hrect(y0=orl, y1=orh, x0=s, x1=e, opacity=0.15, line_width=0, fillcolor="LightSkyBlue")
        fig.add_hline(y=orh, line_dash="dot", opacity=0.5)
        fig.add_hline(y=orl, line_dash="dot", opacity=0.5)

    # RTH shading (09:30–16:00 ET)
    shade_rth(fig, df)

    # Skip dead space: hide 20:00 → 04:00 ET + weekends
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[20, 4], pattern="hour"),
        ]
    )

    # EOD line at the break (20:00 ET)
    try:
        eod = _last_eod_8pm(df.index.max().tz_convert(ET))
        fig.add_vline(
            x=eod, line_dash="dot", line_color="red",
            annotation_text="EOD", annotation_position="top"
        )
    except Exception:
        pass

    # Tighten to data window
    fig.update_xaxes(range=[df.index.min(), df.index.max()])

    fig.update_layout(
        title=f"{ticker} — 24h",
        height=420,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig
