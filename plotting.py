import math
import plotly.graph_objs as go
import pandas as pd
from indicators import compute_opening_range, compute_vwap_from_df

def shade_rth(fig: go.Figure, df: pd.DataFrame, tz="America/New_York", fillcolor="LightGreen", opacity=0.10) -> None:
    """
    Add vertical rectangles for each day's 9:30–16:00 ET RTH window.
    """
    # df index should already be timezone-aware or naive UTC; for simple shading we assume index in ET or localized by data source
    for day in df.index.normalize().unique():
        rth_start = day + pd.Timedelta(hours=9, minutes=30)
        rth_end   = day + pd.Timedelta(hours=16, minutes=0)
        fig.add_vrect(
            x0=rth_start, x1=rth_end,
            fillcolor=fillcolor, opacity=opacity, layer="below", line_width=0,
            annotation_text="RTH", annotation_position="top left"
        )

def plot_with_orb_em(ticker: str, df: pd.DataFrame, orb_minutes: int = 15):
    """
    Candles + VWAP + ORB highlight + Expected-move % band (derived from SPX/VIX passed separately).
    The EM band lines are added by caller if desired to avoid extra dependencies here.
    """
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    vwap_series = compute_vwap_from_df(df)
    fig.add_trace(go.Scatter(x=df.index, y=vwap_series, mode="lines", name="VWAP"))

    s, e, orh, orl = compute_opening_range(df, minutes=orb_minutes)
    if s is not None and e is not None and not (math.isnan(orh) or math.isnan(orl)):
        fig.add_hrect(y0=orl, y1=orh, x0=s, x1=e, opacity=0.15, line_width=0, fillcolor="LightSkyBlue")
        fig.add_hline(y=orh, line_dash="dot", opacity=0.5); fig.add_hline(y=orl, line_dash="dot", opacity=0.5)

    shade_rth(fig, df)

    fig.update_layout(
        title=f"{ticker} — 24h",
        height=420, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig
