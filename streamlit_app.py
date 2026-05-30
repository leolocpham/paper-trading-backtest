"""
streamlit_app.py  —  Paper Trading Backtest Engine
===================================================
Run locally:  streamlit run streamlit_app.py
Deploy:       push to GitHub, connect at share.streamlit.io
"""

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import streamlit as st
import yfinance as yf

from paper_trading_engine import generate_synthetic_ohlcv, PaperTradingEngine


# ─────────────────────────────────────────────
# HELPERS DEFINED BEFORE SIDEBAR
# ─────────────────────────────────────────────

@st.cache_data
def _sample_csv_bytes() -> bytes:
    """500-bar synthetic OHLCV CSV with a ticker column for testing the upload path."""
    df = generate_synthetic_ohlcv(n_bars=500, seed=99)
    df.index.name = "date"
    out = df.reset_index()
    out.insert(1, "ticker", "SAMPLE")
    return out.to_csv(index=False).encode("utf-8")


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Paper Trading Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Paper Trading Backtest Engine")
st.caption(
    "Four independent strategies from the CFI Guide to Trading — "
    "fully simulated with no real money at risk."
)

# ─────────────────────────────────────────────
# INSTRUCTIONS PANEL
# ─────────────────────────────────────────────

with st.expander("How to Use This App", expanded=False):
    tab_use, tab_csv, tab_sources = st.tabs(
        ["Using the App", "CSV Format", "Data Sources"]
    )

    with tab_use:
        st.markdown("""
### Quick Start

**Step 1 — Choose a data source** (sidebar)
- **Yahoo Finance**: Enter any ticker (SPY, AAPL, BTC-USD) and pick a period (default 5 years). Real OHLCV data fetched automatically.
- **Synthetic (GBM)**: Generates realistic price data instantly. Good for parameter sensitivity testing.
- **Upload CSV**: Your own historical OHLCV file. Download the sample CSV from the sidebar to see the expected format.

**Step 2 — Set simulation parameters** (sidebar)

| Parameter | What it controls |
|---|---|
| Starting Capital | Initial portfolio value ($) |
| Risk per Trade | Fraction of equity risked per trade (fixed-fractional position sizing) |
| Period / Interval | For Yahoo Finance: how many years of data and the bar size (daily/weekly) |

**Step 3 — Run Backtest**
Click **Run Backtest**. Results are cached — re-running with identical settings is instant.

**Step 4 — Read the Results**

| Section | What to look for |
|---|---|
| Strategy Summary | Final capital and total return at a glance for all four strategies |
| Equity Curves | Portfolio growth per bar; red shading = drawdown (gap below peak equity) |
| Strategy Detail tabs | Win rate, max drawdown, colour-coded trade log. Filter by **Wins / Losses**, then select any trade to open the drill-down chart. |

**Step 5 — Drill Down into a Trade**
1. Open any Strategy Detail tab.
2. Click **Wins** or **Losses** to filter, or keep **All Trades**.
3. Use the **Select a trade** dropdown to pick a specific trade.
4. The drill-down panel renders a zoomed price + indicator chart with entry/exit markers and the full signal reasoning text.

**The four strategies**

| ID | Name | Signal | Exit |
|---|---|---|---|
| A | Pin Bar Scalping | Candle tail ≥ 2.5× body, extended past 10/21 EMA | 10 EMA touch or 1-tick stop |
| B | Golden / Death Cross | 50 SMA crosses above/below 200 SMA | Held until opposite crossover |
| C | 5-8-13 Fib EMA | 5 EMA crosses 13 EMA with ribbon fanning | Held until opposite crossover |
| D | ADX Breakout | ADX crosses above 25 with positive slope | ADX drops below 25 or slope turns negative |
""")

    with tab_csv:
        st.markdown("""
### CSV Format Requirements

The engine accepts any OHLCV time series at any timeframe (daily, hourly, 5-minute, etc.).

**Required columns** — names are normalised to lowercase automatically:

| Column | Description |
|---|---|
| `open` | Bar opening price |
| `high` | Bar high price |
| `low` | Bar low price |
| `close` | Bar closing price |
| `volume` | Traded volume (any unit) |
| `ticker` | *(optional)* Symbol label — enables multi-ticker CSV with a selectbox |

- Extra columns (dates, adjusted close, etc.) are ignored automatically.
- Rows with any missing OHLCV value are dropped.
- Strategies with long lookbacks (Golden Cross uses 200-bar SMA) need ≥ 300 rows to generate signals.

**Minimal valid example:**
```
open,high,low,close,volume
100.12,102.45,99.87,101.33,1250000
101.33,103.10,100.90,102.78,980000
```

**Yahoo Finance export example** (extra columns dropped automatically):
```
Date,Open,High,Low,Close,Adj Close,Volume
2024-01-02,476.33,479.05,475.82,478.57,478.57,52341200
```
""")

    with tab_sources:
        st.markdown("""
### Recommended Free Data Sources

| Source | Asset Classes | Notes |
|---|---|---|
| **Yahoo Finance** | Stocks, ETFs, Crypto, FX, Indices | Built into this app — just enter a ticker |
| **Alpha Vantage** | Stocks, FX, Crypto | Free API key; 25 req/day on free tier |
| **Binance** | Crypto (spot & futures) | REST API, no key for public OHLCV |
| **Stooq** | Global stocks, Indices, FX | Direct CSV downloads, no account needed |
| **Kaggle** | Curated datasets | Search "OHLCV" for ready-to-use backtest files |

#### Yahoo Finance via Python (`yfinance`)
```python
import yfinance as yf
tkr = yf.Ticker("SPY")
df = tkr.history(period="5y", interval="1d")
df.columns = [c.lower() for c in df.columns]
df[["open","high","low","close","volume"]].to_csv("spy_5y.csv", index=False)
```

#### Binance crypto data
```python
pip install python-binance
from binance.client import Client
import pandas as pd

klines = Client().get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2020")
df = pd.DataFrame(klines, columns=[
    "date","open","high","low","close","volume",
    "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
])
df[["open","high","low","close","volume"]].astype(float).to_csv("btc_daily.csv", index=False)
```

#### Stooq manual download (no account)
1. Go to stooq.com, search a symbol (e.g. spy.us)
2. Click the download icon → upload the `.txt` directly
""")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    st.subheader("Data Source")
    data_source = st.radio(
        "source",
        ["Yahoo Finance", "Synthetic (GBM)", "Upload CSV"],
        horizontal=False,
        label_visibility="collapsed",
    )

    # Initialise variables that some branches set and others don't
    csv_bytes       = None
    selected_ticker = "ASSET"
    n_bars, seed, drift_pct, vol_pct = 1_200, 42, 8, 1.5
    yf_ticker, yf_years, yf_interval = "SPY", 5, "1d"

    # ── Yahoo Finance ──────────────────────────────────────────
    if data_source == "Yahoo Finance":
        selected_ticker = st.text_input(
            "Ticker Symbol",
            value="SPY",
            help="Examples: SPY  AAPL  MSFT  BTC-USD  EURUSD=X",
        ).upper()
        col_y, col_i = st.columns(2)
        with col_y:
            yf_years = st.selectbox("Period (years)", [1, 2, 3, 5, 10], index=3)
        with col_i:
            yf_interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
        st.caption("Data is cached for 1 hour to avoid repeat downloads.")

    # ── Upload CSV ─────────────────────────────────────────────
    elif data_source == "Upload CSV":
        st.download_button(
            label="Download sample CSV",
            data=_sample_csv_bytes(),
            file_name="sample_ohlcv_500bars.csv",
            mime="text/csv",
            use_container_width=True,
            help="500-bar synthetic OHLCV with a ticker column. Upload it back to test the CSV path.",
        )
        uploaded_file = st.file_uploader(
            "Upload your OHLCV CSV",
            type=["csv"],
            help="Required columns: open, high, low, close, volume (case-insensitive).",
        )
        if uploaded_file is not None:
            csv_bytes = uploaded_file.read()
            peek = pd.read_csv(io.BytesIO(csv_bytes))
            peek.columns = [c.strip().lower() for c in peek.columns]
            if "ticker" in peek.columns:
                tickers = sorted(peek["ticker"].dropna().str.upper().unique().tolist())
                if len(tickers) == 1:
                    selected_ticker = tickers[0]
                    st.info(f"Ticker detected: **{selected_ticker}**")
                else:
                    selected_ticker = st.selectbox(
                        "Select Ticker",
                        tickers,
                        help="Multiple tickers found — backtest runs on one at a time.",
                    )
            else:
                selected_ticker = st.text_input(
                    "Ticker Label", value="ASSET",
                    help="No ticker column found — label used in the trade log.",
                ).upper()
        st.caption("See **How to Use** above for format details.")

    # ── Synthetic (GBM) ────────────────────────────────────────
    else:
        selected_ticker = st.text_input(
            "Ticker Label", value="SYNTHETIC",
            help="Label shown in the trade log and performance report.",
        ).upper()
        n_bars    = st.slider("Number of Bars",       300, 3_000, 1_200, 100)
        seed      = int(st.number_input("Random Seed", 1, 9999, 42))
        drift_pct = st.slider("Annual Drift (%)",      -30, 30, 8)
        vol_pct   = st.slider("Daily Volatility (%)",  0.5, 5.0, 1.5, 0.1)

    st.divider()
    st.subheader("Simulation Parameters")
    starting_capital = st.number_input(
        "Starting Capital ($)",
        min_value=1_000, max_value=10_000_000,
        value=100_000, step=10_000, format="%d",
    )
    risk_pct = st.slider(
        "Risk per Trade (%)",
        min_value=0.25, max_value=5.0,
        value=1.0, step=0.25, format="%.2f",
    )

    st.divider()
    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# BACKTEST EXECUTION (CACHED)
# ─────────────────────────────────────────────

def _execute(df: pd.DataFrame, cap: float, risk: float, ticker: str = "ASSET") -> dict:
    """
    Run the engine; return a plain serialisable dict (no custom objects)
    so st.cache_data can pickle/hash it without issues.
    """
    engine = PaperTradingEngine(
        df=df, starting_capital=cap,
        risk_per_trade=risk, tick_size=0.01, ticker=ticker,
    )
    engine.calculate_indicators()
    engine.run()

    out: dict = {
        "starting_capital": cap,
        "n_bars":           len(df),
        "ticker":           ticker,
        # Raw OHLCV lists for drill-down charts (bar index = list index)
        "ohlcv": {
            "open":  df["open"].round(4).tolist(),
            "high":  df["high"].round(4).tolist(),
            "low":   df["low"].round(4).tolist(),
            "close": df["close"].round(4).tolist(),
        },
        # Indicator lists (NaN replaced with 0 for serialisation)
        "indicators": {
            col: [0.0 if pd.isna(v) else round(float(v), 6)
                  for v in engine.ind[col]]
            for col in engine.ind.columns
        },
        "strategies": {},
    }

    for s in engine.STRATEGIES:
        trades  = engine.trades[s]
        equity  = np.array(engine.equity_curve[s])
        n       = len(trades)
        wins    = sum(1 for t in trades if t.pnl > 0)
        win_rt  = wins / n * 100 if n else 0.0
        final   = float(equity[-1])
        ret_pct = (final - cap) / cap * 100.0
        peaks   = np.maximum.accumulate(equity)
        max_dd  = float(((equity - peaks) / peaks * 100.0).min())

        rows = [
            {
                "Ticker":      t.ticker,
                "Direction":   t.direction.capitalize(),
                "Entry Price": round(t.entry_price, 4),
                "Exit Price":  round(t.exit_price,  4),
                "Size":        round(t.size, 4),
                "P&L ($)":     round(t.pnl, 2),
                "P&L (%)":     round(t.pnl_pct, 2),
                "Entry Bar":   t.entry_index,
                "Exit Bar":    t.exit_index,
                "Reason":      t.reason,   # used for drill-down, not displayed in table
            }
            for t in trades
        ]

        out["strategies"][s] = {
            "equity":   equity.tolist(),
            "n_trades": n,
            "wins":     wins,
            "win_rate": win_rt,
            "final":    final,
            "ret_pct":  ret_pct,
            "max_dd":   max_dd,
            "trades":   rows,
        }

    return out


@st.cache_data(show_spinner=False, ttl=3_600)
def run_yfinance(ticker: str, years: int, interval: str,
                 cap: float, risk: float) -> dict:
    end_dt   = date.today()
    start_dt = end_dt.replace(year=end_dt.year - years)
    tkr_obj  = yf.Ticker(ticker)
    df = tkr_obj.history(
        start=str(start_dt), end=str(end_dt),
        interval=interval, auto_adjust=True,
    )
    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. "
            "Check the symbol and try again (e.g. BTC-USD for Bitcoin)."
        )
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
    if len(df) < 210:
        raise ValueError(
            f"Only {len(df)} bars available for '{ticker}'. "
            "The Golden Cross strategy needs ≥ 210 bars. Try a longer period."
        )
    return _execute(df, cap, risk, ticker)


@st.cache_data(show_spinner=False)
def run_synthetic(n_bars, cap, risk, seed,
                  drift_ann_pct, vol_daily_pct, ticker="SYNTHETIC") -> dict:
    df = generate_synthetic_ohlcv(
        n_bars=n_bars,
        volatility=vol_daily_pct / 100,
        drift=drift_ann_pct / 100 / 252,
        seed=seed,
    )
    return _execute(df, cap, risk, ticker)


@st.cache_data(show_spinner=False)
def run_csv(csv_bytes: bytes, cap: float, risk: float, ticker: str = "ASSET") -> dict:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" in df.columns:
        df = df[df["ticker"].str.upper() == ticker.upper()].drop(columns=["ticker"])
    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df[sorted(required)].dropna().reset_index(drop=True)
    return _execute(df, cap, risk, ticker)


# ─────────────────────────────────────────────
# TRIGGER RUN
# ─────────────────────────────────────────────

need_run = run_btn or ("results" not in st.session_state)

if need_run:
    if data_source == "Upload CSV" and csv_bytes is None:
        if run_btn:
            st.warning("Please upload a CSV file before clicking Run.")
            st.stop()
        else:
            # First load, CSV mode but no file — fall back silently to synthetic
            data_source, selected_ticker = "Synthetic (GBM)", "SYNTHETIC"
            n_bars, seed, drift_pct, vol_pct = 1_200, 42, 8, 1.5

    with st.spinner(f"Downloading data and running backtest for {selected_ticker}..."):
        try:
            if data_source == "Yahoo Finance":
                results = run_yfinance(
                    selected_ticker, yf_years, yf_interval,
                    starting_capital, risk_pct / 100,
                )
            elif data_source == "Upload CSV":
                results = run_csv(csv_bytes, starting_capital, risk_pct / 100, selected_ticker)
            else:
                results = run_synthetic(
                    n_bars, starting_capital, risk_pct / 100,
                    seed, drift_pct, vol_pct, selected_ticker,
                )
            st.session_state["results"] = results
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

results = st.session_state["results"]

# ─────────────────────────────────────────────
# DISPLAY CONSTANTS
# ─────────────────────────────────────────────

STRATEGY_LABELS = {
    "A_PinBar":      "A — Pin Bar Scalping",
    "B_GoldenCross": "B — Golden/Death Cross",
    "C_FibEMA":      "C — 5-8-13 Fib EMA",
    "D_ADXBreakout": "D — ADX Breakout",
}
PALETTE = ["steelblue", "darkorange", "seagreen", "crimson"]

# Indicator sets overlaid per strategy in drill-down charts
STRATEGY_INDICATORS = {
    "A_PinBar":      [("ema10", "#e67e22", "10 EMA"), ("ema21", "#8e44ad", "21 EMA")],
    "B_GoldenCross": [("sma50", "#2980b9", "50 SMA"), ("sma200", "#c0392b", "200 SMA")],
    "C_FibEMA":      [("ema5",  "#e74c3c", "5 EMA"),  ("ema8", "#e67e22", "8 EMA"),
                      ("ema13", "#27ae60", "13 EMA")],
    "D_ADXBreakout": [],   # ADX rendered in a sub-panel, not on the price axis
}


# ─────────────────────────────────────────────
# SUMMARY METRICS ROW
# ─────────────────────────────────────────────

st.divider()
st.subheader(f"Strategy Summary  —  {results['ticker']}")

cols = st.columns(4)
for col, (sid, label) in zip(cols, STRATEGY_LABELS.items()):
    s = results["strategies"][sid]
    col.metric(label=label, value=f"${s['final']:,.0f}", delta=f"{s['ret_pct']:+.2f}%")

# ─────────────────────────────────────────────
# EQUITY CURVES
# ─────────────────────────────────────────────

st.divider()
st.subheader("Equity Curves")

fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
axes = axes.flatten()

for ax, (sid, label), color in zip(axes, STRATEGY_LABELS.items(), PALETTE):
    s      = results["strategies"][sid]
    equity = np.array(s["equity"])
    x      = np.arange(len(equity))
    peaks  = np.maximum.accumulate(equity)

    ax.fill_between(x, equity, peaks, alpha=0.18, color="red",    label="Drawdown")
    ax.plot(x, equity,                lw=1.3,     color=color,    label="Equity")
    ax.axhline(results["starting_capital"], lw=0.8, color="dimgrey",
               linestyle="--",                                      label="Start")
    ax.set_title(
        f"{label}  |  {results['ticker']}  |  {s['ret_pct']:+.1f}%  |  {s['n_trades']} trades",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("Bar index", fontsize=8)
    ax.set_ylabel("Portfolio ($)", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)

fig.suptitle("Paper Trading Backtest — Equity Curves", fontsize=12, fontweight="bold")
st.pyplot(fig)
plt.close(fig)


# ─────────────────────────────────────────────
# DRILL-DOWN RENDERER
# ─────────────────────────────────────────────

def _render_drill_down(trade: dict, ohlcv: dict, indicators: dict, sid: str) -> None:
    """
    Render a zoomed price + indicator chart centred on a single trade,
    with entry/exit markers and the full signal reasoning text below.
    """
    entry_bar = trade["Entry Bar"]
    exit_bar  = trade["Exit Bar"]
    n         = len(ohlcv["close"])

    # 25 bars of context before entry, 10 bars after exit
    win_start = max(0, entry_bar - 25)
    win_end   = min(n - 1, exit_bar + 10)
    idx       = list(range(win_start, win_end + 1))

    closes = [ohlcv["close"][i] for i in idx]
    highs  = [ohlcv["high"][i]  for i in idx]
    lows   = [ohlcv["low"][i]   for i in idx]

    entry_rel = entry_bar - win_start
    exit_rel  = exit_bar  - win_start
    x         = range(len(idx))

    profitable = trade["P&L ($)"] >= 0
    direction  = trade["Direction"].lower()

    # ── Build figure (2-panel for ADX strategy, 1-panel for others) ──────
    has_adx_panel = sid == "D_ADXBreakout"
    if has_adx_panel:
        fig, (ax_p, ax_a) = plt.subplots(
            2, 1, figsize=(13, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )
    else:
        fig, ax_p = plt.subplots(1, 1, figsize=(13, 5), constrained_layout=True)

    # ── Price panel ────────────────────────────────────────────────────────
    ax_p.fill_between(x, highs, lows, alpha=0.12, color="steelblue", label="H–L range")
    ax_p.plot(x, closes, color="#1a1a2e", lw=1.2, label="Close")

    # Shade trade region green/red
    ax_p.axvspan(entry_rel, exit_rel,
                 alpha=0.08,
                 color="limegreen" if profitable else "tomato")

    # Vertical entry / exit lines
    ax_p.axvline(entry_rel, color="green", lw=1.8, linestyle="--",
                 label=f"Entry {entry_bar} @ {trade['Entry Price']:.4f}")
    ax_p.axvline(exit_rel,  color="red",   lw=1.8, linestyle="--",
                 label=f"Exit {exit_bar} @ {trade['Exit Price']:.4f}")

    # Horizontal price lines for entry and exit
    ax_p.axhline(trade["Entry Price"], color="green", lw=0.7, linestyle=":", alpha=0.6)
    ax_p.axhline(trade["Exit Price"],  color="red",   lw=0.7, linestyle=":", alpha=0.6)

    # Entry / exit arrow markers on the close line
    entry_marker = "^" if direction == "long" else "v"
    exit_marker  = "v" if direction == "long" else "^"
    ax_p.scatter([entry_rel], [closes[entry_rel]], color="green",
                 s=90, zorder=6, marker=entry_marker)
    ax_p.scatter([exit_rel],  [closes[exit_rel]],  color="red",
                 s=90, zorder=6, marker=exit_marker)

    # Strategy-specific indicator overlays
    for ind_key, ind_color, ind_label in STRATEGY_INDICATORS[sid]:
        vals = [indicators[ind_key][i] for i in idx]
        ax_p.plot(x, vals, color=ind_color, lw=1.1, linestyle="--", label=ind_label)

    ax_p.set_title(
        f"{trade['Ticker']}  |  {sid}  |  {trade['Direction']}  |  "
        f"P&L: ${trade['P&L ($)']:+,.2f} ({trade['P&L (%)']:+.2f}%)",
        fontsize=10, fontweight="bold",
    )
    ax_p.set_ylabel("Price", fontsize=9)
    ax_p.set_xlabel("Bar offset from window start", fontsize=8)
    ax_p.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax_p.legend(fontsize=7, loc="best", ncol=2)
    ax_p.grid(True, alpha=0.22)

    # ── ADX sub-panel ──────────────────────────────────────────────────────
    if has_adx_panel:
        adx_vals = [indicators["adx"][i] for i in idx]
        ax_a.plot(x, adx_vals, color="#8e44ad", lw=1.2, label="ADX (14)")
        ax_a.axhline(25, color="gray", lw=0.9, linestyle="--", label="Threshold = 25")
        ax_a.fill_between(x, 0, adx_vals,
                          where=[v > 25 for v in adx_vals],
                          alpha=0.18, color="#8e44ad")
        ax_a.axvline(entry_rel, color="green", lw=1.8, linestyle="--")
        ax_a.axvline(exit_rel,  color="red",   lw=1.8, linestyle="--")
        ax_a.set_ylabel("ADX", fontsize=9)
        ax_a.legend(fontsize=7)
        ax_a.grid(True, alpha=0.22)

    st.pyplot(fig)
    plt.close(fig)

    # ── Signal reasoning text ──────────────────────────────────────────────
    outcome = "WIN" if profitable else "LOSS"
    badge   = "🟢" if profitable else "🔴"
    st.markdown(f"**{badge} Trade Outcome: {outcome}**")
    st.info(f"**Signal Reasoning:**\n\n{trade['Reason']}")


# ─────────────────────────────────────────────
# PER-STRATEGY DETAIL TABS
# ─────────────────────────────────────────────

st.divider()
st.subheader("Strategy Detail")

tabs = st.tabs(list(STRATEGY_LABELS.values()))

for tab, (sid, _) in zip(tabs, STRATEGY_LABELS.items()):
    s = results["strategies"][sid]
    with tab:

        # ── Summary metrics ──────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Trades",  s["n_trades"])
        m2.metric("Wins",          s["wins"])
        m3.metric("Win Rate",      f"{s['win_rate']:.1f}%")
        m4.metric("Max Drawdown",  f"{s['max_dd']:.2f}%")
        m5.metric("Total Return",  f"{s['ret_pct']:+.2f}%")

        if not s["trades"]:
            st.info(
                "No trades were executed for this strategy on this dataset. "
                "Try increasing the period or number of bars."
            )
            continue

        # ── Win / Loss filter ─────────────────────────────────────────────
        st.write("")
        trade_filter = st.radio(
            "Show trades",
            ["All Trades", "Wins", "Losses"],
            horizontal=True,
            key=f"filter_{sid}",
        )
        if trade_filter == "Wins":
            filtered = [t for t in s["trades"] if t["P&L ($)"] >= 0]
        elif trade_filter == "Losses":
            filtered = [t for t in s["trades"] if t["P&L ($)"] < 0]
        else:
            filtered = s["trades"]

        if not filtered:
            st.info(f"No {trade_filter.lower()} found for this strategy.")
            continue

        # ── Trade log table (Reason and bar-index columns hidden) ─────────
        display_cols = ["Ticker", "Direction", "Entry Price", "Exit Price",
                        "Size", "P&L ($)", "P&L (%)"]
        df_display = pd.DataFrame(filtered)[display_cols]

        def colour_pnl(val):
            return "color: limegreen" if val >= 0 else "color: tomato"

        st.dataframe(
            df_display.style.map(colour_pnl, subset=["P&L ($)", "P&L (%)"]),
            use_container_width=True,
            hide_index=True,
        )

        # ── Trade drill-down selector ─────────────────────────────────────
        st.write("")
        st.markdown("**Drill Down into a Trade**")

        trade_options = {
            f"Trade {i + 1}:  {t['Direction']}  |  "
            f"Entry bar {t['Entry Bar']} @ {t['Entry Price']}  →  "
            f"Exit bar {t['Exit Bar']} @ {t['Exit Price']}  |  "
            f"P&L: ${t['P&L ($)']:+,.2f}": i
            for i, t in enumerate(filtered)
        }

        chosen_label = st.selectbox(
            "Select a trade",
            list(trade_options.keys()),
            key=f"drill_{sid}",
            label_visibility="collapsed",
        )
        chosen_idx   = trade_options[chosen_label]
        chosen_trade = filtered[chosen_idx]

        _render_drill_down(
            chosen_trade,
            results["ohlcv"],
            results["indicators"],
            sid,
        )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.caption(
    f"Paper trading only — no real orders are placed.  "
    f"Ticker: {results['ticker']}  |  "
    f"Bars: {results['n_bars']:,}  |  "
    f"Starting capital: ${results['starting_capital']:,.0f}"
)
