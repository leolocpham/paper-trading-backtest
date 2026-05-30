"""
streamlit_app.py  —  Paper Trading Backtest Engine
===================================================
Run locally:  streamlit run streamlit_app.py
Deploy:       push to GitHub, connect at share.streamlit.io
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

from paper_trading_engine import generate_synthetic_ohlcv, PaperTradingEngine


@st.cache_data
def _sample_csv_bytes() -> bytes:
    """500-bar synthetic OHLCV file with a ticker column for testing the CSV upload path."""
    df = generate_synthetic_ohlcv(n_bars=500, seed=99)
    df.index.name = "date"
    out = df.reset_index()
    out.insert(1, "ticker", "SAMPLE")   # ticker column right after date
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
- **Synthetic (GBM)**: Generates realistic price data instantly via Geometric Brownian Motion.
  No setup required — good for exploring strategy behaviour and parameter sensitivity.
- **Upload CSV**: Run the backtest on real market data you provide.
  Download a pre-formatted sample file from the sidebar to see the expected layout.

**Step 2 — Set simulation parameters** (sidebar)

| Parameter | What it controls |
|---|---|
| Starting Capital | Initial portfolio value ($) |
| Risk per Trade | Fraction of equity risked on each trade (position sizing via fixed-fractional) |
| Number of Bars | Candles in the synthetic dataset — more bars produce more trades |
| Annual Drift | Directional trend of the synthetic price series |
| Daily Volatility | Day-to-day price swing magnitude |

**Step 3 — Run Backtest**
Click **Run Backtest**. Results are cached — re-running with identical settings is instant.
Change any parameter and click again to compare.

**Step 4 — Read the Results**

| Section | What to look for |
|---|---|
| Strategy Summary | Final capital and total return at a glance for all four strategies |
| Equity Curves | Portfolio growth per bar; red shading = drawdown (gap below peak equity) |
| Strategy Detail tabs | Win rate, max drawdown, and a colour-coded trade log (green = profit, red = loss) |

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

- Extra columns (dates, ticker symbols, adjusted close) are ignored.
- Rows with any missing OHLCV value are dropped automatically.
- There is no minimum row count, but strategies with long lookback periods
  (e.g. Golden Cross uses a 200-bar SMA) need at least 300–500 rows to fire signals.

**Minimal valid example:**
```
open,high,low,close,volume
100.12,102.45,99.87,101.33,1250000
101.33,103.10,100.90,102.78,980000
102.78,104.22,101.50,103.91,870000
```

**Yahoo Finance export example** (extra columns are fine — automatically dropped):
```
Date,Open,High,Low,Close,Adj Close,Volume
2024-01-02,476.33,479.05,475.82,478.57,478.57,52341200
2024-01-03,478.20,479.44,472.71,473.15,473.15,61234800
```

Download the sample file from the sidebar (**Upload CSV** mode) to see a ready-to-use 500-bar CSV.
""")

    with tab_sources:
        st.markdown("""
### Recommended Free Data Sources

| Source | Asset Classes | Notes |
|---|---|---|
| **Yahoo Finance** | Stocks, ETFs, Crypto, FX, Indices | Easiest. Manual download or `yfinance` library |
| **Alpha Vantage** | Stocks, FX, Crypto | Free API key; 25 requests/day on free tier |
| **Binance** | Crypto (spot & futures) | Best crypto OHLCV; REST API, no key for public data |
| **Stooq** | Global stocks, Indices, FX | Direct CSV downloads; no account needed |
| **Kaggle** | Curated datasets (various) | Search "OHLCV" for ready-to-use backtest files |
| **FRED** | Bonds, rates, commodities | Macro data; good for macro overlay strategies |

---

#### Yahoo Finance — quickest manual path
1. Go to [finance.yahoo.com](https://finance.yahoo.com) and search for any ticker (e.g. **SPY**)
2. Click **Historical Data** tab
3. Set your date range and click **Download**
4. Upload the downloaded `.csv` directly — column names are normalised automatically

---

#### Yahoo Finance via Python (`yfinance`)
```python
pip install yfinance

import yfinance as yf

df = yf.download("SPY", start="2018-01-01", end="2024-01-01")
df.columns = [c.lower() for c in df.columns]
df = df[["open", "high", "low", "close", "volume"]].dropna()
df.to_csv("spy_daily.csv", index=False)
# Upload spy_daily.csv to this app
```

#### Binance crypto data via Python
```python
pip install python-binance

from binance.client import Client
import pandas as pd

client = Client()          # no API key needed for public endpoints
klines = client.get_historical_klines(
    "BTCUSDT", Client.KLINE_INTERVAL_1DAY,
    "1 Jan, 2021", "1 Jan, 2024"
)
df = pd.DataFrame(klines, columns=[
    "date","open","high","low","close","volume",
    "close_time","quote_vol","trades","taker_buy_base",
    "taker_buy_quote","ignore"
])
df = df[["open","high","low","close","volume"]].astype(float)
df.to_csv("btcusdt_daily.csv", index=False)
```

#### Stooq manual download (no account needed)
1. Go to [stooq.com](https://stooq.com/q/d/?s=spy.us) (example: SPY)
2. Click the **Download Data** icon
3. Upload the `.txt` file — it uses comma-separated OHLCV format
""")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    # Data source
    st.subheader("Data Source")
    data_source = st.radio(
        "source",
        ["Synthetic (GBM)", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    csv_bytes       = None   # raw bytes of the uploaded file (read once here)
    selected_ticker = "ASSET"

    if data_source == "Upload CSV":
        st.download_button(
            label="Download sample CSV",
            data=_sample_csv_bytes(),
            file_name="sample_ohlcv_500bars.csv",
            mime="text/csv",
            use_container_width=True,
            help="500-bar synthetic OHLCV with a ticker column — upload it back to test the CSV path.",
        )
        uploaded_file = st.file_uploader(
            "Upload your OHLCV CSV",
            type=["csv"],
            help="Required columns: open, high, low, close, volume (case-insensitive).",
        )
        if uploaded_file is not None:
            csv_bytes = uploaded_file.read()   # read once; reuse via io.BytesIO later

            # Detect any ticker column and let the user pick
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
                        help="The CSV contains multiple tickers. The backtest runs on one at a time.",
                    )
            else:
                selected_ticker = st.text_input(
                    "Ticker Label",
                    value="ASSET",
                    help="No ticker column found — enter a label that will appear in the trade log.",
                ).upper()

        st.caption("See the **How to Use** panel above for format details and data sources.")

    st.divider()

    # Simulation parameters
    st.subheader("Simulation Parameters")
    starting_capital = st.number_input(
        "Starting Capital ($)",
        min_value=1_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        format="%d",
    )
    risk_pct = st.slider(
        "Risk per Trade (%)",
        min_value=0.25,
        max_value=5.0,
        value=1.0,
        step=0.25,
        format="%.2f",
    )

    # Synthetic data settings (only shown for GBM source)
    if data_source == "Synthetic (GBM)":
        st.divider()
        st.subheader("Synthetic Data Settings")
        selected_ticker = st.text_input(
            "Ticker Label",
            value="SYNTHETIC",
            help="Label shown in the trade log and performance report.",
        ).upper()
        n_bars    = st.slider("Number of Bars",       300, 3_000, 1_200, 100)
        seed      = int(st.number_input("Random Seed", 1, 9999, 42))
        drift_pct = st.slider("Annual Drift (%)",      -30, 30, 8)
        vol_pct   = st.slider("Daily Volatility (%)",  0.5, 5.0, 1.5, 0.1)

    st.divider()
    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# BACKTEST EXECUTION (CACHED)
# ─────────────────────────────────────────────

def _execute(df: pd.DataFrame, cap: float, risk: float, ticker: str = "ASSET") -> dict:
    """
    Run the engine and return a plain serialisable dict so
    st.cache_data can hash/pickle the result without issues.
    """
    engine = PaperTradingEngine(df=df, starting_capital=cap,
                                risk_per_trade=risk, tick_size=0.01,
                                ticker=ticker)
    engine.calculate_indicators()
    engine.run()

    out: dict = {"starting_capital": cap, "n_bars": len(df), "strategies": {}}

    for s in engine.STRATEGIES:
        trades  = engine.trades[s]
        equity  = np.array(engine.equity_curve[s])
        n       = len(trades)
        wins    = sum(1 for t in trades if t.pnl > 0)
        win_rt  = wins / n * 100 if n else 0.0
        final   = float(equity[-1])
        ret_pct = (final - cap) / cap * 100.0

        peaks  = np.maximum.accumulate(equity)
        max_dd = float(((equity - peaks) / peaks * 100.0).min())

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


@st.cache_data(show_spinner=False)
def run_synthetic(n_bars, cap, risk, seed, drift_ann_pct, vol_daily_pct, ticker="SYNTHETIC"):
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

    # Filter to the requested ticker if the column is present
    if "ticker" in df.columns:
        df = df[df["ticker"].str.upper() == ticker.upper()].drop(columns=["ticker"])

    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df[sorted(required)].dropna().reset_index(drop=True)
    return _execute(df, cap, risk, ticker)


# Determine whether to (re-)run
need_run = run_btn or ("results" not in st.session_state)

if need_run:
    if data_source == "Upload CSV":
        if csv_bytes is None:
            if run_btn:
                st.warning("Please upload a CSV file before clicking Run.")
                st.stop()
            else:
                # First load with CSV mode selected but no file yet — fall back to synthetic
                data_source     = "Synthetic (GBM)"
                selected_ticker = "SYNTHETIC"
                n_bars, seed, drift_pct, vol_pct = 1_200, 42, 8, 1.5

    with st.spinner("Running backtest across all four strategies..."):
        try:
            if data_source == "Upload CSV":
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
# DISPLAY HELPERS
# ─────────────────────────────────────────────

STRATEGY_LABELS = {
    "A_PinBar":      "A — Pin Bar Scalping",
    "B_GoldenCross": "B — Golden/Death Cross",
    "C_FibEMA":      "C — 5-8-13 Fib EMA",
    "D_ADXBreakout": "D — ADX Breakout",
}
PALETTE = ["steelblue", "darkorange", "seagreen", "crimson"]

# ─────────────────────────────────────────────
# SUMMARY METRICS ROW
# ─────────────────────────────────────────────

st.divider()
st.subheader("Strategy Summary")

cols = st.columns(4)
for col, (sid, label) in zip(cols, STRATEGY_LABELS.items()):
    s = results["strategies"][sid]
    col.metric(
        label=label,
        value=f"${s['final']:,.0f}",
        delta=f"{s['ret_pct']:+.2f}%",
    )

# ─────────────────────────────────────────────
# EQUITY CURVE CHART
# ─────────────────────────────────────────────

st.divider()
st.subheader("Equity Curves")

fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
axes      = axes.flatten()

for ax, (sid, label), color in zip(axes, STRATEGY_LABELS.items(), PALETTE):
    s      = results["strategies"][sid]
    equity = np.array(s["equity"])
    x      = np.arange(len(equity))
    peaks  = np.maximum.accumulate(equity)

    ax.fill_between(x, equity, peaks, alpha=0.18, color="red",      label="Drawdown")
    ax.plot(x, equity,                lw=1.3,     color=color,      label="Equity")
    ax.axhline(results["starting_capital"], lw=0.8, color="dimgrey",
               linestyle="--",                                        label="Start")

    ax.set_title(
        f"{label}  |  Return: {s['ret_pct']:+.1f}%  |  Trades: {s['n_trades']}",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("Bar index", fontsize=8)
    ax.set_ylabel("Portfolio ($)", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)

fig.suptitle(
    "Paper Trading Backtest — Equity Curves",
    fontsize=12, fontweight="bold",
)
st.pyplot(fig)
plt.close(fig)

# ─────────────────────────────────────────────
# PER-STRATEGY DETAIL TABS
# ─────────────────────────────────────────────

st.divider()
st.subheader("Strategy Detail")

tabs = st.tabs(list(STRATEGY_LABELS.values()))

for tab, (sid, _) in zip(tabs, STRATEGY_LABELS.items()):
    s = results["strategies"][sid]
    with tab:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Trades",  s["n_trades"])
        m2.metric("Wins",          s["wins"])
        m3.metric("Win Rate",      f"{s['win_rate']:.1f}%")
        m4.metric("Max Drawdown",  f"{s['max_dd']:.2f}%")
        m5.metric("Total Return",  f"{s['ret_pct']:+.2f}%")

        st.write("")  # spacing

        if s["trades"]:
            df_trades = pd.DataFrame(s["trades"])

            # Colour-code P&L column
            def colour_pnl(val):
                colour = "color: limegreen" if val >= 0 else "color: tomato"
                return colour

            st.dataframe(
                df_trades.style.map(colour_pnl, subset=["P&L ($)", "P&L (%)"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No trades were executed for this strategy on this dataset.  "
                "Try increasing the number of bars or adjusting the parameters."
            )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.caption(
    "Paper trading only — no real orders are placed.  "
    f"Backtest ran over {results['n_bars']:,} bars.  "
    "Starting capital: $"
    f"{results['starting_capital']:,.0f}."
)
