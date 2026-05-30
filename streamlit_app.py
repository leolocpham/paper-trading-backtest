"""
streamlit_app.py  —  Paper Trading Backtest Engine
===================================================
Run locally:  streamlit run streamlit_app.py
Deploy:       push to GitHub → connect at share.streamlit.io

Two modes
---------
Tab 1  Single Ticker   – any ticker via Yahoo Finance, synthetic GBM, or CSV upload
Tab 2  S&P 500 Scan    – backtest all 500 constituents, view results as an interactive
                         heatmap, click any cell to open the full per-stock drill-down
"""

import io
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from paper_trading_engine import generate_synthetic_ohlcv, PaperTradingEngine

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

STRATEGY_LABELS = {
    "A_PinBar":      "A — Pin Bar",
    "B_GoldenCross": "B — Golden Cross",
    "C_FibEMA":      "C — Fib EMA",
    "D_ADXBreakout": "D — ADX Breakout",
}
PALETTE       = ["steelblue", "darkorange", "seagreen", "crimson"]
BATCH_SIZE    = 50      # tickers per yfinance download batch
MIN_BARS      = 210     # minimum bars needed for all strategies

STRATEGY_INDICATORS = {
    "A_PinBar":      [("ema10", "#e67e22", "10 EMA"), ("ema21", "#8e44ad", "21 EMA")],
    "B_GoldenCross": [("sma50", "#2980b9", "50 SMA"), ("sma200", "#c0392b", "200 SMA")],
    "C_FibEMA":      [("ema5",  "#e74c3c", "5 EMA"), ("ema8", "#e67e22", "8 EMA"),
                      ("ema13", "#27ae60", "13 EMA")],
    "D_ADXBreakout": [],
}


# ═══════════════════════════════════════════════════════════════
# CACHED HELPERS
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def _sample_csv_bytes() -> bytes:
    df = generate_synthetic_ohlcv(n_bars=500, seed=99)
    df.index.name = "date"
    out = df.reset_index()
    out.insert(1, "ticker", "SAMPLE")
    return out.to_csv(index=False).encode("utf-8")


@st.cache_data(ttl=86_400)
def get_sp500_info() -> tuple:
    """
    Fetch S&P 500 constituents from Wikipedia.
    Returns (sorted_tickers, {ticker: sector}, {ticker: company_name}).
    Cached for 24 h.
    Uses a browser User-Agent so Wikipedia doesn't 403 the request.
    """
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; paper-trading-bot/1.0)"}
    html    = requests.get(url, headers=headers, timeout=15).text
    tbl     = pd.read_html(io.StringIO(html), header=0)[0]
    tbl.columns = [str(c).strip() for c in tbl.columns]

    sym_col = next((c for c in tbl.columns if "symbol" in c.lower()), tbl.columns[0])
    sec_col = next((c for c in tbl.columns if "sector" in c.lower()), None)
    nam_col = next((c for c in tbl.columns if "security" in c.lower()
                    or "name" in c.lower()), None)

    tbl[sym_col] = tbl[sym_col].str.replace(".", "-", regex=False)
    tickers = sorted(tbl[sym_col].dropna().tolist())
    sectors = dict(zip(tbl[sym_col], tbl[sec_col] if sec_col else ["Unknown"] * len(tbl)))
    names   = dict(zip(tbl[sym_col], tbl[nam_col] if nam_col else tbl[sym_col]))
    return tickers, sectors, names


# ─────────────────────────────────────────────
# BACKTEST EXECUTION HELPERS
# ─────────────────────────────────────────────

def _extract_one_ticker(raw: pd.DataFrame, ticker: str) -> "pd.DataFrame | None":
    """
    Pull a single ticker's OHLCV out of a yfinance multi-ticker download.
    Handles both column layouts yfinance produces:
      • group_by='ticker'  → level 0 = ticker, level 1 = metric
      • default            → level 0 = metric, level 1 = ticker
    """
    if raw is None or raw.empty:
        return None
    try:
        cols = raw.columns
        if isinstance(cols, pd.MultiIndex):
            lvl0 = cols.get_level_values(0).unique().tolist()
            lvl1 = cols.get_level_values(1).unique().tolist()
            if ticker in lvl0:
                df = raw[ticker].copy()
            elif ticker in lvl1:
                df = raw.xs(ticker, axis=1, level=1).copy()
            else:
                return None
        else:
            df = raw.copy()

        df.columns = [str(c).lower() for c in df.columns]
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(set(df.columns)):
            return None
        return df[["open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
    except Exception:
        return None


def _execute(df: pd.DataFrame, cap: float, risk: float, ticker: str = "ASSET") -> dict:
    """Full backtest — stores OHLCV, indicators, and per-trade records for drill-down."""
    engine = PaperTradingEngine(df=df, starting_capital=cap,
                                risk_per_trade=risk, tick_size=0.01, ticker=ticker)
    engine.calculate_indicators()
    engine.run()

    out: dict = {
        "starting_capital": cap,
        "n_bars":  len(df),
        "ticker":  ticker,
        "ohlcv": {
            "open":  df["open"].round(4).tolist(),
            "high":  df["high"].round(4).tolist(),
            "low":   df["low"].round(4).tolist(),
            "close": df["close"].round(4).tolist(),
        },
        "indicators": {
            col: [0.0 if pd.isna(v) else round(float(v), 6) for v in engine.ind[col]]
            for col in engine.ind.columns
        },
        "strategies": {},
    }

    for s in engine.STRATEGIES:
        trades  = engine.trades[s]
        equity  = np.array(engine.equity_curve[s])
        n       = len(trades)
        wins    = sum(1 for t in trades if t.pnl > 0)
        final   = float(equity[-1])
        ret_pct = (final - cap) / cap * 100.0
        peaks   = np.maximum.accumulate(equity)
        max_dd  = float(((equity - peaks) / peaks * 100.0).min())

        out["strategies"][s] = {
            "equity":   equity.tolist(),
            "n_trades": n,
            "wins":     wins,
            "win_rate": wins / n * 100 if n else 0.0,
            "final":    final,
            "ret_pct":  ret_pct,
            "max_dd":   max_dd,
            "trades": [
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
                    "Reason":      t.reason,
                }
                for t in trades
            ],
        }
    return out


def _execute_summary(df: pd.DataFrame, cap: float, risk: float, ticker: str) -> "dict | None":
    """
    Lightweight backtest for the S&P 500 scan.
    Returns only aggregate metrics — no OHLCV/indicator/trade storage,
    so the scan stays fast and memory-efficient.
    """
    try:
        engine = PaperTradingEngine(df=df, starting_capital=cap,
                                    risk_per_trade=risk, tick_size=0.01, ticker=ticker)
        engine.calculate_indicators()
        engine.run()
        row = {"ticker": ticker}
        for s in engine.STRATEGIES:
            trades = engine.trades[s]
            equity = np.array(engine.equity_curve[s])
            n      = len(trades)
            wins   = sum(1 for t in trades if t.pnl > 0)
            final  = float(equity[-1])
            ret    = (final - cap) / cap * 100.0
            peaks  = np.maximum.accumulate(equity)
            max_dd = float(((equity - peaks) / peaks * 100.0).min())
            row[f"{s}_return"]   = round(ret,  2)
            row[f"{s}_win_rate"] = round(wins / n * 100 if n else 0, 1)
            row[f"{s}_max_dd"]   = round(max_dd, 2)
            row[f"{s}_n_trades"] = n
        return row
    except Exception:
        return None


# ─────────────────────────────────────────────
# CACHED BACKTEST RUNNERS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3_600)
def run_yfinance(ticker: str, years: int, interval: str, cap: float, risk: float) -> dict:
    end_dt   = date.today()
    start_dt = end_dt.replace(year=end_dt.year - years)
    df = yf.Ticker(ticker).history(
        start=str(start_dt), end=str(end_dt),
        interval=interval, auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol.")
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
    if len(df) < MIN_BARS:
        raise ValueError(f"Only {len(df)} bars for '{ticker}' — need {MIN_BARS}+. Try a longer period.")
    return _execute(df, cap, risk, ticker)


@st.cache_data(show_spinner=False)
def run_synthetic(n_bars, cap, risk, seed, drift_ann_pct, vol_pct, ticker="SYNTHETIC") -> dict:
    df = generate_synthetic_ohlcv(
        n_bars=n_bars, volatility=vol_pct / 100,
        drift=drift_ann_pct / 100 / 252, seed=seed,
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
        raise ValueError(f"CSV missing columns: {missing}")
    df = df[sorted(required)].dropna().reset_index(drop=True)
    return _execute(df, cap, risk, ticker)


# ═══════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _render_drill_down(trade: dict, ohlcv: dict, indicators: dict, sid: str) -> None:
    """Zoomed price + indicator chart with entry/exit markers and signal reasoning."""
    eb, xb = trade["Entry Bar"], trade["Exit Bar"]
    n = len(ohlcv["close"])
    ws  = max(0, eb - 25)
    we  = min(n - 1, xb + 10)
    idx = list(range(ws, we + 1))

    closes = [ohlcv["close"][i] for i in idx]
    highs  = [ohlcv["high"][i]  for i in idx]
    lows   = [ohlcv["low"][i]   for i in idx]
    er, xr = eb - ws, xb - ws
    x      = range(len(idx))
    profit = trade["P&L ($)"] >= 0
    dirn   = trade["Direction"].lower()

    has_adx = sid == "D_ADXBreakout"
    if has_adx:
        fig, (ax_p, ax_a) = plt.subplots(2, 1, figsize=(13, 7),
                                          gridspec_kw={"height_ratios": [3, 1]},
                                          constrained_layout=True)
    else:
        fig, ax_p = plt.subplots(1, 1, figsize=(13, 5), constrained_layout=True)

    ax_p.fill_between(x, highs, lows, alpha=0.12, color="steelblue", label="H-L range")
    ax_p.plot(x, closes, color="#1a1a2e", lw=1.2, label="Close")
    ax_p.axvspan(er, xr, alpha=0.08, color="limegreen" if profit else "tomato")
    ax_p.axvline(er, color="green", lw=1.8, linestyle="--",
                 label=f"Entry {eb} @ {trade['Entry Price']:.4f}")
    ax_p.axvline(xr, color="red",   lw=1.8, linestyle="--",
                 label=f"Exit {xb} @ {trade['Exit Price']:.4f}")
    ax_p.axhline(trade["Entry Price"], color="green", lw=0.7, linestyle=":", alpha=0.6)
    ax_p.axhline(trade["Exit Price"],  color="red",   lw=0.7, linestyle=":", alpha=0.6)
    ax_p.scatter([er], [closes[er]], color="green", s=90, zorder=6,
                 marker="^" if dirn == "long" else "v")
    ax_p.scatter([xr], [closes[xr]], color="red",   s=90, zorder=6,
                 marker="v" if dirn == "long" else "^")

    for ind_key, ind_color, ind_lbl in STRATEGY_INDICATORS[sid]:
        vals = [indicators[ind_key][i] for i in idx]
        ax_p.plot(x, vals, color=ind_color, lw=1.1, linestyle="--", label=ind_lbl)

    ax_p.set_title(
        f"{trade['Ticker']}  |  {sid}  |  {trade['Direction']}  |  "
        f"P&L: ${trade['P&L ($)']:+,.2f} ({trade['P&L (%)']:+.2f}%)",
        fontsize=10, fontweight="bold",
    )
    ax_p.set_ylabel("Price", fontsize=9)
    ax_p.set_xlabel("Bar offset from window start", fontsize=8)
    ax_p.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax_p.legend(fontsize=7, ncol=2)
    ax_p.grid(True, alpha=0.22)

    if has_adx:
        adx_vals = [indicators["adx"][i] for i in idx]
        ax_a.plot(x, adx_vals, color="#8e44ad", lw=1.2, label="ADX (14)")
        ax_a.axhline(25, color="gray", lw=0.9, linestyle="--", label="25")
        ax_a.fill_between(x, 0, adx_vals,
                          where=[v > 25 for v in adx_vals],
                          alpha=0.18, color="#8e44ad")
        ax_a.axvline(er, color="green", lw=1.8, linestyle="--")
        ax_a.axvline(xr, color="red",   lw=1.8, linestyle="--")
        ax_a.set_ylabel("ADX", fontsize=9)
        ax_a.legend(fontsize=7)
        ax_a.grid(True, alpha=0.22)

    st.pyplot(fig)
    plt.close(fig)

    badge = "🟢 WIN" if profit else "🔴 LOSS"
    st.markdown(f"**{badge}**")
    st.info(f"**Signal Reasoning:**\n\n{trade['Reason']}")


def _render_single_stock_results(results: dict) -> None:
    """
    Render the full single-stock backtest output:
    summary metrics → equity curves → per-strategy tabs with
    win/loss filter, trade log, and trade drill-down.
    """
    st.markdown(f"### {results['ticker']}  —  {results['n_bars']:,} bars")

    # Summary row
    cols = st.columns(4)
    for col, (sid, label) in zip(cols, STRATEGY_LABELS.items()):
        s = results["strategies"][sid]
        col.metric(label, f"${s['final']:,.0f}", f"{s['ret_pct']:+.2f}%")

    # Equity curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    for ax, (sid, label), color in zip(axes.flatten(), STRATEGY_LABELS.items(), PALETTE):
        s      = results["strategies"][sid]
        equity = np.array(s["equity"])
        x      = np.arange(len(equity))
        peaks  = np.maximum.accumulate(equity)
        ax.fill_between(x, equity, peaks, alpha=0.18, color="red", label="Drawdown")
        ax.plot(x, equity, lw=1.3, color=color, label="Equity")
        ax.axhline(results["starting_capital"], lw=0.8, color="dimgrey",
                   linestyle="--", label="Start")
        ax.set_title(
            f"{label}  |  {s['ret_pct']:+.1f}%  |  {s['n_trades']} trades",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("Bar index", fontsize=8)
        ax.set_ylabel("Portfolio ($)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Equity Curves — {results['ticker']}", fontsize=12, fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)

    # Per-strategy detail tabs
    st.markdown("#### Strategy Detail")
    tabs = st.tabs(list(STRATEGY_LABELS.values()))

    for tab, (sid, _) in zip(tabs, STRATEGY_LABELS.items()):
        s = results["strategies"][sid]
        with tab:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Trades",       s["n_trades"])
            m2.metric("Wins",         s["wins"])
            m3.metric("Win Rate",     f"{s['win_rate']:.1f}%")
            m4.metric("Max Drawdown", f"{s['max_dd']:.2f}%")
            m5.metric("Total Return", f"{s['ret_pct']:+.2f}%")

            if not s["trades"]:
                st.info("No trades for this strategy on this dataset.")
                continue

            # Win / Loss filter
            fltr = st.radio("Show", ["All Trades", "Wins", "Losses"],
                            horizontal=True, key=f"fltr_{results['ticker']}_{sid}")
            if fltr == "Wins":
                filtered = [t for t in s["trades"] if t["P&L ($)"] >= 0]
            elif fltr == "Losses":
                filtered = [t for t in s["trades"] if t["P&L ($)"] < 0]
            else:
                filtered = s["trades"]

            if not filtered:
                st.info(f"No {fltr.lower()} for this strategy.")
                continue

            disp_cols = ["Ticker","Direction","Entry Price","Exit Price",
                         "Size","P&L ($)","P&L (%)"]
            df_disp = pd.DataFrame(filtered)[disp_cols]

            def colour_pnl(v):
                return "color: limegreen" if v >= 0 else "color: tomato"

            st.dataframe(df_disp.style.map(colour_pnl, subset=["P&L ($)","P&L (%)"]),
                         use_container_width=True, hide_index=True)

            # Drill-down selector
            st.markdown("**Drill Down into a Trade**")
            opts = {
                f"Trade {i+1}: {t['Direction']} | "
                f"Bar {t['Entry Bar']} @ {t['Entry Price']} -> {t['Exit Bar']} @ {t['Exit Price']} | "
                f"P&L ${t['P&L ($)']:+,.2f}": i
                for i, t in enumerate(filtered)
            }
            label_sel = st.selectbox("Select a trade", list(opts.keys()),
                                     key=f"sel_{results['ticker']}_{sid}",
                                     label_visibility="collapsed")
            trade = filtered[opts[label_sel]]
            _render_drill_down(trade, results["ohlcv"], results["indicators"], sid)


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Paper Trading Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Paper Trading Backtest Engine")
st.caption(
    "Four independent CFI strategies · Single-ticker drill-down · "
    "Full S&P 500 universe scan with interactive heatmap"
)

# ─────────────────────────────────────────────
# INSTRUCTIONS
# ─────────────────────────────────────────────

with st.expander("How to Use This App", expanded=False):
    st.markdown("""
**Tab 1 — Single Ticker Backtest**
1. Choose a data source in the sidebar: **Yahoo Finance** (any ticker, 1-10 yr), **Synthetic GBM**, or **Upload CSV**.
2. Set Starting Capital and Risk per Trade.
3. Click **Run Backtest**.
4. In Strategy Detail, use **Wins / Losses** to filter the trade log, then select any trade for a zoomed drill-down chart with the full signal reasoning.

**Tab 2 — S&P 500 Universe Scan**
1. Pick a period, interval, and optional sector filter.
2. Click **Start S&P 500 Scan** — progress bar updates as each batch of 50 stocks completes (~3-5 min for all 500).
3. Choose a colour metric (Return %, Win Rate, Max Drawdown) to recolour the heatmap.
4. Click any heatmap cell, or use the selectbox below, to open the full per-stock drill-down.

**CSV format:** columns `open, high, low, close, volume` (lowercase). Optional `ticker` column enables multi-symbol files.

**Strategies:**

| | Name | Entry | Exit |
|---|---|---|---|
| A | Pin Bar Scalping | Tail ≥ 2.5× body, extended past 10/21 EMA | 10 EMA touch or 1-tick stop |
| B | Golden/Death Cross | 50 SMA crosses 200 SMA | Opposite crossover |
| C | 5-8-13 Fib EMA | 5 EMA crosses 13 EMA, ribbon fanning | Opposite crossover |
| D | ADX Breakout | ADX crosses above 25, positive slope | ADX < 25 or slope negative |
""")

# ─────────────────────────────────────────────
# GLOBAL SIDEBAR  (parameters used by both tabs)
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Simulation Parameters")
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
    st.caption("These apply to both tabs.")


# ═══════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════

tab_single, tab_scan = st.tabs(["Single Ticker Backtest", "S&P 500 Universe Scan"])


# ───────────────────────────────────────────────────────────────
# TAB 1 — SINGLE TICKER
# ───────────────────────────────────────────────────────────────

with tab_single:
    # ── Data source controls ───────────────────────────────────
    src_col, param_col = st.columns([1, 2])

    with src_col:
        st.subheader("Data Source")
        data_source = st.radio(
            "source", ["Yahoo Finance", "Synthetic (GBM)", "Upload CSV"],
            label_visibility="collapsed",
        )

    # Shared defaults so every branch defines every variable
    csv_bytes       = None
    selected_ticker = "ASSET"
    n_bars, seed, drift_pct, vol_pct = 1_200, 42, 8, 1.5
    yf_years, yf_interval = 5, "1d"

    with param_col:
        st.subheader("Settings")

        if data_source == "Yahoo Finance":
            c1, c2, c3 = st.columns(3)
            with c1:
                selected_ticker = st.text_input("Ticker", "SPY",
                    help="e.g. AAPL  BTC-USD  EURUSD=X").upper()
            with c2:
                yf_years = st.selectbox("Period (years)", [1,2,3,5,10], index=3)
            with c3:
                yf_interval = st.selectbox("Interval", ["1d","1wk","1mo"])

        elif data_source == "Synthetic (GBM)":
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                selected_ticker = st.text_input("Label", "SYNTHETIC").upper()
            with c2:
                n_bars = st.number_input("Bars", 300, 3000, 1200, 100)
            with c3:
                seed = int(st.number_input("Seed", 1, 9999, 42))
            with c4:
                drift_pct = st.slider("Drift (%/yr)", -30, 30, 8)
            with c5:
                vol_pct = st.slider("Vol (%/day)", 0.5, 5.0, 1.5, 0.1)

        else:   # Upload CSV
            c1, c2 = st.columns([1, 2])
            with c1:
                st.download_button(
                    "Download sample CSV", _sample_csv_bytes(),
                    "sample_ohlcv.csv", "text/csv",
                    use_container_width=True,
                )
            with c2:
                uf = st.file_uploader("Upload OHLCV CSV", type=["csv"])
                if uf is not None:
                    csv_bytes = uf.read()
                    pk = pd.read_csv(io.BytesIO(csv_bytes))
                    pk.columns = [c.strip().lower() for c in pk.columns]
                    if "ticker" in pk.columns:
                        tickers_found = sorted(pk["ticker"].dropna().str.upper().unique().tolist())
                        if len(tickers_found) == 1:
                            selected_ticker = tickers_found[0]
                            st.info(f"Ticker: **{selected_ticker}**")
                        else:
                            selected_ticker = st.selectbox("Ticker", tickers_found)
                    else:
                        selected_ticker = st.text_input("Ticker Label", "ASSET").upper()

    run_btn = st.button("Run Backtest", type="primary", key="run_single")

    # ── Execution ──────────────────────────────────────────────
    need_run = run_btn or ("single_results" not in st.session_state)

    if need_run:
        if data_source == "Upload CSV" and csv_bytes is None:
            if run_btn:
                st.warning("Upload a CSV file first.")
                st.stop()
            else:
                data_source, selected_ticker = "Synthetic (GBM)", "SYNTHETIC"

        with st.spinner(f"Running backtest for {selected_ticker}..."):
            try:
                if data_source == "Yahoo Finance":
                    res = run_yfinance(selected_ticker, yf_years, yf_interval,
                                       starting_capital, risk_pct / 100)
                elif data_source == "Upload CSV":
                    res = run_csv(csv_bytes, starting_capital, risk_pct / 100, selected_ticker)
                else:
                    res = run_synthetic(n_bars, starting_capital, risk_pct / 100,
                                        seed, drift_pct, vol_pct, selected_ticker)
                st.session_state["single_results"] = res
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                st.stop()

    _render_single_stock_results(st.session_state["single_results"])


# ───────────────────────────────────────────────────────────────
# TAB 2 — S&P 500 UNIVERSE SCAN
# ───────────────────────────────────────────────────────────────

with tab_scan:
    st.subheader("S&P 500 Universe Scan")
    st.caption(
        "Downloads 5 years of daily OHLCV for all 500 constituents in batches of 50, "
        "runs four strategies on each stock, and plots results as an interactive heatmap."
    )

    # ── Scan controls ──────────────────────────────────────────
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        scan_years    = st.selectbox("Period (years)", [1, 2, 3, 5], index=3, key="sc_yrs")
    with cc2:
        scan_interval = st.selectbox("Interval", ["1d", "1wk"], key="sc_int")
    with cc3:
        try:
            _, _sectors, _ = get_sp500_info()
            all_sectors = ["All Sectors"] + sorted(set(_sectors.values()))
        except Exception:
            all_sectors = ["All Sectors"]
        scan_sector = st.selectbox("Sector filter", all_sectors, key="sc_sec")
    with cc4:
        scan_metric = st.radio(
            "Colour by",
            ["Return %", "Win Rate %", "Max Drawdown %"],
            key="sc_met",
        )

    scan_btn = st.button("Start S&P 500 Scan", type="primary", key="scan_btn")

    # ── Run scan ───────────────────────────────────────────────
    if scan_btn:
        with st.status("Fetching S&P 500 constituent list...", expanded=True) as status:
            try:
                sp500_tickers, sp500_sectors, sp500_names = get_sp500_info()
            except Exception as e:
                st.error(f"Could not load S&P 500 list: {e}")
                st.stop()

            # Apply sector filter
            if scan_sector != "All Sectors":
                sp500_tickers = [t for t in sp500_tickers
                                 if sp500_sectors.get(t) == scan_sector]

            end_dt   = date.today()
            start_dt = end_dt.replace(year=end_dt.year - scan_years)
            batches  = [sp500_tickers[i:i + BATCH_SIZE]
                        for i in range(0, len(sp500_tickers), BATCH_SIZE)]

            progress_bar  = st.progress(0)
            status_holder = st.empty()
            all_rows      = []
            n_failed      = 0

            for b_idx, batch in enumerate(batches):
                pct = b_idx / len(batches)
                progress_bar.progress(pct)
                status_holder.write(
                    f"Batch {b_idx + 1}/{len(batches)}  |  "
                    f"Tickers {batch[0]} ... {batch[-1]}  |  "
                    f"{len(all_rows)} done, {n_failed} skipped"
                )

                try:
                    raw = yf.download(
                        batch,
                        start=str(start_dt), end=str(end_dt),
                        interval=scan_interval,
                        auto_adjust=True, threads=True,
                        progress=False, group_by="ticker",
                    )

                    for ticker in batch:
                        df_t = _extract_one_ticker(raw, ticker)
                        if df_t is None or len(df_t) < MIN_BARS:
                            n_failed += 1
                            continue
                        row = _execute_summary(df_t, starting_capital, risk_pct / 100, ticker)
                        if row:
                            row["sector"] = sp500_sectors.get(ticker, "Unknown")
                            row["name"]   = sp500_names.get(ticker, ticker)
                            all_rows.append(row)
                        else:
                            n_failed += 1
                except Exception:
                    n_failed += len(batch)

            progress_bar.progress(1.0)
            status_holder.empty()
            status.update(
                label=f"Scan complete — {len(all_rows)} stocks processed, {n_failed} skipped",
                state="complete",
            )

        if all_rows:
            st.session_state["sp500_summary"]  = pd.DataFrame(all_rows)
            st.session_state["sp500_scan_cfg"] = {
                "years": scan_years, "interval": scan_interval,
            }
        else:
            st.error("No results — check your network connection and try again.")

    # ── Heatmap ────────────────────────────────────────────────
    if "sp500_summary" not in st.session_state:
        st.info("Configure the scan parameters above and click **Start S&P 500 Scan**.")
    else:
        summary_df = st.session_state["sp500_summary"]

        # Metric column suffix
        metric_suffix = {
            "Return %":       "_return",
            "Win Rate %":     "_win_rate",
            "Max Drawdown %": "_max_dd",
        }[scan_metric]
        metric_label = scan_metric

        # Build heatmap matrix (ticker × strategy, value = chosen metric)
        strategy_ids = list(STRATEGY_LABELS.keys())
        hm_cols      = [f"{s}{metric_suffix}" for s in strategy_ids]
        hm_df        = summary_df.set_index("ticker")[hm_cols].copy()
        hm_df.columns = list(STRATEGY_LABELS.values())   # short display names

        # Sort by row average (best performers at top)
        hm_df["_avg"] = hm_df.mean(axis=1)
        hm_df = hm_df.sort_values("_avg", ascending=False).drop(columns="_avg")

        # Optional sector filter on display
        if scan_sector != "All Sectors":
            keep = summary_df[summary_df["sector"] == scan_sector]["ticker"].tolist()
            hm_df = hm_df[hm_df.index.isin(keep)]

        # Slider to limit visible rows (default: top 100)
        n_show = st.slider(
            "Stocks shown in heatmap (sorted by avg metric, best first)",
            min_value=25, max_value=len(hm_df), value=min(100, len(hm_df)), step=25,
            key="hm_nshow",
        )
        hm_show = hm_df.head(n_show)

        # Colour scale: diverge at 0 for return/drawdown, 50 for win rate
        zmid = 50.0 if metric_suffix == "_win_rate" else 0.0
        colorscale = "RdYlGn" if metric_suffix != "_max_dd" else "RdYlGn_r"

        fig_hm = go.Figure(go.Heatmap(
            z               = hm_show.values.tolist(),
            x               = hm_show.columns.tolist(),
            y               = hm_show.index.tolist(),
            colorscale      = colorscale,
            zmid            = zmid,
            colorbar        = dict(title=metric_label, thickness=14),
            hovertemplate   = (
                "<b>%{y}</b><br>"
                "Strategy: %{x}<br>"
                f"{metric_label}: " + "%{z:.2f}<extra></extra>"
            ),
        ))
        fig_hm.update_layout(
            title       = f"S&P 500 Universe Scan — {metric_label} (top {n_show} stocks by avg)",
            xaxis       = dict(side="top", tickfont=dict(size=11)),
            yaxis       = dict(autorange="reversed", tickfont=dict(size=9)),
            height      = max(500, n_show * 16),
            margin      = dict(l=80, r=30, t=80, b=20),
        )

        # Render heatmap; capture clicks for pre-filling the drill-down
        hm_event = st.plotly_chart(
            fig_hm, use_container_width=True,
            on_select="rerun", key="sp500_heatmap",
        )

        # ── Summary leaderboard below heatmap ──────────────────
        with st.expander("Full results table (sortable)", expanded=False):
            display_summary = summary_df.copy()
            display_summary.columns = [
                c.replace("_return", " Ret%")
                 .replace("_win_rate", " WR%")
                 .replace("_max_dd", " MDD%")
                 .replace("_n_trades", " #T")
                for c in display_summary.columns
            ]
            st.dataframe(display_summary, use_container_width=True, hide_index=True)

        # ── Drill-down ─────────────────────────────────────────
        st.divider()
        st.subheader("Individual Stock Drill-Down")

        # Pre-fill ticker from heatmap click
        clicked_ticker = None
        if (hm_event and hasattr(hm_event, "selection")
                and hm_event.selection
                and hm_event.selection.points):
            pt = hm_event.selection.points[0]
            clicked_ticker = pt.get("y") or pt.get("label")
            if clicked_ticker:
                st.info(f"Selected from heatmap: **{clicked_ticker}**")

        drill_tickers = sorted(summary_df["ticker"].tolist())
        default_idx   = (drill_tickers.index(clicked_ticker)
                         if clicked_ticker in drill_tickers else 0)

        drill_sel = st.selectbox(
            "Select stock",
            drill_tickers,
            index=default_idx,
            key="sp500_drill",
            format_func=lambda t: (
                f"{t}  —  "
                f"{sp500_names.get(t, '')}  |  "  # type: ignore[name-defined]
                f"{sp500_sectors.get(t, '')}"      # type: ignore[name-defined]
            ) if "sp500_names" in dir() else t,
        )

        if drill_sel:
            cfg = st.session_state.get("sp500_scan_cfg", {"years": 5, "interval": "1d"})
            with st.spinner(f"Loading full backtest for {drill_sel}..."):
                try:
                    drill_res = run_yfinance(
                        drill_sel, cfg["years"], cfg["interval"],
                        starting_capital, risk_pct / 100,
                    )
                    _render_single_stock_results(drill_res)
                except Exception as exc:
                    st.error(f"Could not load drill-down for {drill_sel}: {exc}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.caption("Paper trading only — no real orders are placed.")
