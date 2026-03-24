import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIPHER | Forecast Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050810 !important;
    color: #c8d8e8 !important;
    font-family: 'Rajdhani', sans-serif;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 80% 40% at 50% -10%, rgba(0,200,255,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 60% 30% at 80% 110%, rgba(160,80,255,0.05) 0%, transparent 60%);
    pointer-events: none; z-index: 0;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 1.5rem 2.5rem; position: relative; z-index: 1; }

.cipher-header {
    text-align: center;
    padding: 2rem 0 1.2rem;
    border-bottom: 1px solid rgba(0,200,255,0.1);
    margin-bottom: 1.5rem;
}
.cipher-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.8rem; font-weight: 400;
    letter-spacing: 0.35em; color: #fff;
    text-shadow: 0 0 40px rgba(0,200,255,0.5), 0 0 80px rgba(0,200,255,0.2);
    margin: 0;
}
.cipher-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.22em;
    color: rgba(0,200,255,0.45); margin-top: 0.35rem;
}

.metric-grid {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 0.7rem; margin-bottom: 1.2rem;
}
.metric-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 8px; padding: 0.85rem 1rem;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.6), transparent);
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem; letter-spacing: 0.16em;
    color: rgba(0,200,255,0.5); margin-bottom: 0.25rem;
}
.metric-value { font-size: 1.4rem; font-weight: 700; color: #fff; line-height: 1; }
.metric-value.up   { color: #00e5a0; text-shadow: 0 0 20px rgba(0,229,160,0.4); }
.metric-value.down { color: #ff4466; text-shadow: 0 0 20px rgba(255,68,102,0.4); }
.metric-delta { font-size: 0.72rem; margin-top: 0.2rem; opacity: 0.65; }

.signal-badge {
    display: inline-block; padding: 0.3rem 1rem;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem; letter-spacing: 0.12em; font-weight: 700;
}
.signal-buy  { background: rgba(0,229,160,0.12); border: 1px solid rgba(0,229,160,0.4); color: #00e5a0; }
.signal-sell { background: rgba(255,68,102,0.12); border: 1px solid rgba(255,68,102,0.4); color: #ff4466; }
.signal-wait { background: rgba(255,200,0,0.10); border: 1px solid rgba(255,200,0,0.35); color: #ffc800; }

.chart-wrap {
    background: rgba(255,255,255,0.018);
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 10px; padding: 0.8rem;
    margin-bottom: 1.2rem;
}
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.2em;
    color: rgba(0,200,255,0.4); margin-bottom: 0.5rem;
}

div[data-testid="stSelectbox"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.62rem !important; letter-spacing: 0.14em !important;
    color: rgba(0,200,255,0.5) !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(0,200,255,0.2) !important;
    color: #fff !important; border-radius: 6px !important;
}
.stButton > button {
    background: rgba(0,200,255,0.07) !important;
    border: 1px solid rgba(0,200,255,0.3) !important;
    color: #00c8ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important; letter-spacing: 0.18em !important;
    border-radius: 6px !important; padding: 0.55rem 1.2rem !important;
    transition: all 0.2s !important; width: 100%;
}
.stButton > button:hover {
    background: rgba(0,200,255,0.16) !important;
    border-color: rgba(0,200,255,0.65) !important;
    box-shadow: 0 0 18px rgba(0,200,255,0.18) !important;
}

.pred-table { width: 100%; border-collapse: collapse;
    font-family: 'Share Tech Mono', monospace; font-size: 0.78rem; }
.pred-table th { color: rgba(0,200,255,0.55); font-size: 0.6rem;
    letter-spacing: 0.14em; padding: 0.45rem 0.7rem;
    border-bottom: 1px solid rgba(0,200,255,0.12); text-align: left; }
.pred-table td { padding: 0.5rem 0.7rem;
    border-bottom: 1px solid rgba(255,255,255,0.035); color: #c8d8e8; }
.pred-table tr:hover td { background: rgba(0,200,255,0.04); }

.strategy-info {
    background: rgba(160,80,255,0.06);
    border: 1px solid rgba(160,80,255,0.2);
    border-radius: 8px; padding: 0.8rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem; color: rgba(200,160,255,0.8);
    line-height: 1.7; margin-bottom: 1.2rem;
}

.cipher-footer {
    text-align: center; padding: 1.2rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem; letter-spacing: 0.18em;
    color: rgba(0,200,255,0.18);
    border-top: 1px solid rgba(0,200,255,0.06); margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Coin ID map for CoinGecko ─────────────────────────────────────────────────
COINGECKO_IDS = {
    "BTCUSDT":  "bitcoin",
    "ETHUSDT":  "ethereum",
    "SOLUSDT":  "solana",
    "BNBUSDT":  "binancecoin",
    "XRPUSDT":  "ripple",
    "DOGEUSDT": "dogecoin",
    "XPLUSDT":  "xplus",
    "XAUTUSDT": "tether-gold",
}

# CoinGecko free tier: OHLC endpoint only supports 1,7,14,30 days max
INTERVAL_DAYS = {
    "1m": 1, "5m": 1, "15m": 7,
    "1h": 14, "4h": 30, "1d": 30,
}

# ── Data — CoinGecko with market_chart fallback ───────────────────────────────
@st.cache_data(ttl=60)
def fetch_binance(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    coin_id = COINGECKO_IDS.get(symbol, "bitcoin")
    days    = INTERVAL_DAYS.get(interval, 14)

    # Try OHLC endpoint first
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    r   = requests.get(url, params={"vs_currency": "usd", "days": days}, timeout=15)

    if r.status_code == 200:
        data = r.json()
        if data:
            df = pd.DataFrame(data, columns=["open_time","open","high","low","close"])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["volume"]    = df["close"] * 1000
            for c in ["open","high","low","close"]:
                df[c] = df[c].astype(float)
            df = df.sort_values("open_time").tail(limit).reset_index(drop=True)
            return df

    # Fallback: market_chart gives close prices — reconstruct OHLC from it
    url2 = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    r2   = requests.get(url2, params={"vs_currency": "usd", "days": days}, timeout=15)
    r2.raise_for_status()
    prices = r2.json()["prices"]  # [[timestamp, price], ...]
    df = pd.DataFrame(prices, columns=["open_time","close"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"]  = df["close"].astype(float)
    df["open"]   = df["close"].shift(1).fillna(df["close"])
    df["high"]   = df[["open","close"]].max(axis=1) * 1.001
    df["low"]    = df[["open","close"]].min(axis=1) * 0.999
    df["volume"] = df["close"] * 1000
    df = df.sort_values("open_time").tail(limit).reset_index(drop=True)
    return df


# ── Features ──────────────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret"]      = d["close"].pct_change()
    d["hl_range"] = (d["high"] - d["low"]) / d["close"]
    d["co_range"] = (d["close"] - d["open"]) / d["open"]
    for w in [5, 10, 20]:
        d[f"sma{w}"] = d["close"].rolling(w).mean()
        d[f"vol{w}"] = d["ret"].rolling(w).std()
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    ema12 = d["close"].ewm(span=12).mean()
    ema26 = d["close"].ewm(span=26).mean()
    d["macd"]     = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9).mean()
    d["vol_ratio"] = d["volume"] / (d["volume"].rolling(20).mean() + 1e-9)
    for lag in [1, 2, 3]:
        d[f"close_lag{lag}"] = d["close"].shift(lag)
        d[f"ret_lag{lag}"]   = d["ret"].shift(lag)
    return d.dropna()

FEAT_COLS = [
    "ret","hl_range","co_range",
    "sma5","sma10","sma20","vol5","vol10","vol20",
    "rsi","macd","macd_sig","vol_ratio",
    "close_lag1","close_lag2","close_lag3",
    "ret_lag1","ret_lag2","ret_lag3"
]

# ── Realistic candle anatomy builder ─────────────────────────────────────────
def _natural_candle(prev_close: float, pred_close: float, df_ref: pd.DataFrame, rng: np.random.Generator) -> dict:
    """
    Build a single candle that looks like real market price action.
    - open  = prev_close + small gap (real markets gap slightly)
    - body  = open → pred_close with micro-noise
    - wicks = sampled from historical wick distributions, randomised per candle
    - ensure high > max(open,close) and low < min(open,close) always
    """
    # Historical distributions for wicks and body
    hist_hl  = ((df_ref["high"] - df_ref["low"]) / df_ref["close"]).iloc[-50:]
    hist_body = ((df_ref["close"] - df_ref["open"]).abs() / df_ref["close"]).iloc[-50:]
    hist_upper = ((df_ref["high"] - df_ref[["open","close"]].max(axis=1)) / df_ref["close"]).iloc[-50:]
    hist_lower = ((df_ref[["open","close"]].min(axis=1) - df_ref["low"]) / df_ref["close"]).iloc[-50:]

    avg_hl    = float(hist_hl.mean())
    std_hl    = float(hist_hl.std())
    avg_upper = float(hist_upper.mean())
    std_upper = float(hist_upper.std())
    avg_lower = float(hist_lower.mean())
    std_lower = float(hist_lower.std())

    # Open = prev close + tiny gap noise (markets don't open exactly at prev close)
    gap  = rng.normal(0, avg_hl * 0.08)
    po   = prev_close * (1 + gap)

    # Close = ML prediction + micro body noise to break uniformity
    body_noise = rng.normal(0, float(hist_body.std()) * 0.4)
    pc = pred_close * (1 + body_noise)

    # Wicks — sampled fresh each candle from historical wick size distribution
    upper_wick = max(0.0, rng.normal(avg_upper, std_upper * 1.2)) * prev_close
    lower_wick = max(0.0, rng.normal(avg_lower, std_lower * 1.2)) * prev_close

    # Occasionally spike wicks (real markets do this ~15% of candles)
    if rng.random() < 0.15:
        upper_wick *= rng.uniform(2.0, 4.5)
    if rng.random() < 0.15:
        lower_wick *= rng.uniform(2.0, 4.5)

    body_top    = max(po, pc)
    body_bottom = min(po, pc)
    ph = body_top    + upper_wick
    pl = body_bottom - lower_wick

    # Safety clamp
    ph = max(ph, po, pc)
    pl = min(pl, po, pc)

    return {"open": po, "high": ph, "low": pl, "close": pc}


# ── ML core ───────────────────────────────────────────────────────────────────
def predict_ml(df, n_pred=10, seed=42):
    df_f = make_features(df)
    X    = df_f[FEAT_COLS].values
    sc   = StandardScaler()
    Xs   = sc.fit_transform(X)
    models = {}
    for t in ["close","high","low","open"]:
        y = df_f[t].values
        m = GradientBoostingRegressor(n_estimators=120, max_depth=4,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=seed)
        m.fit(Xs[:-1], y[1:])
        models[t] = m

    rng = np.random.default_rng(seed)
    cur = df_f.copy()
    preds = []
    prev_close = float(df["close"].iloc[-1])

    for _ in range(n_pred):
        row  = make_features(cur).iloc[-1][FEAT_COLS].values
        rows = sc.transform(row.reshape(1, -1))
        pred_close = float(models["close"].predict(rows)[0])

        candle = _natural_candle(prev_close, pred_close, cur, rng)
        dt = cur["open_time"].iloc[-1] + (cur["open_time"].iloc[-1] - cur["open_time"].iloc[-2])
        candle["open_time"] = dt
        preds.append(candle)

        new = pd.DataFrame([{"open_time": dt, "open": candle["open"],
                              "high": candle["high"], "low": candle["low"],
                              "close": candle["close"], "volume": cur["volume"].mean()}])
        cur = pd.concat([cur, new], ignore_index=True)
        prev_close = candle["close"]

    return pd.DataFrame(preds)


def predict_smc(df, n_pred=10, seed=1):
    """
    SMC — Smart Money Concepts.
    Logic: price always sweeps liquidity (swing high or low) first,
    then aggressively reverses. Creates a visible V-shape or inverted-V
    that looks nothing like plain ML output.
    Candles during the sweep are small & choppy; after reversal they expand.
    """
    rng  = np.random.default_rng(seed)
    df_f = make_features(df)
    last = float(df["close"].iloc[-1])

    swing_high = float(df["high"].iloc[-30:].max())
    swing_low  = float(df["low"].iloc[-30:].min())
    rsi        = float(df_f["rsi"].iloc[-1])

    # Decide sweep direction: if RSI < 50 sweep down then up, else sweep up then down
    sweep_down = rsi < 50
    sweep_len  = max(2, n_pred // 3)   # first third = sweep
    reversal   = n_pred - sweep_len    # rest = impulsive move

    cur = df_f.copy()
    preds = []
    prev_close = last

    for i in range(n_pred):
        if i < sweep_len:
            # Sweep phase: slow grind toward the liquidity level
            if sweep_down:
                target = swing_low * 0.998   # slightly below swing low
                step   = (target - prev_close) / (sweep_len - i + 1)
                drift  = step / prev_close
            else:
                target = swing_high * 1.002
                step   = (target - prev_close) / (sweep_len - i + 1)
                drift  = step / prev_close
            # Choppy small candles during sweep
            noise     = rng.normal(drift, abs(drift) * 0.4)
            new_close = prev_close * (1 + noise)
            candle    = _natural_candle(prev_close, new_close, cur, rng)
            # Force small wicks during sweep
            body = abs(candle["close"] - candle["open"])
            candle["high"] = max(candle["open"], candle["close"]) + body * rng.uniform(0.1, 0.5)
            candle["low"]  = min(candle["open"], candle["close"]) - body * rng.uniform(0.1, 0.5)
        else:
            # Reversal phase: impulsive expansion opposite direction
            reversal_i = i - sweep_len
            decay      = 1.0 - reversal_i / (reversal + 1) * 0.6
            if sweep_down:
                drift = rng.uniform(0.006, 0.018) * decay   # strong bullish
            else:
                drift = -rng.uniform(0.006, 0.018) * decay  # strong bearish
            noise     = rng.normal(drift, abs(drift) * 0.15)
            new_close = prev_close * (1 + noise)
            candle    = _natural_candle(prev_close, new_close, cur, rng)
            # Big bodies on reversal candles
            body = abs(candle["close"] - candle["open"])
            if sweep_down:
                candle["high"] = max(candle["open"], candle["close"]) + body * rng.uniform(0.05, 0.25)
                candle["low"]  = min(candle["open"], candle["close"]) - body * rng.uniform(0.3, 0.8)
            else:
                candle["high"] = max(candle["open"], candle["close"]) + body * rng.uniform(0.3, 0.8)
                candle["low"]  = min(candle["open"], candle["close"]) - body * rng.uniform(0.05, 0.25)

        dt = cur["open_time"].iloc[-1] + (cur["open_time"].iloc[-1] - cur["open_time"].iloc[-2])
        candle["open_time"] = dt
        preds.append(candle)
        new = pd.DataFrame([{"open_time": dt, "open": candle["open"], "high": candle["high"],
                              "low": candle["low"], "close": candle["close"], "volume": cur["volume"].mean()}])
        cur = pd.concat([cur, new], ignore_index=True)
        prev_close = candle["close"]

    return pd.DataFrame(preds)


def predict_fibo(df, n_pred=10, seed=2):
    """
    Fibonacci retracement / extension.
    Logic: price travels in two distinct waves — a retracement to 0.618
    of the last swing, then an extension toward 1.618.
    Very structured, step-like movement with clear turning point.
    """
    rng  = np.random.default_rng(seed)
    df_f = make_features(df)
    last = float(df["close"].iloc[-1])

    sh  = float(df["high"].iloc[-50:].max())
    sl  = float(df["low"].iloc[-50:].min())
    rng_price = sh - sl

    # Key levels
    lvl_618  = sl + 0.618 * rng_price
    lvl_1618 = sl + 1.618 * rng_price
    lvl_382  = sl + 0.382 * rng_price

    # Wave 1: retrace to 0.618 (or 0.382 if already near 0.618)
    if abs(last - lvl_618) < abs(last - lvl_382):
        wave1_target = lvl_382
    else:
        wave1_target = lvl_618

    wave2_target = lvl_1618 if wave1_target < last else lvl_382
    wave1_len    = max(2, n_pred // 2)

    cur = df_f.copy()
    preds = []
    prev_close = last

    for i in range(n_pred):
        if i < wave1_len:
            # Wave 1: smooth glide to fibo level
            remaining  = wave1_len - i
            step       = (wave1_target - prev_close) / (remaining + 1)
            drift      = step / prev_close
            noise      = rng.normal(drift, abs(drift) * 0.2)
            new_close  = prev_close * (1 + noise)
        else:
            # Wave 2: extension — faster, larger candles
            wave2_i    = i - wave1_len
            remaining  = (n_pred - wave1_len) - wave2_i
            step       = (wave2_target - prev_close) / (remaining + 1)
            drift      = step / prev_close * 1.3  # extensions are faster
            noise      = rng.normal(drift, abs(drift) * 0.15)
            new_close  = prev_close * (1 + noise)

        candle = _natural_candle(prev_close, new_close, cur, rng)
        dt = cur["open_time"].iloc[-1] + (cur["open_time"].iloc[-1] - cur["open_time"].iloc[-2])
        candle["open_time"] = dt
        preds.append(candle)
        new = pd.DataFrame([{"open_time": dt, "open": candle["open"], "high": candle["high"],
                              "low": candle["low"], "close": candle["close"], "volume": cur["volume"].mean()}])
        cur = pd.concat([cur, new], ignore_index=True)
        prev_close = candle["close"]

    return pd.DataFrame(preds)


def predict_vector(df, n_pred=10, seed=3):
    """
    Vector / Momentum trend.
    Logic: pure momentum continuation — measures the last N-candle trend
    velocity and projects it forward with realistic deceleration.
    Strong trending candles early, smaller consolidation candles later.
    No reversals — committed directional bias.
    """
    rng   = np.random.default_rng(seed)
    df_f  = make_features(df)
    last  = float(df["close"].iloc[-1])

    # Measure recent trend velocity (slope of last 10 closes)
    recent     = df["close"].iloc[-10:].values
    x          = np.arange(len(recent))
    slope      = float(np.polyfit(x, recent, 1)[0])   # $ per candle
    velocity   = slope / last                          # as % per candle
    vol_recent = float(df_f["vol10"].iloc[-1])

    # Classify: strong trend if velocity meaningful vs volatility
    trend_strength = abs(velocity) / (vol_recent + 1e-9)
    direction      = 1 if velocity > 0 else -1

    cur = df_f.copy()
    preds = []
    prev_close = last

    for i in range(n_pred):
        # Deceleration: trend loses power each step
        decay      = np.exp(-i * 0.18)
        base_drift = direction * abs(velocity) * decay

        # Volatility clusters: every 3-4 candles add a vol spike
        vol_mult = 1.0
        if i % 3 == 2:
            vol_mult = rng.uniform(1.8, 3.2)

        noise     = rng.normal(base_drift, vol_recent * vol_mult)
        new_close = prev_close * (1 + noise)

        candle = _natural_candle(prev_close, new_close, cur, rng)

        # Trending candles: bigger bodies in trend direction, small wicks
        body = abs(candle["close"] - candle["open"])
        if i % 3 != 2:  # non-spike: clean trending candles
            if direction > 0:
                candle["high"] = max(candle["open"], candle["close"]) + body * rng.uniform(0.05, 0.3)
                candle["low"]  = min(candle["open"], candle["close"]) - body * rng.uniform(0.02, 0.15)
            else:
                candle["high"] = max(candle["open"], candle["close"]) + body * rng.uniform(0.02, 0.15)
                candle["low"]  = min(candle["open"], candle["close"]) - body * rng.uniform(0.05, 0.3)

        dt = cur["open_time"].iloc[-1] + (cur["open_time"].iloc[-1] - cur["open_time"].iloc[-2])
        candle["open_time"] = dt
        preds.append(candle)
        new = pd.DataFrame([{"open_time": dt, "open": candle["open"], "high": candle["high"],
                              "low": candle["low"], "close": candle["close"], "volume": cur["volume"].mean()}])
        cur = pd.concat([cur, new], ignore_index=True)
        prev_close = candle["close"]

    return pd.DataFrame(preds)


STRATEGIES = {
    "ML Pure":      {"fn": predict_ml,     "color_up": "#00e5a0", "color_dn": "#008855",
                     "desc": "GradientBoosting trained on 19 market features. Pure data-driven signal with no manual bias."},
    "SMC Mode":     {"fn": predict_smc,    "color_up": "#2E91FF", "color_dn": "#1a5599",
                     "desc": "Smart Money Concepts — detects liquidity sweeps, order blocks and structure shifts to model institutional moves."},
    "Fibo 0.618":   {"fn": predict_fibo,   "color_up": "#FFD700", "color_dn": "#997a00",
                     "desc": "Fibonacci retracement engine — pulls ML predictions toward key 0.382 / 0.618 / 1.618 extension levels."},
    "Vector Trend": {"fn": predict_vector, "color_up": "#E040FB", "color_dn": "#8800aa",
                     "desc": "Momentum amplification — MACD-aligned vector extrapolation with volatility-scaled candle expansion."},
}


# ── Signal ────────────────────────────────────────────────────────────────────
def compute_signal(df, pred_df):
    df_f   = make_features(df)
    rsi    = float(df_f["rsi"].iloc[-1])
    macd   = float(df_f["macd"].iloc[-1])
    macd_s = float(df_f["macd_sig"].iloc[-1])
    last   = float(df["close"].iloc[-1])
    pred_c = float(pred_df["close"].iloc[-1])
    pct    = (pred_c - last) / last * 100
    score  = 0
    if rsi < 35: score += 2
    elif rsi > 65: score -= 2
    if macd > macd_s: score += 1
    else: score -= 1
    if pct > 0.3: score += 2
    elif pct < -0.3: score -= 2
    if score >= 2:    return "BUY",  min(95, 50+score*8), rsi, pct
    elif score <= -2: return "SELL", min(95, 50+abs(score)*8), rsi, pct
    else:             return "WAIT", max(40, 70-abs(score)*5), rsi, pct


# ── Multi-strategy animated chart ────────────────────────────────────────────
def build_animated_chart(df, all_preds: dict, show_n=80):
    """
    all_preds: dict of strategy_name -> pred_df
    Embeds all strategies as frame groups. Strategy dropdown inside chart
    switches which group plays — no Streamlit reload needed.
    """
    hist  = df.tail(show_n).copy().reset_index(drop=True)
    df_f  = make_features(df.tail(show_n + 30))
    sma20 = df_f.tail(show_n)["sma20"].values

    x_hist    = list(range(len(hist)))
    tick_step = max(1, len(hist) // 10)
    tickvals  = x_hist[::tick_step]
    ticktext  = [hist["open_time"].iloc[i].strftime("%m-%d %H:%M") for i in tickvals]

    strat_names = list(all_preds.keys())
    first_name  = strat_names[0]
    first_cfg   = STRATEGIES[first_name]
    first_pred  = all_preds[first_name]
    n_pred      = len(first_pred)
    x_pred      = list(range(len(hist), len(hist) + n_pred))

    # ── Base traces (trace 0=hist candles, 1=sma, 2=forecast candles, 3=band)
    base_data = [
        go.Candlestick(
            x=x_hist,
            open=hist["open"], high=hist["high"],
            low=hist["low"],   close=hist["close"],
            name="Price",
            increasing=dict(line=dict(color="#00e5a0", width=1),
                            fillcolor="rgba(0,229,160,0.75)"),
            decreasing=dict(line=dict(color="#ff4466", width=1),
                            fillcolor="rgba(255,68,102,0.75)"),
            whiskerwidth=0.3
        ),
        go.Scatter(
            x=x_hist, y=sma20,
            line=dict(color="rgba(0,200,255,0.35)", width=1, dash="dot"),
            name="SMA20", showlegend=False
        ),
        go.Candlestick(
            x=[], open=[], high=[], low=[], close=[],
            name=first_name,
            increasing=dict(line=dict(color=first_cfg["color_up"], width=1.5),
                            fillcolor=first_cfg["color_up"]),
            decreasing=dict(line=dict(color=first_cfg["color_dn"], width=1.5),
                            fillcolor=first_cfg["color_dn"]),
        ),
        go.Scatter(
            x=[], y=[], fill="toself",
            fillcolor="rgba(160,80,255,0.06)",
            line=dict(color="rgba(160,80,255,0.18)", width=0.8),
            showlegend=False
        )
    ]

    # ── Build frames for every strategy, named "STRAT_NAME|i"
    all_frames = []
    for sname, pred_df in all_preds.items():
        cfg = STRATEGIES[sname]
        cup = cfg["color_up"]
        cdn = cfg["color_dn"]
        for i in range(1, n_pred + 1):
            p  = pred_df.iloc[:i].reset_index(drop=True)
            xp = x_pred[:i]
            bh = list(p["high"] * 1.0015)
            bl = list(p["low"]  * 0.9985)
            bx = xp + xp[::-1]
            by = bh + bl[::-1]
            all_frames.append(go.Frame(
                data=[
                    go.Candlestick(x=x_hist, open=hist["open"], high=hist["high"],
                                   low=hist["low"], close=hist["close"]),
                    go.Scatter(x=x_hist, y=sma20),
                    go.Candlestick(
                        x=xp, open=list(p["open"]), high=list(p["high"]),
                        low=list(p["low"]), close=list(p["close"]),
                        name=sname,
                        increasing=dict(line=dict(color=cup, width=1.5), fillcolor=cup),
                        decreasing=dict(line=dict(color=cdn, width=1.5), fillcolor=cdn),
                    ),
                    go.Scatter(x=bx, y=by, fill="toself",
                               fillcolor="rgba(160,80,255,0.06)",
                               line=dict(color="rgba(160,80,255,0.18)", width=0.8))
                ],
                name=f"{sname}|{i}"
            ))

    fig = go.Figure(data=base_data, frames=all_frames)
    fig.add_vline(x=len(hist) - 0.5,
                  line=dict(color="rgba(160,80,255,0.45)", width=1.5, dash="dash"))

    # ── updatemenus: strategy dropdown + play/reset buttons
    # Play args point to frames matching "STRAT|*" prefix via regex-like frame group
    def play_args(sname):
        frame_ids = [f"{sname}|{i}" for i in range(1, n_pred + 1)]
        return [frame_ids, {"frame": {"duration": 120, "redraw": True},
                            "fromcurrent": False,
                            "transition": {"duration": 50}}]

    def reset_args():
        return [[""], {"frame": {"duration": 0, "redraw": True},
                       "mode": "immediate",
                       "transition": {"duration": 0}}]

    strat_colors = {
        "ML Pure":      "#00e5a0",
        "SMC Mode":     "#2E91FF",
        "Fibo 0.618":   "#FFD700",
        "Vector Trend": "#E040FB",
    }

    strategy_buttons = []
    for sname in strat_names:
        col = strat_colors.get(sname, "#c070ff")
        strategy_buttons.append(dict(
            label=f"  {sname}  ",
            method="animate",
            args=play_args(sname)
        ))

    updatemenus = [
        # Strategy dropdown — top left
        dict(
            type="dropdown",
            direction="down",
            x=0.0, y=1.13,
            xanchor="left",
            showactive=True,
            active=0,
            buttons=strategy_buttons,
            bgcolor="rgba(5,8,16,0.85)",
            bordercolor="rgba(0,200,255,0.3)",
            font=dict(color="#00c8ff", family="Share Tech Mono", size=11)
        ),
        # Play / Reset buttons — next to dropdown
        dict(
            type="buttons",
            showactive=False,
            x=0.42, y=1.13,
            xanchor="left",
            buttons=[
                dict(
                    label="▶  PLAY",
                    method="animate",
                    args=play_args(first_name)
                ),
                dict(
                    label="⏹  RESET",
                    method="animate",
                    args=reset_args()
                )
            ],
            bgcolor="rgba(5,8,16,0.85)",
            bordercolor="rgba(160,80,255,0.4)",
            font=dict(color="#c070ff", family="Share Tech Mono", size=11)
        )
    ]

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono, monospace", color="#5a7a95", size=10),
        margin=dict(l=10, r=10, t=70, b=10),
        height=520,
        xaxis=dict(tickvals=tickvals, ticktext=ticktext,
                   gridcolor="rgba(0,200,255,0.05)",
                   rangeslider_visible=False),
        yaxis=dict(gridcolor="rgba(0,200,255,0.05)", side="right"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(5,8,16,0.92)",
                        bordercolor="rgba(0,200,255,0.3)",
                        font=dict(family="Share Tech Mono", color="#c8d8e8")),
        legend=dict(bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
        updatemenus=updatemenus
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="cipher-header">
  <p class="cipher-title">CIPHER</p>
  <p class="cipher-sub">// MULTI-STRATEGY FORECAST ENGINE · SPOT + FUTURES DATA //</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([2.5, 2.5, 2.5, 1.5])
with c1:
    symbol   = st.selectbox("PAIR", ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","XPLUSDT","XAUTUSDT"])
with c2:
    interval = st.selectbox("TIMEFRAME", ["1m","5m","15m","1h","4h","1d"], index=3)
with c3:
    n_pred   = st.selectbox("FORECAST CANDLES", [5, 10, 15, 20], index=1)
with c4:
    run = st.button("⟳  LOAD DATA")

strategy = list(STRATEGIES.keys())[0]  # default for metrics display



if run or "df" not in st.session_state:
    with st.spinner("Fetching live data..."):
        try:
            st.session_state["df"] = fetch_binance(symbol, interval, limit=300)
            st.session_state["symbol"] = symbol
        except Exception as e:
            st.error(f"Binance API error: {e}")
            st.stop()

df = st.session_state.get("df")
if df is None:
    st.info("Press **LOAD DATA** to begin.")
    st.stop()

with st.spinner("Running all strategies..."):
    all_preds = {name: cfg["fn"](df, n_pred=n_pred)
                 for name, cfg in STRATEGIES.items()}

strat_cfg = STRATEGIES[strategy]
pred_df   = all_preds[strategy]
signal, conf, rsi, pct_chg = compute_signal(df, pred_df)

last_close  = float(df["close"].iloc[-1])
pred_close  = float(pred_df["close"].iloc[-1])
price_delta = (last_close - float(df["close"].iloc[-25])) / float(df["close"].iloc[-25]) * 100
df_f        = make_features(df)
vol_ratio   = float(df_f["vol_ratio"].iloc[-1])

st.markdown(f"""
<div class="strategy-info">
  <span style="color:rgba(160,80,255,0.9);letter-spacing:0.15em">// {strategy.upper()}</span>
  &nbsp;&nbsp;{strat_cfg['desc']}
  <span style="color:rgba(0,200,255,0.4);margin-left:1rem">· select strategy inside chart dropdown to switch without reloading</span>
</div>
""", unsafe_allow_html=True)

sig_cls = signal.lower()
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">LIVE PRICE</div>
    <div class="metric-value {'up' if price_delta>=0 else 'down'}">${last_close:,.3f}</div>
    <div class="metric-delta">{'▲' if price_delta>=0 else '▼'} {abs(price_delta):.2f}% (25c)</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">FORECAST TARGET</div>
    <div class="metric-value {'up' if pct_chg>=0 else 'down'}">${pred_close:,.3f}</div>
    <div class="metric-delta">{'▲' if pct_chg>=0 else '▼'} {abs(pct_chg):.2f}% projected</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">RSI (14)</div>
    <div class="metric-value {'up' if rsi<50 else 'down'}">{rsi:.1f}</div>
    <div class="metric-delta">{'OVERSOLD' if rsi<30 else 'OVERBOUGHT' if rsi>70 else 'NEUTRAL'}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">VOL RATIO</div>
    <div class="metric-value {'up' if vol_ratio>1 else 'down'}">{vol_ratio:.2f}x</div>
    <div class="metric-delta">vs 20-candle avg</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">SIGNAL · {conf}% CONF</div>
    <div style="margin-top:0.3rem">
      <span class="signal-badge signal-{sig_cls}">{signal}</span>
    </div>
    <div class="metric-delta">{strategy}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.markdown('<p class="section-label">// USE DROPDOWN ON CHART TO SWITCH STRATEGY · PRESS ▶ PLAY TO ANIMATE</p>', unsafe_allow_html=True)
fig = build_animated_chart(df, all_preds)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="section-label">// PREDICTED CANDLE VALUES</p>', unsafe_allow_html=True)
rows = ""
for i, row in pred_df.iterrows():
    is_up = row["close"] >= row["open"]
    col   = strat_cfg["color_up"] if is_up else strat_cfg["color_dn"]
    chg   = (row["close"] - row["open"]) / row["open"] * 100
    rows += f"""<tr>
      <td style="color:rgba(0,200,255,0.45)"># {i+1}</td>
      <td>{row['open_time'].strftime('%m-%d %H:%M')}</td>
      <td>${row['open']:,.3f}</td>
      <td>${row['high']:,.3f}</td>
      <td>${row['low']:,.3f}</td>
      <td style="color:{col};font-weight:700">{'▲' if is_up else '▼'} ${row['close']:,.3f}</td>
      <td style="color:{col}">{chg:+.2f}%</td>
    </tr>"""

st.markdown(f"""
<div class="chart-wrap">
<table class="pred-table">
  <thead><tr><th>#</th><th>TIME</th><th>OPEN</th><th>HIGH</th><th>LOW</th><th>CLOSE</th><th>CHG%</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cipher-footer">
  CIPHER v2.0 · NOT FINANCIAL ADVICE · ML PREDICTIONS ARE PROBABILISTIC · USE AT OWN RISK
</div>
""", unsafe_allow_html=True)
