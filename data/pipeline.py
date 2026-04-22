# ─────────────────────────────────────────────
# data/pipeline.py  —  Stages 1, 2, 3
#   Stage 1: download OHLCV data via yfinance
#   Stage 2: engineer features (log returns, volatility, rolling z-score)
#   Stage 3: split into train / val / test sets
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    DOWNLOAD_START, DOWNLOAD_END,
    TRAIN_START, TRAIN_END,
    VAL_START,   VAL_END,
    TEST_START,  TEST_END,
    RETURN_WINDOWS, VOLATILITY_WINDOW, NORM_WINDOW,
)


# ── Stage 1: Data Collection ─────────────────────────────────────────────────

def download_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Download daily adjusted close prices for the given tickers.

    We pull from DOWNLOAD_START (2009) so the 60-day and 252-day rolling
    windows produce valid values by TRAIN_START (2010).

    Returns: DataFrame, shape (dates, tickers), no NaN rows.
    """
    df = yf.download(
        tickers,
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        auto_adjust=True,   # adjusts for splits and dividends
        progress=False,
    )["Close"]

    # Keep only the requested tickers in consistent order
    df = df[tickers]

    # Drop any date where any ticker is missing (holidays, data gaps)
    df = df.dropna()

    print(f"Downloaded {len(df)} trading days for {len(tickers)} assets.")
    return df


# ── Stage 2: Feature Engineering ─────────────────────────────────────────────

def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each asset compute:
      - Log returns at 1, 5, 20, 60 day horizons
      - Rolling 20-day volatility (std of daily log returns)

    Returns: DataFrame with MultiIndex columns (ticker, feature_name).
    NaN rows from the initial lookback period are dropped.
    """
    asset_frames = {}

    for ticker in prices.columns:
        p = prices[ticker]
        f = pd.DataFrame(index=prices.index)

        # Log returns: log(P_t / P_{t-n})  — stationary, scale-invariant across assets
        for w in RETURN_WINDOWS:
            f[f"log_return_{w}d"] = np.log(p / p.shift(w))

        # Rolling volatility of daily returns — captures recent riskiness of each asset
        daily_ret = np.log(p / p.shift(1))
        f["volatility_20d"] = daily_ret.rolling(VOLATILITY_WINDOW).std()

        asset_frames[ticker] = f

    # Combine into MultiIndex columns: (ticker, feature)
    feature_df = pd.concat(asset_frames, axis=1)
    feature_df = feature_df.dropna()   # first ~60 rows will be NaN

    return feature_df


def normalize_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a rolling z-score (window = NORM_WINDOW = 252 days) to every column.

    Rolling — not global — normalization is critical: using a global mean/std
    would leak future information into the training set (lookahead bias).

    Returns: normalized DataFrame, NaN rows from the initial window are dropped.
    """
    normed = pd.DataFrame(index=feature_df.index, columns=feature_df.columns, dtype=float)

    for col in feature_df.columns:
        mu  = feature_df[col].rolling(NORM_WINDOW).mean()
        sig = feature_df[col].rolling(NORM_WINDOW).std()
        normed[col] = (feature_df[col] - mu) / (sig + 1e-8)

    normed = normed.dropna()
    print(f"Features ready: {len(normed)} dates × {len(normed.columns)} columns.")
    return normed


def build_features(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: download → engineer → normalize.
    Returns (normalized_features, raw_prices) aligned on the same date index.
    """
    prices  = download_prices(tickers)
    raw_features = compute_features(prices)
    features = normalize_features(raw_features)

    # Align prices to the same dates as features (features lose early rows)
    prices = prices.loc[features.index]

    return features, prices


# ── Stage 3: Train / Val / Test Split ────────────────────────────────────────

def split_data(
    features: pd.DataFrame,
    prices:   pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Slice features and prices into train, val, and test windows.

    Returns a dict with keys:
      train_features, train_prices,
      val_features,   val_prices,
      test_features,  test_prices
    """
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val":   (VAL_START,   VAL_END),
        "test":  (TEST_START,  TEST_END),
    }

    result = {}
    for name, (start, end) in splits.items():
        result[f"{name}_features"] = features[start:end]
        result[f"{name}_prices"]   = prices[start:end]
        print(f"  {name:5s}: {len(result[f'{name}_features'])} days")

    return result


# ── Entry point (smoke test) ──────────────────────────────────────────────────

if __name__ == "__main__":
    from config import AGENT1_TICKERS

    features, prices = build_features(AGENT1_TICKERS)
    splits = split_data(features, prices)

    print("\nSample feature row:")
    print(features.iloc[-1])
