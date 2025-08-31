"""data_fetch.py

Module to fetch and preprocess S&P500 (^GSPC) time-series data for TimeGrad.
Handles yfinance fetching, missing data, normalization; fits data pipeline for synthetic generation.
"""

from typing import Tuple, Optional
import datetime
import pandas as pd
import torch
import yfinance as yf
from pydantic import BaseModel, ValidationError, field_validator
from gluonts.dataset.common import ListDataset

class DataConfig(BaseModel):
    """Configuration for data fetching and preprocessing."""
    symbol: str = '^GSPC'
    start_date: str
    end_date: Optional[str] = None
    interval: str = '1h'
    context_length: int = 24
    prediction_length: int = 24

    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v: str) -> str:
        if v not in ['1h', '1d']:
            raise ValueError("Interval must be '1h' or '1d'.")
        return v

def fetch_sp500_data(config: DataConfig) -> pd.DataFrame:
    """Fetch S&P500 data from yfinance.

    Args:
        config: DataConfig with symbol, dates, and interval.

    Returns:
        pd.DataFrame: OHLCV data with datetime index.

    Raises:
        ValueError: If fetch fails or data is empty.
    """
    try:
        df = yf.download(config.symbol, start=config.start_date, end=config.end_date or datetime.date.today().isoformat(),
                         interval=config.interval)
        if df.empty:
            raise ValueError("Fetched data is empty.")
        df = df.ffill().bfill()  # Handle missing data
        return df[['Close']].rename(columns={'Close': 'close'})  # Use 'close' for consistency
    except Exception as e:
        raise ValueError(f"Data fetch failed: {str(e)}")

def prepare_gluonts_dataset(df: pd.DataFrame, config: DataConfig) -> ListDataset:
    """Convert DataFrame to GluonTS dataset for TimeGrad.

    Args:
        df: DataFrame with time-series data.
        config: DataConfig with context and prediction lengths.

    Returns:
        ListDataset: GluonTS-compatible dataset.
    """
    try:
        data = [{'target': df['close'].values, 'start': df.index[0]}]
        return ListDataset(
            data,
            freq='1H' if config.interval == '1h' else 'D',
            one_dim_target=True
        )
    except Exception as e:
        raise ValueError(f"Dataset preparation failed: {str(e)}")

if __name__ == "__main__":
    try:
        config = DataConfig(start_date='2024-01-01')
        df = fetch_sp500_data(config)
        dataset = prepare_gluonts_dataset(df, config)
        print(f"Data shape: {df.shape}, Dataset: {len(dataset)} series")
    except ValidationError as ve:
        print(f"Config error: {ve}")
    except ValueError as ve:
        print(f"Processing error: {ve}")