#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
import traceback
from datetime import datetime, timezone, timedelta

# 使用app的日志
import logging
logger = logging.getLogger(__name__) # 获取logger

# 导入app的K线工具，用于间隔转换（虽然指标计算本身不直接用，但依赖的 calculate_vwap 用到）
# from app.utils.kline_utils import interval_to_timedelta # 实际上 calculate_vwap 是内部函数，不需要这里导入

def calculate_vwap(df: pd.DataFrame, period: str = 'D') -> pd.Series:
    """
    Calculate the Volume Weighted Average Price (VWAP) for a DataFrame.
    Internal helper function for calculate_indicators.

    Args:
        df (pd.DataFrame): The DataFrame with OHLCV data (must have DatetimeIndex or timestamp column).
                           Expected columns: 'high', 'low', 'close', 'volume', 'timestamp'.
        period (str): The period to use for VWAP calculation ('D', 'W', or 'M').

    Returns:
        pd.Series: The VWAP values, aligned to the input DataFrame's index. Returns empty Series on error or insufficient data.
    """
    # Ensure DatetimeIndex
    if 'timestamp' in df.columns:
        # Ensure timestamp is datetime and set as index if it's a column
        df_calc = df.copy()
        df_calc['timestamp'] = pd.to_datetime(df_calc['timestamp'], utc=True)
        df_calc = df_calc.set_index('timestamp').sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df_calc = df.copy()
    else:
        logger.error("VWAP calculation requires a DatetimeIndex or 'timestamp' column.")
        return pd.Series(index=df.index, dtype=float) # Return empty Series with original index shape

    try:
        required_cols = ['high', 'low', 'close', 'volume']
        # Ensure required columns exist and are numeric
        for col in required_cols:
            if col not in df_calc.columns:
                 # logger.warning(f"VWAP calc: Missing required column '{col}'. Adding with NaN.") # Less noisy
                 df_calc[col] = np.nan # Add missing column with NaN
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce') # Coerce errors to NaN

        # Drop rows where essential data (HLCV) is missing for calculation
        # This ensures valid price_volume calculation
        df_calc = df_calc.dropna(subset=['high', 'low', 'close', 'volume'])

        if df_calc.empty:
            logger.warning("No valid data for VWAP calculation after dropping NaNs from HLCV.")
            return pd.Series(index=df.index, dtype=float) # Return empty Series matching original index shape

        # Calculate Typical Price (HLC/3) and Price * Volume
        # Ensure prices are positive for HLC3 calculation, replace <=0 with NaN
        df_calc[['high', 'low', 'close']] = df_calc[['high', 'low', 'close']].apply(lambda x: x if x > 0 else np.nan)
        df_calc['hlc3'] = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
        df_calc['price_volume'] = df_calc['hlc3'] * df_calc['volume']

        # Determine the resampling/grouping frequency for pandas based on the period
        # 'W-SUN' aligns with PineScript's weekly candle start logic (Sunday 23:59:59.999 end -> Monday 00:00:00.000 start)
        # 'W' in pandas defaults to Sunday 23:59:59.999 boundary (label='right', closed='right')
        # Using 'W-MON' aligns to Monday start (boundary=end, label=left, closed=left -> Monday 00:00 boundary)
        # For VWAP, PineScript's vwap() function with 'D', 'W', 'M' resets the calculation daily, weekly, or monthly.
        # Pandas Grouper handles this correctly based on frequency.
        freq_map = {'D': 'D', 'W': 'W', 'M': 'MS'} # Use 'W' for weekly default pandas behavior, 'MS' for month start
        # Note: PineScript's 'W' might behave like 'W-SUN' depending on exact version/context. 'W-SUN' aligns to Sunday *end*.
        # 'W' (default in pandas) is Sunday end. Let's use 'W'.
        grouper_freq = freq_map.get(period, 'D') # Default to Daily if period is invalid

        # Perform groupby and cumulative sums
        try:
             # Use pd.Grouper with freq to handle calendar-based grouping
             # The index must be a DatetimeIndex
             grouped = df_calc.groupby(pd.Grouper(freq=grouper_freq))
        except Exception as group_e:
             # This indicates a serious issue with the index or freq
             logger.error(f"Error creating pandas Grouper with freq='{grouper_freq}': {group_e}. Ensure DatetimeIndex is valid/timezone-aware or naive consistently. Cannot calculate VWAP.")
             return pd.Series(index=df.index, dtype=float) # Return empty Series on critical grouping failure


        cumulative_price_volume = grouped['price_volume'].cumsum()
        cumulative_volume = grouped['volume'].cumsum()

        # Calculate VWAP, handling potential division by zero if volume is zero
        vwap_series_partial = cumulative_price_volume / cumulative_volume
        vwap_series_partial = vwap_series_partial.replace([np.inf, -np.inf], np.nan) # Replace inf results with NaN

        # Reindex the calculated VWAP series back to the original input DataFrame's index.
        # This ensures the resulting Series has the same length and index as the input df.
        # NaNs introduced by dropna or division by zero will remain at this stage.
        # Use df.index to reindex correctly, whether it was originally index or timestamp column
        vwap_series_reindexed = vwap_series_partial.reindex(df_calc.index) # Reindex to the temporary calculation index

        # Fill any NaNs forward and then backward to propagate the last known VWAP value
        # This is common practice for time-series indicators where values should persist
        # across periods where calculation was impossible (e.g., zero volume) or missing early data.
        vwap_series = vwap_series_reindexed.ffill().bfill()

        # Return the VWAP series aligned to the *original* input DataFrame's index.
        # If the input DF had a 'timestamp' column and not a DatetimeIndex, we need to return
        # a Series with the original index or align back to the original shape.
        # Assuming the input df *will* be treated as a DatetimeIndex internally for indicator calcs.
        # pandas_ta adds columns to the input DF based on its index. So returning a Series
        # indexed by the DatetimeIndex from df_calc is correct.
        return vwap_series
    except Exception as e:
        logger.error(f"Unexpected error calculating VWAP: {e}\n{traceback.format_exc()}")
        return pd.Series(index=df_calc.index if 'timestamp' in df.columns else df.index, dtype=float) # Return empty Series on exception


def calculate_indicators(
    df: pd.DataFrame, # This DF should be the AGGREGATED DF passed from on_candle_close
    short_ema_len: int,
    long_ema_len: int,
    rsi_len: int,
    vwap_period: str, # e.g., 'D', 'W', 'M'
    rsi_overbought: float,
    rsi_oversold: float
) -> pd.DataFrame:
    """
    Calculate technical indicators (EMA, RSI, VWAP) and strategy signals for a DataFrame.
    Adds indicators and signal condition columns to the DataFrame.
    Assumes input df is the AGGREGATED DataFrame with a DatetimeIndex.

    Args:
        df (pd.DataFrame): The DataFrame with AGGREGATED OHLCV data (must have DatetimeIndex).
                           Expected columns: 'open', 'high', 'low', 'close', 'volume'.
        short_ema_len (int): Length for the shorter EMA.
        long_ema_len (int): Length for the longer EMA.
        rsi_len (int): Length for the RSI calculation.
        vwap_period (str): Reset period for VWAP ('D', 'W', 'M').
        rsi_overbought (float): RSI overbought level.
        rsi_oversold (float): RSI oversold level.

    Returns:
        pd.DataFrame: The input DataFrame with added indicator and per-bar signal columns.
                      Returns original DF with signal columns initialized to False on error
                      or insufficient data for basic calculation.
    """
    # Minimum required data length for the longest indicator to produce a value.
    # pandas_ta indicators typically need length periods for the first valid value.
    # Max lookback for indicator calculation: max(short_ema_len, long_ema_len, rsi_len)
    max_indicator_lookback = max(short_ema_len, long_ema_len, rsi_len)
    required_length_for_indicators = max_indicator_lookback + 1 # Need enough bars for indicators to be non-NaN at the end


    # Ensure required columns exist and are numeric
    required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
    for col in required_ohlcv:
        if col not in df.columns:
            logger.warning(f"Indicator calc: Missing required OHLCV column '{col}'. Adding with NaN.") # Less noisy
            df[col] = np.nan # Add missing column
        df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

    # Ensure DatetimeIndex and sort
    if not isinstance(df.index, pd.DatetimeIndex):
         if 'timestamp' in df.columns:
              try:
                   df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                   df = df.set_index('timestamp').sort_index()
              except Exception as e_idx:
                   logger.error(f"Indicator calc: Could not set 'timestamp' as DatetimeIndex: {e_idx}. Cannot proceed.")
                   return df # Return original DF with signal cols initialized
         else:
              logger.error("Indicator calc: DataFrame has no DatetimeIndex or 'timestamp' column. Cannot proceed.")
              return df # Return original DF with signal cols initialized

    # Create a copy to avoid modifying the original DataFrame in place during calculations
    df_calc = df.copy()

    # Initialize signal columns to False. This ensures they exist even if indicators cannot be calculated.
    # These columns will hold the signal state *at the close of each bar*.
    condition_cols = ['ema_crossover', 'ema_crossunder', 'entry_long_cond', 'entry_short_cond', 'exit_long_rsi_cond', 'exit_short_rsi_cond']
    for col in condition_cols:
        df_calc[col] = False # Initialize as False boolean


    if df_calc.empty or len(df_calc) < required_length_for_indicators:
        # Not enough data for indicator calculation itself
        logger.warning(f"Not enough data ({len(df_calc) if df_calc is not None else 'None'} rows) for indicator calculation. Need at least {required_length_for_indicators}.")
        # Return the df_calc with signal columns initialized to False
        return df_calc

    try:
        # Calculate indicators using pandas_ta
        # pandas_ta automatically handles NaN for the initial warm-up periods
        df_calc[f'EMA_{short_ema_len}'] = ta.ema(df_calc['close'], length=short_ema_len)
        df_calc[f'EMA_{long_ema_len}'] = ta.ema(df_calc['close'], length=long_ema_len)
        df_calc['RSI'] = ta.rsi(df_calc['close'], length=rsi_len)
        # Calculate VWAP
        df_calc['VWAP'] = calculate_vwap(df_calc, vwap_period)

        # --- Calculate signals *per bar* based on Pine Script logic ---
        # These conditions evaluate based on the state *at the close* of each bar
        # Use the calculated indicator columns directly from df_calc

        short_ema = df_calc[f'EMA_{short_ema_len}']
        long_ema = df_calc[f'EMA_{long_ema_len}']
        close = df_calc['close']
        vwap = df_calc['VWAP'] # VWAP is calculated on the HLC3 of the data for each bar in its period
        rsi = df_calc['RSI']

        # Calculate shifted values for crossover/crossunder (requires at least 2 bars)
        # Pandas shift handles NaN for the first row(s) correctly
        prev_short_ema = short_ema.shift(1)
        prev_long_ema = long_ema.shift(1)

        # Crossover/Crossunder detection (condition is true at the end of the bar where it occurs)
        # Use fillna(-inf/+inf) for comparisons involving None/NaN shifted values to avoid issues.
        # (Current short EMA > long EMA) AND (Previous short EMA <= previous long EMA)
        df_calc['ema_crossover'] = (short_ema > long_ema) & (prev_short_ema.fillna(-np.inf) <= prev_long_ema.fillna(np.inf))
        # (Current short EMA < long EMA) AND (Previous short EMA >= previous long EMA)
        df_calc['ema_crossunder'] = (short_ema < long_ema) & (prev_short_ema.fillna(np.inf) >= prev_long_ema.fillna(-np.inf))

        # Entry conditions (based on conditions at the close of each bar where signals are valid)
        # Use `.fillna(False)` to treat rows with NaN indicators as not meeting the condition
        # Also ensure close and vwap are valid numbers before comparison
        df_calc['entry_long_cond'] = (
                df_calc['ema_crossover'].fillna(False) &
                (close.notna() & vwap.notna() & (close > vwap)).fillna(False) & # Check if close > VWAP and handle potential NaNs
                (rsi.notna() & (rsi < rsi_overbought)).fillna(False) # Check RSI condition and handle potential NaNs
        )

        df_calc['entry_short_cond'] = (
                df_calc['ema_crossunder'].fillna(False) &
                (close.notna() & vwap.notna() & (close < vwap)).fillna(False) & # Check if close < VWAP and handle potential NaNs
                (rsi.notna() & (rsi > rsi_oversold)).fillna(False) # Check RSI condition and handle potential NaNs
        )

        # Exit conditions (based on conditions at the close of each bar where signals are valid)
        df_calc['exit_long_rsi_cond'] = (rsi.notna() & (rsi > rsi_overbought)).fillna(False)
        df_calc['exit_short_rsi_cond'] = (rsi.notna() & (rsi < rsi_oversold)).fillna(False)


        # Ensure condition columns are explicitly boolean type
        for col_cond in condition_cols:
            # Use nullable boolean type first to handle potential NaNs, then convert to non-nullable bool
            # fillna(False) before final astype(bool) is robust
            try:
                 df_calc[col_cond] = df_calc[col_cond].fillna(False).astype(bool)
            except Exception as e_bool_cast:
                 # Fallback if fillna(False).astype(bool) fails for some reason
                 logger.warning(f"Failed to cast column '{col_cond}' to boolean safely ({e_bool_cast}). Using fallback list comprehension.")
                 df_calc[col_cond] = [bool(x) for x in df_calc[col_cond].fillna(False)] # Less efficient fallback

        # Return the DataFrame with calculated indicators and signals.
        # Rows with NaN indicators (due to warm-up period) will have signal columns set to False.
        return df_calc

    except Exception as e:
        logger.error(f"Error during indicator calculation: {e}\n{traceback.format_exc()}")
        # On error, return the df_calc DataFrame with signal columns initialized to False
        # Ensure the DataFrame has the initial signal columns even if other calculations fail
        for col in condition_cols:
             if col not in df_calc.columns:
                 df_calc[col] = False
        return df_calc

