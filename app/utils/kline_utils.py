#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import timedelta, datetime, timezone # 导入datetime和timedelta
# 使用app的日志
import logging
logger = logging.getLogger(__name__) # 获取logger

def format_kline_from_api(kline_data):
    """将币安API的单个K线列表转换为字典。"""
    return {
        'timestamp': pd.to_datetime(kline_data[0], unit='ms', utc=True),
        'open': float(kline_data[1]),
        'high': float(kline_data[2]),
        'low': float(kline_data[3]),
        'close': float(kline_data[4]),
        'volume': float(kline_data[5]),
        'quote_volume': float(kline_data[7]),
        # Note: kline_data[6] is close_time, kline_data[8] is number_of_trades
        # Add them if needed by the strategy indicators (e.g., number_of_trades)
        'close_time': pd.to_datetime(kline_data[6], unit='ms', utc=True),
        'number_of_trades': int(kline_data[8]), # Ensure int
    }

def get_pandas_resample_interval(binance_interval_str):
    """将币安间隔字符串转换为Pandas重采样兼容字符串。"""
    # Pandas resample interval strings: 'min', 'H', 'D', 'W', 'M'
    # Binance intervals: '1m', '3m', '1h', '1d', '1w', '1M', etc.
    # Note: Pandas 'W' defaults to Sunday end, 'W-MON' to Monday end etc.
    # Binance '1w' starts on Monday. Need to check alignment.
    # For simple resampling, 'W' might be sufficient or need 'W-MON'.
    # Let's use the simpler mapping for now.
    if binance_interval_str.endswith('s'): # Seconds
        return binance_interval_str[:-1] + 'S' # Pandas uses S for seconds
    elif binance_interval_str.endswith('m'): # Minutes
        return binance_interval_str[:-1] + 'min' # Pandas uses min
    elif binance_interval_str.endswith('h'): # Hours
        return binance_interval_str[:-1] + 'H' # Pandas uses H
    elif binance_interval_str.endswith('d'): # Days
        return binance_interval_str[:-1] + 'D' # Pandas uses D
    elif binance_interval_str.endswith('w'): # Weeks (Binance starts Mon)
        return binance_interval_str[:-1] + 'W-MON' # Pandas 'W-MON' starts Monday
    elif binance_interval_str.endswith('M'): # Months (Binance is calendar month)
        return binance_interval_str[:-1] + 'MS' # Pandas 'MS' is month start frequency
    else:
        # Fallback or error
        logging.warning(f"Unknown Binance interval format for pandas resampling: {binance_interval_str}. Returning as-is (may fail).")
        return binance_interval_str


# --- 从原始项目 utils 提取的函数 ---

def interval_to_timedelta(interval_str: str) -> timedelta:
    """Converts Binance/CoinAPI interval string (like '1m', '1h', '1d') to timedelta."""
    try:
        if interval_str.endswith('s'):
            return timedelta(seconds=int(interval_str[:-1]))
        elif interval_str.endswith('m'):
            return timedelta(minutes=int(interval_str[:-1]))
        elif interval_str.endswith('h'):
            return timedelta(hours=int(interval_str[:-1]))
        elif interval_str.endswith('d'):
            return timedelta(days=int(interval_str[:-1]))
        elif interval_str.endswith('w'):
            return timedelta(weeks=int(interval_str[:-1]))
        elif interval_str.endswith('M'):
            # Approximation, assumes 30 days for a month. Accurate month logic is complex.
            # For strategy based on candles, timedelta approximation might be okay,
            # but for precise time calculations across month boundaries, use calendar logic.
            # Given the original Pine Script likely treats months as fixed duration for backtesting,
            # this approximation might align.
            logger.warning(f"Using approximated timedelta for month interval '{interval_str}'. Calculation assumes 30 days per month.")
            return timedelta(days=int(interval_str[:-1]) * 30) # Standard approximation

        else:
            raise ValueError(f"Unknown interval format: {interval_str}")
    except ValueError as e:
        logger.error(f"Failed to convert interval string '{interval_str}' to timedelta: {e}")
        # Re-raise the ValueError as it's a configuration issue or fundamental data problem
        raise ValueError(f"Invalid interval string for timedelta: {interval_str}") from e

def align_timestamp_to_interval(timestamp: datetime, interval_delta: timedelta, direction: str = 'start') -> datetime:
    """
    Aligns a timezone-aware timestamp to the start or end of the nearest interval boundary
    relative to the Unix epoch (UTC).

    Args:
        timestamp (datetime): The input timezone-aware timestamp.
        interval_delta (timedelta): The duration of the interval.
        direction (str): 'start' to align to the start of the current/previous interval,
                         'end' to align to the end of the current/next interval.
                         'nearest' to align to the nearest interval boundary.

    Returns:
        datetime: The aligned timezone-aware timestamp (always UTC).
    """
    # Ensure timestamp is timezone-aware UTC before calculation
    if timestamp.tzinfo is None:
        # logger.warning(f"Aligning timezone-naive timestamp {timestamp}. Assuming UTC.")
        timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp_utc = timestamp.astimezone(timezone.utc)

    # Unix epoch start (UTC)
    epoch_utc = datetime(1970, 1, 1, tzinfo=timezone.utc)

    # Time elapsed since epoch in seconds
    elapsed_seconds = (timestamp_utc - epoch_utc).total_seconds()
    interval_seconds = interval_delta.total_seconds()

    if interval_seconds <= 0:
         logger.error(f"Invalid interval_delta for alignment: {interval_delta}")
         # Return original timestamp as UTC fallback if interval is invalid
         return timestamp_utc

    # Number of full intervals elapsed since epoch (using floor division)
    num_intervals = elapsed_seconds // interval_seconds

    if direction == 'start':
        aligned_seconds = num_intervals * interval_seconds
    elif direction == 'end':
        # The end of the current interval is the start of the next interval boundary.
        # This corresponds to the ceiling of (elapsed_seconds / interval_seconds) * interval_seconds.
        # For a timestamp EXACTLY on an interval start (e.g., 08:00:00 for 1h), its 'end' should be 09:00:00.
        # For a timestamp WITHIN an interval (e.g., 08:05:00 for 1h), its 'end' should be 09:00:00.
        # For a timestamp EXACTLY on an interval end (e.g., 09:00:00 for 1h), its 'end' should be 09:00:00.
        # Using ceil logic: ceil(x/y) * y.
        if elapsed_seconds == 0: # Epoch start
            aligned_seconds = interval_seconds # End of the first interval
        else:
            # Small tolerance for floats near boundary
            if abs(elapsed_seconds % interval_seconds) < 1e-9: # Check if it's exactly on a boundary
                 aligned_seconds = elapsed_seconds # If exactly on boundary, end is that timestamp itself (for historical data) or next boundary (for live)
                 # For aligning *data points*, if the data point is T, its interval ended at T.
                 # If aligning a *current time* T to the *next* interval end, use ceil.
                 # Given this is used for aligning timestamps *of klines* which represent interval starts,
                 # the 'end' of the interval starting at T is T + interval_delta.
                 # Let's return the timestamp itself if it's on a boundary, plus delta.
                 aligned_seconds = num_intervals * interval_seconds + interval_seconds # Start of current + delta
                 # Refined logic: If timestamp is *exactly* aligned to the start of an interval,
                 # the "end" of that interval is timestamp + interval_delta.
                 # If timestamp is within an interval, the "end" is the start of the *next* interval.
                 # This is equivalent to ceiling(elapsed_seconds / interval_seconds) * interval_seconds IF using 0 as base
                 # But safer to align to start then add delta.
                 start_of_current_interval_seconds = num_intervals * interval_seconds
                 aligned_seconds = start_of_current_interval_seconds + interval_seconds


    elif direction == 'nearest':
         # Nearest interval boundary
         aligned_seconds = round(elapsed_seconds / interval_seconds) * interval_seconds
    else:
        logger.error(f"Invalid alignment direction: {direction}. Using 'start'.")
        aligned_seconds = num_intervals * interval_seconds

    # Calculate the aligned timestamp, ensure it's timezone-aware UTC
    aligned_timestamp_utc = epoch_utc + timedelta(seconds=aligned_seconds)
    return aligned_timestamp_utc
