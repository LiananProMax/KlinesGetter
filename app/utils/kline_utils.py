#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import timedelta, datetime, timezone # 导入datetime和timedelta
import decimal
from decimal import Decimal
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
        logger.warning(f"Unknown Binance interval format for pandas resampling: {binance_interval_str}. Returning as-is (may fail).")
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
            # this approximation might align. Ensure this aligns with pandas 'MS' frequency behavior if used.
            # Note: pandas 'MS' is calendar month start, not fixed 30-day duration.
            return timedelta(days=int(interval_str[:-1]) * 30) # Standard approximation

        else:
            raise ValueError(f"Unknown interval format: {interval_str}")
    except ValueError as e:
        logger.error(f"Failed to convert interval string '{interval_str}' to timedelta: {e}")
        # Re-raise the ValueError as it's a configuration issue or fundamental data problem
        raise ValueError(f"Invalid interval string for timedelta: {interval_str}") from e

def align_timestamp_to_interval(timestamp: datetime, interval_delta: timedelta, direction: str = 'start') -> datetime:
    """
    将带时区的时间戳对齐到相对于Unix纪元(UTC)的最近间隔边界的开始或结束。

    参数:
        timestamp (datetime): 输入的带时区的时间戳。
        interval_delta (timedelta): 间隔的持续时间。
        direction (str): 'start' 表示对齐到当前/前一个间隔的开始，
                         'end' 表示对齐到当前/下一个间隔的结束，
                         'nearest' 表示对齐到最近的间隔边界。

    返回:
        datetime: 对齐后的带时区的时间戳(始终为UTC)。
    """
    # 确保时间戳在计算前带有UTC时区信息
    if timestamp.tzinfo is None:
        # logger.warning(f"对齐无时区的时间戳 {timestamp}。假定为UTC。")
        timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp_utc = timestamp.astimezone(timezone.utc)

    # Unix纪元起始时间 (UTC)
    epoch_utc = datetime(1970, 1, 1, tzinfo=timezone.utc)

    # 从纪元开始到现在经过的秒数
    elapsed_seconds = (timestamp_utc - epoch_utc).total_seconds()
    interval_seconds = interval_delta.total_seconds()

    if interval_seconds <= 0:
         logger.error(f"用于对齐的间隔无效: {interval_delta}")
         # 如果间隔无效，返回原始UTC时间戳作为回退方案
         return timestamp_utc

    # 从纪元开始经过的完整间隔数（使用向下取整除法）
    # 使用Decimal来提高边界计算的精度
    elapsed_seconds_dec = Decimal(str(elapsed_seconds))
    interval_seconds_dec = Decimal(str(interval_seconds))

    # 计算包含该时间戳的间隔的开始
    # 整数除法会截断，对于正数有效地向下取整
    start_of_interval_seconds_dec = (elapsed_seconds_dec // interval_seconds_dec) * interval_seconds_dec

    if direction == 'start':
        aligned_seconds_dec = start_of_interval_seconds_dec
    elif direction == 'end':
        # 从 T 开始持续时间 I 的间隔的'结束'是 T + I。
        # 因此，包含“时间戳”的间隔的结束是（start_of_interval）+ interval_delta。
        #
        # 从 `start_of_interval_seconds_dec` 开始的间隔的结束时间戳
        # 是 `start_of_interval_seconds_dec + interval_seconds_dec`。
        # 如果输入的 `timestamp_utc` 正好是 `start_of_interval_seconds_dec`，
        # 其K线*开始*于该时间，并*结束*于 `start_of_interval_seconds_dec + interval_seconds_dec`。
        # 如果我们对齐到'结束'，而输入时间戳正好*在*间隔的结束处，
        # 这意味着从 `timestamp - interval_delta` 到 `timestamp` 的间隔刚刚关闭。
        # 用于对齐目的的'结束'时间戳应该是 `timestamp`。
        # 如果时间戳在间隔内（不在其开始或结束处），'结束'是该间隔的结束。
        # 这个逻辑可能很棘手，取决于确切的需求（包含/排除边界）。
        # 一种常见的方法是找到时间戳所在的间隔的开始，然后添加间隔增量。
        # 如果时间戳正好在间隔边界上，这个计算可能需要注意。
        # 让我们假设'结束'表示*包含*时间戳的间隔的结束。
        # 示例：时间戳=08:59:00，间隔=1h。开始=08:00:00。结束=09:00:00。
        # 示例：时间戳=09:00:00，间隔=1h。开始=09:00:00。结束=10:00:00。
        # 这意味着如果时间戳正好在开始处，简单的start_of_interval + delta可能会偏移一个间隔。
        # 更安全的方法：找到小于或等于时间戳的开始时间，然后添加interval_delta。
        # 如果时间戳正好在间隔的*结束*处（例如，08:00-09:00 K线的 09:00:00），
        # 间隔开始是 timestamp - interval_delta。结束是 timestamp。
        # 让我们对齐到时间戳*之后*或*正好在*时间戳处的边界（如果它在边界上）。
        # 这是 ceiling(elapsed_seconds / interval_seconds) * interval_seconds，但需要处理零/负值。
        # 对于正的elapsed_seconds和interval_seconds：aligned_seconds = ceil(elapsed_seconds / interval_seconds) * interval_seconds
        # 或：使用整数计算的 (elapsed_seconds + interval_seconds - 1) // interval_seconds * interval_seconds
        # 使用Decimal来提高精度：
        if elapsed_seconds_dec <= Decimal('0'): # 处理纪元或负数时间
             aligned_seconds_dec = Decimal('0') # 第一个间隔的结束是开始 + 增量
        else:
            # 使用Decimal直接向上取整
            # 对于正整数a和b，ceil(a/b) = (a + b - 1) // b
            # 使用Decimal更简单：
            aligned_seconds_dec = (elapsed_seconds_dec / interval_seconds_dec).quantize(Decimal('1'), rounding=decimal.ROUND_CEILING) * interval_seconds_dec

    elif direction == 'nearest':
         # 最近的间隔边界
         aligned_seconds_dec = (elapsed_seconds_dec / interval_seconds_dec).quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP) * interval_seconds_dec
    else:
        logger.error(f"无效的对齐方向: {direction}。使用 'start'。")
        aligned_seconds_dec = start_of_interval_seconds_dec

    # 计算对齐后的时间戳，确保它是带时区的UTC
    # 将aligned_seconds_dec转换回浮点数作为timedelta构造函数的参数
    aligned_timestamp_utc = epoch_utc + timedelta(seconds=float(aligned_seconds_dec))
    return aligned_timestamp_utc
