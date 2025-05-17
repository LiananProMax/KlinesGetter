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
    计算DataFrame的成交量加权平均价（VWAP）。
    用于calculate_indicators的内部辅助函数。

    Args:
        df (pd.DataFrame): 带有OHLCV数据的DataFrame（必须有DatetimeIndex或timestamp列）。
                           期望列：'high', 'low', 'close', 'volume', 'timestamp'。
        period (str): 用于VWAP计算的周期（'D', 'W', 或 'M'）。

    Returns:
        pd.Series: VWAP值，与输入DataFrame的索引对齐。错误或数据不足时返回空Series。
    """
    # 确保是DatetimeIndex
    if 'timestamp' in df.columns:
        # 确保timestamp是datetime类型，如果它是列，则设为索引
        df_calc = df.copy()
        df_calc['timestamp'] = pd.to_datetime(df_calc['timestamp'], utc=True)
        df_calc = df_calc.set_index('timestamp').sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df_calc = df.copy()
    else:
        logger.error("VWAP计算需要一个DatetimeIndex或'timestamp'列。")
        # 返回与原始索引形状匹配的空Series
        return pd.Series(index=df.index, dtype=float) # Return empty Series with original index shape

    try:
        required_cols = ['high', 'low', 'close', 'volume']
        # 确保所需的列存在且为数字类型
        for col in required_cols:
            if col not in df_calc.columns:
                 # logger.warning(f"VWAP calc: Missing required column '{col}'. Adding with NaN.") # Less noisy
                 # logger.warning(f"VWAP计算：缺少必需的列'{col}'。添加NaN。") # 不那么嘈杂
                 df_calc[col] = np.nan # 添加缺失列并用NaN填充
            # 将错误强制转换为NaN
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce') # Coerce errors to NaN

        # 丢弃计算所需关键数据（HLCV）缺失的行
        # 这确保了price_volume计算的有效性
        df_calc = df_calc.dropna(subset=['high', 'low', 'close', 'volume'])

        if df_calc.empty:
            logger.warning("丢弃HLCV中的NaN后，没有有效数据进行VWAP计算。")
            # 返回与原始索引形状匹配的空Series
            return pd.Series(index=df.index, dtype=float) # Return empty Series matching original index shape

        # 计算典型价格(HLC/3)和价格*成交量
        # 确保价格为正数以便计算HLC3，将<=0的值替换为NaN
        df_calc[['high', 'low', 'close']] = df_calc[['high', 'low', 'close']].apply(lambda x: x if x > 0 else np.nan)
        df_calc['hlc3'] = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
        df_calc['price_volume'] = df_calc['hlc3'] * df_calc['volume']

        # 根据周期确定pandas的重采样/分组频率
        # 'W-SUN'与PineScript的周K线开始逻辑对齐（周日 23:59:59.999 结束 -> 周一 00:00:00.000 开始）
        # pandas中的'W'默认边界是周日 23:59:59.999 （label='right', closed='right'）
        # 使用'W-MON'对齐到周一开始（boundary=end, label=left, closed=left -> 周一 00:00 边界）
        # 对于VWAP，PineScript的vwap()函数使用'D'、'W'、'M'时每天、每周或每月重置计算。
        # Pandas Grouper根据频率正确处理这一点。
        freq_map = {'D': 'D', 'W': 'W', 'M': 'MS'} # 使用'W'表示pandas默认的周行为，'MS'表示月开始
        # 注意：PineScript的'W'根据具体版本/上下文可能行为像'W-SUN'。'W-SUN'对齐到周日*结束*。
        # 'W'（pandas中默认）是周日结束。我们使用'W'。
        grouper_freq = freq_map.get(period, 'D') # 如果周期无效，默认为每日

        # 执行groupby和累积求和
        try:
             # 使用带有freq的pd.Grouper来处理基于日历的分组
             # 索引必须是DatetimeIndex
             grouped = df_calc.groupby(pd.Grouper(freq=grouper_freq))
        except Exception as group_e:
             # 这表明索引或频率存在严重问题
             logger.error(f"创建pandas Grouper时出错，freq='{grouper_freq}'：{group_e}。请确保DatetimeIndex有效/时区感知或始终是朴素的。无法计算VWAP。")
             # 关键分组失败时返回空Series
             return pd.Series(index=df.index, dtype=float) # Return empty Series on critical grouping failure


        cumulative_price_volume = grouped['price_volume'].cumsum()
        cumulative_volume = grouped['volume'].cumsum()

        # 计算VWAP，处理成交量为零时可能出现的除以零错误
        vwap_series_partial = cumulative_price_volume / cumulative_volume
        vwap_series_partial = vwap_series_partial.replace([np.inf, -np.inf], np.nan) # 将inf结果替换为NaN

        # 将计算出的VWAP Series重新索引回原始输入DataFrame的索引。
        # 这确保了结果Series与输入df具有相同的长度和索引。
        # 此时将保留因dropna或除以零而引入的NaN。
        # 使用df.index正确地重新索引，无论它最初是索引还是timestamp列
        vwap_series_reindexed = vwap_series_partial.reindex(df_calc.index) # 重新索引到临时计算索引

        # 将所有NaN向前填充，然后向后填充，以传播最后一个已知的VWAP值
        # 这在时间序列指标中是常见的做法，其中值应该在无法计算（例如成交量为零）或早期数据缺失的周期中持续存在。
        vwap_series = vwap_series_reindexed.ffill().bfill()

        # 返回与*原始*输入DataFrame的索引对齐的VWAP Series。
        # 如果输入DF有一个'timestamp'列而不是DatetimeIndex，我们需要返回
        # 一个具有原始索引的Series或对齐回原始形状。
        # 假设输入df在指标计算内部*将*被视为DatetimeIndex。
        # pandas_ta根据其索引向输入DF添加列。因此返回一个Series
        # 使用df_calc中的DatetimeIndex作为索引是正确的。
        return vwap_series
    except Exception as e:
        logger.error(f"计算VWAP时发生意外错误: {e}\n{traceback.format_exc()}")
        # 异常时返回空Series
        return pd.Series(index=df_calc.index if 'timestamp' in df.columns else df.index, dtype=float) # Return empty Series on exception


def calculate_indicators(
    df: pd.DataFrame, # 此DF应为从on_candle_close传递的聚合DF
    short_ema_len: int,
    long_ema_len: int,
    rsi_len: int,
    vwap_period: str, # 例如：'D', 'W', 'M'
    rsi_overbought: float,
    rsi_oversold: float
) -> pd.DataFrame:
    """
    计算DataFrame的技术指标（EMA, RSI, VWAP）和策略信号。
    向DataFrame添加指标列和信号条件列。
    假设输入df是具有DatetimeIndex的聚合DataFrame。

    Args:
        df (pd.DataFrame): 带有聚合OHLCV数据的DataFrame（必须有DatetimeIndex）。
                           期望列：'open', 'high', 'low', 'close', 'volume'。
        short_ema_len (int): 短期EMA的周期长度。
        long_ema_len (int): 长期EMA的周期长度。
        rsi_len (int): RSI计算的周期长度。
        vwap_period (str): VWAP的重置周期（'D', 'W', 'M'）。
        rsi_overbought (float): RSI超买水平。
        rsi_oversold (float): RSI超卖水平。

    Returns:
        pd.DataFrame: 输入DataFrame，添加了指标列和每根K线的信号列。
                      错误或数据不足以进行基本计算时，返回原始DF，信号列初始化为False。
    """
    # 最长指标产生有效值所需的最小数据长度。
    # pandas_ta指标通常需要其周期长度的数据才能产生第一个有效值。
    # 指标计算的最大回溯期：max(short_ema_len, long_ema_len, rsi_len)
    max_indicator_lookback = max(short_ema_len, long_ema_len, rsi_len)
    required_length_for_indicators = max_indicator_lookback + 1 # 需要足够的K线以使指标在末尾不是NaN


    # 确保所需的列存在且为数字类型
    required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
    for col in required_ohlcv:
        if col not in df.columns:
            # logger.warning(f"Indicator calc: Missing required OHLCV column '{col}'. Adding with NaN.") # Less noisy
            logger.warning(f"指标计算：缺少必需的OHLCV列'{col}'。添加NaN。") # 不那么嘈杂
            df[col] = np.nan # 添加缺失列
        # 将错误强制转换为NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 确保是DatetimeIndex并排序
    if not isinstance(df.index, pd.DatetimeIndex):
         if 'timestamp' in df.columns:
              try:
                   df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                   df = df.set_index('timestamp').sort_index()
              except Exception as e_idx:
                   logger.error(f"指标计算：无法将'timestamp'设置为DatetimeIndex：{e_idx}。无法继续。")
                   # 返回原始DF，信号列已初始化
                   return df # Return original DF with signal cols initialized
         else:
              logger.error("指标计算：DataFrame没有DatetimeIndex或'timestamp'列。无法继续。")
              # 返回原始DF，信号列已初始化
              return df # Return original DF with signal cols initialized

    # 创建一个副本，以避免在计算过程中修改原始DataFrame
    df_calc = df.copy()

    # 将信号列初始化为False。这确保即使指标无法计算，这些列也存在。
    # 这些列将保存每根K线*收盘时*的信号状态。
    condition_cols = ['ema_crossover', 'ema_crossunder', 'entry_long_cond', 'entry_short_cond', 'exit_long_rsi_cond', 'exit_short_rsi_cond']
    for col in condition_cols:
        df_calc[col] = False # 初始化为布尔值False


    if df_calc.empty or len(df_calc) < required_length_for_indicators:
        # 数据不足，无法进行指标计算本身
        logger.warning(f"数据不足（{len(df_calc) if df_calc is not None else 'None'}行），无法进行指标计算。至少需要{required_length_for_indicators}行。")
        # 返回df_calc，信号列已初始化为False
        return df_calc

    try:
        # 使用pandas_ta计算指标
        # pandas_ta会自动处理初始预热期的NaN值
        df_calc[f'EMA_{short_ema_len}'] = ta.ema(df_calc['close'], length=short_ema_len)
        df_calc[f'EMA_{long_ema_len}'] = ta.ema(df_calc['close'], length=long_ema_len)
        df_calc['RSI'] = ta.rsi(df_calc['close'], length=rsi_len)
        # 计算VWAP
        df_calc['VWAP'] = calculate_vwap(df_calc, vwap_period)

        # --- 根据Pine Script逻辑计算*每根K线*的信号 ---
        # 这些条件基于每根K线*收盘时*的状态进行评估
        # 直接使用从df_calc中计算出的指标列

        short_ema = df_calc[f'EMA_{short_ema_len}']
        long_ema = df_calc[f'EMA_{long_ema_len}']
        close = df_calc['close']
        # VWAP是基于其周期内每根K线的HLC3数据计算的
        vwap = df_calc['VWAP']
        rsi = df_calc['RSI']

        # 计算用于交叉/向下交叉的移位值（至少需要2根K线）
        # Pandas的shift函数正确处理第一行（或前几行）的NaN
        prev_short_ema = short_ema.shift(1)
        prev_long_ema = long_ema.shift(1)

        # 向上/向下交叉检测（条件在发生该事件的K线结束时为真）
        # 对于涉及None/NaN移位值的比较，使用fillna(-inf/+inf)避免问题。
        # (当前短期EMA > 长期EMA) AND (前一根短期EMA <= 前一根长期EMA)
        df_calc['ema_crossover'] = (short_ema > long_ema) & (prev_short_ema.fillna(-np.inf) <= prev_long_ema.fillna(np.inf))
        # (当前短期EMA < 长期EMA) AND (前一根短期EMA >= 前一根长期EMA)
        df_calc['ema_crossunder'] = (short_ema < long_ema) & (prev_short_ema.fillna(np.inf) >= prev_long_ema.fillna(-np.inf))

        # 入场条件（基于信号有效的每根K线收盘时的条件）
        # 使用`.fillna(False)`将指标为NaN的行视为不满足条件
        # 还要确保在比较之前close和vwap是有效数字
        df_calc['entry_long_cond'] = (
                df_calc['ema_crossover'].fillna(False) &
                # 检查close > VWAP并处理潜在的NaN
                (close.notna() & vwap.notna() & (close > vwap)).fillna(False) & # Check if close > VWAP and handle potential NaNs
                # 检查RSI条件并处理潜在的NaN
                (rsi.notna() & (rsi < rsi_overbought)).fillna(False) # Check RSI condition and handle potential NaNs
        )

        df_calc['entry_short_cond'] = (
                df_calc['ema_crossunder'].fillna(False) &
                # 检查close < VWAP并处理潜在的NaN
                (close.notna() & vwap.notna() & (close < vwap)).fillna(False) & # Check if close < VWAP and handle potential NaNs
                # 检查RSI条件并处理潜在的NaN
                (rsi.notna() & (rsi > rsi_oversold)).fillna(False) # Check RSI condition and handle potential NaNs
        )

        # 出场条件（基于信号有效的每根K线收盘时的条件）
        df_calc['exit_long_rsi_cond'] = (rsi.notna() & (rsi > rsi_overbought)).fillna(False)
        df_calc['exit_short_rsi_cond'] = (rsi.notna() & (rsi < rsi_oversold)).fillna(False)


        # 确保条件列显式为布尔类型
        # 先使用可空布尔类型处理潜在的NaN，然后转换为不可空的bool
        # 在最终astype(bool)之前使用fillna(False)是稳健的
        for col_cond in condition_cols:
            try:
                 df_calc[col_cond] = df_calc[col_cond].fillna(False).astype(bool)
            except Exception as e_bool_cast:
                 # 如果fillna(False).astype(bool)因某种原因失败，使用备用方案
                 logger.warning(f"未能安全地将列'{col_cond}'转换为布尔类型 ({e_bool_cast})。使用备用列表推导。")
                 # 效率较低的备用方案
                 df_calc[col_cond] = [bool(x) for x in df_calc[col_cond].fillna(False)] # Less efficient fallback

        # 返回带有计算出的指标和信号的DataFrame。
        # 指标为NaN（由于预热期）的行，其信号列将设置为False。
        return df_calc

    except Exception as e:
        logger.error(f"指标计算过程中出错: {e}\n{traceback.format_exc()}")
        # 错误时，返回df_calc DataFrame，信号列已初始化为False
        # 即使其他计算失败，也要确保DataFrame包含初始信号列
        for col in condition_cols:
             if col not in df_calc.columns:
                 df_calc[col] = False
        return df_calc
