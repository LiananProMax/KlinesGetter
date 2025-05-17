#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd

def display_historical_aggregated_klines(df_aggregated: pd.DataFrame, symbol: str, agg_interval: str, display_count: int):
    """显示初始历史聚合K线集。"""
    if df_aggregated.empty:
        logging.warning(f"没有{agg_interval}的历史数据可显示（{symbol}）。")
        return

    actual_display_count = min(display_count, len(df_aggregated))
    logging.info(f"\n--- 初始{actual_display_count}个历史{agg_interval} OHLCV数据（{symbol}）---")
    
    output_lines = []
    # 确保timestamp列在潜在的reset_index聚合后存在
    if 'timestamp' not in df_aggregated.columns:
        logging.error("display_historical_aggregated_klines: 聚合DataFrame中缺少'timestamp'列。")
        return

    for _, row in df_aggregated.tail(actual_display_count).iterrows():
        line = (
            f"  开始：{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}, "
            f"开：{row['open']:.2f}, 高：{row['high']:.2f}, 低：{row['low']:.2f}, 收：{row['close']:.2f}, "
            f"量：{row['volume']:.2f}, 交易额：{row['quote_volume']:.2f}, 状态：已关闭"
        )
        output_lines.append(line)
    print("\n".join(output_lines))
    print("-" * 70)
    if len(df_aggregated) < display_count:
        logging.info(f"注意：聚合历史K线（{len(df_aggregated)}）少于期望的数量（{display_count}）。")


def display_realtime_update(df_aggregated: pd.DataFrame, symbol_ws: str, agg_interval_str: str, base_interval_str: str, all_base_klines_df: pd.DataFrame):
    """显示实时更新的最新聚合K线。"""
    if df_aggregated.empty:
        # 如果基础K线在形成完整的聚合K线之前逐个到达，这个日志可能过于频繁
        # logging.info(f"尚未有足够的{base_interval_str}数据形成完整的{agg_interval_str}K线。")
        return

    logging.info(f"\n--- 当前{agg_interval_str} OHLCV数据（{symbol_ws}）---")
    display_recent_count = 5 
    
    output_lines = []

    # 确保timestamp列存在
    if 'timestamp' not in df_aggregated.columns:
        logging.error("display_realtime_update: 聚合DataFrame中缺少'timestamp'列。")
        return
    if not all_base_klines_df.empty and 'timestamp' not in all_base_klines_df.columns:
        logging.error("display_realtime_update: all_base_klines_df DataFrame中缺少'timestamp'列。")
        return

    try:
        base_interval_duration = pd.Timedelta(base_interval_str)
        agg_interval_duration = pd.Timedelta(agg_interval_str)
    except ValueError as e:
        logging.error(f"无法解析间隔字符串: {base_interval_str}, {agg_interval_str}. 错误: {e}")
        return

    for i, row_agg in df_aggregated.tail(display_recent_count).iterrows():
        status = "正在形成" 
        
        agg_candle_start_time = row_agg['timestamp']
        agg_candle_end_time = agg_candle_start_time + agg_interval_duration
        
        if not all_base_klines_df.empty:
            latest_base_kline_start_time = all_base_klines_df['timestamp'].iloc[-1]
            latest_base_kline_close_time = latest_base_kline_start_time + base_interval_duration

            if latest_base_kline_close_time >= agg_candle_end_time:
                status = "已关闭"
        
        if i < len(df_aggregated) - 1: 
             status = "已关闭"

        line = (
            f"  开始：{row_agg['timestamp'].strftime('%Y-%m-%d %H:%M:%S %Z')}, "
            f"开：{row_agg['open']:.2f}, 高：{row_agg['high']:.2f}, 低：{row_agg['low']:.2f}, 收：{row_agg['close']:.2f}, "
            f"量：{row_agg['volume']:.2f}, 交易额：{row_agg['quote_volume']:.2f}, 状态：{status}"
        )
        output_lines.append(line)
    print("\n".join(output_lines))
    print("-" * 70)
