# display_manager.py
import logging
import pandas as pd # 导入pandas用于pd.Timedelta

def display_historical_aggregated_klines(df_aggregated, symbol, agg_interval, display_count):
    """显示初始历史聚合K线集。"""
    if df_aggregated.empty:
        logging.warning(f"没有{agg_interval}的历史数据可显示（{symbol}）。")
        return

    actual_display_count = min(display_count, len(df_aggregated))
    logging.info(f"\n--- 初始{actual_display_count}个历史{agg_interval} OHLCV数据（{symbol}）---")
    
    output_lines = []
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


def display_realtime_update(df_aggregated, symbol_ws, agg_interval_str, base_interval_str, all_base_klines_df):
    """显示实时更新的最新聚合K线。"""
    if df_aggregated.empty:
        logging.info(f"尚未有足够的{base_interval_str}数据形成完整的{agg_interval_str}K线。")
        return

    logging.info(f"\n--- 当前{agg_interval_str} OHLCV数据（{symbol_ws}）---")
    display_recent_count = 5 # 显示最新N个聚合K线
    
    output_lines = []
    base_interval_duration = pd.Timedelta(base_interval_str)
    agg_interval_duration = pd.Timedelta(agg_interval_str)

    for i, row_agg in df_aggregated.tail(display_recent_count).iterrows():
        status = "正在形成" # 最新蜡烛的默认状态
        
        agg_candle_start_time = row_agg['timestamp']
        agg_candle_end_time = agg_candle_start_time + agg_interval_duration
        
        if not all_base_klines_df.empty:
            latest_base_kline_start_time = all_base_klines_df['timestamp'].iloc[-1]
            latest_base_kline_close_time = latest_base_kline_start_time + base_interval_duration

            if latest_base_kline_close_time >= agg_candle_end_time:
                status = "已关闭"
        
        # df_aggregated中不是绝对最后一个的任何蜡烛都被视为已关闭
        # （假设数据持续流动于先前周期）
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
