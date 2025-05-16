# main_app.py
import logging
import time
import json
import pandas as pd
from datetime import datetime, timezone # 为binance_api_manager中的使用添加

# 导入本地模块
import config
from kline_utils import format_kline_from_api
from data_store import KlineDataStore
from kline_aggregator import aggregate_klines_df
from binance_api_manager import fetch_historical_klines, BinanceWebsocketManager
from display_manager import display_historical_aggregated_klines, display_realtime_update

from binance.lib.utils import config_logging # 用于日志设置

# --- 全局实例（由此主应用模块管理）---
# 这些不再是其他模块可以直接访问的真正全局变量，
# 而是由main_app.py管理和传递的中央实例。
kline_store = None
ws_manager = None

def setup_logging():
    """配置应用程序范围的日志。"""
    # 使用币安的config_logging或您自己喜欢的设置
    config_logging(logging, config.LOG_LEVEL)
    if config.OPERATING_MODE == "TEST":
        logging.info(f"运行模式：TEST（{config.BASE_INTERVAL}数据 -> {config.AGG_INTERVAL}聚合）")
    elif config.OPERATING_MODE == "PRODUCTION":
        logging.info(f"运行模式：PRODUCTION（{config.BASE_INTERVAL}数据 -> {config.AGG_INTERVAL}聚合）")


def websocket_message_handler(_, message_str):
    """处理传入的WebSocket K线消息。"""
    global kline_store # 访问kline_store实例

    try:
        data = json.loads(message_str)

        if 'e' in data and data['e'] == 'kline':
            kline_payload = data['k']
            if kline_payload['x']:  # 仅处理已关闭的K线
                symbol_ws = kline_payload['s']
                base_interval_stream = kline_payload['i']
                
                kline_timestamp_dt = pd.to_datetime(kline_payload['t'], unit='ms', utc=True)
                logging.debug(f"收到{symbol_ws}的已关闭{base_interval_stream}K线 @ {kline_timestamp_dt}")
                
                new_kline_dict = {
                    'timestamp': kline_timestamp_dt,
                    'open': float(kline_payload['o']), 'high': float(kline_payload['h']),
                    'low': float(kline_payload['l']), 'close': float(kline_payload['c']),
                    'volume': float(kline_payload['v']), 'quote_volume': float(kline_payload['q'])
                }
                
                if kline_store:
                    kline_store.add_single_kline(new_kline_dict)
                    
                    # 聚合并显示
                    current_base_df = kline_store.get_base_klines_df()
                    df_aggregated = aggregate_klines_df(current_base_df, config.AGG_INTERVAL)
                    display_realtime_update(df_aggregated, symbol_ws, config.AGG_INTERVAL, config.BASE_INTERVAL, current_base_df)
                else:
                    logging.error("在websocket_message_handler中KlineDataStore未初始化。")

        elif 'e' in data and data['e'] == 'error':
            logging.error(f"WebSocket API错误：{data.get('m', '未提供错误消息')}")
        # else:
            # logging.debug(f"其他WebSocket消息：{message_str[:200]}")

    except Exception as e:
        logging.error(f"websocket_message_handler中出错：{e}", exc_info=True)
        logging.error(f"问题原始消息：{message_str}")


def main_application():
    global kline_store, ws_manager # 分配给模块级变量

    setup_logging()

    # 初始化DataStore
    kline_store = KlineDataStore(
        base_interval_str=config.BASE_INTERVAL,
        agg_interval_str=config.AGG_INTERVAL,
        historical_candles_to_display_count=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
    )

    # 1. 获取历史数据
    try:
        base_td = pd.Timedelta(config.BASE_INTERVAL)
        agg_td = pd.Timedelta(config.AGG_INTERVAL)
        if base_td.total_seconds() == 0 or agg_td.total_seconds() == 0 or \
           (agg_td.total_seconds() / base_td.total_seconds()) <= 0:
            raise ValueError("历史数据计算的间隔配置无效。")
        base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()
    except Exception as e_calc:
        logging.error(f"计算间隔比率时出错：{e_calc}。无法获取历史K线。")
        return

    num_base_klines_needed = int((config.HISTORICAL_AGG_CANDLES_TO_DISPLAY + 5) * base_intervals_per_agg)
    
    historical_klines_list = fetch_historical_klines(
        symbol=config.SYMBOL,
        interval=config.BASE_INTERVAL,
        num_klines_to_fetch=num_base_klines_needed,
        api_base_url=config.API_BASE_URL_FUTURES,
        max_limit_per_request=config.MAX_KLINE_LIMIT_PER_REQUEST
    )

    if historical_klines_list:
        kline_store.add_klines(historical_klines_list)
        logging.info(f"成功处理{len(historical_klines_list)}个历史基础K线。")
        base_df_for_initial_agg = kline_store.get_base_klines_df()
        if not base_df_for_initial_agg.empty:
            min_ts = base_df_for_initial_agg['timestamp'].min().isoformat()
            max_ts = base_df_for_initial_agg['timestamp'].max().isoformat()
            logging.info(f"历史基础K线范围：{min_ts}到{max_ts}")
            
            initial_agg_df = aggregate_klines_df(base_df_for_initial_agg, config.AGG_INTERVAL)
            display_historical_aggregated_klines(
                initial_agg_df,
                config.SYMBOL,
                config.AGG_INTERVAL,
                config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
        else:
            logging.warning("初始聚合后存储中没有基础K线。")
    else:
        logging.warning("未获取历史K线。仅继续使用实时数据。")

    # 2. 设置并启动WebSocket
    ws_manager = BinanceWebsocketManager(
        symbol=config.SYMBOL,
        base_interval=config.BASE_INTERVAL,
        on_message_callback=websocket_message_handler
    )
    ws_manager.start()

    # 3. 主循环（保持活动并检查WebSocket状态）
    try:
        while True:
            time.sleep(30) # 检查状态或其他任务的间隔
            if ws_manager and not ws_manager.is_active():
                logging.warning("WebSocket管理器报告未激活。终止应用程序。")
                break
            elif not ws_manager:
                logging.error("WebSocket管理器未初始化。终止。")
                break
    except KeyboardInterrupt:
        logging.info("检测到键盘中断。正在关闭...")
    except Exception as e:
        logging.error(f"主循环中意外错误：{e}", exc_info=True)
    finally:
        if ws_manager:
            ws_manager.stop()
        logging.info("应用程序已终止。")

if __name__ == "__main__":
    main_application()
