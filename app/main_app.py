#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
import json
import pandas as pd
from datetime import datetime, timezone 

# 配置和工具
from app.core import config
from app.utils.kline_utils import format_kline_from_api # 虽然现在这里没有直接使用，但为了保持一致性保留

# 数据处理组件
from app.data_handling.data_interfaces import KlinePersistenceInterface
from app.data_handling.kline_data_store import KlineDataStore # 默认内存存储
from app.data_handling.db_manager import DBManager # 数据库管理器
from app.data_handling.kline_aggregator import aggregate_klines_df

# API和显示组件
from app.api_clients.binance_api_manager import fetch_historical_klines, BinanceWebsocketManager
from app.ui.display_manager import display_historical_aggregated_klines, display_realtime_update

from binance.lib.utils import config_logging # 用于日志设置

# --- 全局实例（由此main_app模块管理）---
# 这些将基于新结构持有实例。
# kline_persistence将持有所选的数据存储（内存或数据库）
kline_persistence: KlinePersistenceInterface = None
ws_manager: BinanceWebsocketManager = None

def setup_logging():
    """配置应用程序范围的日志。"""
    config_logging(logging, config.LOG_LEVEL)
    logging.info(f"运行模式：{config.OPERATING_MODE}（{config.BASE_INTERVAL}数据 -> {config.AGG_INTERVAL}聚合）")

def websocket_message_handler(_, message_str: str):
    """处理传入的WebSocket K线消息。"""
    global kline_persistence # 访问kline_persistence实例

    if not kline_persistence:
        logging.error("websocket_message_handler中未初始化Kline持久化服务。")
        return

    try:
        data = json.loads(message_str)

        if 'e' in data and data['e'] == 'kline':
            kline_payload = data['k']
            if kline_payload['x']:  # 仅处理已关闭的K线
                symbol_ws = kline_payload['s']
                base_interval_stream = kline_payload['i'] # 匹配kline_persistence.get_base_interval_str()
                
                kline_timestamp_dt = pd.to_datetime(kline_payload['t'], unit='ms', utc=True)
                logging.debug(f"收到{symbol_ws}的已关闭{base_interval_stream}K线 @ {kline_timestamp_dt}")
                
                new_kline_dict = {
                    'timestamp': kline_timestamp_dt,
                    'open': float(kline_payload['o']), 'high': float(kline_payload['h']),
                    'low': float(kline_payload['l']), 'close': float(kline_payload['c']),
                    'volume': float(kline_payload['v']), 'quote_volume': float(kline_payload['q'])
                }
                
                kline_persistence.add_single_kline(new_kline_dict)
                
                current_base_df = kline_persistence.get_klines_df()
                df_aggregated = aggregate_klines_df(current_base_df, kline_persistence.get_agg_interval_str())
                display_realtime_update(
                    df_aggregated, 
                    symbol_ws, 
                    kline_persistence.get_agg_interval_str(),
                    kline_persistence.get_base_interval_str(),
                    current_base_df
                )

        elif 'e' in data and data['e'] == 'error':
            logging.error(f"WebSocket API错误：{data.get('m', '未提供错误消息')}")

    except Exception as e:
        logging.error(f"websocket_message_handler中出错：{e}", exc_info=True)
        logging.error(f"问题原始消息：{message_str}")

def main_application():
    global kline_persistence, ws_manager 

    setup_logging()

    # --- 初始化数据持久化服务 ---
    try:
        if config.DATA_STORE_TYPE == "database":
            logging.info("使用数据库存储（PostgreSQL）。")
            kline_persistence = DBManager(
                symbol=config.SYMBOL, # 传递交易对以命名表
                base_interval_str=config.BASE_INTERVAL,
                agg_interval_str=config.AGG_INTERVAL,
                historical_candles_to_display_count=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
        else: # 默认为内存存储
            logging.info("使用内存存储。")
            kline_persistence = KlineDataStore(
                base_interval_str=config.BASE_INTERVAL,
                agg_interval_str=config.AGG_INTERVAL,
                historical_candles_to_display_count=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
    except Exception as e_store_init:
        logging.error(f"初始化数据持久化服务失败：{e_store_init}", exc_info=True)
        return # 没有数据存储就无法继续

    logging.info(f"使用 {type(kline_persistence).__name__} 进行数据存储。")


    # 1. 获取历史数据
    try:
        base_td = pd.Timedelta(config.BASE_INTERVAL)
        agg_td = pd.Timedelta(config.AGG_INTERVAL)
        if base_td.total_seconds() <= 0 or agg_td.total_seconds() <= 0: # 检查是否为正持续时间
            raise ValueError("基础间隔或聚合间隔必须是正持续时间。")
        if agg_td.total_seconds() < base_td.total_seconds(): # 聚合间隔应 >= 基础间隔
             raise ValueError("聚合间隔必须大于或等于基础间隔。")

        base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()
        if base_intervals_per_agg <= 0: # 应该被前面的检查捕获，但还是检查一下
            raise ValueError("计算的 base_intervals_per_agg 无效。")
            
    except Exception as e_calc:
        logging.error(f"计算间隔比率时出错：{e_calc}。"
                      f"BASE_INTERVAL: '{config.BASE_INTERVAL}', AGG_INTERVAL: '{config.AGG_INTERVAL}'. "
                      "无法获取历史K线。")
        return

    # 添加缓冲区（例如，再增加5个聚合间隔）以进行计算并确保有足够的数据
    num_base_klines_needed = int((config.HISTORICAL_AGG_CANDLES_TO_DISPLAY + 5) * base_intervals_per_agg)
    logging.info(f"计算得到需要获取 {num_base_klines_needed} 个 '{config.BASE_INTERVAL}' {config.SYMBOL} 历史K线...")
    
    historical_klines_list = fetch_historical_klines(
        symbol=config.SYMBOL,
        interval=config.BASE_INTERVAL,
        num_klines_to_fetch=num_base_klines_needed,
        api_base_url=config.API_BASE_URL_FUTURES,
        max_limit_per_request=config.MAX_KLINE_LIMIT_PER_REQUEST
    )

    if historical_klines_list:
        kline_persistence.add_klines(historical_klines_list)
        logging.info(f"成功处理{len(historical_klines_list)}个历史基础K线。")
        
        base_df_for_initial_agg = kline_persistence.get_klines_df()
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
            time.sleep(30) 
            if ws_manager and not ws_manager.is_active():
                logging.warning("WebSocket管理器报告未激活。尝试重启...")
                ws_manager.stop() # 确保在重启前完全停止
                time.sleep(5) # 在重启前稍等片刻
                ws_manager.start()
                if not ws_manager.is_active(): # 重启尝试后再次检查
                    logging.error("WebSocket重启失败。终止应用程序。")
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
        if hasattr(kline_persistence, 'close') and callable(getattr(kline_persistence, 'close')):
            kline_persistence.close() # 如果DBManager有close方法，显式关闭它
        logging.info("应用程序已终止。")

# if __name__ == "__main__": # 移除此部分，因为run.py是入口点
#     main_application()
