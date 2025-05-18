#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 设置时区为UTC，确保所有时间处理一致
import os
import time
os.environ['TZ'] = 'UTC'
try:
    time.tzset() # This might not work on all systems, especially Windows
except AttributeError:
    pass # Ignore if tzset is not available

import logging
import structlog
# 导入 Structlog 处理器
from structlog.stdlib import filter_by_level
from structlog.processors import (
    add_log_level,
    TimeStamper,
    StackInfoRenderer,
    format_exc_info,
)
from structlog.contextvars import merge_contextvars, bind_contextvars, clear_contextvars
from structlog.dev import ConsoleRenderer
import json
import pandas as pd
from datetime import datetime, timezone
import ssl
import certifi
import requests 

# 配置和工具
from app.core.config import config
from app.utils.kline_utils import format_kline_from_api # 虽然现在这里没有直接使用，但为了保持一致性保留

# 数据处理组件
from app.data_handling.data_interfaces import KlinePersistenceInterface
from app.data_handling.kline_data_store import KlineDataStore # 默认内存存储
from app.data_handling.db_manager import DBManager # 数据库管理器
from app.data_handling.kline_aggregator import aggregate_klines_df

# API和显示组件
from app.api_clients.binance_api_manager import fetch_historical_klines, BinanceWebsocketManager
from app.ui.display_manager import display_historical_aggregated_klines, display_realtime_update

# 可选：如果仍需要兼容 binance-futures-connector 的日志设置
# from binance.lib.utils import config_logging

# --- 全局实例（由此main_app模块管理）---
# 这些将基于新结构持有实例。
# kline_persistence将持有所选的数据存储（内存或数据库）
kline_persistence: KlinePersistenceInterface = None
ws_manager: BinanceWebsocketManager = None

def setup_logging():
    """配置应用程序范围的结构化日志，包括级别过滤。"""
    # 获取日志级别
    log_level_str = config.validated_log_level.upper()
    try:
        log_level_int = getattr(logging, log_level_str)
    except AttributeError:
        # 回退到INFO，如果配置的级别无效
        logging.error(f"配置的日志级别 '{log_level_str}' 无效，回退到 INFO。")
        log_level_int = logging.INFO

    # 自定义日志格式化器，移除多余空格和格式化问题
    class SimpleLogFormatter:
        """简洁的日志格式化器，生成并格式化日志及其属性"""
        def __call__(self, logger, method_name, event_dict):
            # 获取级别并格式化
            if 'level' in event_dict:
                level = event_dict['level'].upper()
                event_dict['level'] = f"[{level}]"
            return event_dict

    # 配置标准库日志格式
    logging.basicConfig(
        level=log_level_int,
        format="%(message)s",  # 只输出消息部分，避免重复前缀
        force=True  # 强制重置现有配置
    )
    
    # 使用简单的日志处理流程
    structlog.configure(
        processors=[
            merge_contextvars,              # 合并上下文变量
            add_log_level,                 # 添加日志级别
            SimpleLogFormatter(),          # 自定义格式化器
            filter_by_level,               # 按级别过滤
            TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),  # 简洁的时间戳格式
            format_exc_info,               # 格式化异常信息
            # 简洁的控制台渲染器
            ConsoleRenderer(
                colors=False,               # 禁用颜色以保持简洁
                sort_keys=False,            # 不强制排序键值对
                repr_native_str=False,      # 不使用repr()处理字符串
                pad_event=0,                # 不使用填充
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 配置日志文件
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        "binance_kline_app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level_int)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    
    # 配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_int)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    
    # 获取根日志器并添加处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_int)
    # 清除现有处理器
    for handler in root_logger.handlers[:]: 
        root_logger.removeHandler(handler)
    # 添加新的处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 调整其他库的日志级别
    logging.getLogger("websocket").setLevel(logging.DEBUG)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # 获取日志器并绑定全局上下文
    log = structlog.get_logger().bind(service="BinanceKlineApp")
    log.info("日志配置完成", effective_log_level=log_level_str)
    log.info("运行模式", operating_mode=config.OPERATING_MODE,
             base_interval=config.BASE_INTERVAL, agg_interval=config.AGG_INTERVAL)

def _is_valid_kline(kline_payload):
    """验证K线数据的有效性"""
    return (
        kline_payload and 
        'x' in kline_payload and 
        kline_payload['x']  # 仅处理已关闭的K线
    )

def _process_kline_data(kline_payload):
    """从K线数据载荷中提取并格式化K线数据"""
    if not _is_valid_kline(kline_payload): 
        return None

    return {
        'timestamp': pd.to_datetime(kline_payload['t'], unit='ms', utc=True),
        'open': float(kline_payload['o']), 
        'high': float(kline_payload['h']),
        'low': float(kline_payload['l']), 
        'close': float(kline_payload['c']),
        'volume': float(kline_payload['v']), 
        'quote_volume': float(kline_payload['q'])
    }

def _handle_error(data):
    """处理WebSocket错误消息"""
    log = structlog.get_logger()
    log.error("WebSocket API错误", error_message=data.get('m', '未提供错误消息'))

def _update_kline_store(kline_dict, symbol_ws):
    """更新K线存储并刷新显示"""
    global kline_persistence

    kline_persistence.add_single_kline(kline_dict)

    current_base_df = kline_persistence.get_klines_df()
    df_aggregated = aggregate_klines_df(current_base_df, kline_persistence.get_agg_interval_str())

    _refresh_display(df_aggregated, symbol_ws, current_base_df)

def _refresh_display(df_aggregated, symbol_ws, current_base_df):
    """刷新UI显示"""
    display_realtime_update(
        df_aggregated, 
        symbol_ws, 
        kline_persistence.get_agg_interval_str(),
        kline_persistence.get_base_interval_str(),
        current_base_df
    )

def websocket_message_handler(_, message_str: str):
    """处理传入的WebSocket K线消息。"""
    global kline_persistence # 访问kline_persistence实例
    log = structlog.get_logger()

    if not kline_persistence:
        log.error("K线持久化服务未初始化", handler="websocket_message_handler")
        return

    try:
        data = json.loads(message_str)

        # 处理订阅确认消息（Binance WebSocket 响应）
        if 'result' in data and 'id' in data:
            log.debug("收到订阅确认", subscription_id=data['id'])
            return  # 不再处理此消息

        # 处理 K 线消息
        if 'e' in data and data['e'] == 'kline':
            kline_dict = _process_kline_data(data['k'])
            if kline_dict:
                symbol_ws = data['k']['s']
                base_interval_stream = data['k']['i']
                log.info(
                    "收到已关闭K线",
                    symbol=symbol_ws,
                    interval=base_interval_stream,
                    timestamp=kline_dict['timestamp'].isoformat()
                )
                _update_kline_store(kline_dict, symbol_ws)
        # 处理错误消息
        elif 'e' in data and data['e'] == 'error':
            _handle_error(data)
        # 处理其他未知消息
        else:
            log.warning("未知消息类型", data=data)

    except Exception as e:
        log.error("websocket_message_handler中出错", 
                  error=str(e), 
                  exc_info=True, 
                  original_message=message_str)

def main_application():
    global kline_persistence, ws_manager
    bind_contextvars(symbol=config.SYMBOL, operating_mode=config.OPERATING_MODE)
    setup_logging()
    log = structlog.get_logger()
    # 设置全局SSL上下文
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.check_hostname = True  # 启用主机名验证
    ssl_context.verify_mode = ssl.CERT_REQUIRED  # 强制证书验证

    # 创建一个使用certifi证书的requests Session
    requests_session = requests.Session()
    requests_session.verify = certifi.where()

    # --- 初始化数据持久化服务 ---
    try:
        if config.DATA_STORE_TYPE == "database":
            log.info("使用数据库存储", storage_type="PostgreSQL")
            kline_persistence = DBManager(
                symbol=config.SYMBOL, # 传递交易对以命名表
                base_interval_str=config.BASE_INTERVAL,
                agg_interval_str=config.AGG_INTERVAL,
                historical_candles_to_display_count=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
        else: # 默认为内存存储
            log.info("使用内存存储", storage_type="memory")
            kline_persistence = KlineDataStore(
                base_interval_str=config.BASE_INTERVAL,
                agg_interval_str=config.AGG_INTERVAL,
                historical_candles_to_display_count=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
    except Exception as e_store_init:
        log.error("初始化数据持久化服务失败", error=str(e_store_init), exc_info=True)
        return # 没有数据存储就无法继续

    log.debug("数据存储初始化完成", storage_class=type(kline_persistence).__name__)


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
        log.error("计算间隔比率时出错", 
                  error=str(e_calc), 
                  base_interval=config.BASE_INTERVAL, 
                  agg_interval=config.AGG_INTERVAL)
        return

    # 添加缓冲区（例如，再增加5个聚合间隔）以进行计算并确保有足够的数据
    num_base_klines_needed = int((config.HISTORICAL_AGG_CANDLES_TO_DISPLAY + 5) * base_intervals_per_agg)
    log.debug("计算历史K线需求", 
             klines_needed=num_base_klines_needed, 
             interval=config.BASE_INTERVAL)

    historical_klines_list = fetch_historical_klines(
        symbol=config.SYMBOL,
        interval=config.BASE_INTERVAL,
        num_klines_to_fetch=num_base_klines_needed,
        api_base_url=config.API_BASE_URL_FUTURES,
        max_limit_per_request=config.MAX_KLINE_LIMIT_PER_REQUEST,
        session=requests_session  # 传递自定义session
    )

    if historical_klines_list:
        kline_persistence.add_klines(historical_klines_list)
        log.debug("历史基础K线处理完成", count=len(historical_klines_list))

        base_df_for_initial_agg = kline_persistence.get_klines_df()
        if not base_df_for_initial_agg.empty:
            min_ts = base_df_for_initial_agg['timestamp'].min().isoformat()
            max_ts = base_df_for_initial_agg['timestamp'].max().isoformat()
            log.debug("历史基础K线范围", min_timestamp=min_ts, max_timestamp=max_ts)

            initial_agg_df = aggregate_klines_df(base_df_for_initial_agg, config.AGG_INTERVAL)
            display_historical_aggregated_klines(
                initial_agg_df,
                config.SYMBOL,
                config.AGG_INTERVAL,
                config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
        else:
            log.warning("初始聚合后存储中没有基础K线")
    else:
        log.warning("未获取历史K线，仅继续使用实时数据")

    # 2. 设置并启动WebSocket，传递SSL上下文
    ws_manager = BinanceWebsocketManager(
        symbol=config.SYMBOL,
        base_interval=config.BASE_INTERVAL,
        on_message_callback=websocket_message_handler,
        ssl_context=ssl_context  # 传递自定义SSL上下文
    )
    ws_manager.start()

    # 3. 主循环（保持活动并检查WebSocket状态）
    try:
        while True:
            time.sleep(30) 
            if ws_manager and not ws_manager.is_active():
                log.warning("WebSocket管理器报告未激活，尝试重启")
                ws_manager.stop() # 确保在重启前完全停止
                time.sleep(5) # 在重启前稍等片刻
                ws_manager.start()
                if not ws_manager.is_active(): # 重启尝试后再次检查
                    log.error("WebSocket重启失败，终止应用程序")
                    break
            elif not ws_manager:
                log.error("WebSocket管理器未初始化，终止")
                break
    except KeyboardInterrupt:
        log.info("检测到键盘中断，正在关闭")
    except Exception as e:
        log.error("主循环中意外错误", error=str(e), exc_info=True)
    finally:
        if ws_manager:
            ws_manager.stop()
        if hasattr(kline_persistence, 'close') and callable(getattr(kline_persistence, 'close')):
            kline_persistence.close() # 如果DBManager有close方法，显式关闭它
        log.info("应用程序终止")
        clear_contextvars()  # 清理structlog上下文变量，防止内存泄漏
        clear_contextvars()

# if __name__ == "__main__": # 移除此部分，因为run.py是入口点
#     main_application()
