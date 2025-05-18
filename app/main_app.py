#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
os.environ['TZ'] = 'UTC'
try:
    time.tzset()
except AttributeError:
    pass

import logging
import structlog
from structlog.stdlib import ProcessorFormatter, add_log_level as add_stdlib_log_level
from structlog.processors import (
    TimeStamper,
    StackInfoRenderer,
    format_exc_info,
)
from structlog.contextvars import merge_contextvars, bind_contextvars, clear_contextvars
from structlog.dev import ConsoleRenderer, better_traceback
from logging.handlers import RotatingFileHandler
import sys
import json
import pandas as pd
from datetime import datetime, timezone
import ssl
import certifi
import requests

# 自定义处理器：将日志级别转为大写
def uppercase_level_processor(_, __, event_dict):
    """
    将 event_dict 中的 'level' 键的值转换为大写。
    """
    level = event_dict.get("level")
    if level and isinstance(level, str):  # 检查 level 是否存在且为字符串
        event_dict["level"] = level.upper()
    return event_dict

from app.core.config import config
from app.utils.kline_utils import format_kline_from_api

from app.data_handling.data_interfaces import KlinePersistenceInterface
from app.data_handling.kline_data_store import KlineDataStore
from app.data_handling.db_manager import DBManager
from app.data_handling.kline_aggregator import aggregate_klines_df

from app.api_clients.binance_api_manager import fetch_historical_klines, BinanceWebsocketManager
from app.ui.display_manager import display_historical_aggregated_klines, display_realtime_update


# --- 全局实例 ---
kline_persistence: KlinePersistenceInterface = None
ws_manager: BinanceWebsocketManager = None
log = structlog.get_logger("BinanceKlineApp")


def setup_logging():
    """配置应用程序范围的结构化日志系统。"""
    global log

    log_level_str = config.validated_log_level.upper()
    log_level_int = getattr(logging, log_level_str, logging.INFO)

    # 1. 配置 structlog 的核心处理器链
    # 这些处理器会按顺序处理来自 structlog 的日志事件。
    # 最后一个处理器 wrap_for_formatter 会将事件传递给标准库 logging。
    structlog_core_processors = [
        merge_contextvars,                # 合并上下文变量
        structlog.stdlib.add_logger_name,   # 添加 logger 名称 (例如: "BinanceKlineApp")
        add_stdlib_log_level,             # 添加日志级别 (例如: "info", "error")
        uppercase_level_processor,        # 将日志级别转为大写 (例如: "INFO", "ERROR")
        TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True), # 添加时间戳
        StackInfoRenderer(),              # 渲染堆栈信息 (主要用于异常)
        format_exc_info,                  # 格式化异常信息
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter, # **必须是最后一个**
    ]

    structlog.configure(
        processors=structlog_core_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 2. 为标准库 logging 的处理器 (handlers) 创建格式化器 (formatters)
    # 这些格式化器会处理来自 structlog (通过 wrap_for_formatter)
    # 以及直接来自标准库 logging 的日志记录。

    # `foreign_pre_chain` 用于处理非 structlog 产生的日志记录，
    # 使它们在最终渲染前看起来像 structlog 的事件字典。
    foreign_processors_chain = [
        structlog.stdlib.add_logger_name, # 从 record.name 添加 logger
        add_stdlib_log_level,             # 从 record.levelname 添加 level
        uppercase_level_processor,        # 将日志级别转为大写 (例如: "INFO", "ERROR")
        structlog.stdlib.ExtraAdder(),    # 添加 record.extra 中的额外字段
        # TimeStamper is not typically needed here as LogRecord has `created`
        # StackInfoRenderer and format_exc_info are also usually for structlog events,
        # stdlib handles its own exc_info formatting by default if not handled by ProcessorFormatter's processor.
        structlog.stdlib.ProcessorFormatter.remove_processors_meta, # 清理 _record, _logger 等内部键
    ]

    # 控制台格式化器
    console_formatter = ProcessorFormatter(
        # `processor` 参数用于将事件字典渲染为最终的字符串输出
        processor=ConsoleRenderer(colors=True, exception_formatter=better_traceback),
        foreign_pre_chain=foreign_processors_chain,
    )

    # 文件格式化器
    file_formatter = ProcessorFormatter(
        processor=ConsoleRenderer(colors=False, exception_formatter=better_traceback), # 无颜色输出到文件
        # processor=structlog.processors.JSONRenderer(), # 或者使用JSON格式输出到文件
        foreign_pre_chain=foreign_processors_chain,
    )

    # 3. 创建和配置标准库的处理器 (handlers)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level_int)

    file_handler = RotatingFileHandler(
        "binance_kline_app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level_int)

    # 4. 配置根 logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: # 清理已存在的处理器
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level_int) # 根 logger 的最低级别

    # 5. 调整其他库的日志级别
    for lib_logger_name in ["websocket", "urllib3", "requests", "chardet", "charset_normalizer"]:
        logging.getLogger(lib_logger_name).setLevel(logging.WARNING)

    # 6. 重新获取 logger 实例以确保它使用新配置
    log = structlog.get_logger("BinanceKlineApp")
# ... (rest of main_app.py remains the same as in the previous good version)
# _is_valid_kline, _process_kline_data, _handle_error, _update_kline_store,
# _refresh_display, websocket_message_handler, main_application
# Ensure main_application calls setup_logging() after bind_contextvars()

def _is_valid_kline(kline_payload):
    return (
        kline_payload and
        'x' in kline_payload and
        kline_payload['x']
    )

def _process_kline_data(kline_payload):
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
    log.error("WebSocket API错误", error_message=data.get('m', '未提供错误消息'))


def _update_kline_store(kline_dict, symbol_ws):
    global kline_persistence
    kline_persistence.add_single_kline(kline_dict)
    current_base_df = kline_persistence.get_klines_df()
    df_aggregated = aggregate_klines_df(current_base_df, kline_persistence.get_agg_interval_str())
    _refresh_display(df_aggregated, symbol_ws, current_base_df)

def _refresh_display(df_aggregated, symbol_ws, current_base_df):
    display_realtime_update(
        df_aggregated,
        symbol_ws,
        kline_persistence.get_agg_interval_str(),
        kline_persistence.get_base_interval_str(),
        current_base_df
    )

def websocket_message_handler(_, message_str: str):
    global kline_persistence

    if not kline_persistence:
        log.error("K线持久化服务未初始化", handler="websocket_message_handler")
        return

    try:
        data = json.loads(message_str)
        if 'result' in data and 'id' in data:
            log.debug("收到订阅确认", subscription_id=data['id'])
            return

        if 'e' in data and data['e'] == 'kline':
            kline_payload = data.get('k')
            if kline_payload and kline_payload.get('x'): 
                kline_dict = _process_kline_data(kline_payload)
                if kline_dict:
                    symbol_ws = kline_payload['s']
                    base_interval_stream = kline_payload['i']
                    log.debug( 
                        "收到已关闭K线(WebSocket)",
                        interval=base_interval_stream,
                        timestamp=kline_dict['timestamp'].isoformat(),
                    )
                    _update_kline_store(kline_dict, symbol_ws)
        elif 'e' in data and data['e'] == 'error':
            _handle_error(data)
        else:
            log.warning("未知WebSocket消息类型", raw_message=message_str[:200]) 

    except json.JSONDecodeError:
        log.error("WebSocket消息JSON解码失败", raw_message=message_str[:200])
    except Exception as e:
        log.error("websocket_message_handler中出错",
                  error=str(e),
                  exc_info=True,
                  original_message=message_str[:200])

def main_application():
    global kline_persistence, ws_manager, log

    clear_contextvars()
    bind_contextvars(
        symbol=config.SYMBOL,
        operating_mode=config.OPERATING_MODE
    )

    setup_logging() 

    log.info("日志系统初始化完毕", effective_log_level=config.validated_log_level.upper())
    log.info("配置加载完成",
             base_interval=config.BASE_INTERVAL,
             agg_interval=config.AGG_INTERVAL,
             data_store=config.DATA_STORE_TYPE,
             log_level_setting=config.LOG_LEVEL) 
    log.debug("详细配置内容", config_json=config.model_dump_json(indent=2))

    log.info("应用程序启动", data_store_type=config.DATA_STORE_TYPE)

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED

    requests_session = requests.Session()
    requests_session.verify = certifi.where()

    try:
        if config.DATA_STORE_TYPE == "database":
            log.info("使用数据库存储", storage_type="PostgreSQL")
            kline_persistence = DBManager(
                symbol=config.SYMBOL,
                base_interval_str=config.BASE_INTERVAL,
                agg_interval_str=config.AGG_INTERVAL,
                historical_candles_to_display_count=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
        else:
            log.info("使用内存存储", storage_type="memory")
            kline_persistence = KlineDataStore(
                base_interval_str=config.BASE_INTERVAL,
                agg_interval_str=config.AGG_INTERVAL,
                historical_candles_to_display_count=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
    except Exception as e_store_init:
        log.error("初始化数据持久化服务失败", error=str(e_store_init), exc_info=True)
        return

    log.debug("数据存储初始化完成", storage_class=type(kline_persistence).__name__)

    try:
        base_td = pd.Timedelta(config.BASE_INTERVAL)
        agg_td = pd.Timedelta(config.AGG_INTERVAL)
        if base_td.total_seconds() <= 0 or agg_td.total_seconds() <= 0:
            raise ValueError("基础间隔或聚合间隔必须是正持续时间。")
        if agg_td.total_seconds() < base_td.total_seconds():
             raise ValueError("聚合间隔必须大于或等于基础间隔。")
        base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()
        if base_intervals_per_agg <= 0:
            raise ValueError("计算的 base_intervals_per_agg 无效。")
    except Exception as e_calc:
        log.error("计算间隔比率时出错", error=str(e_calc), exc_info=True)
        return

    num_base_klines_needed = int((config.HISTORICAL_AGG_CANDLES_TO_DISPLAY + 20) * base_intervals_per_agg) 
    log.debug("计算历史K线需求",
             klines_to_fetch=num_base_klines_needed,
             for_agg_candles=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY,
             base_interval=config.BASE_INTERVAL)

    historical_klines_list = fetch_historical_klines(
        symbol=config.SYMBOL,
        interval=config.BASE_INTERVAL,
        num_klines_to_fetch=num_base_klines_needed,
        api_base_url=config.API_BASE_URL_FUTURES,
        max_limit_per_request=config.MAX_KLINE_LIMIT_PER_REQUEST,
        session=requests_session
    )

    if historical_klines_list:
        kline_persistence.add_klines(historical_klines_list)
        log.info("历史基础K线已获取并存储", count=len(historical_klines_list))

        base_df_for_initial_agg = kline_persistence.get_klines_df()
        if not base_df_for_initial_agg.empty:
            min_ts = base_df_for_initial_agg['timestamp'].min().isoformat()
            max_ts = base_df_for_initial_agg['timestamp'].max().isoformat()
            log.debug("用于初始聚合的基础K线范围", count=len(base_df_for_initial_agg), min_timestamp=min_ts, max_timestamp=max_ts)

            initial_agg_df = aggregate_klines_df(base_df_for_initial_agg, config.AGG_INTERVAL)
            display_historical_aggregated_klines(
                initial_agg_df,
                config.SYMBOL,
                config.AGG_INTERVAL,
                config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
        else:
            log.warning("获取历史数据后，存储中没有基础K线用于初始聚合")
    else:
        log.warning("未获取到历史K线，将仅使用实时数据")

    ws_manager = BinanceWebsocketManager(
        symbol=config.SYMBOL,
        base_interval=config.BASE_INTERVAL,
        on_message_callback=websocket_message_handler,
        ssl_context=ssl_context
    )
    ws_manager.start()
    log.info("WebSocket管理器已启动", stream_symbol=config.SYMBOL.lower(), stream_interval=config.BASE_INTERVAL)

    try:
        while True:
            time.sleep(30)
            if ws_manager and not ws_manager.is_active():
                log.warning("WebSocket管理器报告未激活，尝试重启...")
                ws_manager.stop()
                time.sleep(5)
                ws_manager.start()
                if not ws_manager.is_active():
                    log.error("WebSocket重启失败，应用程序即将终止")
                    break
                else:
                    log.info("WebSocket管理器已成功重启")
            elif not ws_manager:
                log.error("WebSocket管理器未初始化，应用程序即将终止")
                break
    except KeyboardInterrupt:
        log.info("检测到键盘中断 (Ctrl+C)，正在关闭...")
    except Exception as e:
        log.error("主循环中发生意外错误", error=str(e), exc_info=True)
    finally:
        log.info("开始应用程序关闭流程...")
        if ws_manager:
            log.debug("正在停止WebSocket管理器...")
            ws_manager.stop()
            log.info("WebSocket管理器已停止")
        if hasattr(kline_persistence, 'close') and callable(getattr(kline_persistence, 'close')):
            log.debug("正在关闭数据持久化连接...")
            kline_persistence.close()
            log.info("数据持久化连接已关闭")
        log.info("应用程序已终止")
        clear_contextvars()