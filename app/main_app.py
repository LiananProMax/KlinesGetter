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
from datetime import datetime, timezone # 确保导入 datetime
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
from app.data_handling.db_manager import DBManager # 确保 DBManager 已导入
from app.data_handling.kline_aggregator import aggregate_klines_df

from app.api_clients.binance_api_manager import fetch_historical_klines, BinanceWebsocketManager
from app.ui.display_manager import display_historical_aggregated_klines, display_realtime_update


# --- 全局实例 ---
kline_persistence: KlinePersistenceInterface = None
ws_manager: BinanceWebsocketManager = None
log = structlog.get_logger("BinanceKlineApp")


def _ensure_record_types_for_db(records_list: list, context_msg: str) -> list:
    """确保记录列表中的时间戳是datetime对象，数值是float，供数据库使用。"""
    valid_records = []
    if not records_list:
        return []
        
    for record in records_list:
        # 检查并转换 timestamp
        ts_val = record.get('timestamp')
        if isinstance(ts_val, pd.Timestamp):
            record['timestamp'] = ts_val.to_pydatetime()
        elif not isinstance(ts_val, datetime):
            log.error(f"{context_msg}: 聚合记录中的时间戳类型不正确，将跳过此记录",
                      timestamp_val=ts_val, type_val=type(ts_val), record_data=str(record)[:200])
            continue
        
        # 检查并转换数值字段
        valid_record = True
        for key in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            if key in record and record[key] is not None:
                try:
                    record[key] = float(record[key])
                except (ValueError, TypeError) as e_float:
                    log.error(f"{context_msg}: 聚合记录中字段 '{key}' 转换为float失败，将跳过此记录",
                              value=record[key], error=str(e_float), record_data=str(record)[:200])
                    valid_record = False
                    break # 当前记录无效
        
        if valid_record:
            valid_records.append(record)
            
    return valid_records


def setup_logging():
    """配置应用程序范围的结构化日志系统。"""
    global log

    log_level_str = config.validated_log_level.upper()
    log_level_int = getattr(logging, log_level_str, logging.INFO)

    structlog_core_processors = [
        merge_contextvars,
        structlog.stdlib.add_logger_name,
        add_stdlib_log_level,
        uppercase_level_processor,
        TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
        StackInfoRenderer(),
        format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        processors=structlog_core_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    foreign_processors_chain = [
        structlog.stdlib.add_logger_name,
        add_stdlib_log_level,
        uppercase_level_processor,
        structlog.stdlib.ExtraAdder(),
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ]

    console_formatter = ProcessorFormatter(
        processor=ConsoleRenderer(colors=True, exception_formatter=better_traceback),
        foreign_pre_chain=foreign_processors_chain,
    )

    file_formatter = ProcessorFormatter(
        processor=ConsoleRenderer(colors=False, exception_formatter=better_traceback),
        foreign_pre_chain=foreign_processors_chain,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level_int)

    file_handler = RotatingFileHandler(
        "binance_kline_app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level_int)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level_int)

    for lib_logger_name in ["websocket", "urllib3", "requests", "chardet", "charset_normalizer", "psycopg2"]:
        logging.getLogger(lib_logger_name).setLevel(logging.WARNING)

    log = structlog.get_logger("BinanceKlineApp")


def _is_valid_kline(kline_payload):
    return (
        kline_payload and
        'x' in kline_payload and # 'x' key indicates if the kline is closed
        kline_payload['x'] # The value of 'x' should be true for a closed kline
    )

def _process_kline_data(kline_payload):
    # No need to check _is_valid_kline here again if it's checked before calling
    return {
        'timestamp': pd.to_datetime(kline_payload['t'], unit='ms', utc=True), # Start time of the kline
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
    if not kline_persistence:
        log.error("K线持久化服务未初始化，无法更新", for_symbol=symbol_ws)
        return

    kline_persistence.add_single_kline(kline_dict)
    log.debug("基础K线已添加到持久化存储", timestamp=kline_dict['timestamp'].isoformat())

    current_base_df = kline_persistence.get_klines_df()
    if current_base_df.empty:
        log.warning("从持久化存储获取的基础K线数据为空，无法进行聚合", for_symbol=symbol_ws)
        _refresh_display(pd.DataFrame(), symbol_ws, current_base_df)
        return

    df_aggregated = aggregate_klines_df(current_base_df, kline_persistence.get_agg_interval_str())
    log.debug("基础K线已聚合", agg_count=len(df_aggregated), base_count_used=len(current_base_df))

    if not df_aggregated.empty and isinstance(kline_persistence, DBManager):
        aggregated_data_to_store_raw = df_aggregated.to_dict('records')
        # 使用辅助函数确保类型正确
        aggregated_data_to_store_final = _ensure_record_types_for_db(aggregated_data_to_store_raw, "实时聚合")

        if aggregated_data_to_store_final:
            log.info("准备存储实时聚合K线到数据库", count=len(aggregated_data_to_store_final), symbol=symbol_ws)
            try:
                kline_persistence.store_aggregated_data(aggregated_data_to_store_final)
                log.debug("实时聚合K线数据已提交到DBManager进行存储")
            except Exception as e_store_agg:
                log.error("存储实时聚合K线到数据库时出错", error=str(e_store_agg), exc_info=True)
        else:
            log.debug("没有有效的实时聚合数据需要存储到数据库", symbol=symbol_ws, raw_count=len(aggregated_data_to_store_raw))


    # 5. 刷新显示
    _refresh_display(df_aggregated, symbol_ws, current_base_df)

def _refresh_display(df_aggregated, symbol_ws, current_base_df):
    # 此函数现在只负责调用display_manager中的函数
    display_realtime_update(
        df_aggregated,
        symbol_ws,
        kline_persistence.get_agg_interval_str(), # 从持久化层获取聚合间隔
        kline_persistence.get_base_interval_str(),  # 从持久化层获取基础间隔
        current_base_df # 传递当前的基础数据副本
    )

def websocket_message_handler(_, message_str: str):
    global kline_persistence # 确保全局变量被正确引用

    if not kline_persistence: # 早期检查
        log.error("K线持久化服务未初始化", handler="websocket_message_handler")
        return

    try:
        data = json.loads(message_str)
        if 'result' in data and 'id' in data: # 订阅确认消息
            log.debug("收到订阅确认", subscription_id=data['id'])
            return

        if 'e' in data and data['e'] == 'kline':
            kline_payload = data.get('k')
            # 只有当K线关闭时才处理 ('x': true)
            if kline_payload and _is_valid_kline(kline_payload):
                kline_dict = _process_kline_data(kline_payload)
                if kline_dict:
                    symbol_ws = kline_payload['s'] # 交易对
                    base_interval_stream = kline_payload['i'] # K线间隔
                    log.debug( 
                        "收到已关闭的基础K线(WebSocket)",
                        symbol=symbol_ws,
                        interval=base_interval_stream,
                        timestamp=kline_dict['timestamp'].isoformat(), # 使用ISO格式的时间戳
                        # open=kline_dict['open'], high=kline_dict['high'],
                        # low=kline_dict['low'], close=kline_dict['close']
                    )
                    _update_kline_store(kline_dict, symbol_ws) # 调用更新和聚合逻辑
                else:
                    log.warning("处理后的K线数据为空", raw_payload=kline_payload)
            # else: # K线未关闭或payload无效，可以选择性记录日志
            #     if kline_payload and not kline_payload.get('x', False):
            #         log.trace("收到未关闭的K线更新(WebSocket)", symbol=kline_payload.get('s'), interval=kline_payload.get('i'))
            #     elif not kline_payload:
            #         log.warning("WebSocket kline事件中缺少kline_payload", raw_data=data)

        elif 'e' in data and data['e'] == 'error': # 错误消息
            _handle_error(data)
        else: # 未知消息类型
            log.warning("未知WebSocket消息类型", raw_message=message_str[:200]) # 截断长消息

    except json.JSONDecodeError:
        log.error("WebSocket消息JSON解码失败", raw_message=message_str[:200])
    except Exception as e: # 捕获其他潜在错误
        log.error("websocket_message_handler中出错",
                  error=str(e),
                  exc_info=True, # 自动添加堆栈信息
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
    # log.debug("详细配置内容", config_json=config.model_dump_json(indent=2)) # 避免过于详细的配置打印到标准输出

    log.info("应用程序启动", data_store_type=config.DATA_STORE_TYPE, symbol=config.SYMBOL)


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
        else: # 默认或 "memory"
            log.info("使用内存存储", storage_type="memory")
            kline_persistence = KlineDataStore(
                base_interval_str=config.BASE_INTERVAL,
                agg_interval_str=config.AGG_INTERVAL,
                historical_candles_to_display_count=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
    except Exception as e_store_init:
        log.critical("初始化数据持久化服务失败，应用程序将退出", error=str(e_store_init), exc_info=True)
        return # 关键组件失败，退出

    log.debug("数据存储初始化完成", storage_class=type(kline_persistence).__name__)

    try:
        base_td = pd.Timedelta(config.BASE_INTERVAL)
        agg_td = pd.Timedelta(config.AGG_INTERVAL)
        if base_td.total_seconds() <= 0 or agg_td.total_seconds() <= 0:
            raise ValueError("基础间隔或聚合间隔必须是正持续时间。")
        if agg_td.total_seconds() < base_td.total_seconds(): # 聚合间隔不能小于基础间隔
             raise ValueError("聚合间隔必须大于或等于基础间隔。")
        base_intervals_per_agg = agg_td.total_seconds() / base_td.total_seconds()
        if base_intervals_per_agg <= 0 : # 应该总是 >0 如果上面的检查通过了
            raise ValueError("计算的 base_intervals_per_agg 无效或为零。")
    except Exception as e_calc:
        log.critical("计算间隔比率时出错，应用程序将退出", error=str(e_calc), exc_info=True)
        return # 关键计算失败，退出

    # HISTORICAL_AGG_CANDLES_TO_DISPLAY 控制的是最终聚合后要展示的K线数量
    # 为了得到这么多聚合K线，我们需要获取更多的基础K线
    # 例如：展示50个3分钟K线，基础是1分钟，需要 50 * 3 = 150 个基础K线 (至少)
    # 加上一些缓冲，例如额外20个聚合周期的数据，以确保聚合边缘的准确性
    num_base_klines_needed = int((config.HISTORICAL_AGG_CANDLES_TO_DISPLAY + 20) * base_intervals_per_agg) 
    log.debug("计算历史K线需求",
             base_klines_to_fetch=num_base_klines_needed,
             target_agg_candles_display=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY,
             base_interval=config.BASE_INTERVAL,
             agg_interval=config.AGG_INTERVAL)


    historical_klines_list = fetch_historical_klines(
        symbol=config.SYMBOL,
        interval=config.BASE_INTERVAL,
        num_klines_to_fetch=num_base_klines_needed, # 获取计算出来的基础K线数量
        api_base_url=config.API_BASE_URL_FUTURES,
        max_limit_per_request=config.MAX_KLINE_LIMIT_PER_REQUEST,
        session=requests_session
    )

    if historical_klines_list:
        kline_persistence.add(historical_klines_list) # 使用接口的add方法
        log.info("历史基础K线已获取并存入持久化层", count=len(historical_klines_list))

        # 从持久化层获取数据进行初始聚合 (确保获取的是最新状态)
        base_df_for_initial_agg = kline_persistence.get_klines_df()
        if not base_df_for_initial_agg.empty:
            min_ts = base_df_for_initial_agg['timestamp'].min().isoformat()
            max_ts = base_df_for_initial_agg['timestamp'].max().isoformat()
            log.debug("用于初始聚合的基础K线范围", count=len(base_df_for_initial_agg), min_timestamp=min_ts, max_timestamp=max_ts)

            initial_agg_df = aggregate_klines_df(base_df_for_initial_agg, config.AGG_INTERVAL)
            log.info("初始聚合K线已生成", count=len(initial_agg_df))

            # 如果是数据库存储，将初始聚合数据存入数据库
            if not initial_agg_df.empty and isinstance(kline_persistence, DBManager):
                log.info("准备存储初始聚合K线到数据库", count=len(initial_agg_df))
                initial_agg_data_to_store_raw = initial_agg_df.to_dict('records')
                # 使用辅助函数确保类型正确
                initial_agg_data_to_store_final = _ensure_record_types_for_db(initial_agg_data_to_store_raw, "初始聚合")
                
                if initial_agg_data_to_store_final:
                    try:
                        kline_persistence.store_aggregated_data(initial_agg_data_to_store_final)
                        log.debug("初始聚合K线数据已存储到数据库")
                    except Exception as e_store_hist_agg:
                        log.error("存储历史聚合K线到数据库时出错", error=str(e_store_hist_agg), exc_info=True)
                else:
                    log.warning("没有有效的初始聚合数据可存储到数据库。", raw_count=len(initial_agg_data_to_store_raw))


            display_historical_aggregated_klines(
                initial_agg_df,
                config.SYMBOL,
                config.AGG_INTERVAL,
                config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
        else:
            log.warning("获取历史数据后，持久化存储中没有基础K线用于初始聚合")
    else:
        log.warning("未获取到历史K线，将仅使用实时数据（如果有）")


    ws_manager = BinanceWebsocketManager(
        symbol=config.SYMBOL,
        base_interval=config.BASE_INTERVAL, # WebSocket订阅基础间隔
        on_message_callback=websocket_message_handler,
        ssl_context=ssl_context # 传递SSL上下文
    )
    ws_manager.start()
    log.info("WebSocket管理器已启动", stream_symbol=config.SYMBOL.lower(), stream_interval=config.BASE_INTERVAL)


    try:
        while True:
            time.sleep(30) # 主线程的监控周期
            if ws_manager and not ws_manager.is_active():
                log.warning("WebSocket管理器报告未激活，尝试重启...")
                ws_manager.stop() # 确保先停止
                time.sleep(5) # 等待资源释放
                # 重新创建SSL上下文可能更安全，如果之前的已损坏
                new_ssl_context = ssl.create_default_context(cafile=certifi.where())
                new_ssl_context.check_hostname = True
                new_ssl_context.verify_mode = ssl.CERT_REQUIRED
                ws_manager.ssl_context = new_ssl_context # 更新SSL上下文
                ws_manager.start() # 尝试重启
                if not ws_manager.is_active():
                    log.error("WebSocket重启失败，应用程序即将终止")
                    break # 重启失败，退出主循环
                else:
                    log.info("WebSocket管理器已成功重启")
            elif not ws_manager: # ws_manager 实例丢失 (理论上不应发生)
                log.critical("WebSocket管理器未初始化或已丢失，应用程序即将终止")
                break
    except KeyboardInterrupt:
        log.info("检测到键盘中断 (Ctrl+C)，正在关闭...")
    except Exception as e: # 捕获主循环中的其他意外错误
        log.error("主循环中发生意外错误", error=str(e), exc_info=True)
    finally:
        log.info("开始应用程序关闭流程...")
        if ws_manager:
            log.debug("正在停止WebSocket管理器...")
            ws_manager.stop()
            log.info("WebSocket管理器已停止")
        
        # 如果持久化层有 close 方法 (例如DBManager)
        if hasattr(kline_persistence, 'close') and callable(getattr(kline_persistence, 'close')):
            log.debug("正在关闭数据持久化连接...")
            kline_persistence.close()
            log.info("数据持久化连接已关闭")
        
        log.info("应用程序已终止")
        clear_contextvars() # 清理上下文变量