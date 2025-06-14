#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import time
import json
import structlog
import pandas as pd
from datetime import datetime, timezone
from websocket import WebSocketApp
import threading
from app.utils.kline_utils import format_kline_from_api
import certifi
import ssl

log_fetch = structlog.get_logger(__name__) # 使用模块级logger，避免在函数内部反复获取

def fetch_historical_klines(symbol, interval, num_klines_to_fetch, api_base_url, max_limit_per_request, session=None, max_retries=3, base_delay=1.0):
    """从币安期货REST API获取历史K线。
    
    Args:
        symbol: 交易对符号
        interval: K线间隔
        num_klines_to_fetch: 要获取的K线数量
        api_base_url: API基础URL
        max_limit_per_request: 每次请求的最大限制
        session: requests会话对象
        max_retries: 最大重试次数
        base_delay: 基础重试延迟时间（秒）
    """
    # log = structlog.get_logger() # 改用模块级 logger: log_fetch
    klines_fetched_so_far = 0
    all_klines_raw_list = []
    current_end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    initial_requested_end_time_iso = pd.to_datetime(current_end_time_ms, unit='ms', utc=True).isoformat()
    log_fetch.debug(f"开始获取{symbol}：{num_klines_to_fetch}个'{interval}'K线。"
                    f"初始结束时间：{initial_requested_end_time_iso}")

    while klines_fetched_so_far < num_klines_to_fetch:
        remaining_klines = num_klines_to_fetch - klines_fetched_so_far
        current_batch_limit = min(remaining_klines, max_limit_per_request)
        params = {
            'symbol': symbol, 'interval': interval,
            'limit': current_batch_limit, 'endTime': current_end_time_ms
        }
        request_end_time_iso = pd.to_datetime(current_end_time_ms, unit='ms', utc=True).isoformat()
        log_fetch.debug(f"获取{symbol}批次：{current_batch_limit}个'{interval}'K线，结束时间：{request_end_time_iso}")
        
        # 实现重试机制
        retry_count = 0
        data_batch_raw = None
        
        while retry_count <= max_retries:
            try:
                if session is None:
                    session = requests.Session()
                    session.verify = certifi.where()
                response = session.get(f"{api_base_url}/fapi/v1/klines", params=params, timeout=30)
                response.raise_for_status()
                data_batch_raw = response.json()
                break  # 成功获取数据，退出重试循环
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count <= max_retries:
                    delay = base_delay * (2 ** (retry_count - 1))  # 指数退避
                    log_fetch.warning(f"获取历史K线时出错（第{retry_count}次重试）：{e}，"
                                     f"{delay:.1f}秒后重试...")
                    time.sleep(delay)
                else:
                    log_fetch.error(f"获取历史K线失败，已达最大重试次数{max_retries}：{e}")
                    return []
                    
            except json.JSONDecodeError as e:
                retry_count += 1
                if retry_count <= max_retries:
                    delay = base_delay * (2 ** (retry_count - 1))  # 指数退避
                    log_fetch.warning(f"从历史K线响应解码JSON时出错（第{retry_count}次重试）：{e}，"
                                     f"{delay:.1f}秒后重试...")
                    if 'response' in locals(): 
                        log_fetch.debug(f"响应文本：{response.text[:200]}...")
                    time.sleep(delay)
                else:
                    log_fetch.error(f"JSON解码失败，已达最大重试次数{max_retries}：{e}")
                    if 'response' in locals(): 
                        log_fetch.error(f"响应文本：{response.text}")
                    return []
        
        # 检查是否成功获取到数据
        if data_batch_raw is None:
            log_fetch.error("重试后仍未能获取到数据")
            return []
            
        if not data_batch_raw:
            log_fetch.debug("此期间内没有更多的历史数据。")
            break

        first_k_ts = pd.to_datetime(data_batch_raw[0][0], unit='ms', utc=True).isoformat()
        last_k_ts = pd.to_datetime(data_batch_raw[-1][0], unit='ms', utc=True).isoformat()
        log_fetch.debug(f"  收到批次：{len(data_batch_raw)}个K线。从{first_k_ts}到{last_k_ts}")

        all_klines_raw_list = data_batch_raw + all_klines_raw_list
        klines_fetched_so_far += len(data_batch_raw)
        if len(data_batch_raw) < current_batch_limit:
            log_fetch.debug(f"  获取了{len(data_batch_raw)}个K线，少于请求的{current_batch_limit}。假设没有更旧的数据。")
            break
        current_end_time_ms = data_batch_raw[0][0]
        time.sleep(0.25) # 保持礼谐的请求间隔

    # formatted_klines = [format_kline_from_api(k) for k in all_klines_raw_list] # 移动到去重逆辑中
    
    # --- 新增：去重处理 ---
    if all_klines_raw_list:
        # 1. 先格式化所有获取到的原始K线数据
        initial_formatted_klines = []
        for k_raw in all_klines_raw_list:
            try:
                initial_formatted_klines.append(format_kline_from_api(k_raw))
            except Exception as e_fmt_single:
                log_fetch.error("格式化单个原始K线数据时出错 (去重前)", 
                                kline_data_snippet=str(k_raw)[:60], error=str(e_fmt_single))
                continue # 跳过无法格式化的数据

        if not initial_formatted_klines:
            log_fetch.warning(f"{symbol}：所有原始K线数据都无法格式化，返回空列表。")
            return []

        # 2. 使用Pandas DataFrame进行去重
        #    确保 timestamp 是 datetime 对象，并且已设为 UTC (format_kline_from_api 保证了这点)
        df_temp = pd.DataFrame(initial_formatted_klines)
        
        # 按时间戳排序 (升序), 然后对于重复的时间戳，保留最后出现的那个。
        # 币安API通常按时间倒序返回数据，我们是 data_batch_raw + all_klines_raw_list，
        # 所以越新的批次（时间越晚）在 all_klines_raw_list 的越前面。
        # 如果要保留"最新获取"的（通常意味着数据更可靠或最终），
        # 排序后用 keep='last' (对于相同时间戳，保留排序后的最后一个，即最新的)
        # 或者如果信任原始顺序，先反转列表，再 drop_duplicates(keep='first')
        # 这里我们假设 format_kline_from_api 产生的 'timestamp' 是可比较的
        df_temp = df_temp.sort_values(by='timestamp', ascending=True)
        df_temp = df_temp.drop_duplicates(subset=['timestamp'], keep='last')
        
        formatted_klines = df_temp.to_dict('records')
        log_fetch.debug(f"{symbol}：原始获取 {len(all_klines_raw_list)} 条，"
                        f"初步格式化 {len(initial_formatted_klines)} 条，"
                        f"最终去重后剩 {len(formatted_klines)} 条历史K线。")
    else:
        formatted_klines = []
        log_fetch.debug(f"{symbol}：没有获取到任何原始K线数据。")
        
    # log_fetch.debug(f"{symbol}总共格式化{len(formatted_klines)}个历史K线。") # 这句被上面的日志替代
    return formatted_klines

class BinanceWebsocketManager:
    def __init__(self, symbol, base_interval, on_message_callback, ssl_context=None):
        self.symbol = symbol.lower()
        self.base_interval = base_interval
        self.on_message_callback = on_message_callback
        self.ssl_context = ssl_context or ssl.create_default_context(cafile=certifi.where())
        self.ws = None
        self.ws_thread = None
        log = structlog.get_logger()
        log.debug(f"为{self.symbol}@{self.base_interval}初始化BinanceWebsocketManager")

    def start(self):
        log = structlog.get_logger()
        if self.ws:
            log.warning("WebSocket客户端已经启动。如果需要重启，请先调用stop()。")
            return

        def on_open(ws):
            subscription_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{self.symbol}@kline_{self.base_interval}"],
                "id": 1
            }
            ws.send(json.dumps(subscription_msg))
            log.debug(f"WebSocket连接已打开并订阅了{self.symbol}@{self.base_interval}_kline流。")

        def on_close(ws, close_status_code, close_msg):
            log.debug(f"WebSocket连接已关闭：{close_status_code} - {close_msg}")

        def on_error(ws, error):
            log.error(f"WebSocket错误：{error}")

        self.ws = WebSocketApp(
            "wss://fstream.binance.com/ws",
            on_message=self.on_message_callback,
            on_open=on_open,
            on_close=on_close,
            on_error=on_error
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={"sslopt": {"context": self.ssl_context}})
        self.ws_thread.start()
        log.debug(f"WebSocket线程已启动。")

    def stop(self):
        log = structlog.get_logger()
        if self.ws:
            self.ws.close()
            self.ws_thread.join()
            self.ws = None
            self.ws_thread = None
            log.debug("WebSocket客户端已停止。")
        else:
            log.debug("WebSocket客户端未运行或已停止。")

    def is_active(self):
        return self.ws is not None and self.ws.sock is not None and self.ws.sock.connected
