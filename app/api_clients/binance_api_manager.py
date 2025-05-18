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

def fetch_historical_klines(symbol, interval, num_klines_to_fetch, api_base_url, max_limit_per_request, session=None):
    """从币安期货REST API获取历史K线。"""
    log = structlog.get_logger()
    klines_fetched_so_far = 0
    all_klines_raw_list = []
    current_end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    initial_requested_end_time_iso = pd.to_datetime(current_end_time_ms, unit='ms', utc=True).isoformat()
    log.debug(f"开始获取{symbol}：{num_klines_to_fetch}个'{interval}'K线。"
              f"初始结束时间：{initial_requested_end_time_iso}")

    while klines_fetched_so_far < num_klines_to_fetch:
        remaining_klines = num_klines_to_fetch - klines_fetched_so_far
        current_batch_limit = min(remaining_klines, max_limit_per_request)
        params = {
            'symbol': symbol, 'interval': interval,
            'limit': current_batch_limit, 'endTime': current_end_time_ms
        }
        request_end_time_iso = pd.to_datetime(current_end_time_ms, unit='ms', utc=True).isoformat()
        log.debug(f"获取{symbol}批次：{current_batch_limit}个'{interval}'K线，结束时间：{request_end_time_iso}")
        try:
            if session is None:
                session = requests.Session()
                session.verify = certifi.where()
            response = session.get(f"{api_base_url}/fapi/v1/klines", params=params)
            response.raise_for_status()
            data_batch_raw = response.json()
            if not data_batch_raw:
                log.debug("此期间内没有更多的历史数据。")
                break

            first_k_ts = pd.to_datetime(data_batch_raw[0][0], unit='ms', utc=True).isoformat()
            last_k_ts = pd.to_datetime(data_batch_raw[-1][0], unit='ms', utc=True).isoformat()
            log.debug(f"  收到批次：{len(data_batch_raw)}个K线。从{first_k_ts}到{last_k_ts}")

            all_klines_raw_list = data_batch_raw + all_klines_raw_list
            klines_fetched_so_far += len(data_batch_raw)
            if len(data_batch_raw) < current_batch_limit:
                log.debug(f"  获取了{len(data_batch_raw)}个K线，少于请求的{current_batch_limit}。假设没有更旧的数据。")
                break
            current_end_time_ms = data_batch_raw[0][0]
            time.sleep(0.25)
        except requests.exceptions.RequestException as e:
            log.error(f"获取历史K线时出错：{e}")
            return []
        except json.JSONDecodeError as e:
            log.error(f"从历史K线响应解码JSON时出错：{e}")
            if response: log.error(f"响应文本：{response.text}")
            return []

    formatted_klines = [format_kline_from_api(k) for k in all_klines_raw_list]
    log.debug(f"{symbol}总共格式化{len(formatted_klines)}个历史K线。")
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
