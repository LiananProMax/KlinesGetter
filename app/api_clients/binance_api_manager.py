#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import time
import json
import logging
import pandas as pd
from datetime import datetime, timezone
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from app.utils.kline_utils import format_kline_from_api # 从本地模块导入

def fetch_historical_klines(symbol, interval, num_klines_to_fetch, api_base_url, max_limit_per_request):
    """从币安期货REST API获取历史K线。"""
    klines_fetched_so_far = 0
    all_klines_raw_list = []
    current_end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    initial_requested_end_time_iso = pd.to_datetime(current_end_time_ms, unit='ms', utc=True).isoformat()
    logging.info(f"开始获取{symbol}：{num_klines_to_fetch}个'{interval}'K线。"
                 f"初始结束时间：{initial_requested_end_time_iso}")

    # formatted_klines = [] # 此变量在原始版本中在循环前初始化但未使用
    while klines_fetched_so_far < num_klines_to_fetch:
        remaining_klines = num_klines_to_fetch - klines_fetched_so_far
        current_batch_limit = min(remaining_klines, max_limit_per_request)
        params = {
            'symbol': symbol, 'interval': interval,
            'limit': current_batch_limit, 'endTime': current_end_time_ms
        }
        request_end_time_iso = pd.to_datetime(current_end_time_ms, unit='ms', utc=True).isoformat()
        logging.info(f"获取{symbol}批次：{current_batch_limit}个'{interval}'K线，结束时间：{request_end_time_iso}")
        try:
            response = requests.get(f"{api_base_url}/fapi/v1/klines", params=params)
            response.raise_for_status()
            data_batch_raw = response.json()
            if not data_batch_raw:
                logging.info("此期间内没有更多的历史数据。")
                break
            
            first_k_ts = pd.to_datetime(data_batch_raw[0][0], unit='ms', utc=True).isoformat()
            last_k_ts = pd.to_datetime(data_batch_raw[-1][0], unit='ms', utc=True).isoformat()
            logging.info(f"  收到批次：{len(data_batch_raw)}个K线。从{first_k_ts}到{last_k_ts}")

            all_klines_raw_list = data_batch_raw + all_klines_raw_list # 前置以保持顺序
            klines_fetched_so_far += len(data_batch_raw)
            if len(data_batch_raw) < current_batch_limit:
                logging.info(f"  获取了{len(data_batch_raw)}个K线，少于请求的{current_batch_limit}。假设没有更旧的数据。")
                break
            current_end_time_ms = data_batch_raw[0][0] # 下一批的最旧K线开盘时间
            time.sleep(0.25) # 尊重API速率限制
        except requests.exceptions.RequestException as e:
            logging.error(f"获取历史K线时出错：{e}")
            return [] # 如果出错则返回空
        except json.JSONDecodeError as e:
            logging.error(f"从历史K线响应解码JSON时出错：{e}")
            if response: logging.error(f"响应文本：{response.text}")
            return []
            
    formatted_klines = [format_kline_from_api(k) for k in all_klines_raw_list]
    logging.info(f"{symbol}总共格式化{len(formatted_klines)}个历史K线。")
    return formatted_klines

class BinanceWebsocketManager:
    def __init__(self, symbol, base_interval, on_message_callback):
        self.symbol = symbol.lower() # WebSocket API通常偏好小写
        self.base_interval = base_interval
        self.on_message_callback = on_message_callback
        self.ws_client = None
        logging.info(f"为{self.symbol}@{self.base_interval}初始化BinanceWebsocketManager")

    def start(self):
        if self.ws_client:
            logging.warning("WebSocket客户端已经启动。如果需要重启，请先调用stop()。")
            return
        
        self.ws_client = UMFuturesWebsocketClient(on_message=self.on_message_callback)
        logging.info(f"正在连接到{self.symbol}@{self.base_interval}_kline流的WebSocket...")
        self.ws_client.kline(
            symbol=self.symbol,
            id=1, # 订阅请求的唯一ID
            interval=self.base_interval
        )
        logging.info(f"通过WebSocket订阅了{self.symbol}@{self.base_interval}_kline流。")

    def stop(self):
        if self.ws_client:
            logging.info("正在停止WebSocket客户端...")
            if hasattr(self.ws_client, 'stop') and callable(getattr(self.ws_client, 'stop')):
                self.ws_client.stop() # 这应该正确关闭websocket连接
                logging.info("已发出WebSocket客户端停止命令。")
            else:
                logging.warning("WebSocket客户端没有可调用的'stop'方法。")
            self.ws_client = None 
        else:
            logging.info("WebSocket客户端未运行或已停止。")

    def is_active(self):
        if not self.ws_client:
            return False
        try:
            # 基于底层websocket-client库的WebSocketApp状态检查
            if hasattr(self.ws_client, 'ws_connection') and \
               self.ws_client.ws_connection and \
               hasattr(self.ws_client.ws_connection, 'keep_running'):
                return self.ws_client.ws_connection.keep_running
            else:
                logging.debug("无法通过ws_connection.keep_running确定WebSocket活动状态。可能正在连接中。")
                # 如果ws_client存在但状态无法确定，假设它正在尝试连接或已连接。
                # 这种状态很棘手；新创建的客户端可能尚未完全初始化ws_connection。
                return True # 乐观：如果客户端存在且没有明确的"已停止"状态，假设活动/尝试中。
        except Exception as e:
            logging.error(f"检查WebSocket活动状态时出错：{e}")
            return False
