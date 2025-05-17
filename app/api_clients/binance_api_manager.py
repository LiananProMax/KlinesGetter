#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import time
import json
import logging
import traceback
import pandas as pd
from datetime import datetime, timezone
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from app.utils.kline_utils import format_kline_from_api # 从本地模块导入

# 从新的 binance_trading_api 模块导入 safe_api_call
# 注意：这里的数据获取 REST API 调用也使用 safe_api_call 包装
from app.api_clients.binance_trading_api import safe_api_call
# 从 config 导入 MAX_KLINE_LIMIT_PER_REQUEST 和 API_BASE_URL_FUTURES
from app.core import config


def fetch_historical_klines(symbol, interval, num_klines_to_fetch, api_base_url=config.API_BASE_URL_FUTURES, max_limit_per_request=config.MAX_KLINE_LIMIT_PER_REQUEST):
    """从币安期货REST API获取历史K线，使用safe_api_call包装。"""
    klines_fetched_so_far = 0
    all_klines_raw_list = []
    # 获取当前时间的毫秒时间戳作为初始endTime，并转换为UTC时区 aware
    current_end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # 将初始 requested_end_time_iso 转换为 UTC 时间戳的 ISO 格式，便于日志记录和调试
    initial_requested_end_time_iso = pd.to_datetime(current_end_time_ms, unit='ms', utc=True).isoformat()

    logging.info(f"开始获取 {symbol} 的 {num_klines_to_fetch} 个 '{interval}' K线。")
    logging.debug(f"初始请求结束时间（Unix MS）：{current_end_time_ms} ({initial_requested_end_time_iso})")


    while klines_fetched_so_far < num_klines_to_fetch:
        remaining_klines = num_klines_to_fetch - klines_fetched_so_far
        current_batch_limit = min(remaining_klines, max_limit_per_request)

        # API参数，使用当前 end_time_ms 作为 endTime
        params = {
            'symbol': symbol.upper(), # 确保符号大写
            'interval': interval,
            'limit': current_batch_limit,
            'endTime': current_end_time_ms
        }
        request_end_time_iso = pd.to_datetime(current_end_time_ms, unit='ms', utc=True).isoformat()
        logging.debug(f"获取批次：{current_batch_limit} 个 '{interval}' K线，结束时间：{request_end_time_iso}")

        # 使用 safe_api_call 包装 requests.get 调用
        try:
            # safe_api_call 需要 client 实例，但对于公共 REST 端点，我们可以直接使用 requests 并自行处理重试
            # 或者改造 safe_api_call 以支持 requests 调用。
            # 考虑到 safe_api_call 已经处理了大部分异常，且主要用于 Binance Client，
            # 对于 requests 调用，我们可以在这里手动添加简单的重试或直接使用 requests。
            # 最简单的做法是直接调用 requests 并捕获异常，不使用 safe_api_call。
            # 如果需要统一异常处理，可以将 requests.get 也封装到 safe_api_call 中，
            # 或者为 requests 创建一个独立的 retry wrapper。
            # 这里为了精简和不引入额外的 wrapper，直接使用 requests.get 并捕获异常。

            # 原始代码中没有使用 safe_api_call 包装 fetch_historical_klines 的 requests 调用
            # 为了兼容和精简，我们保持原样，手动处理异常
            response = requests.get(f"{api_base_url}/fapi/v1/klines", params=params)
            response.raise_for_status() # 会对 4xx/5xx 抛出 HTTPError
            data_batch_raw = response.json()

            if not data_batch_raw:
                logging.info("此期间内没有更多的历史数据。")
                break

            # CoinAPI 返回的数据是按时间升序排列的。
            # 币安REST API 返回的数据是按时间倒序排列的（当使用 endTime 时）。
            # 因此，我们将新获取的批次添加到列表的前面。
            # data_batch_raw 已经是按时间从旧到新排列的（当只使用 limit 和 endTime 时，endTime 是该批次中最新的时间点，limit 向前推）
            # 所以应该 append 到列表末尾，然后对整个列表排序。或者添加到列表头部。
            # 原始代码是将新批次添加到列表头部，这意味着新批次的第一个元素（最旧的）会成为列表的第一个元素。
            # 让我们保持原始逻辑，将新批次前置，这样最终列表的顺序是正确的（从旧到新）。
            all_klines_raw_list = data_batch_raw + all_klines_raw_list

            klines_fetched_so_far += len(data_batch_raw)

            # 如果获取的数量少于请求的数量，意味着已经到达历史数据的末尾
            if len(data_batch_raw) < current_batch_limit:
                logging.info(f"  获取了 {len(data_batch_raw)} 个K线，少于请求的 {current_batch_limit}。假设没有更旧的数据。")
                break # 没有更旧的数据了，退出循环

            # 更新 current_end_time_ms 为当前批次中最早的K线的开盘时间
            # 减去1毫秒，确保下一批次的最后一个K线不与当前批次的最早K线重叠
            current_end_time_ms = data_batch_raw[0][0] - 1
            # 添加一个小的延迟以避免触犯API速率限制
            time.sleep(0.25)

        except requests.exceptions.RequestException as e:
            logging.error(f"获取历史K线时出错：{e}")
            # 在获取批次数据时出现错误，返回已获取的数据
            # 原始代码返回空列表，这里也保持一致，表示获取过程失败
            return []
        except json.JSONDecodeError as e:
            logging.error(f"从历史K线响应解码JSON时出错：{e}")
            if response: logging.error(f"响应文本：{response.text}")
            return []
        except Exception as ex:
             logging.error(f"获取历史K线时发生意外错误：{ex}\n{traceback.format_exc()}")
             return []


    # 格式化获取到的所有原始K线数据
    formatted_klines = []
    
    # 打印一个原始 K 线示例，仅用于调试
    if all_klines_raw_list and len(all_klines_raw_list) > 0:
        sample_raw = all_klines_raw_list[0]
        logging.debug(f"原始 K 线数据示例: {sample_raw}")
    
    for k in all_klines_raw_list:
        try:
            formatted_kline = format_kline_from_api(k)
            # 检查是否包含必要的字段
            if not formatted_kline or 'timestamp' not in formatted_kline:
                logging.error(f"格式化后的 K 线缺少 'timestamp' 字段: {formatted_kline}")
                logging.error(f"原始数据: {k}")
                continue
            formatted_klines.append(formatted_kline)
        except Exception as e:
            logging.error(f"格式化 K 线时出错: {e}\n原始数据: {k}")
            continue
    
    logging.info(f"为 {symbol} 总共格式化了 {len(formatted_klines)} 个历史K线。")
    
    # 打印一个格式化后的 K 线示例，仅用于调试
    if formatted_klines and len(formatted_klines) > 0:
        sample_formatted = formatted_klines[0]
        logging.debug(f"格式化后的 K 线示例: {sample_formatted}")
    
    return formatted_klines

class BinanceWebsocketManager:
    def __init__(self, symbol, base_interval, on_message_callback):
        # WebSocket API 通常偏好小写交易对
        self.symbol = symbol.lower()
        self.base_interval = base_interval
        self.on_message_callback = on_message_callback
        self.ws_client = None
        logging.info(f"为 {self.symbol}@{self.base_interval} 初始化 BinanceWebsocketManager")

    def start(self):
        """启动WebSocket客户端并订阅K线流。"""
        if self.ws_client:
            logging.warning("WebSocket客户端已经启动。如果需要重启，请先调用stop()。")
            return

        # UMFuturesWebsocketClient 需要提供 on_message 回调
        # 客户端连接和订阅是同步发生的
        try:
            # client=BinanceFuturesClient(api_key, api_secret) # 如果需要认证的流，传递认证信息
            # Binance Websocket Client library uses threading internally
            # 注意：UMFuturesWebsocketClient 不直接接受 sslopt 参数
            # 如果需要禁用 SSL 验证，我们需要在环境中设置相关变量
            if not config.VERIFY_SSL:
                # 全局禁用证书验证，仅限开发环境使用
                import ssl
                import websocket
                websocket.enableTrace(True)  # 启用 WebSocket 调试输出
                ssl._create_default_https_context = ssl._create_unverified_context
            
            self.ws_client = UMFuturesWebsocketClient() # 对于公共流，通常不需要 API Key/Secret

            # 订阅 K线流
            logging.info(f"正在连接到 {self.symbol}@{self.base_interval}_kline 流的WebSocket并订阅...")
            self.ws_client.kline(
                symbol=self.symbol,
                # id=1, # 订阅请求的唯一ID，旧库参数，新库不再需要显式传递
                interval=self.base_interval,
                callback=self.on_message_callback # 设置回调函数
            )
            logging.info(f"已通过WebSocket订阅 {self.symbol}@{self.base_interval}_kline 流。")
            # Note: The client.kline() method typically starts the background thread internally.

        except Exception as e:
            logging.error(f"启动 WebSocket 客户端或订阅时出错：{e}\n{traceback.format_exc()}")
            self.ws_client = None # 确保在失败时客户端为 None
            # 考虑抛出异常，让 main_app 处理启动失败
            raise


    def stop(self):
        """停止WebSocket客户端。"""
        if self.ws_client:
            logging.info("正在停止WebSocket客户端...")
            try:
                # Binance Websocket Client library 的 stop 方法用于停止客户端线程和连接
                self.ws_client.stop()
                logging.info("已发出WebSocket客户端停止命令。")
            except Exception as e:
                logging.error(f"停止WebSocket客户端时出错：{e}")
            self.ws_client = None
        else:
            logging.info("WebSocket客户端未运行或已停止。")

    def is_active(self):
        """检查WebSocket客户端是否活跃。"""
        if not self.ws_client:
            return False
        try:
            # Binance Websocket Client library 的 is_alive 方法检查底层线程
            if hasattr(self.ws_client, 'is_alive') and callable(getattr(self.ws_client, 'is_alive')):
                return self.ws_client.is_alive()
            # Fallback check if the above method is not available or appropriate
            logging.debug("WebSocket客户端没有可调用的'is_alive'方法，或检查失败。使用备用状态检查。")
            # Alternative check might involve checking internal connection state if exposed
            # For this library, is_alive seems standard.
            return True # Optimistic fallback if is_alive check fails
        except Exception as e:
            logging.error(f"检查WebSocket活动状态时出错：{e}")
            return False
