# src/api_client/websocket_client.py
import websocket # websocket-client库
import json
import time
import threading
import logging
from src.utils.network_utils import resolve_hostname
from src.config import WS_INACTIVITY_TIMEOUT

logger = logging.getLogger(__name__)

class CoinWebsocketClient:
    def __init__(self, ws_url, api_key, symbol_id, message_processor_callback, shutdown_event):
        if not api_key:
            logger.error("CoinWebsocketClient未配置API_KEY。")
            raise ValueError("CoinWebsocketClient需要API_KEY")
        if not symbol_id:
            logger.error("未向CoinWebsocketClient提供symbol_id。")
            raise ValueError("CoinWebsocketClient需要symbol_id")

        self.ws_url = ws_url
        self.api_key = api_key
        self.selected_symbol_id = symbol_id
        self.message_processor_callback = message_processor_callback
        self.shutdown_event = shutdown_event
        
        self.ws_app = None
        self.inactivity_timer = None
        self.ws_lock = threading.Lock() # 保护ws_app和inactivity_timer

    def _on_message(self, ws, message_str):
        with self.ws_lock:
            self._reset_inactivity_timer_unsafe() # 假设已持有锁
        self.message_processor_callback(message_str)

    def _on_error(self, ws, error):
        logger.error(f"WebSocket错误：{error}")
        # 错误后通常会调用on_close，处理重连逻辑。

    def _on_close(self, ws, close_status_code, close_msg):
        with self.ws_lock:
            if self.inactivity_timer:
                self.inactivity_timer.cancel()
                self.inactivity_timer = None
        
        if self.shutdown_event.is_set():
            logger.info("WebSocket连接已优雅关闭。")
        else:
            logger.warning(f"WebSocket连接意外关闭。状态：{close_status_code}，消息：{close_msg}。主循环将尝试重新连接。")

    def _on_open(self, ws):
        logger.info("WebSocket连接已打开。")
        hello_message = {
            "type": "hello",
            "apikey": self.api_key,
            "subscribe_data_type": ["ohlcv"], # 只订阅 "ohlcv"
            "subscribe_filter_symbol_id": [self.selected_symbol_id]
            # 如果API支持通过WebSocket进行OHLCV周期的服务器端过滤，
            # 你可能想添加 "subscribe_filter_period_id": config.WS_OHLCV_INTERESTED_PERIODS
            # CoinAPI的WebSocket通常是通过发送所有交易对的数据来进行主要的OHLCV过滤。
        }
        try:
            ws.send(json.dumps(hello_message))
            logger.info(f"已发送交易对订阅请求 (仅OHLCV)：{self.selected_symbol_id}")
            with self.ws_lock:
                self._reset_inactivity_timer_unsafe()
        except websocket.WebSocketConnectionClosedException:
            logger.error("发送hello消息失败：WebSocket连接已关闭。")
        except Exception as e:
            logger.error(f"发送hello消息时出错：{e}")
            self.close() # 如果发送hello消息严重失败则关闭

    def _reset_inactivity_timer_unsafe(self):
        """重置或启动不活动计时器。调用者必须持有ws_lock。"""
        if self.inactivity_timer:
            self.inactivity_timer.cancel()

        if self.ws_app and self.ws_app.sock and self.ws_app.sock.connected:
            self.inactivity_timer = threading.Timer(WS_INACTIVITY_TIMEOUT, self._handle_inactivity)
            self.inactivity_timer.daemon = True
            self.inactivity_timer.start()
        else:
            self.inactivity_timer = None # 如果ws不活动，确保清除

    def _handle_inactivity(self):
        logger.warning(f"{WS_INACTIVITY_TIMEOUT}秒内未收到消息。连接可能已过时。关闭以触发重新连接。")
        self.close() # 这将触发on_close，主应用循环处理重新连接

    def connect(self):
        """建立并运行WebSocket连接。"""
        if not self.selected_symbol_id:
            logger.error("无法连接WebSocket：未设置symbol_id。")
            return False # 表示设置失败

        try:
            ws_host = self.ws_url.split('//')[1].split('/')[0]
            if not resolve_hostname(ws_host):
                return False # DNS解析失败
        except IndexError:
            logger.error(f"WS_URL格式无效：{self.ws_url}")
            return False

        logger.info(f"尝试连接到WebSocket：{self.ws_url}")
        
        with self.ws_lock:
            self.ws_app = websocket.WebSocketApp(self.ws_url,
                                             on_open=self._on_open,
                                             on_message=self._on_message,
                                             on_error=self._on_error,
                                             on_close=self._on_close)
        try:
            # run_forever是阻塞的。当连接关闭时它会返回。
            self.ws_app.run_forever(ping_interval=20, ping_timeout=10) # CoinAPI推荐ping
        except Exception as e:
            logger.error(f"WebSocket run_forever遇到错误：{e}")
        finally:
            with self.ws_lock:
                self.ws_app = None # run_forever退出后清除实例

        if self.shutdown_event.is_set():
            logger.info("由于关闭信号，WebSocket run_forever循环已退出。")
            return False # 如果正在关闭，不要尝试重新连接
        else:
            logger.info("WebSocket run_forever循环已退出。如果允许，将尝试重新连接。")
            return True # 表示正常退出（例如，断开连接）可能需要重新连接

    def close(self):
        """如果WebSocket连接处于活动状态，则关闭它。"""
        with self.ws_lock:
            if self.inactivity_timer:
                self.inactivity_timer.cancel()
                self.inactivity_timer = None
            if self.ws_app:
                logger.info("正在关闭WebSocket连接...")
                try:
                    self.ws_app.close()
                except Exception as e:
                    logger.error(f"关闭WebSocket时出错：{e}")