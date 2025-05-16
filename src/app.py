# src/app.py
import logging
import threading
import time
from datetime import datetime, timedelta

from src import config
from src.api_client.rest_client import CoinRestApiClient
from src.api_client.websocket_client import CoinWebsocketClient
from src.data_handlers.message_processor import process_websocket_message
from src.utils.signal_utils import setup_signal_handlers

logger = logging.getLogger(__name__)

class Application:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.selected_symbol_id = None
        self.rest_client = None
        self.ws_client = None

        if not config.API_KEY:
            logger.critical("API_KEY未设置。应用程序无法继续。")
            # 在实际应用中，你可能会引发异常或使用sys.exit()
            # 对于这个示例，我们将让它尝试并在客户端初始化时失败。
            return


        self.rest_client = CoinRestApiClient(api_key=config.API_KEY, base_url=config.REST_API_URL)

    def _initialize_symbol(self):
        """获取交易对并选择一个。"""
        logger.info("正在初始化交易对...")
        symbols_data = self.rest_client.get_symbols()
        if not symbols_data:
            logger.error("获取交易对数据失败。无法继续。")
            return False

        self.selected_symbol_id = self.rest_client.filter_and_select_symbol(symbols_data)
        if not self.selected_symbol_id:
            logger.error("选择交易对失败。无法继续。")
            return False
        logger.info(f"应用程序将使用交易对：{self.selected_symbol_id}")
        return True

    def _fetch_sample_historical_data(self):
        """获取并记录示例历史OHLCV数据。"""
        if not self.selected_symbol_id or not self.rest_client:
            logger.warning("无法获取历史数据：交易对或REST客户端未初始化。")
            return

        logger.info("获取示例历史OHLCV数据...")
        try:
            time_end = datetime.now().isoformat()
            
            # 最近一小时的1分钟数据
            time_start_1min = (datetime.now() - timedelta(hours=1)).isoformat()
            if "1MIN" in config.PERIODS_FOR_HISTORY:
                ohlcv_1min = self.rest_client.get_ohlcv_history(self.selected_symbol_id, "1MIN", time_start_1min, time_end)
                if ohlcv_1min:
                    logger.info(f"1分钟OHLCV数据示例（第一条）：{ohlcv_1min[0] if ohlcv_1min else '无数据'}")

            # 最近7天的1小时数据（如果已配置）
            if "1HRS" in config.PERIODS_FOR_HISTORY:
                time_start_1h = (datetime.now() - timedelta(days=7)).isoformat()
                ohlcv_1h = self.rest_client.get_ohlcv_history(self.selected_symbol_id, "1HRS", time_start_1h, time_end)
                if ohlcv_1h:
                    logger.info(f"1小时OHLCV数据示例（第一条）：{ohlcv_1h[0] if ohlcv_1h else '无数据'}")

        except Exception as e:
            logger.error(f"获取示例历史数据时出错：{e}")

    def run(self):
        """主应用程序运行循环。"""
        if not config.API_KEY: # 初始化日志后再次检查
            logger.error("由于缺少API_KEY，应用程序退出。")
            return

        setup_signal_handlers(self.shutdown_event)
        logger.info("应用程序启动中...")

        if not self._initialize_symbol():
            logger.error("应用程序初始化失败。退出。")
            return

        # 获取示例历史数据（可选，与原始脚本一样）
        self._fetch_sample_historical_data()

        if not self.selected_symbol_id:
             logger.error("交易对ID不可用，无法启动WebSocket客户端。退出。")
             return

        self.ws_client = CoinWebsocketClient(
            ws_url=config.WS_URL,
            api_key=config.API_KEY,
            symbol_id=self.selected_symbol_id,
            message_processor_callback=process_websocket_message,
            shutdown_event=self.shutdown_event
        )

        reconnect_attempts = 0
        backoff_time = config.WS_INITIAL_BACKOFF_SECONDS

        while not self.shutdown_event.is_set():
            if config.MAX_WS_RECONNECT_ATTEMPTS > 0 and reconnect_attempts >= config.MAX_WS_RECONNECT_ATTEMPTS:
                logger.error("达到最大WebSocket重连尝试次数。退出。")
                break
            
            should_reconnect = self.ws_client.connect() # 这是一个阻塞调用

            if self.shutdown_event.is_set():
                break # 如果在connect或run_forever期间发出关闭信号，则退出

            if should_reconnect: # connect()返回True，表示它已退出并可能需要重新连接
                reconnect_attempts += 1
                logger.info(f"WebSocket连接丢失/失败。尝试在{backoff_time}秒内重新连接 {reconnect_attempts}/"
                            f"{config.MAX_WS_RECONNECT_ATTEMPTS if config.MAX_WS_RECONNECT_ATTEMPTS > 0 else '无限'} "
                            f"...")
                
                # 等待backoff_time，但频繁检查shutdown_event
                wait_start_time = time.time()
                while time.time() - wait_start_time < backoff_time:
                    if self.shutdown_event.is_set():
                        logger.info("在退避期间启动关闭。中止重新连接。")
                        break
                    time.sleep(0.5) # 每0.5秒检查一次
                
                if self.shutdown_event.is_set():
                    break

                backoff_time = min(backoff_time * 2, config.WS_MAX_BACKOFF_SECONDS)
            else:
                # connect()返回False，表示设置错误或关闭
                if not self.shutdown_event.is_set():
                    logger.error("WebSocket客户端表示不重试（例如，设置错误）。退出。")
                break
        
        if self.ws_client and self.ws_client.ws_app: # 确保在退出前关闭WS
             logger.info("在退出应用程序前确保WebSocket已关闭...")
             self.ws_client.close()

        logger.info("应用程序关闭完成。")