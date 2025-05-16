# src/config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# API和WebSocket配置
API_KEY = os.getenv('API_KEY')
REST_API_URL = os.getenv('REST_API_URL', 'https://rest.coinapi.io/v1/symbols')
WS_URL = os.getenv('WS_URL', 'wss://ws.coinapi.io/v1/')

# REST API历史数据的时间周期
DEFAULT_PERIODS_LIST = ['1MIN', '5MIN', '15MIN', '30MIN', '1HRS', '4HRS', '1DAY']
periods_str = os.getenv('PERIODS')
PERIODS_FOR_HISTORY = periods_str.split(',') if periods_str else DEFAULT_PERIODS_LIST
logger.info(f"加载REST API历史数据周期: {PERIODS_FOR_HISTORY}")

# WebSocket OHLCV流处理的时间周期
DEFAULT_WS_OHLCV_PERIODS = ['1MIN', '3MIN', '1HRS']
ws_ohlcv_periods_str = os.getenv('WS_OHLCV_INTERESTED_PERIODS')
WS_OHLCV_INTERESTED_PERIODS = ws_ohlcv_periods_str.split(',') if ws_ohlcv_periods_str else DEFAULT_WS_OHLCV_PERIODS
logger.info(f"加载WebSocket OHLCV关注周期: {WS_OHLCV_INTERESTED_PERIODS}")


# 重试和超时配置
MAX_REST_RETRIES = int(os.getenv('MAX_REST_RETRIES', 5))
INITIAL_REST_WAIT_TIME = int(os.getenv('INITIAL_REST_WAIT_TIME', 2))
MAX_WS_RECONNECT_ATTEMPTS = int(os.getenv('MAX_WS_RECONNECT_ATTEMPTS', 10))
WS_INITIAL_BACKOFF_SECONDS = int(os.getenv('WS_INITIAL_BACKOFF_SECONDS', 5))
WS_MAX_BACKOFF_SECONDS = int(os.getenv('WS_MAX_BACKOFF_SECONDS', 60))
WS_INACTIVITY_TIMEOUT = int(os.getenv('WS_INACTIVITY_TIMEOUT', 70)) # 秒（CoinAPI每55秒发送一次心跳）

# 验证必要的配置
if not API_KEY:
    logger.critical("严重错误: 在环境变量中未找到API_KEY。应用程序无法启动。")
    # 考虑在这里引发异常或使用sys.exit()立即停止执行
    # 现在，它将记录日志并可能在稍后失败。