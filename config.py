# config.py
import logging

# --- API配置 ---
API_BASE_URL_FUTURES = "https://fapi.binance.com"  # 用于USDⓈ-M期货

# --- 交易对设置 ---
SYMBOL = "BTCUSDT"

# --- 运行模式 ---
# "TEST": 基础"1m" -> 聚合"3m"
# "PRODUCTION": 基础"1h" -> 聚合"3h"
OPERATING_MODE = "TEST"

# --- K线设置 ---
HISTORICAL_AGG_CANDLES_TO_DISPLAY = 50  # 显示的初始聚合K线数量
MAX_KLINE_LIMIT_PER_REQUEST = 1000     # 每次请求K线的币安API限制

# --- 动态间隔设置（由OPERATING_MODE决定） ---
BASE_INTERVAL = ""
AGG_INTERVAL = ""

if OPERATING_MODE == "TEST":
    BASE_INTERVAL = "1m"
    AGG_INTERVAL = "3m"
elif OPERATING_MODE == "PRODUCTION":
    BASE_INTERVAL = "1h"
    AGG_INTERVAL = "3h"
else:
    raise ValueError("config.py中的OPERATING_MODE无效。选择'TEST'或'PRODUCTION'。")

# --- 日志配置 ---
LOG_LEVEL = logging.INFO # 使用logging.DEBUG获取详细输出，使用logging.INFO获取常规流程
