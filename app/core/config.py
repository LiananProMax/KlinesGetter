#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from dotenv import load_dotenv

# 从项目根目录中的.env文件加载环境变量
# 假设.env文件与run.py在同一目录或app/core的上两级目录
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
if not os.path.exists(dotenv_path):
    # 当脚本从项目根目录直接运行时的回退方案
    dotenv_path = os.path.join(os.getcwd(), '.env')

load_dotenv(dotenv_path=dotenv_path)

# --- 获取环境变量的辅助函数（带默认值和类型转换） ---
def get_env_variable(var_name, default_value, var_type=str):
    """
    获取环境变量并将其转换为指定类型。
    如果变量不存在或转换失败，则提供默认值。
    """
    value = os.getenv(var_name)
    if value is None:
        # logging.debug(f"环境变量 {var_name} 未找到。使用默认值：{default_value}")
        return default_value

    try:
        if var_type == bool: # 布尔值的特殊处理
            if value.lower() in ('true', '1', 't', 'yes', 'y'):
                return True
            elif value.lower() in ('false', '0', 'f', 'no', 'n'):
                return False
            else:
                raise ValueError(f"{var_name} 的布尔值无效：{value}")
        return var_type(value)
    except ValueError:
        # 如果日志尚未配置，使用基本的print
        print(
            f"警告：环境变量 {var_name} 类型无效。"
            f"期望 {var_type}，得到 '{value}'。使用默认值：{default_value}"
        )
        return default_value

# --- API 相关设置 ---
API_KEY = get_env_variable('API_KEY', '', str)
API_SECRET = get_env_variable('API_SECRET', '', str)
# 是否使用测试网 (True/False)
IS_TESTNET = get_env_variable('IS_TESTNET', 'True', str).lower() == 'true'
# Binance 测试网络基本 URL
API_BASE_URL = get_env_variable('API_BASE_URL', 'https://testnet.binance.vision', str)
# Binance 测试网络期货 API URL
API_BASE_URL_FUTURES = get_env_variable('API_BASE_URL_FUTURES', 'https://testnet.binance.vision', str)
# Binance 测试网络 WebSocket URL
WS_BASE_URL = get_env_variable('WS_BASE_URL', 'wss://testnet.binance.vision/ws', str)
# Binance 测试网络期货 WebSocket URL
WS_BASE_URL_FUTURES = get_env_variable('WS_BASE_URL_FUTURES', 'wss://testnet.binance.vision/ws', str)

# SSL 证书验证设置
# 注意: 在生产环境中应始终保持 VERIFY_SSL=True
VERIFY_SSL = get_env_variable('VERIFY_SSL', 'True', str).lower() == 'true'

# API调用重试配置
MAX_API_RETRIES = get_env_variable('MAX_API_RETRIES', '3', int)
API_RETRY_DELAY = get_env_variable('API_RETRY_DELAY', '5', int) # 秒

# --- 交易对设置 ---
SYMBOL = get_env_variable("SYMBOL", "BTCUSDT")

# --- 运行模式 ---
# 决定基础和聚合间隔。
# "TEST": 基础 "1m" -> 聚合 "3m" (测试期间频繁更新)
# "PRODUCTION": 基础 "1h" -> 聚合 "3h" (更少频率，更稳定的数据)
OPERATING_MODE = get_env_variable("OPERATING_MODE", "TEST").upper()

# --- K线设置 ---
# 初始获取和显示的聚合K线数量
HISTORICAL_AGG_CANDLES_TO_DISPLAY = get_env_variable("HISTORICAL_AGG_CANDLES_TO_DISPLAY", 50, int)
# 币安API每个请求允许的最大K线数量
MAX_KLINE_LIMIT_PER_REQUEST = get_env_variable("MAX_KLINE_LIMIT_PER_REQUEST", 1000, int)

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
    # 如果日志尚未配置，使用基本的print
    print(f"错误：.env文件中的OPERATING_MODE '{OPERATING_MODE}'无效。默认使用'TEST'模式。")
    OPERATING_MODE = "TEST" # 无效时回退到TEST
    BASE_INTERVAL = "1m"
    AGG_INTERVAL = "3m"


# --- 日志配置 ---
# 添加自定义日志级别

# SYSTEM 级别 - 定义在 INFO (20) 和 WARNING (30) 之间，值为 25
SYSTEM_LOG_LEVEL = 25
logging.addLevelName(SYSTEM_LOG_LEVEL, "SYSTEM")

# 添加 system 方法到 Logger 类
def system(self, message, *args, **kwargs):
    if self.isEnabledFor(SYSTEM_LOG_LEVEL):
        self._log(SYSTEM_LOG_LEVEL, message, args, **kwargs)

# 将 system 方法添加到 Logger 类
logging.Logger.system = system

# SUCCESS 级别 - 定义在 INFO (20) 和 SYSTEM (25) 之间，值为 22
SUCCESS_LOG_LEVEL = 22
logging.addLevelName(SUCCESS_LOG_LEVEL, "SUCCESS")

# 添加 success 方法到 Logger 类
def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LOG_LEVEL):
        self._log(SUCCESS_LOG_LEVEL, message, args, **kwargs)

# 将 success 方法添加到 Logger 类
logging.Logger.success = success

# STRATEGY 级别 - 定义在 INFO (20) 和 SUCCESS (22) 之间，值为 21
STRATEGY_LOG_LEVEL = 21
logging.addLevelName(STRATEGY_LOG_LEVEL, "STRATEGY")

# 添加 strategy 方法到 Logger 类
def strategy(self, message, *args, **kwargs):
    if self.isEnabledFor(STRATEGY_LOG_LEVEL):
        self._log(STRATEGY_LOG_LEVEL, message, args, **kwargs)

# 将 strategy 方法添加到 Logger 类
logging.Logger.strategy = strategy

# 从.env获取日志级别字符串并映射到相应的logging常量
LOG_LEVEL_STR = get_env_variable("LOG_LEVEL", "INFO").upper()
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "STRATEGY": STRATEGY_LOG_LEVEL,  # 添加 STRATEGY 级别到映射中
    "SUCCESS": SUCCESS_LOG_LEVEL,  # 添加 SUCCESS 级别到映射中
    "SYSTEM": SYSTEM_LOG_LEVEL,  # 添加 SYSTEM 级别到映射中
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOG_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL_STR, logging.INFO) # 如果字符串无效，默认为INFO

if LOG_LEVEL_STR not in LOG_LEVEL_MAP:
    # 这可能在Rich日志完全设置之前执行。
    # 使用基本的print作为备用警告机制。
    # main_app的setup_logging应使用配置好的日志器再次警告如有必要。
    # 如果RichHandler设置能够保证早期日志可用，可以移除此基本的print。
    # 如果日志尚未配置，使用基本的print
    print(
        f"警告：.env文件中的LOG_LEVEL '{LOG_LEVEL_STR}'无效。默认使用INFO。"
    )


# --- 数据存储配置 ---
DATA_STORE_TYPE = get_env_variable("DATA_STORE_TYPE", "memory").lower() # "memory" 或 "database"

# --- PostgreSQL数据库配置 (仅当 DATA_STORE_TYPE="database" 时相关) ---
DB_HOST = get_env_variable("DB_HOST", "localhost")
DB_PORT = get_env_variable("DB_PORT", "5432")
DB_NAME = get_env_variable("DB_NAME", "binance_data")
DB_USER = get_env_variable("DB_USER", "postgres")
DB_PASSWORD = get_env_variable("DB_PASSWORD", "")

# --- 数据同步/验证配置 ---
DATA_SYNC_VERIFY_DELAY = get_env_variable("DATA_SYNC_VERIFY_DELAY", 5, int) # 秒
DATA_SYNC_VERIFY_ATTEMPTS = get_env_variable("DATA_SYNC_VERIFY_ATTEMPTS", 12, int) # 次

# --- 策略参数 (从原始项目提取) ---
# EMA Settings
SHORT_EMA_LEN = get_env_variable('SHORT_EMA_LEN', 9, int)
LONG_EMA_LEN = get_env_variable('LONG_EMA_LEN', 21, int)

# RSI Settings
RSI_LEN = get_env_variable('RSI_LEN', 14, int)
RSI_OVERBOUGHT = get_env_variable('RSI_OVERBOUGHT', 70.0, float)
RSI_OVERSOLD = get_env_variable('RSI_OVERSOLD', 30.0, float)

# VWAP Settings
VWAP_PERIOD = get_env_variable('VWAP_PERIOD', 'D', str).upper()

# Stop Loss / Take Profit Settings
USE_SLTP = get_env_variable('USE_SLTP', True, bool)
STOP_LOSS_TICKS = get_env_variable('STOP_LOSS_TICKS', 100, float)
TAKE_PROFIT_TICKS = get_env_variable('TAKE_PROFIT_TICKS', 200, float)

# Order Management Parameters
QTY_PERCENT = get_env_variable('QTY_PERCENT', 0.90, float)

# 交易类型枚举 (从原始项目提取，放在这里方便全局访问)
SIDE_BUY = 'BUY'
SIDE_SELL = 'SELL'
SIDE_LONG = 'LONG' # 期货多头仓位标识
SIDE_SHORT = 'SHORT' # 期货空头仓位标识

ORDER_TYPE_MARKET = 'MARKET'
ORDER_TYPE_LIMIT = 'LIMIT'
ORDER_TYPE_STOP_LOSS = 'STOP_LOSS'
ORDER_TYPE_TAKE_PROFIT = 'TAKE_PROFIT'

FUTURE_ORDER_TYPE_MARKET = 'MARKET'
FUTURE_ORDER_TYPE_LIMIT = 'LIMIT'
FUTURE_ORDER_TYPE_STOP = 'STOP' # 期货止损单
FUTURE_ORDER_TYPE_STOP_MARKET = 'STOP_MARKET' # 期货市价止损
FUTURE_ORDER_TYPE_TAKE_PROFIT = 'TAKE_PROFIT' # 期货止盈单
FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET = 'TAKE_PROFIT_MARKET' # 期货市价止盈

TIME_IN_FORCE_GTC = 'GTC' # Good Till Cancelled
TIME_IN_FORCE_IOC = 'IOC' # Immediate or Cancel

# Order Status Check (用于简单的阻塞式订单状态检查，非推荐方式)
ORDER_STATUS_CHECK_DELAY = get_env_variable('ORDER_STATUS_CHECK_DELAY', 2, int) # 秒
POSITION_CLOSE_VERIFY_DELAY = get_env_variable('POSITION_CLOSE_VERIFY_DELAY', 3, int) # 秒
POSITION_CLOSE_VERIFY_ATTEMPTS = get_env_variable('POSITION_CLOSE_VERIFY_ATTEMPTS', 5, int) # 次

# K线数据处理需要的最小历史窗口长度
MAX_DF_LEN_STRATEGY = get_env_variable('MAX_DF_LEN_STRATEGY', 1000, int) # 策略模块内部维护的历史K线长度

# 将所有配置参数整合到一个字典中，方便传递给策略类
STRATEGY_CONFIG = {
    'SYMBOL': SYMBOL,
    'MARKET_TYPE': OPERATING_MODE, # 使用OPERATING_MODE映射期货/现货，需要进一步细化
    'INTERVAL_STR': AGG_INTERVAL, # 策略运行在聚合间隔上
    'SHORT_EMA_LEN': SHORT_EMA_LEN,
    'LONG_EMA_LEN': LONG_EMA_LEN,
    'RSI_LEN': RSI_LEN,
    'RSI_OVERBOUGHT': RSI_OVERBOUGHT,
    'RSI_OVERSOLD': RSI_OVERSOLD,
    'VWAP_PERIOD': VWAP_PERIOD,
    'USE_SLTP': USE_SLTP,
    'STOP_LOSS_TICKS': STOP_LOSS_TICKS,
    'TAKE_PROFIT_TICKS': TAKE_PROFIT_TICKS,
    'QTY_PERCENT': QTY_PERCENT,
    'ORDER_STATUS_CHECK_DELAY': ORDER_STATUS_CHECK_DELAY,
    'POSITION_CLOSE_VERIFY_DELAY': POSITION_CLOSE_VERIFY_DELAY,
    'POSITION_CLOSE_VERIFY_ATTEMPTS': POSITION_CLOSE_VERIFY_ATTEMPTS,
    'MAX_DF_LEN_STRATEGY': MAX_DF_LEN_STRATEGY, # 传递给策略的DF长度限制
    # 需要根据 OPERATING_MODE 映射到 FUTURES/SPOT，这在 main_app 或策略初始化中处理
    # 需要 Spot 的引用资产 SPOT_QUOTE_ASSET，如果支持 Spot 交易
    # 'SPOT_QUOTE_ASSET': SPOT_QUOTE_ASSET # 如果需要 Spot
}

# 修正：根据 OPERATING_MODE 确定 MARKET_TYPE
if OPERATING_MODE == "TEST":
    # 假设TEST模式用于期货测试网
    STRATEGY_CONFIG['MARKET_TYPE'] = 'FUTURES'
elif OPERATING_MODE == "PRODUCTION":
    # 假设PRODUCTION模式用于期货主网
    STRATEGY_CONFIG['MARKET_TYPE'] = 'FUTURES' # 如果也支持 SPOT，需要额外的配置或逻辑判断
    # 如果 PRODUCTION 需要在现货主网运行，则设置为 'SPOT'
    # STRATEGY_CONFIG['MARKET_TYPE'] = 'SPOT'
    # 如果是SPOT，则需要 SPOT_QUOTE_ASSET
    # STRATEGY_CONFIG['SPOT_QUOTE_ASSET'] = get_env_variable('SPOT_QUOTE_ASSET', 'USDT', str).upper()

# 校验基础配置完整性
if not API_KEY or not API_SECRET:
    # 如果RichHandler设置能够保证日志可用，可以移除此基本的print。
    print("警告：API_KEY或API_SECRET未配置。交易功能将不可用。")

# 校验策略配置（部分基本校验，更严格的校验在策略类中进行）
if STRATEGY_CONFIG['SHORT_EMA_LEN'] <= 0 or STRATEGY_CONFIG['LONG_EMA_LEN'] <= 0 or STRATEGY_CONFIG['RSI_LEN'] <= 0 or STRATEGY_CONFIG['SHORT_EMA_LEN'] >= STRATEGY_CONFIG['LONG_EMA_LEN']:
    # 如果RichHandler设置能够保证日志可用，可以移除此基本的print。
    print("警告：EMA/RSI长度配置无效。请检查SHORT_EMA_LEN和LONG_EMA_LEN。")
if not (0 <= STRATEGY_CONFIG['RSI_OVERSOLD'] < STRATEGY_CONFIG['RSI_OVERBOUGHT'] <= 100):
    # 如果RichHandler设置能够保证日志可用，可以移除此基本的print。
    print("警告：RSI超买/超卖配置无效。请检查RSI_OVERBOUGHT和RSI_OVERSOLD。")
if STRATEGY_CONFIG['VWAP_PERIOD'] not in ['D', 'W', 'M']:
     # 如果RichHandler设置能够保证日志可用，可以移除此基本的print。
     print(f"警告：VWAP_PERIOD '{STRATEGY_CONFIG['VWAP_PERIOD']}' 无效。使用'D', 'W', 'M'。")
if not (0.0 < STRATEGY_CONFIG['QTY_PERCENT'] <= 1.0):
    # 如果RichHandler设置能够保证日志可用，可以移除此基本的print。
    print(f"警告：QTY_PERCENT '{STRATEGY_CONFIG['QTY_PERCENT']}' 无效。应在(0, 1]范围内。")


# Log configuration details after logging is set up in main_app
