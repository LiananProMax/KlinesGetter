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

# --- API配置 ---
API_BASE_URL_FUTURES = get_env_variable("API_BASE_URL_FUTURES", "https://fapi.binance.com")  # 用于USDⓈ-M期货

# --- 交易对设置 ---
SYMBOL = get_env_variable("SYMBOL", "BTCUSDT")

# --- 运行模式 ---
# "TEST": 基础"1m" -> 聚合"3m"
# "PRODUCTION": 基础"1h" -> 聚合"3h"
OPERATING_MODE = get_env_variable("OPERATING_MODE", "TEST").upper()

# --- K线设置 ---
HISTORICAL_AGG_CANDLES_TO_DISPLAY = get_env_variable("HISTORICAL_AGG_CANDLES_TO_DISPLAY", 50, int)  # 显示的初始聚合K线数量
MAX_KLINE_LIMIT_PER_REQUEST = get_env_variable("MAX_KLINE_LIMIT_PER_REQUEST", 1000, int)  # 每次请求K线的币安API限制

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
    # 如果有效模式对应用程序启动至关重要，可以考虑引发ValueError
    # raise ValueError(".env中的OPERATING_MODE无效。选择'TEST'或'PRODUCTION'。")


# --- 日志配置 ---
# 从.env获取日志级别字符串并映射到相应的logging常量
LOG_LEVEL_STR = get_env_variable("LOG_LEVEL", "INFO").upper()
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOG_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL_STR, logging.INFO) # 如果字符串无效，默认为INFO

if LOG_LEVEL_STR not in LOG_LEVEL_MAP:
    # 如果日志尚未配置，使用基本的print
    print(
        f"警告：.env文件中的LOG_LEVEL '{LOG_LEVEL_STR}'无效。默认使用INFO。"
    )

# --- (未来可能添加) 数据存储配置 ---
# 未来使用的示例：
# DATA_STORE_TYPE = get_env_variable("DATA_STORE_TYPE", "memory")
# DB_CONNECTION_STRING = get_env_variable("DB_CONNECTION_STRING", "")

# 日志初始化后，记录已加载的配置是个好习惯，
# 通常在主应用程序设置中。例如：
# logging.info(f"--- 配置已加载 ---")
# logging.info(f"API_BASE_URL_FUTURES: {API_BASE_URL_FUTURES}")
# logging.info(f"SYMBOL: {SYMBOL}")
# logging.info(f"OPERATING_MODE: {OPERATING_MODE} (基础: {BASE_INTERVAL}, 聚合: {AGG_INTERVAL})")
# logging.info(f"HISTORICAL_AGG_CANDLES_TO_DISPLAY: {HISTORICAL_AGG_CANDLES_TO_DISPLAY}")
# logging.info(f"MAX_KLINE_LIMIT_PER_REQUEST: {MAX_KLINE_LIMIT_PER_REQUEST}")
# logging.info(f"LOG_LEVEL: {LOG_LEVEL_STR} (生效: {LOG_LEVEL})")
# logging.info(f"-----------------------------")
