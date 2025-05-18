#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog # 导入structlog

# 使用模块级logger
log = structlog.get_logger(__name__)

class Config(BaseSettings):
    # --- API配置 ---
    API_BASE_URL_FUTURES: str = "https://fapi.binance.com"  # 用于USDⓈ-M期货

    # --- 交易对设置 ---
    SYMBOL: str = "BTCUSDT"

    # --- 运行模式 ---
    OPERATING_MODE: str = "TEST"  # "TEST" 或 "PRODUCTION"

    # --- K线设置 ---
    HISTORICAL_AGG_CANDLES_TO_DISPLAY: int = 50  # 显示的初始聚合K线数量
    MAX_KLINE_LIMIT_PER_REQUEST: int = 1000  # 每次请求K线的币安API限制

    # --- 动态间隔设置（由OPERATING_MODE决定） ---
    BASE_INTERVAL: str = ""
    AGG_INTERVAL: str = ""

    # --- 日志配置 ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "console" # 此设置更多是概念性的，structlog有自己的渲染方式

    # --- 数据存储配置 ---
    DATA_STORE_TYPE: str = "memory"

    # --- PostgreSQL数据库配置 ---
    DB_HOST: str = "localhost"
    DB_PORT: str = "5432"
    DB_NAME: str = "binance_data"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def is_valid_operating_mode(self) -> bool:
        return self.OPERATING_MODE.upper() in ["TEST", "PRODUCTION"]

    @property
    def validated_log_level(self) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level = self.LOG_LEVEL.upper()
        if level not in valid_levels:
            # 注意: 此时日志系统可能还未完全配置，此处的ValueError可能比log.error更早被看到
            raise ValueError(f"无效的 LOG_LEVEL '{level}'。必须是 {valid_levels} 中的一个。")
        return level

# 全局配置实例
try:
    config = Config()

    if not config.is_valid_operating_mode:
        raise ValueError(f"OPERATING_MODE '{config.OPERATING_MODE}' 无效。必须是 'TEST' 或 'PRODUCTION'。")

    if config.OPERATING_MODE.upper() == "TEST":
        config.BASE_INTERVAL = "1m"
        config.AGG_INTERVAL = "3m"
    elif config.OPERATING_MODE.upper() == "PRODUCTION":
        config.BASE_INTERVAL = "1h"
        config.AGG_INTERVAL = "3h"

    # 在日志系统配置完成后，此日志才会按预期格式输出
    # 我们将在main_app.py中日志系统配置完成后再打印配置信息
except ValueError as e:
    log.critical("配置验证失败，应用程序可能无法启动", error=str(e)) # 使用 logger
    raise RuntimeError(f"配置验证失败：{e}") from e
except Exception as e_global_config:
    log.critical("加载配置时发生严重错误", error=str(e_global_config))
    raise RuntimeError(f"加载配置时发生严重错误：{e_global_config}") from e_global_config

# 移除这里的 log.info("配置加载成功"...)，移至 main_app.py 中日志系统初始化之后