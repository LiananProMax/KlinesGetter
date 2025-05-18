#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    # 注意：Pydantic 无法直接处理逻辑，这里我们保留在 main_app.py 中计算
    BASE_INTERVAL: str = ""  # 基础间隔，由 OPERATING_MODE 决定
    AGG_INTERVAL: str = ""   # 聚合间隔，由 OPERATING_MODE 决定

    # --- 日志配置 ---
    LOG_LEVEL: str = "INFO"  # 可能的值: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

    # --- 数据存储配置 ---
    DATA_STORE_TYPE: str = "memory"  # "memory" 或 "database"

    # --- PostgreSQL数据库配置 (仅当 DATA_STORE_TYPE="database" 时相关) ---
    DB_HOST: str = "localhost"
    DB_PORT: str = "5432"
    DB_NAME: str = "binance_data"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""

    # Pydantic 配置：从 .env 文件加载，并验证
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # 自定义验证：确保 OPERATING_MODE 是有效的
    @property
    def is_valid_operating_mode(self) -> bool:
        return self.OPERATING_MODE.upper() in ["TEST", "PRODUCTION"]

    # 自定义验证：确保 LOG_LEVEL 是有效的
    @property
    def validated_log_level(self) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level = self.LOG_LEVEL.upper()
        if level not in valid_levels:
            raise ValueError(f"无效的 LOG_LEVEL '{level}'。必须是 {valid_levels} 中的一个。")
        return level

# 全局配置实例
config = Config()

# 导入structlog并添加配置加载日志
import structlog
log = structlog.get_logger()

# 在应用启动时，验证配置
try:
    # 验证 OPERATING_MODE
    if not config.is_valid_operating_mode:
        raise ValueError(f"OPERATING_MODE '{config.OPERATING_MODE}' 无效。必须是 'TEST' 或 'PRODUCTION'。")

    # 设置 BASE_INTERVAL 和 AGG_INTERVAL 基于 OPERATING_MODE
    if config.OPERATING_MODE.upper() == "TEST":
        config.BASE_INTERVAL = "1m"
        config.AGG_INTERVAL = "3m"
    elif config.OPERATING_MODE.upper() == "PRODUCTION":
        config.BASE_INTERVAL = "1h"
        config.AGG_INTERVAL = "3h"

    # 其他验证可以在这里添加
except ValueError as e:
    raise RuntimeError(f"配置验证失败：{e}") from e

# 添加配置加载成功的INFO日志（确保在LOG_LEVEL=INFO时可见）
log.info("配置加载成功", config=config.model_dump_json(indent=2))
