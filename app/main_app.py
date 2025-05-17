#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
import json
import pandas as pd
from datetime import datetime, timezone
import traceback # Import traceback for detailed error logging
import sys # Import sys for exiting

from binance.client import Client # 导入Binance Client

# 配置和工具
from app.core import config
from app.utils.kline_utils import format_kline_from_api, interval_to_timedelta, align_timestamp_to_interval # Import interval_to_timedelta and align_timestamp_to_interval
# 导入交易相关的工具函数和API客户端
from app.api_clients.binance_trading_api import get_precisions_and_tick_size # 导入获取精度函数

# 数据处理组件
from app.data_handling.data_interfaces import KlinePersistenceInterface
from app.data_handling.kline_data_store import KlineDataStore # 默认内存存储
from app.data_handling.db_manager import DBManager # 数据库管理器
from app.data_handling.kline_aggregator import aggregate_klines_df

# API和显示组件
from app.api_clients.binance_api_manager import fetch_historical_klines, BinanceWebsocketManager

from app.ui.display_manager import display_historical_aggregated_klines, display_realtime_update

# 导入策略模块
from app.trading.trading_strategy import TradingStrategy # 导入策略类

# 日志设置（使用app.core.config中的日志级别）
# binance.lib.utils.config_logging 依赖 logging 模块
# 确保 logging 模块在调用 config_logging 之前可用
# 在 app.core.config 中已经设置了基础日志 handler 和 formatter
# 并且 logging.setLoggerClass 应该已经被 RichHandler 设置
# 所以这里只需要获取logger即可
# logging.basicConfig(level=config.LOG_LEVEL) # This might overwrite custom handlers from app.core.config, maybe remove if RichHandler is set up elsewhere
# Let's ensure logging is configured once, likely in app.core.config or run.py setup.
# If basicConfig is used here, ensure it's not conflicting. A simple way is to set level on root logger.
logging.getLogger().setLevel(config.LOG_LEVEL) # Set level on root logger


logger = logging.getLogger(__name__) # 获取logger

# --- 全局实例（由此main_app模块管理）---
# 这些将持有实例，方便在不同函数（如WebSocket回调）中访问。
binance_client: Client | None = None # Add type hint and allow None initially
kline_persistence: KlinePersistenceInterface | None = None # Add type hint and allow None initially
ws_manager: BinanceWebsocketManager | None = None # Add type hint and allow None initially
trading_strategy: TradingStrategy | None = None # Add type hint and allow None initially

# Keep track of the last processed timestamp for aggregated candles to trigger strategy only once per closed candle
last_processed_agg_candle_timestamp: datetime | None = None


def setup_logging():
    """配置应用程序范围的日志。"""
    # config_logging(logging, config.LOG_LEVEL) # 注释掉或移除，已经在app.core.config中设置了基础日志或RichHandler
    logging.getLogger().setLevel(config.LOG_LEVEL) # 确保根logger级别设置

    logger.info(f"--- 应用程序启动配置 ---")
    logger.info(f"交易对: {config.SYMBOL}")
    logger.info(f"运行模式: {config.OPERATING_MODE} (基础: {config.BASE_INTERVAL}, 聚合: {config.AGG_INTERVAL})")
    logger.info(f"市场类型 (策略): {config.STRATEGY_CONFIG.get('MARKET_TYPE', '未知')}") # 从策略配置中获取市场类型
    logger.info(f"显示历史聚合K线数量: {config.HISTORICAL_AGG_CANDLES_TO_DISPLAY}")
    # LOG_LEVEL_STR is not a standard logging attribute, log the level value instead
    logger.info(f"日志级别: {logging.getLevelName(config.LOG_LEVEL)} (配置值: {config.LOG_LEVEL_STR})")
    logger.info(f"数据存储类型: {config.DATA_STORE_TYPE}")
    if config.DATA_STORE_TYPE == "database":
        logger.info(f"  DB_HOST: {config.DB_HOST}")
        logger.info(f"  DB_PORT: {config.DB_PORT}")
        logger.info(f"  DB_NAME: {config.DB_NAME}")
    # Log key strategy parameters
    logger.info(f"策略参数: ShortEMA={config.SHORT_EMA_LEN}, LongEMA={config.LONG_EMA_LEN}, RSI_Len={config.RSI_LEN}, RSI_OB={config.RSI_OVERBOUGHT}, RSI_OS={config.RSI_OVERSOLD}, VWAP_Period={config.VWAP_PERIOD}, USE_SLTP={config.USE_SLTP}, SL_Ticks={config.STOP_LOSS_TICKS}, TP_Ticks={config.TAKE_PROFIT_TICKS}, QTY_PERCENT={config.QTY_PERCENT}")
    logger.info(f"--- 配置加载完成 ---")


def websocket_message_handler(_, message_str: str):
    """处理传入的WebSocket K线消息，仅负责更新数据存储。"""

    if not kline_persistence or not trading_strategy:
        # 这可能发生在关闭时或初始化失败时。
        # 如果是正常关闭序列，避免过多的日志记录。
        # if ws_manager and ws_manager.is_active(): # 仅当WS仍然意外活跃时记录
        #      logger.error("websocket_message_handler中未初始化Kline持久化服务或交易策略。")
        return

    try:
        data = json.loads(message_str)

        # 检查kline事件数据
        if 'e' in data and data['e'] == 'kline' and 'k' in data:
            kline_payload = data['k']
            symbol_stream = kline_payload.get('s')
            base_interval_stream = kline_payload.get('i')

            # 验证符号和间隔是否与配置的匹配
            if symbol_stream.upper() != config.SYMBOL.upper() or base_interval_stream != config.BASE_INTERVAL:
                logger.warning(f"收到不匹配配置的WS K线数据: {symbol_stream}@{base_interval_stream}. 忽略.")
                return

            is_base_candle_closed = kline_payload.get('x', False) # 'x'标志表示基础K线是否已关闭

            # 格式化传入的K线数据
            # 使用format_kline_from_api辅助函数（假设结构与原始兼容）
            # 时间戳（kline[0]）是基础K线的开始时间。收盘时间（kline[6]）是结束时间。
            # 字典应包含所有format_kline_from_api需要的字段或提供它们。
            # WS K线负载直接包含这些字段。
            new_kline_dict = {
                'timestamp': kline_payload.get('t'), # 开始时间毫秒
                'open': kline_payload.get('o', '0'),
                'high': kline_payload.get('h', '0'),
                'low': kline_payload.get('l', '0'),
                'close': kline_payload.get('c', '0'),
                'volume': kline_payload.get('v', '0'),
                'close_time': kline_payload.get('T'), # 收盘时间毫秒
                'quote_volume': kline_payload.get('q', '0'),
                'number_of_trades': kline_payload.get('n', 0),
                'ignore': kline_payload.get('i', 0), # 负载中的'i'是忽略，不是间隔
                'is_closed': is_base_candle_closed # 负载中的'x'
            }
            # 将毫秒时间戳转换为带时区的datetime对象（UTC）
            new_kline_dict['timestamp'] = pd.to_datetime(new_kline_dict['timestamp'], unit='ms', utc=True)
            new_kline_dict['close_time'] = pd.to_datetime(new_kline_dict['close_time'], unit='ms', utc=True)

            # 将新的基础K线添加到持久化层
            kline_persistence.add_single_kline(new_kline_dict)
            logger.debug(f"收到并存储了 {config.SYMBOL} 的 {config.BASE_INTERVAL} K线更新 @ {new_kline_dict['timestamp'].isoformat()} (Closed: {is_base_candle_closed})")
            
            # 数据已存储。主循环将处理检查聚合K线关闭和触发策略。
            # 可选：在此处添加逻辑，如果需要在任何新的基础K线上立即触发显示更新。
            # 现在，显示更新与策略触发/调度相关联。


        elif 'e' in data and data['e'] == 'error':
            error_code = data.get('code')
            error_message = data.get('msg', '未提供错误消息')
            logger.error(f"Binance WebSocket错误 ({error_code}): {error_message}")
        # Add handler for other message types if needed (e.g., 'ping', 'pong', subscription confirmations)
        # The binance-futures-connector library's client might handle these internally.

    except json.JSONDecodeError:
        logger.error(f"从WebSocket消息解码JSON时出错：{message_str}")
    except Exception as e:
        logger.error(f"websocket_message_handler中处理消息时出错：{e}", exc_info=True)
        logger.error(f"问题原始消息：{message_str}")


def main_application():
    global binance_client, kline_persistence, ws_manager, trading_strategy # 声明全局变量

    setup_logging()

    # --- 初始化 Binance 客户端（用于 REST 和 WebSocket API） ---
    # 客户端需要 API Key 和 Secret 来执行交易操作，但对于公共数据（K线），有时不需要。
    # 策略需要客户端来进行交易和查仓位。
    can_trade = False
    if not config.API_KEY or not config.API_SECRET or config.API_KEY == "替换为您的API密钥" or config.API_SECRET == "替换为您的API密钥":
        logger.warning("API_KEY 或 API_SECRET 未配置/使用示例值。交易功能将不可用。策略将只能模拟信号。")
    else:
        can_trade = True
        logger.info("API_KEY 和 API_SECRET 已配置。交易功能已启用。")


    logger.system("正在初始化 Binance Client...")
    try:
        # Client初始化需要API Key/Secret，以及测试网/主网设置
        # 根据 app.core.config 的 IS_TESTNET 来设置
        # Use base_url specific to futures if MARKET_TYPE is FUTURES (though python-binance Client abstracts this)
        # Use requests_params for default timeout
        binance_client = Client(
            config.API_KEY,
            config.API_SECRET,
            testnet=config.IS_TESTNET,
            requests_params={'timeout': 30}
        )

        # Determine which market type client to use for initial ping/exchange info
        market_type_for_init = config.STRATEGY_CONFIG.get('MARKET_TYPE', 'FUTURES') # Default to FUTURES

        # Attempt to ping API to verify connection and get exchange info
        if market_type_for_init == 'FUTURES':
            logger.system("Attempting Binance FUTURES API ping...")
            binance_client.futures_ping()
            logger.success("Binance Futures API ping successful.")
            # Fetch futures exchange info
            logger.system("Fetching Binance FUTURES exchange info...")
            exchange_info = binance_client.futures_exchange_info()
            if not exchange_info:
                 raise RuntimeError("Failed to fetch Futures exchange info.")
            logger.success("Binance Futures exchange info fetched.")

        elif market_type_for_init == 'SPOT':
            logger.system("Attempting Binance SPOT API ping...")
            binance_client.ping()
            logger.success("Binance Spot API ping successful.")
            # Fetch spot exchange info
            logger.system("Fetching Binance SPOT exchange info...")
            exchange_info = binance_client.get_exchange_info()
            if not exchange_info:
                 raise RuntimeError("Failed to fetch Spot exchange info.")
            logger.success("Binance Spot exchange info fetched.")

        else:
            logger.warning(f"Configured MARKET_TYPE '{market_type_for_init}' unknown. Skipping market-specific ping and exchange info fetch.")
            # Proceed without market-specific checks if type is unknown

        logger.success("Binance Client initialized and connected successfully.")

    except Exception as e:
        logger.critical(f"初始化 Binance Client 失败：{e}\n{traceback.format_exc()}")
        logger.critical("无法连接到 Binance API 或获取交易信息。请检查网络连接、API 密钥和配置。")
        # If no valid client or exchange info, application cannot function.
        sys.exit(1) # Critical failure, exit.

    # 获取聚合间隔时间差以进行调度
    agg_interval_timedelta = interval_to_timedelta(config.AGG_INTERVAL)


    # --- 初始化数据持久化服务（内存或数据库） ---
    try:
        if config.DATA_STORE_TYPE == "database":
            logger.info("使用数据库存储（PostgreSQL）。")
            # DBManager constructor needs symbol and intervals
            kline_persistence = DBManager(
                symbol=config.SYMBOL,
                base_interval_str=config.BASE_INTERVAL,
                agg_interval_str=config.AGG_INTERVAL,
                historical_candles_to_display_count=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
        else: # 默认为内存存储
            logger.info("使用内存存储。")
            # KlineDataStore constructor needs intervals and display count
            kline_persistence = KlineDataStore(
                base_interval_str=config.BASE_INTERVAL,
                agg_interval_str=config.AGG_INTERVAL,
                historical_candles_to_display_count=config.HISTORICAL_AGG_CANDLES_TO_DISPLAY
            )
        logger.success(f"数据存储使用：{type(kline_persistence).__name__}。")

    except Exception as e_store_init:
        logger.critical(f"初始化数据持久化服务失败：{e_store_init}\n{traceback.format_exc()}")
        # If data storage cannot initialize (especially database), application cannot run
        sys.exit(1) # Critical failure, exit.


    # --- 初始化交易策略 ---
    # 策略需要 Binance Client 实例和策略配置
    # Also pass the can_trade flag if strategy needs to know if trading is enabled
    try:
        # Pass initialized client and STRATEGY_CONFIG dictionary
        # STRATEGY_CONFIG already contains all necessary params from .env
        trading_strategy = TradingStrategy(binance_client, config.STRATEGY_CONFIG)
        logger.success("交易策略实例初始化成功。")
        # Perform initial state synchronization
        trading_strategy.initialize_state()
    except Exception as e_strategy_init:
        logger.critical(f"初始化交易策略失败：{e_strategy_init}\n{traceback.format_exc()}")
        # If strategy cannot initialize (e.g., getting symbol info failed), application cannot run.
        sys.exit(1) # Critical failure, exit.


    # 1. 获取历史基础K线数据并加载到存储
    # 获取足够的历史基础K线，以确保策略运行时有足够的数据进行聚合和指标计算。
    # 策略内部需要 MAX_DF_LEN_STRATEGY + indicator_warmup_buffer worth of AGGREGATED data.
    # This requires a larger amount of BASE data depending on the interval ratio.
    # The data layer fetches enough base klines to satisfy this (based on MAX_DF_LEN_STRATEGY config).
    # We fetch initial historical data here to seed the storage.
    try:
        base_td = pd.Timedelta(config.BASE_INTERVAL)
        agg_td = pd.Timedelta(config.AGG_INTERVAL)

        if base_td.total_seconds() <= 0 or agg_td.total_seconds() <= 0:
            raise ValueError("基础间隔或聚合间隔必须是正持续时间。")
        if agg_td.total_seconds() < base_td.total_seconds():
             raise ValueError("聚合间隔必须大于或等于基础间隔。")

        # The strategy needs enough base klines to produce `strategy.required_agg_df_length_for_signals`
        # plus maybe a buffer for the AGGREGATED DF.
        # Number of base intervals per aggregated interval
        base_intervals_per_agg = int(agg_td.total_seconds() / base_td.total_seconds())
        # Number of aggregated bars needed * strategy.required_agg_df_length_for_signals
        # Add a buffer of aggregated bars to be safe, e.g., 20 more agg bars
        num_agg_bars_for_history = config.HISTORICAL_AGG_CANDLES_TO_DISPLAY + 20 # Display count + buffer
        min_base_klines_to_fetch = num_agg_bars_for_history * base_intervals_per_agg
        # Also ensure we fetch enough base klines to meet the strategy's MAX_DF_LEN_STRATEGY
        # This MAX_DF_LEN_STRATEGY is intended for the AGGREGATED DF in the strategy.
        # So we need base data covering the time span of MAX_DF_LEN_STRATEGY AGGREGATED bars.
        min_base_klines_for_strategy_logic = config.MAX_DF_LEN_STRATEGY * base_intervals_per_agg # This might be large
        # Use the larger of the two calculations (display needs vs. strategy logic needs)
        # Let's simplify for now and use a fixed multiplier of MAX_DF_LEN_STRATEGY
        # The data layer's get_klines_df should be responsible for fetching enough base klines
        # based on agg interval and strategy's MAX_DF_LEN_STRATEGY.
        # The DBManager already implements this logic to fetch based on historical_candles_to_display_count * ratio.
        # Let's just fetch a reasonable amount for display + buffer initially.
        num_klines_to_fetch_initial = config.HISTORICAL_AGG_CANDLES_TO_DISPLAY * base_intervals_per_agg + 50 # Display count * ratio + buffer

        logger.info(f"正在获取 {num_klines_to_fetch_initial} 个 '{config.BASE_INTERVAL}' {config.SYMBOL} 历史K线...")

        # Use binance_api_manager.fetch_historical_klines (uses requests, not Client)
        # Need to pass the correct base URL for futures if configured.
        historical_klines_list_raw = fetch_historical_klines(
            symbol=config.SYMBOL,
            interval=config.BASE_INTERVAL,
            num_klines_to_fetch=num_klines_to_fetch_initial,
            api_base_url=config.API_BASE_URL_FUTURES, # Pass futures specific URL
            max_limit_per_request=config.MAX_KLINE_LIMIT_PER_REQUEST
        )

        if historical_klines_list_raw:
            # Format the raw list of lists into list of dicts
            historical_klines_list_formatted = [format_kline_from_api(k) for k in historical_klines_list_raw]

            # Add the fetched historical Klines to the data storage
            kline_persistence.add_klines(historical_klines_list_formatted)
            logger.success(f"成功获取并存储了 {len(historical_klines_list_formatted)} 个历史基础K线。")

            # Load all available base klines from storage for initial aggregation and display
            base_df_for_initial_agg = kline_persistence.get_klines_df()

            if not base_df_for_initial_agg.empty:
                # Ensure base_df has DatetimeIndex for aggregation
                if 'timestamp' in base_df_for_initial_agg.columns:
                     base_df_for_initial_agg['timestamp'] = pd.to_datetime(base_df_for_initial_agg['timestamp'], utc=True)
                     base_df_for_initial_agg = base_df_for_initial_agg.set_index('timestamp').sort_index()
                # else: Assume DatetimeIndex if not 'timestamp' col

                min_ts = base_df_for_initial_agg.index.min().isoformat()
                max_ts = base_df_for_initial_agg.index.max().isoformat()
                logger.info(f"当前存储中的历史基础K线范围：{min_ts}到{max_ts}")

                # Aggregate the historical data for initial display
                initial_agg_df = aggregate_klines_df(base_df_for_initial_agg, config.AGG_INTERVAL)

                display_historical_aggregated_klines(
                    initial_agg_df,
                    config.SYMBOL,
                    config.AGG_INTERVAL,
                    config.HISTORICAL_AGG_CANDLES_TO_DISPLAY # Display configured number of agg klines
                )
            else:
                logger.warning("存储中没有基础K线数据。无法进行初始聚合显示。")
        else:
            logger.error("未能获取历史K线数据。WebSocket可能会提供实时数据，但历史回溯将缺失，策略可能无法立即计算指标。")

    except Exception as e_hist:
        logger.error(f"获取或处理历史K线数据时出错：{e_hist}\n{traceback.format_exc()}")
        # If historical data acquisition fails, indicator calculation might not be possible initially,
        # but WebSocket might still work. Don't exit, continue attempting to start WebSocket.


    # 2. 设置并启动WebSocket
    # WebSocket用于接收实时的 BASE_INTERVAL K线数据更新
    logger.system(f"正在启动 WebSocket 连接到 {config.SYMBOL}@{config.BASE_INTERVAL} K线流...")
    try:
        # UMFuturesWebsocketClient needs the symbol in lowercase
        ws_manager = BinanceWebsocketManager(
            symbol=config.SYMBOL.lower(), # Pass lowercase symbol for WS
            base_interval=config.BASE_INTERVAL, # Subscribe to base interval
            on_message_callback=websocket_message_handler # Pass handler function
        )
        ws_manager.start()
        logger.success("Binance WebSocket Manager 启动成功。")
    except Exception as e_ws:
        logger.critical(f"启动 WebSocket Manager 失败：{e_ws}\n{traceback.format_exc()}")
        logger.critical("无法接收实时数据。应用程序无法运行。")
        sys.exit(1)


    # 3. 主循环（实现定时策略执行和WebSocket状态检查）
    # 主线程在这里实现定时策略执行和WebSocket状态检查
    try:
        logger.info("应用程序正在运行。按 Ctrl+C 停止。")

        # 确定当前聚合间隔的开始时间
        # 第一次策略运行应该发生在*下一个*聚合K线关闭时。
        now_utc = datetime.now(timezone.utc)
        # 将当前时间对齐到当前聚合间隔的开始
        current_agg_interval_start = align_timestamp_to_interval(now_utc, agg_interval_timedelta, direction='start')
        # 下一个聚合K线关闭时间是current_agg_interval_start + agg_interval_timedelta
        next_agg_candle_close_time = current_agg_interval_start + agg_interval_timedelta

        logger.info(f"当前UTC时间: {now_utc.isoformat()}")
        logger.info(f"当前聚合间隔 '{config.AGG_INTERVAL}' 开始时间: {current_agg_interval_start.isoformat()}")
        logger.info(f"下一次策略执行（聚合K线关闭时间）计划在: {next_agg_candle_close_time.isoformat()}")

        while True:
            now_utc = datetime.now(timezone.utc)

            # 如果当前时间大于或等于下一个计划的策略执行时间
            if now_utc >= next_agg_candle_close_time:
                logger.system(f"检测到聚合K线关闭时间 @ {next_agg_candle_close_time.isoformat()}. 准备触发策略.")

                # --- 触发策略执行 ---
                if trading_strategy and kline_persistence:
                    # 从存储中获取所有可用的基础K线传递给策略
                    all_base_klines_df = kline_persistence.get_klines_df()
                    if all_base_klines_df is None or all_base_klines_df.empty:
                         logger.warning("主循环: 获取用于策略的基础K线DataFrame为空。无法运行策略此周期。")
                    else:
                        # 将完整的基础DF和计划的关闭时间传递给策略
                        # 策略负责验证数据并执行逻辑。
                        trading_strategy.on_candle_close(all_base_klines_df, next_agg_candle_close_time)

                else:
                    logger.error("主循环: 交易策略或K线持久化服务未初始化。无法触发策略。")

                # 计算下一次计划的策略执行时间
                # 在当前计划的关闭时间上添加一个聚合间隔持续时间
                next_agg_candle_close_time = next_agg_candle_close_time + agg_interval_timedelta
                logger.info(f"下一次策略执行计划在: {next_agg_candle_close_time.isoformat()}")

            # 计算到下一次计划执行或状态检查所需的睡眠时间
            time_to_sleep = (next_agg_candle_close_time - datetime.now(timezone.utc)).total_seconds()
            # 确保睡眠时间不为负值且不过长（例如，最多1分钟或直到下次检查）
            sleep_duration = max(1, min(time_to_sleep, 60)) # 至少1秒，最多60秒
            time.sleep(config.POSITION_CLOSE_VERIFY_DELAY) # 使用close verify delay作为循环间隔（任意选择）
            # 如果WS管理器存在，检查其状态
            if ws_manager:
                 if not ws_manager.is_active():
                     logger.warning("WebSocket 管理器报告未激活。尝试重启...")
                     ws_manager.stop() # Ensure stopped
                     time.sleep(config.API_RETRY_DELAY) # Wait before restart attempt
                     try:
                         ws_manager.start() # Attempt restart
                         if ws_manager.is_active():
                              logger.success("WebSocket 已成功重启。")
                         else:
                              logger.error("WebSocket 重启尝试失败。管理器未激活。正在退出。")
                              # If restart failed, break the loop to terminate
                              break
                     except Exception as e_ws_restart:
                          logger.error(f"重启 WebSocket 时发生错误：{e_ws_restart}\n{traceback.format_exc()}")
                          # If error during restart, break the loop to terminate
                          break
                 # else: ws_manager is active, continue loop

            else: # ws_manager was not initialized (critical failure)
                logger.error("WebSocket 管理器未初始化。无法继续。正在退出。")
                break # If manager is None, cannot continue


    except KeyboardInterrupt:
        logger.info("检测到键盘中断。正在关闭...")
    except Exception as e:
        logger.error(f"主循环中发生意外错误：{e}\n{traceback.format_exc()}")
    finally:
        # --- 应用程序清理 ---
        logger.system("正在执行应用程序关闭流程...")
        # Cancel any open orders placed by the bot before closing
        if binance_client and trading_strategy and trading_strategy._is_initialized:
             try:
                  # Attempt to cancel all bot's open orders on the symbol and market type
                  # This is a best-effort attempt during shutdown.
                  cancel_all_open_orders_for_symbol(binance_client, config.SYMBOL, config.STRATEGY_CONFIG.get('MARKET_TYPE', 'FUTURES'))
                  logger.success("已尝试取消所有Bot的关联订单。")
             except Exception as e_cancel:
                  logger.error(f"关闭时取消订单失败：{e_cancel}")


        if ws_manager:
            logger.system("停止 WebSocket 管理器...")
            ws_manager.stop()
            # Note: The binance-futures-connector library might manage threading internally.
            # Adding a small sleep might help ensure threads close, but robust joining
            # would require access to the internal thread objects.
            time.sleep(2) # Give a moment for WS thread to potentially clean up
            logger.success("WebSocket 管理器停止完成。")

        if kline_persistence and hasattr(kline_persistence, 'close') and callable(getattr(kline_persistence, 'close')):
            logger.system("关闭数据持久化连接...")
            try:
                 kline_persistence.close() # If DBManager or KlineDataStore has close method, call it
                 logger.success("数据持久化连接关闭完成。")
            except Exception as e_persist_close:
                 logger.error(f"关闭数据持久化连接时出错：{e_persist_close}")

        # Binance Client does not typically require an explicit close method.

        logger.system("应用程序已终止。")

