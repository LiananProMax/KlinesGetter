#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
from datetime import datetime, timezone
from decimal import Decimal
from binance.client import Client
from app.trading.order_management import OrderManagement  # 导入订单管理模块
from app.trading.state_sync import StateSync  # 导入状态同步模块
from app.trading.trade_execution import TradeExecution  # 导入交易执行模块
from app.core import config
from app.utils.trading_utils import (
    calculate_trade_qty, format_price, format_quantity,
    adjust_price_to_tick_size, generate_client_order_id
)
from app.api_clients.binance_trading_api import (
    get_precisions_and_tick_size, get_current_market_status,
    place_trade_order, cancel_trade_order, close_current_position,
    verify_position_closed, cancel_all_open_orders_for_symbol,
    get_open_orders_for_symbol_binance, get_open_oco_lists_binance, cancel_spot_oco_order, place_spot_oco_sell_order,
    is_bot_order
)
from app.trading.indicators import calculate_indicators

# 获取日志器
logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    简化后的交易策略类，基于 EMA-VWAP-RSI 逻辑。
    管理状态并根据提供的 K 线数据执行交易。
    不处理数据获取或持久化，依赖外部调用提供数据和触发执行。
    """

    def __init__(self, binance_client: Client, strategy_config: dict):
        """
        初始化交易策略。

        参数:
            binance_client: 已初始化的 Binance Client 实例。
            strategy_config: 包含策略参数的字典，从 config.STRATEGY_CONFIG 加载。
        """
        self.client = binance_client  # Binance 客户端实例
        self.config = strategy_config  # 策略配置字典
        self.symbol_str = str(self.config.get('SYMBOL', '')).upper()  # 交易对符号，大写
        self.market_type = str(self.config.get('MARKET_TYPE', '')).upper()  # 市场类型，大写
        self.interval_str = self.config.get('INTERVAL_STR', '')  # 聚合间隔字符串

        self.logger = logger  # 使用日志器

        # 验证基本配置
        if not all(self.config.get(k) for k in ['SYMBOL', 'MARKET_TYPE', 'INTERVAL_STR']):
            self.logger.critical("TradingStrategy: 缺少或为空的基本配置键 (SYMBOL, MARKET_TYPE, INTERVAL_STR)。")
            raise ValueError("缺少基本策略配置。")

        # 验证市场类型
        if self.market_type not in ['FUTURES', 'SPOT']:
            self.logger.critical(f"TradingStrategy: 无效的市场类型 '{self.market_type}'。必须是 'FUTURES' 或 'SPOT'。")
            raise ValueError(f"不支持的市场类型: {self.market_type}")

        # 获取间隔时间差
        try:
            self.interval_timedelta = config.interval_to_timedelta(self.interval_str)  # 从 utils.kline_utils 导入
            if self.interval_timedelta.total_seconds() <= 0:
                self.logger.critical(f"TradingStrategy: 无效的 INTERVAL_STR '{self.interval_str}'，导致非正的时间差。")
                raise ValueError(f"无效的 INTERVAL_STR: {self.interval_str}")
        except ValueError as e:
            self.logger.critical(f"TradingStrategy: 将 INTERVAL_STR '{self.interval_str}' 转换为时间差时出错: {e}。")
            raise

        # 获取精度和 tick 大小
        try:
            self.tick_size, self.price_precision, self.quantity_precision = get_precisions_and_tick_size(
                self.client, self.symbol_str, self.market_type
            )
            if None in (self.tick_size, self.price_precision, self.quantity_precision):
                self.logger.critical(f"TradingStrategy: 无法获取 Binance 精度信息 {self.symbol_str} ({self.market_type})。策略无法初始化。")
                raise RuntimeError(f"Binance 精度信息不可用。")
        except Exception as e_prec:
            self.logger.critical(f"TradingStrategy: 获取 Binance 精度信息时出错 {self.symbol_str} ({self.market_type}): {e_prec}\n{traceback.format_exc()}")
            raise RuntimeError("获取精度信息出错。") from e_prec

        self.logger.info(f"TradingStrategy 初始化完成，符号: {self.symbol_str}，市场: {self.market_type}，间隔: {self.interval_str}")
        self.logger.info(f"精度信息: TickSize={self.tick_size}，PricePrecision={self.price_precision}，QtyPrecision={self.quantity_precision}")

        # 策略参数
        self.short_ema_len = self.config.get('SHORT_EMA_LEN', 9)
        self.long_ema_len = self.config.get('LONG_EMA_LEN', 21)
        self.rsi_len = self.config.get('RSI_LEN', 14)
        self.rsi_overbought = self.config.get('RSI_OVERBOUGHT', 70.0)
        self.rsi_oversold = self.config.get('RSI_OVERSOLD', 30.0)
        self.vwap_period = self.config.get('VWAP_PERIOD', 'D').upper()
        self.use_sltp = self.config.get('USE_SLTP', True)
        self.stop_loss_ticks = self.config.get('STOP_LOSS_TICKS', 100)
        self.take_profit_ticks = self.config.get('TAKE_PROFIT_TICKS', 200)
        self.qty_percent = self.config.get('QTY_PERCENT', 0.90)
        self.max_df_len_strategy = self.config.get('MAX_DF_LEN_STRATEGY', 1000)

        # 计算聚合 DataFrame 所需的最小长度
        max_indicator_lookback = max(self.short_ema_len, self.long_ema_len, self.rsi_len)
        self.required_agg_df_length_for_signals = max_indicator_lookback + 1  # 至少需要这么多聚合柱才能计算信号

        # 验证策略参数
        if not (isinstance(self.short_ema_len, int) and self.short_ema_len > 0 and
                isinstance(self.long_ema_len, int) and self.long_ema_len > 0 and
                isinstance(self.rsi_len, int) and self.rsi_len > 0 and
                self.short_ema_len < self.long_ema_len):
            self.logger.critical("TradingStrategy: 策略参数错误: EMA/RSI 长度必须是正整数，且 SHORT_EMA_LEN < LONG_EMA_LEN。")
            raise ValueError("无效的策略参数: EMA/RSI 长度。")
        if not (isinstance(self.rsi_oversold, (int, float)) and isinstance(self.rsi_overbought, (int, float)) and
                0 <= self.rsi_oversold < self.rsi_overbought <= 100):
            self.logger.critical("TradingStrategy: 策略参数错误: RSI_OVERSOLD/OVERBOUGHT 必须在 0-100 范围内，且 OVERSOLD < OVERBOUGHT。")
            raise ValueError("无效的策略参数: RSI 水平。")
        if self.vwap_period not in ['D', 'W', 'M']:
            self.logger.critical(f"TradingStrategy: 无效的 VWAP_PERIOD '{self.vwap_period}'。必须使用 'D'、'W' 或 'M'。")
            raise ValueError(f"无效的 VWAP_PERIOD: {self.vwap_period}")
        if not (isinstance(self.qty_percent, (int, float)) and 0.0 < self.qty_percent <= 1.0):
            self.logger.critical(f"TradingStrategy: QTY_PERCENT 必须在 (0.0, 1.0] 范围内，但值为 {self.qty_percent}。")
            raise ValueError(f"无效的 QTY_PERCENT: {self.qty_percent}")
        if not (isinstance(self.max_df_len_strategy, int) and self.max_df_len_strategy > 0):
            self.logger.critical(f"TradingStrategy: MAX_DF_LEN_STRATEGY 必须是正整数，但值为 {self.max_df_len_strategy}。")
            raise ValueError(f"无效的 MAX_DF_LEN_STRATEGY: {self.max_df_len_strategy}")
        if self.use_sltp and (not isinstance(self.stop_loss_ticks, (int, float)) or self.stop_loss_ticks < 0 or
                              not isinstance(self.take_profit_ticks, (int, float)) or self.take_profit_ticks < 0):
            self.logger.warning(f"TradingStrategy: USE_SLTP 为 True，但 STOP_LOSS_TICKS ({self.stop_loss_ticks}) 或 TAKE_PROFIT_TICKS ({self.take_profit_ticks}) 无效/负数。止损/止盈可能无法正确放置。")

        self.logger.info(f"策略参数: ShortEMA={self.short_ema_len}，LongEMA={self.long_ema_len}，RSI_Len={self.rsi_len}，RSI_OB={self.rsi_overbought}，RSI_OS={self.rsi_oversold}，VWAP_Period={self.vwap_period}，USE_SLTP={self.use_sltp}，SL_Ticks={self.stop_loss_ticks}，TP_Ticks={self.take_profit_ticks}，QTY_PERCENT={self.qty_percent}，MAX_DF_LEN_STRATEGY={self.max_df_len_strategy}，RequiredAggDFLen={self.required_agg_df_length_for_signals}")

        # 状态变量
        self.current_position_side: str | None = None  # 'LONG'、'SHORT' 或 None
        self.current_position_qty: float = 0.0  # 当前仓位数量（绝对值）
        self.entry_price: float | None = None  # 平均入场价格
        # 关联订单状态 ID
        if self.market_type == 'FUTURES':
            self.active_sl_order_id: int | None = None  # 止损订单 ID
            self.active_tp_order_id: int | None = None  # 止盈订单 ID
        elif self.market_type == 'SPOT':
            self.active_oco_order_list_id: int | None = None  # OCO 订单列表 ID
        self._is_initialized = False  # 初始化标志

        # 初始化子模块实例
        self.order_manager = OrderManagement(self)  # 订单管理实例，传递自身引用
        self.state_syncer = StateSync(self)  # 状态同步实例，传递自身引用
        self.trade_executor = TradeExecution(self)  # 交易执行实例，传递自身引用

    def initialize_state(self):
        """进行初始状态同步。"""
        if self._is_initialized:
            self.logger.warning("TradingStrategy: 状态已初始化。")
            return
        self.logger.info("TradingStrategy: 进行初始状态同步...")
        self.state_syncer.sync_state_with_binance()  # 调用状态同步模块
        self._is_initialized = True
        self.log_current_state("初始同步后")

    def log_current_state(self, label: str = ""):
        """记录当前策略状态。"""
        state_log = f"TradingStrategy: 当前状态 ({label}) - {self.symbol_str} ({self.market_type}): "
        position_info = "无仓位"
        if self.current_position_side:
            position_info = f"{self.current_position_side} {self.current_position_qty:.{self.quantity_precision if self.quantity_precision is not None else 8}f}"
            if self.entry_price is not None:
                position_info += f" @ 入场价 {self.entry_price:.{self.price_precision if self.price_precision is not None else 8}f}"

        state_log += f"仓位: {position_info} | "

        associated_orders_info = "无活跃订单"
        if self.market_type == 'FUTURES':
            associated_orders_info = f"止损 ID: {self.active_sl_order_id}, 止盈 ID: {self.active_tp_order_id}"
        elif self.market_type == 'SPOT':
            associated_orders_info = f"OCO 清单 ID: {self.active_oco_order_list_id}"
        state_log += f"相关订单: {associated_orders_info}"

        self.logger.strategy(state_log)

    def on_candle_close(self, all_base_klines_df: pd.DataFrame, scheduled_agg_candle_close_time: datetime):
        """
        在聚合 K 线关闭时由外部调用，触发策略评估和交易执行。
        包含数据可用性检查。

        参数:
            all_base_klines_df (pd.DataFrame): 包含所有可用历史基础 K 线的 DataFrame。
            scheduled_agg_candle_close_time (datetime): 聚合 K 线关闭的时间。
        """
        if not self._is_initialized:
            self.logger.error("TradingStrategy: 状态未初始化。请先调用 initialize_state()。")
            return

        if all_base_klines_df is None or all_base_klines_df.empty:
            self.logger.warning("TradingStrategy: 接收到的 all_base_klines_df 为空。无法运行策略此周期。")
            return

        # 确保 DataFrame 有 DatetimeIndex 并排序
        if 'timestamp' in all_base_klines_df.columns:
            try:
                all_base_klines_df['timestamp'] = pd.to_datetime(all_base_klines_df['timestamp'], utc=True)
                base_df_indexed = all_base_klines_df.set_index('timestamp').sort_index()
            except Exception as e_idx:
                self.logger.error(f"TradingStrategy: on_candle_close: 无法将 'timestamp' 设置为 DatetimeIndex: {e_idx}。无法继续。")
                return
        elif isinstance(all_base_klines_df.index, pd.DatetimeIndex):
            base_df_indexed = all_base_klines_df.copy()
        else:
            self.logger.error("TradingStrategy: on_candle_close: 输入 DataFrame 没有 DatetimeIndex 或 'timestamp' 列。无法继续。")
            return

        # 数据可用性检查
        try:
            base_interval_timedelta = config.interval_to_timedelta(config.BASE_INTERVAL)
            required_last_base_candle_start_time = scheduled_agg_candle_close_time - base_interval_timedelta
        except ValueError:
            self.logger.error(f"TradingStrategy: 无法从 '{config.BASE_INTERVAL}' 计算基础间隔时间差。无法验证数据可用性。")
            return

        data_is_ready = False
        for attempt in range(config.DATA_SYNC_VERIFY_ATTEMPTS):
            latest_base_timestamp_in_df = base_df_indexed.index[-1] if not base_df_indexed.empty else None
            if latest_base_timestamp_in_df is not None and latest_base_timestamp_in_df >= required_last_base_candle_start_time:
                self.logger.info(f"数据验证通过。DF 中最新基础 K 线时间戳 ({latest_base_timestamp_in_df.isoformat()}) >= 聚合 K 线关闭于 {scheduled_agg_candle_close_time.isoformat()} 所需的开始时间 ({required_last_base_candle_start_time.isoformat()})。")
                data_is_ready = True
                break
            else:
                self.logger.warning(f"数据验证失败 (尝试 {attempt + 1}/{config.DATA_SYNC_VERIFY_ATTEMPTS})。DF 中最新基础 K 线时间戳 ({latest_base_timestamp_in_df.isoformat() if latest_base_timestamp_in_df else 'None'}) < 聚合 K 线关闭于 {scheduled_agg_candle_close_time.isoformat()} 所需的开始时间 ({required_last_base_candle_start_time.isoformat()})。{config.DATA_SYNC_VERIFY_DELAY}秒后重试...")
                if attempt < config.DATA_SYNC_VERIFY_ATTEMPTS - 1:
                    time.sleep(config.DATA_SYNC_VERIFY_DELAY)

        if not data_is_ready:
            self.logger.error(f"TradingStrategy: 经过 {config.DATA_SYNC_VERIFY_ATTEMPTS} 次尝试后，数据仍未就绪，聚合 K 线关闭时间为 {scheduled_agg_candle_close_time.isoformat()}。跳过此周期的策略评估。")
            return

        # 与 Binance 同步状态
        self.state_syncer.sync_state_with_binance(base_df_indexed)  # 传递基础 DF
        self.log_current_state("同步后")

        # 聚合基础 K 线数据
        try:
            df_aggregated = self.aggregate_klines_df(base_df_indexed, self.interval_str)  # 调用内部聚合函数或外部工具
            if df_aggregated is None or df_aggregated.empty:
                self.logger.warning(f"TradingStrategy: 间隔 {self.interval_str} 的聚合 DataFrame 为空。跳过策略评估。")
                return
        except Exception as e_agg:
            self.logger.error(f"TradingStrategy: 将数据聚合到 {self.interval_str} 时出错: {e_agg}\n{traceback.format_exc()}")
            self.logger.warning("TradingStrategy: 由于聚合失败，跳过本周期的策略执行。")
            return

        # 计算指标
        try:
            df_with_indicators = calculate_indicators(
                df_aggregated.copy(),
                self.short_ema_len, self.long_ema_len, self.rsi_len,
                self.vwap_period, self.rsi_overbought, self.rsi_oversold
            )
        except Exception as e_calc_ind:
            self.logger.error(f"TradingStrategy: 计算指标时出错: {e_calc_ind}\n{traceback.format_exc()}")
            self.logger.warning("TradingStrategy: 由于指标计算失败，跳过本周期的策略执行。")
            return

        # 验证指标是否有效
        if df_with_indicators.empty or len(df_with_indicators.dropna(subset=[f'EMA_{self.short_ema_len}', f'EMA_{self.long_ema_len}', 'RSI', 'VWAP'])) < 1:
            self.logger.warning(f"TradingStrategy: 指标计算后有效数据不足，无法进行策略决策。跳过本周期的策略运行。")
            return

        # 获取最新蜡烛数据
        closed_agg_candle = df_with_indicators.iloc[-1]
        latest_base_candle = base_df_indexed.iloc[-1]
        current_price_for_checks = latest_base_candle['close']

        if pd.isna(current_price_for_checks) or current_price_for_checks <= 0:
            self.logger.warning("TradingStrategy: 最新基础 K 线收盘价无效。跳过策略评估。")
            return

        # 执行交易逻辑
        self.trade_executor.evaluate_and_execute_trades(closed_agg_candle, current_price_for_checks)

    # 辅助函数：聚合 K 线数据（如果需要，可以移动到其他模块）
    def aggregate_klines_df(self, df_source: pd.DataFrame, agg_interval_str: str) -> pd.DataFrame:
        """聚合 K 线数据到指定间隔。"""
        from app.data_handling.kline_aggregator import aggregate_klines_df as agg_func  # 导入聚合函数
        return agg_func(df_source, agg_interval_str)
