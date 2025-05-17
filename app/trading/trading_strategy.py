#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
import traceback
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal # Import Decimal

# 使用app的日志和配置
import logging
logger = logging.getLogger(__name__) # 获取logger
from app.core import config # 导入配置常量

# 导入app的API客户端
from app.api_clients.binance_trading_api import (
    get_precisions_and_tick_size, get_current_market_status,
    place_trade_order, cancel_trade_order, close_current_position,
    verify_position_closed, cancel_all_open_orders_for_symbol,
    get_open_orders_for_symbol_binance, get_open_oco_lists_binance, cancel_spot_oco_order, place_spot_oco_sell_order,
    is_bot_order # Import bot order check utility
)

# 导入app的工具函数
from app.utils.trading_utils import (
    calculate_trade_qty, format_price, format_quantity,
    adjust_price_to_tick_size, generate_client_order_id # Import necessary utilities
)
from app.utils.kline_utils import interval_to_timedelta # 导入间隔转换函数

# 导入指标计算函数
from app.trading.indicators import calculate_indicators # 导入指标计算函数

# 导入Binance客户端（由main_app初始化并传递）
from binance.client import Client
# 导入异常类型方便捕获
from binance.exceptions import BinanceAPIException, BinanceRequestException


class TradingStrategy:
    """
    Simplified trading strategy based on EMA-VWAP-RSI logic.
    Manages state and executes trades based on provided K线 data DataFrame.
    Does NOT handle data fetching, persistence, or threading.
    Relies on an external caller to provide necessary data and trigger execution.
    """
    def __init__(self, binance_client: Client, strategy_config: dict):
        """
        Initializes the trading strategy.

        Args:
            binance_client: An initialized Binance Client instance.
            strategy_config (dict): Configuration dictionary containing strategy parameters.
                                    Expected to contain keys from config.STRATEGY_CONFIG.
        """
        self.client = binance_client
        self.config = strategy_config # Store the strategy specific config dictionary

        self.symbol_str = str(self.config.get('SYMBOL', '')).upper() # Ensure symbol is uppercase string
        self.market_type = str(self.config.get('MARKET_TYPE', '')).upper() # Ensure market type is uppercase string
        self.interval_str = self.config.get('INTERVAL_STR', '') # Aggregated interval string (e.g., '3m')

        self.logger = logging.getLogger(__name__) # Use the standard logger

        # Validate essential config keys
        if not all(self.config.get(k) for k in ['SYMBOL', 'MARKET_TYPE', 'INTERVAL_STR']):
             self.logger.critical("TradingStrategy: Missing or empty essential configuration keys (SYMBOL, MARKET_TYPE, INTERVAL_STR).")
             raise ValueError("Missing essential strategy configuration.")

        # Validate Market Type
        if self.market_type not in ['FUTURES', 'SPOT']:
            self.logger.critical(f"TradingStrategy: Invalid MARKET_TYPE '{self.market_type}'. Must be 'FUTURES' or 'SPOT'.")
            raise ValueError(f"Unsupported MARKET_TYPE: {self.market_type}")

        # Get interval timedelta (used for calculating candle boundaries)
        try:
            self.interval_timedelta = interval_to_timedelta(self.interval_str)
            if self.interval_timedelta.total_seconds() <= 0:
                 self.logger.critical(f"TradingStrategy: Invalid INTERVAL_STR '{self.interval_str}' resulted in non-positive timedelta.")
                 raise ValueError(f"Invalid INTERVAL_STR: {self.interval_str}")
        except ValueError as e:
            self.logger.critical(f"TradingStrategy: Error converting INTERVAL_STR '{self.interval_str}' to timedelta: {e}.")
            raise

        # Fetch and store precisions and tick size once
        # These are needed for order calculations and placement
        try:
            self.tick_size, self.price_precision, self.quantity_precision = get_precisions_and_tick_size(
                self.client, self.symbol_str, self.market_type
            )
            if None in (self.tick_size, self.price_precision, self.quantity_precision):
                 self.logger.critical(f"TradingStrategy: Failed to get Binance precisions for {self.symbol_str} ({self.market_type}). Strategy cannot initialize.")
                 raise RuntimeError(f"Binance precisions not available for strategy initialization for {self.symbol_str}.")
        except Exception as e_prec:
             self.logger.critical(f"TradingStrategy: Exception getting Binance precisions for {self.symbol_str} ({self.market_type}): {e_prec}\n{traceback.format_exc()}")
             raise RuntimeError("Exception getting Binance precisions.") from e_prec


        self.logger.info(f"TradingStrategy initialized for {self.symbol_str}, Market: {self.market_type}, Interval: {self.interval_str}")
        self.logger.info(f"Precisions: TickSize={self.tick_size}, PricePrecision={self.price_precision}, QtyPrecision={self.quantity_precision}")

        # --- Strategy Parameters (from config) ---
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
        self.max_df_len_strategy = self.config.get('MAX_DF_LEN_STRATEGY', 1000) # Max DF length needed by strategy (historical window)

        # Calculate the required length of the base DataFrame needed to calculate indicators
        # for the aggregated DataFrame of size MAX_DF_LEN_STRATEGY + indicator_warmup_buffer.
        # This is complex as it depends on the base_interval and agg_interval ratio.
        # A simpler (and safer) approach is to request enough *base* klines to cover
        # MAX_DF_LEN_STRATEGY * (agg_interval / base_interval) + a buffer.
        # The data layer fetches enough base klines to allow for aggregation and indicator calculation.
        # The strategy then *receives* the base DF and aggregates it.
        # The number of base klines needed must be sufficient to produce at least
        # MAX_DF_LEN_STRATEGY + some buffer (e.g., max indicator lookback + margin)
        # of *aggregated* candles for the strategy to use.

        # Let's estimate the minimum number of *base* candles needed for the strategy to work.
        # The strategy needs indicators on the *aggregated* candles.
        # To calculate indicators on an aggregated DF of size N, where N includes warm-up,
        # the underlying base DF must cover that time span.
        # Required aggregated bars for strategy = MAX_DF_LEN_STRATEGY + indicator_warmup_buffer
        # Max indicator lookback applies to the AGGREGATED bars.
        # Let's assume the data store provides sufficient base data based on its own logic
        # and MAX_DF_LEN_STRATEGY config. The strategy receives the base DF and works from there.
        # The key is that calculate_indicators needs to be run on the AGGREGATED DF.

        # Let's calculate the required minimum length of the *aggregated* DF for indicators
        # to be available for the *last closed candle* (iloc[-1] after aggregation).
        max_indicator_lookback_agg = max(self.short_ema_len, self.long_ema_len, self.rsi_len)
        # We need enough aggregated bars so that after indicator calculation NaNs,
        # we have at least 1 valid row (the latest closed candle) to check signals on.
        self.required_agg_df_length_for_signals = max_indicator_lookback_agg + 1 # Need this many AGG bars minimum


        # Validate strategy parameters (basic checks)
        if not (isinstance(self.short_ema_len, int) and self.short_ema_len > 0 and
                isinstance(self.long_ema_len, int) and self.long_ema_len > 0 and
                isinstance(self.rsi_len, int) and self.rsi_len > 0 and
                self.short_ema_len < self.long_ema_len):
            self.logger.critical("TradingStrategy: Strategy parameter error: EMA/RSI lengths must be positive integers, and SHORT_EMA_LEN < LONG_EMA_LEN.")
            raise ValueError("Invalid strategy parameters: EMA/RSI lengths.")
        if not (isinstance(self.rsi_oversold, (int, float)) and isinstance(self.rsi_overbought, (int, float)) and
                0 <= self.rsi_oversold < self.rsi_overbought <= 100):
             self.logger.critical("TradingStrategy: Strategy parameter error: RSI_OVERSOLD/OVERBOUGHT must be 0-100 and OVERSOLD < OVERBOUGHT.")
             raise ValueError("Invalid strategy parameters: RSI levels.")
        if self.vwap_period not in ['D', 'W', 'M']:
            self.logger.critical(f"TradingStrategy: Invalid VWAP_PERIOD '{self.vwap_period}'. Use 'D', 'W', or 'M'.")
            raise ValueError(f"Invalid VWAP_PERIOD: {self.vwap_period}")
        if not (isinstance(self.qty_percent, (int, float)) and 0.0 < self.qty_percent <= 1.0):
             self.logger.critical(f"TradingStrategy: QTY_PERCENT must be between 0.0 and 1.0 (exclusive of 0). Value is {self.qty_percent}. Cannot proceed.")
             raise ValueError(f"Invalid QTY_PERCENT: {self.qty_percent}")
        if not (isinstance(self.max_df_len_strategy, int) and self.max_df_len_strategy > 0):
             self.logger.critical(f"TradingStrategy: MAX_DF_LEN_STRATEGY must be a positive integer, but is {self.max_df_len_strategy}.")
             raise ValueError(f"Invalid MAX_DF_LEN_STRATEGY: {self.max_df_len_strategy}")
        # SL/TP ticks can be 0 if USE_SLTP is False, but if USE_SLTP is True, they should ideally be > 0.
        if self.use_sltp and (not isinstance(self.stop_loss_ticks, (int, float)) or self.stop_loss_ticks < 0 or
                              not isinstance(self.take_profit_ticks, (int, float)) or self.take_profit_ticks < 0):
             self.logger.warning(f"TradingStrategy: USE_SLTP is True, but STOP_LOSS_TICKS ({self.stop_loss_ticks}) or TAKE_PROFIT_TICKS ({self.take_profit_ticks}) are invalid/negative. SL/TP might not be placed correctly.")


        self.logger.info(f"Strategy Params: ShortEMA={self.short_ema_len}, LongEMA={self.long_ema_len}, RSI_Len={self.rsi_len}, RSI_OB={self.rsi_overbought}, RSI_OS={self.rsi_oversold}, VWAP_Period={self.vwap_period}, USE_SLTP={self.use_sltp}, SL_Ticks={self.stop_loss_ticks}, TP_Ticks={self.take_profit_ticks}, QTY_PERCENT={self.qty_percent}, MAX_DF_LEN_STRATEGY={self.max_df_len_strategy}, RequiredAggDFLen={self.required_agg_df_length_for_signals}")

        # --- Strategy State Variables (In-memory only, not persisted across restarts) ---
        # Represents the bot's *belief* about the current position based on its actions and API sync
        self.current_position_side: str | None = None   # 'LONG', 'SHORT', or None
        self.current_position_qty: float = 0.0     # Current open position quantity (abs value)
        self.entry_price: float | None = None             # Average entry price of the current position

        # Associated order state (Futures: SL/TP IDs, Spot: OCO List ID)
        if self.market_type == 'FUTURES':
             self.active_sl_order_id: int | None = None # Binance order ID for active SL order
             self.active_tp_order_id: int | None = None # Binance order ID for active TP order
        elif self.market_type == 'SPOT':
             self.active_oco_order_list_id: int | None = None # Binance orderListId for active OCO order

        self._is_initialized = False # Flag to indicate initial sync is needed

    def initialize_state(self):
        """Performs initial state synchronization with Binance."""
        if self._is_initialized:
            self.logger.warning("TradingStrategy: State already initialized.")
            return

        self.logger.system("TradingStrategy: Performing initial state synchronization...")
        self._sync_state_with_binance() # Syncs position and associated orders
        self._is_initialized = True
        self.logger.success("TradingStrategy: Initial state synchronization complete.")
        self.log_current_state("After Initial Sync")


    def log_current_state(self, label: str = ""):
        """Logs the current state of the strategy."""
        state_log = f"TradingStrategy: Current State ({label}) for {self.symbol_str} ({self.market_type}): "
        position_info = "No Position"
        if self.current_position_side:
             position_info = f"{self.current_position_side} {self.current_position_qty:.{self.quantity_precision if self.quantity_precision is not None else 8}f}"
             if self.entry_price is not None:
                 position_info += f" @ Entry {self.entry_price:.{self.price_precision if self.price_precision is not None else 8}f}"

        state_log += f"Position: {position_info} | "

        associated_orders_info = "None Active"
        if self.market_type == 'FUTURES':
             associated_orders_info = f"SL ID: {self.active_sl_order_id}, TP ID: {self.active_tp_order_id}"
        elif self.market_type == 'SPOT':
             associated_orders_info = f"OCO List ID: {self.active_oco_order_list_id}"
        state_log += f"Associated Orders: {associated_orders_info}"

        self.logger.strategy(state_log)


    def on_candle_close(self, all_base_klines_df: pd.DataFrame):
        """
        Called by the external data handling module when a new aggregated candle closes.
        Triggers strategy evaluation and trade execution.

        Args:
            all_base_klines_df (pd.DataFrame): DataFrame containing all available
                                                historical base Klines from the data store.
                                                Must have a DatetimeIndex or 'timestamp' column (UTC).
        """
        if not self._is_initialized:
            self.logger.error("TradingStrategy: State not initialized. Call initialize_state() first.")
            return

        if all_base_klines_df is None or all_base_klines_df.empty:
            self.logger.warning("TradingStrategy: Received empty all_base_klines_df. Cannot run strategy this cycle.")
            return

        # Ensure the DataFrame has a DatetimeIndex for resampling and indicator calculations
        if 'timestamp' in all_base_klines_df.columns:
             try:
                  all_base_klines_df['timestamp'] = pd.to_datetime(all_base_klines_df['timestamp'], utc=True)
                  base_df_indexed = all_base_klines_df.set_index('timestamp').sort_index()
             except Exception as e_idx:
                  self.logger.error(f"TradingStrategy: on_candle_close: Could not set 'timestamp' as DatetimeIndex: {e_idx}. Cannot proceed.")
                  return
        elif isinstance(all_base_klines_df.index, pd.DatetimeIndex):
            base_df_indexed = all_base_klines_df.copy() # Ensure index is DatetimeIndex
        else:
            self.logger.error("TradingStrategy: on_candle_close: Input DataFrame has no DatetimeIndex or 'timestamp' column. Cannot proceed.")
            return


        # 1. Sync state with Binance (to get the *actual* current position and orders)
        # This is done at the start of every cycle based on a closed aggregated candle
        self._sync_state_with_binance(latest_base_kline_df=base_df_indexed) # Pass base DF for spot sync potential needs
        self.log_current_state("After Sync")


        # 2. Aggregate the base Klines DataFrame into the strategy's interval
        try:
            df_aggregated = aggregate_klines_df(base_df_indexed, self.interval_str)
            if df_aggregated is None or df_aggregated.empty:
                 self.logger.warning(f"TradingStrategy: Aggregated DataFrame is empty for interval {self.interval_str}. Skipping strategy evaluation.")
                 return

            # Ensure aggregated DF has a DatetimeIndex
            if 'timestamp' in df_aggregated.columns:
                 df_aggregated['timestamp'] = pd.to_datetime(df_aggregated['timestamp'], utc=True)
                 df_aggregated = df_aggregated.set_index('timestamp').sort_index()
            # else: already assumed to have DatetimeIndex from aggregation utility


            # We need enough aggregated data to calculate indicators for the *last* bar in the aggregated DF.
            # The number of rows required for indicators depends on the longest indicator length.
            if len(df_aggregated) < self.required_agg_df_length_for_signals:
                 self.logger.warning(f"TradingStrategy: Not enough aggregated data ({len(df_aggregated)} rows) for indicator calculation ({self.required_agg_df_length_for_signals} needed). Skipping strategy run this cycle.")
                 return

        except Exception as e_agg:
             self.logger.error(f"TradingStrategy: Error during data aggregation to {self.interval_str}: {e_agg}\n{traceback.format_exc()}")
             self.logger.warning("TradingStrategy: Skipping strategy execution this cycle due to aggregation failure.")
             return # Stop here if aggregation fails


        # 3. Calculate indicators on the AGGREGATED DataFrame
        try:
            df_with_indicators_agg = calculate_indicators(
                df_aggregated.copy(), # Pass a copy to prevent modification
                self.short_ema_len, self.long_ema_len, self.rsi_len,
                self.vwap_period, self.rsi_overbought, self.rsi_oversold
            )
        except Exception as e_calc_ind:
            self.logger.error(f"TradingStrategy: Error calculating indicators on aggregated data: {e_calc_ind}\n{traceback.format_exc()}")
            self.logger.warning("TradingStrategy: Skipping strategy execution this cycle due to indicator calculation failure.")
            return # Stop here if indicators fail

        # Verify indicators were calculated and we still have enough rows in the AGG DF
        # After indicator calculation, some initial rows might have NaNs.
        # We need at least 1 row with valid indicators to check signals on (the last bar).
        if df_with_indicators_agg is None or df_with_indicators_agg.empty or len(df_with_indicators_agg.dropna(subset=[f'EMA_{self.short_ema_len}', f'EMA_{self.long_ema_len}', 'RSI', 'VWAP'])) < 1:
             self.logger.warning(f"TradingStrategy: Not enough valid data ({len(df_with_indicators_agg) if df_with_indicators_agg is not None else 'None'} rows total) after indicator calculation on aggregated data for strategy decision (need at least 1 valid bar). Skipping strategy run this cycle.")
             return

        # The strategy decision is based on the *latest closed aggregated candle*.
        # This is the LAST row (iloc[-1]) of the df_with_indicators_agg DataFrame.
        closed_agg_candle = df_with_indicators_agg.iloc[-1]

        # We also need the *most recent price* for certain checks (like implied SL/TP trigger).
        # Use the close price of the *latest base candle* received.
        # base_df_indexed is already sorted, so iloc[-1] is the latest base candle.
        latest_base_candle = base_df_indexed.iloc[-1]
        current_price_for_checks = latest_base_candle['close']

        if current_price_for_checks is None or pd.isna(current_price_for_checks):
            self.logger.warning("TradingStrategy: Latest base candle close price is invalid (NaN/None). Cannot perform checks relying on current price. Skipping strategy evaluation this cycle.")
            return

        # 4. Evaluate signals on the closed aggregated candle and execute trades
        self._evaluate_and_execute_trades(closed_agg_candle, current_price_for_checks)


    def _sync_state_with_binance(self, latest_base_kline_df: pd.DataFrame | None = None):
        """
        Syncs local strategy state (position, active associated orders) with Binance.
        Args:
             latest_base_kline_df (pd.DataFrame | None): Optional, used for SPOT balance check logging context.
        """
        self.logger.data(f"TradingStrategy: Syncing local state with Binance API for {self.symbol_str} ({self.market_type})...")
        current_qty_before_sync = self.current_position_qty # Store for logging change
        current_side_before_sync = self.current_position_side # Store for logging change

        try:
            # 1. Get Position Status using the shared API helper
            # Pass latest_base_kline_df for potential use in SPOT balance check logging if needed by _get_current_spot_status
            position_data = get_current_market_status(self.client, self.symbol_str, self.market_type, latest_kline_df=latest_base_kline_df) # Pass DF here
            # position_data is expected to be (side, quantity, entry_price, status_code) for FUTURES
            # or (synced_position_side, synced_position_qty, synced_entry_price, status_code) for SPOT
            position_api_status = position_data[-1] if position_data and isinstance(position_data, tuple) and len(position_data) > 0 else 'ERROR'

            synced_position_side = None
            synced_position_qty = 0.0
            synced_entry_price = self.entry_price # Keep existing local entry price by default


            if position_api_status == 'OK':
                # Unpack position_data tuple based on market type
                if self.market_type == 'FUTURES':
                    synced_position_side, synced_position_qty, synced_entry_price_from_api, _ = position_data
                    # Note: Binance Futures API positionAmt can be negative for SHORT.
                    # The helper returns absolute quantity and derives side.
                    if synced_qty is not None and abs(Decimal(str(synced_position_qty or 0))) > Decimal('0'):
                        # Position exists. Update local entry price from API if API provided a valid one.
                        if synced_entry_price_from_api is not None and synced_entry_price_from_api > 0:
                             synced_entry_price = synced_entry_price_from_api
                             self.logger.data(f"Sync: Synced FUTURES entry price with API: {synced_entry_price}")
                        else:
                             self.logger.warning(f"Sync: API reported FUTURES position ({synced_position_qty}) but entry price was invalid ({synced_entry_price_from_api}). Keeping local entry price {self.entry_price}.")
                    else: # API reports zero quantity
                         synced_position_side = None
                         synced_position_qty = 0.0
                         synced_entry_price = None # Clear entry price if position is gone

                elif self.market_type == 'SPOT':
                    synced_position_side, synced_position_qty, synced_entry_price_from_api, _ = position_data # Note: synced_entry_price_from_api is usually None for SPOT API
                    # synced_position_side will be 'LONG' if holding base asset > threshold, None otherwise.
                    # synced_position_qty will be the base asset balance if in position, 0 otherwise.

                    if synced_position_side == config.SIDE_LONG and synced_position_qty is not None and synced_position_qty > 0:
                         # Position exists. SPOT API doesn't give reliable average entry.
                         # Keep local entry price unless it's None (e.g., bot restart), then we cannot reliably recover it without trade history.
                         # For this simple module, if local entry_price is None, it remains None.
                         if self.entry_price is None:
                             self.logger.warning("Sync: SPOT position detected, but local entry_price is None and API price is unreliable. Entry price will be N/A until next entry.")
                             synced_entry_price = None # Explicitly ensure it's None if it was

                    else: # API reports no significant base balance or not LONG side
                         synced_position_side = None
                         synced_position_qty = 0.0
                         synced_entry_price = None # Clear entry price if position is gone

            elif position_api_status == 'NO_POSITION': # API confirms no position for the symbol
                 synced_position_side = None
                 synced_position_qty = 0.0
                 synced_entry_price = None # Clear entry price

            else: # API call failed (API_ERROR, CLIENT_ERROR, PARSE_ERROR)
                self.logger.error(f"Sync: Failed to get reliable Binance position status ({position_api_status}) for {self.symbol_str} ({self.market_type}). Local position state may be inaccurate.")
                # Keep current local state in case of API error, don't overwrite with None/0.

            # Update local position state if it changed
            # Use Decimal comparison for quantity robustness
            qty_changed = abs(Decimal(str(self.current_position_qty or 0))) != abs(Decimal(str(synced_position_qty or 0)))
            side_changed = self.current_position_side != synced_position_side

            if side_changed or qty_changed:
                self.logger.trading(f"Sync: Position state changed: {current_side_before_sync} {current_qty_before_sync} -> {synced_position_side} {synced_position_qty}")
                self.current_position_side = synced_position_side
                self.current_position_qty = synced_position_qty

                # Update entry price based on the synced position (if position exists)
                # Prioritize API entry price for Futures if available, else keep local.
                # For Spot, API entry price is unreliable, stick to local (if any).
                if self.current_position_side:
                    if self.market_type == 'FUTURES' and synced_entry_price is not None and synced_entry_price > 0:
                        self.entry_price = synced_entry_price
                    # For SPOT, entry price is tracked internally upon entry, not synced from API
                    # unless we implement fetching trade history, which is more complex.
                    # If local entry price is None when a position is detected, it stays None.
                else: # Position is now zero
                     self.entry_price = None
                     self.logger.data("Sync: Position is now zero, clearing local entry price.")

            # If we are now out of position based on sync, clear all associated order state IDs
            # This happens if Binance reported a position before, but now reports none (or zero quantity).
            if (self.current_position_qty is None or self.current_position_qty <= 0) and \
               (current_qty_before_sync is not None and current_qty_before_sync > 0): # Position changed from >0 to <=0
                 self.logger.trading(f"Sync: Detected position closed on Binance ({self.symbol_str}). Clearing local associated order state IDs.")
                 self.clear_associated_order_state(reason="Position closed on Binance.") # Clear local IDs


            # 2. Get and sync Associated Order Status (SL/TP for Futures, OCO for Spot)
            # Use shared API helper to get all open orders for this symbol
            open_orders_binance = get_open_orders_for_symbol_binance(self.client, self.symbol_str, self.market_type)

            if open_orders_binance is None:
                self.logger.warning(f"Sync: Failed to fetch open orders for {self.symbol_str}. Local associated order state may be inaccurate.")
                # Keep current local associated order state if API call failed
                # This means self.active_sl/tp/oco will remain what they were before this sync attempt.
                # This is safer than setting to None if the orders might still be active.
            else: # API call succeeded (returned list or empty list)
                if self.market_type == 'FUTURES':
                    synced_sl_id, synced_tp_id = None, None
                    # Find active bot SL/TP orders from the list of open orders
                    for order in open_orders_binance:
                        order_id = order.get('orderId')
                        order_type = order.get('type')
                        order_status = order.get('status')
                        is_reduce_only = order.get('reduceOnly', False) # Default to False if key missing
                        client_order_id = order.get('clientOrderId')

                        # Identify bot's reduceOnly orders that are currently active (NEW or PARTIALLY_FILLED or PENDING_CANCEL)
                        # PENDING_CANCEL might still appear in open orders list briefly.
                        if is_reduce_only is True and order_status in ['NEW', 'PARTIALLY_FILLED', 'PENDING_CANCEL']:
                            # Further filter by clientOrderId pattern for confidence
                            if is_bot_order(client_order_id):
                                if order_type in [config.FUTURE_ORDER_TYPE_STOP_MARKET, config.FUTURE_ORDER_TYPE_STOP]:
                                    # For simplicity, assume only one active SL order at a time placed by the bot
                                    if synced_sl_id is None: # Only sync the first one found
                                         synced_sl_id = order_id
                                    else: logger.warning(f"Sync: Found multiple active bot Futures SL orders for {self.symbol_str}. Syncing the first one found ({synced_sl_id}). Please investigate.")
                                elif order_type in [config.FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET, config.FUTURE_ORDER_TYPE_TAKE_PROFIT]:
                                    # For simplicity, assume only one active TP order at a time placed by the bot
                                    if synced_tp_id is None: # Only sync the first one found
                                         synced_tp_id = order_id
                                    else: logger.warning(f"Sync: Found multiple active bot Futures TP orders for {self.symbol_str}. Syncing the first one found ({synced_tp_id}). Please investigate.")
                                else:
                                     logger.warning(f"Sync: Found unexpected bot open order type '{order_type}' with reduceOnly=True for {self.symbol_str}. Order ID: {order_id}, ClientOrderID: {client_order_id}")

                    # Update local SL/TP state if different from synced state
                    if self.active_sl_order_id != synced_sl_id:
                        self.logger.trading(f"Sync: Futures SL order ID updated: {self.active_sl_order_id} -> {synced_sl_id}")
                        self.active_sl_order_id = synced_sl_id
                    if self.active_tp_order_id != synced_tp_id:
                        self.logger.trading(f"Sync: Futures TP order ID updated: {self.active_tp_order_id} -> {synced_tp_id}")
                        self.active_tp_order_id = synced_tp_id


                elif self.market_type == 'SPOT':
                    synced_oco_list_id = None
                    # Find active bot OCO lists from the list of open orders (which includes OCO legs)
                    # It's simpler to use the get_open_oco_lists_binance helper directly for OCO lists
                    open_oco_lists = get_open_oco_lists_binance(self.client, self.symbol_str)

                    if open_oco_lists is not None: # API call succeeded (returns list or empty list)
                        for oco_list in open_oco_lists:
                            list_id = oco_list.get('orderListId')
                            list_status_type = oco_list.get('listStatusType') # e.g., 'RESPONSE', 'EXECUTION_STARTED', 'ALL_DONE', 'CANCELED', 'REJECT'
                            list_client_order_id = oco_list.get('listClientOrderId')

                            # Check if this OCO list was placed by this bot and is currently active/executing
                            # 'EXECUTING' means one leg was triggered but the other isn't yet canceled/filled.
                            # 'ALL_DONE' or 'CANCELED' mean it's no longer active.
                            # 'RESPONSE' might be briefly active before execution starts? Let's consider EXECUTING as the primary active state.
                            if is_bot_order(list_client_order_id) and list_status_type in ['EXECUTING', 'RESPONSE']: # Also consider 'RESPONSE' briefly
                                if synced_oco_list_id is None: # Only sync the first active OCO list found
                                    synced_oco_list_id = list_id
                                else: logger.warning(f"Sync: Found multiple active bot Spot OCO lists for {self.symbol_str}. Syncing the first one found ({synced_oco_list_id}). Please investigate.")
                                break # Found an active OCO list, assuming only one is managed at a time

                    else: # get_open_oco_lists_binance failed (returned None)
                         self.logger.warning(f"Sync: Failed to fetch open SPOT OCO lists for {self.symbol_str}. Local OCO state may be inaccurate.")
                         # Keep local state in case of API error

                    # Update local OCO list ID state
                    if self.active_oco_order_list_id != synced_oco_list_id:
                        self.logger.trading(f"Sync: Spot OCO List ID updated: {self.active_oco_order_list_id} -> {synced_oco_list_id}")
                        self.active_oco_order_list_id = synced_oco_list_id
            else:
                 # This case should not happen if get_open_orders_for_symbol_binance returned None,
                 # which is handled above. This is a safeguard.
                 self.logger.error(f"Sync: get_open_orders_for_symbol_binance returned unexpected type: {type(open_orders_binance)}")


        except Exception as sync_e:
            self.logger.error(f"TradingStrategy: Error during state sync with Binance for {self.symbol_str}: {sync_e}\n{traceback.format_exc()}")
            # If sync fails critically, local state might be inaccurate.
            # Log error but do NOT raise, allowing strategy run to potentially proceed
            # with potentially stale state (less safe, but avoids bot stopping).

    def clear_state(self, reason: str = ""):
        """Clears all local strategy state variables."""
        self.logger.trading(f"TradingStrategy: Clearing all local state for {self.symbol_str}. Reason: {reason}")
        # Log current state before clearing
        self.log_current_state("Before Clear")

        self.current_position_side = None
        self.current_position_qty = 0.0
        self.entry_price = None
        self.clear_associated_order_state(reason=reason) # Clear associated order IDs


    def clear_associated_order_state(self, reason: str = ""):
        """Clears local state variables related to associated orders (SL/TP/OCO IDs).
           Does NOT cancel orders on the exchange.
        """
        self.logger.data(f"TradingStrategy: Clearing local associated order state for {self.symbol_str}. Reason: {reason}")
        if self.market_type == 'FUTURES':
             if self.active_sl_order_id is not None or self.active_tp_order_id is not None:
                 self.logger.data(f"Current Futures SL_ID: {self.active_sl_order_id}, Current TP_ID: {self.active_tp_order_id}")
                 self.active_sl_order_id = None
                 self.active_tp_order_id = None
                 self.logger.data("Cleared Futures SL/TP IDs.")
             else:
                 self.logger.data("No active Futures SL/TP IDs to clear.")
        elif self.market_type == 'SPOT':
             if self.active_oco_order_list_id is not None:
                 self.logger.data(f"Current Spot OCO_ListID: {self.active_oco_order_list_id}")
                 self.active_oco_order_list_id = None
                 self.logger.data("Cleared Spot OCO List ID.")
             else:
                 self.logger.data("No active Spot OCO List ID to clear.")
        else:
             self.logger.warning(f"TradingStrategy: clear_associated_order_state called for unknown market type {self.market_type}.")


    def cancel_associated_orders(self):
        """Cancels active associated orders (SL/TP/OCO) on the exchange if local state indicates they are active."""
        # This function attempts cancellation based on the *local* state IDs.
        # The _sync_state_with_binance on the next cycle is responsible for
        # clearing the local IDs if the orders are confirmed cancelled on Binance.
        self.logger.trading(f"TradingStrategy: Attempting to cancel associated orders for {self.symbol_str} ({self.market_type})...")
        cancelled_any = False # Flag to indicate if any cancellation API call was *attempted* and succeeded in sending the request
        attempted_cancellations = False # Flag to indicate if any cancellation was attempted at all (based on local IDs)

        if self.market_type == 'FUTURES':
            # Cancel SL order if active locally
            if self.active_sl_order_id is not None:
                attempted_cancellations = True
                self.logger.trading(f"Cancelling Futures SL order {self.active_sl_order_id} for {self.symbol_str}...")
                # cancel_trade_order handles market type internally and uses safe_api_call
                if cancel_trade_order(self.client, self.symbol_str, self.active_sl_order_id, self.market_type):
                    cancelled_any = True
                else:
                    self.logger.warning(f"Failed to send cancel request for Futures SL order {self.active_sl_order_id}.")

            # Cancel TP order if active locally
            if self.active_tp_order_id is not None:
                attempted_cancellations = True
                self.logger.trading(f"Cancelling Futures TP order {self.active_tp_order_id} for {self.symbol_str}...")
                # cancel_trade_order handles market type internally and uses safe_api_call
                if cancel_trade_order(self.client, self.symbol_str, self.active_tp_order_id, self.market_type):
                    cancelled_any = True
                else:
                    self.logger.warning(f"Failed to send cancel request for Futures TP order {self.active_tp_order_id}.")

        elif self.market_type == 'SPOT':
            # Cancel OCO list if active locally
            if self.active_oco_order_list_id is not None:
                 attempted_cancellations = True
                 self.logger.trading(f"Cancelling Spot OCO order list {self.active_oco_order_list_id} for {self.symbol_str}...")
                 # cancel_spot_oco_order uses safe_api_call
                 if cancel_spot_oco_order(self.client, self.symbol_str, order_list_id=self.active_oco_order_list_id):
                      cancelled_any = True
                 else:
                      self.logger.warning(f"Failed to send cancel request for Spot OCO order list {self.active_oco_order_list_id}.")
        else:
             self.logger.warning(f"TradingStrategy: cancel_associated_orders called for unknown market type {self.market_type}.")


        if attempted_cancellations:
            if cancelled_any:
                self.logger.trading("TradingStrategy: Successfully sent cancellation requests for some associated orders.")
            else:
                self.logger.warning("TradingStrategy: Failed to send cancellation requests for any associated orders.")
             # Local state IDs are cleared by _sync_state_with_binance on the *next* cycle
             # if the cancellation is confirmed on Binance. This is more reliable than clearing here.
        else:
             self.logger.data("TradingStrategy: No associated orders active locally to cancel.")


        return cancelled_any # Return True if at least one cancellation request was sent successfully

    def _place_associated_orders(self, current_price_for_checks: float | Decimal):
        """
        Places SL/TP or OCO orders for an existing position if USE_SLTP is True.
        Assumes self.current_position_side, self.entry_price, self.current_position_qty,
        and precisions/tick_size are correctly set BEFORE calling this.
        Updates local state IDs (active_sl/tp_order_id or active_oco_order_list_id) upon
        successful API response (order accepted).
        Args:
             current_price_for_checks (float | Decimal): The current market price used for safety checks.
        Returns:
             bool: True if all required associated orders were successfully requested via API, False otherwise.
        """
        if not self.use_sltp:
            self.logger.data("TradingStrategy: USE_SLTP is False. Not placing associated orders.")
            return False

        # Check if orders already seem active based on local state
        # This check relies on the _sync_state_with_binance having run recently.
        if self.market_type == 'FUTURES':
             is_associated_active_locally = bool(self.active_sl_order_id is not None or self.active_tp_order_id is not None)
        elif self.market_type == 'SPOT':
             is_associated_active_locally = bool(self.active_oco_order_list_id is not None)
        else:
             self.logger.error(f"TradingStrategy: Cannot place associated orders: Unknown market type {self.market_type}.")
             return False

        if is_associated_active_locally:
             # Log the active IDs for context
             if self.market_type == 'FUTURES':
                 self.logger.data(f"TradingStrategy: Futures SL/TP orders already seem active locally (SL: {self.active_sl_order_id}, TP: {self.active_tp_order_id}). Skipping placement.")
             elif self.market_type == 'SPOT':
                  self.logger.data(f"TradingStrategy: Spot OCO order list {self.active_oco_order_list_id} already seems active locally. Skipping placement.")
             # Consider it successful if orders are already active
             return True

        # Ensure position and entry price are set before trying to calculate SL/TP/OCO prices
        if not (self.current_position_side and self.entry_price is not None and self.current_position_qty > 0):
            self.logger.warning("TradingStrategy: Cannot place associated orders: No valid position state found locally (Side: {self.current_position_side}, Qty: {self.current_position_qty}, Entry: {self.entry_price}).")
            return False

        # Ensure precisions and tick size are available (fetched in __init__)
        if self.tick_size is None or self.price_precision is None or self.quantity_precision is None:
            self.logger.error("TradingStrategy: Cannot setup associated orders: Binance precisions not available. Critical error.")
            return False

        self.logger.trading(f"TradingStrategy: Attempting to place associated orders for {self.current_position_side} position of {self.current_position_qty} for {self.symbol_str} @ entry {self.entry_price:.{self.price_precision}f}")

        try:
            # --- Calculate Target Prices for SL/TP/OCO ---
            # Use Decimal for calculations to maintain precision
            base_price_dec = Decimal(str(self.entry_price))
            tick_size_dec = Decimal(str(self.tick_size)) # Ensure tick_size is Decimal
            stop_loss_ticks_count = Decimal(str(self.stop_loss_ticks))
            take_profit_ticks_count = Decimal(str(self.take_profit_ticks))
            current_price_dec = Decimal(str(current_price_for_checks)) # Ensure current price is Decimal

            sl_offset = stop_loss_ticks_count * tick_size_dec
            tp_offset = take_profit_ticks_count * tick_size_dec

            if self.current_position_side == config.SIDE_LONG:
                sl_trigger_price_val = base_price_dec - sl_offset
                tp_trigger_price_val = base_price_dec + tp_offset # For Futures TP trigger or Spot TP limit
                sl_direction, tp_direction = 'DOWN', 'UP' # Rounding direction relative to entry
                order_side_for_sltp = config.SIDE_SELL # To close a LONG position, place a SELL order
            elif self.current_position_side == config.SIDE_SHORT:
                sl_trigger_price_val = base_price_dec + sl_offset
                tp_trigger_price_val = base_price_dec - tp_offset
                sl_direction, tp_direction = 'UP', 'DOWN' # Rounding direction relative to entry
                order_side_for_sltp = config.SIDE_BUY # To close a SHORT position, place a BUY order
            else:
                self.logger.error(f"TradingStrategy: Invalid position side '{self.current_position_side}' for associated order calculation for {self.symbol_str}. Cannot place orders."); return False # Should not happen if position state is valid


            # Quantity to use for the associated orders is the current open position quantity
            qty_to_close_float = self.current_position_qty
            formatted_qty_to_close_str = format_quantity(qty_to_close_float, self.quantity_precision)
            if formatted_qty_to_close_str is None or float(formatted_qty_to_close_str) <= 0:
                 self.logger.warning(f"TradingStrategy: Calculated quantity to close {qty_to_close_float} resulted in zero/invalid formatted quantity '{formatted_qty_to_close_str}'. Cannot place associated orders."); return False


            placed_successfully = False # Flag to track if all necessary orders were successfully requested via API

            # --- Place Market Specific Associated Orders ---
            if self.market_type == 'FUTURES':
                 # Futures uses separate STOP_MARKET (SL) and TAKE_PROFIT_MARKET (TP) orders with reduceOnly=True

                 # Adjust Futures SL/TP trigger prices to tick size
                 sl_stop_price_adj = adjust_price_to_tick_size(float(sl_trigger_price_val), tick_size_dec, self.price_precision, direction=sl_direction)
                 tp_stop_price_adj = adjust_price_to_tick_size(float(tp_trigger_price_val), tick_size_dec, self.price_precision, direction=tp_direction)

                 if sl_stop_price_adj is None or tp_stop_price_adj is None or sl_stop_price_adj <= 0 or tp_stop_price_adj <= 0:
                     self.logger.error(f"TradingStrategy: FUTURES SL/TP: Invalid calculated adjusted prices: SL={sl_stop_price_adj}, TP={tp_stop_price_adj}. Cannot place orders."); return False

                 # Safety check: Prevent placing SL/TP that would trigger immediately based on current price.
                 # Compare adjusted stop prices against the current market price.
                 if (self.current_position_side == config.SIDE_LONG and (Decimal(str(sl_stop_price_adj)) >= current_price_dec or Decimal(str(tp_stop_price_adj)) <= current_price_dec)) or \
                     (self.current_position_side == config.SIDE_SHORT and (Decimal(str(sl_stop_price_adj)) <= current_price_dec or Decimal(str(tp_stop_price_adj)) >= current_price_dec)):
                      self.logger.error(f"TradingStrategy: FUTURES SL/TP: Calculated adjusted SL/TP prices {sl_stop_price_adj}/{tp_stop_price_adj} appear to trigger immediately based on current price {current_price_for_checks:.{self.price_precision}f} for {self.current_position_side} position. Cannot place orders.")
                      # This check is important. If the current price has crossed the SL/TP level since entry,
                      # the strategy exit logic (checking implied triggers in _evaluate_and_execute_trades)
                      # should have triggered an exit via MARKET order *before* trying to place these.
                      # If we reach here, it implies the market moved sharply *between* the signal bar close
                      # and the current moment, or there's a logic error. Fail placement for safety.
                      return False


                 self.logger.trading(f"TradingStrategy: Calculated FUTURES targets for {self.current_position_side} position on {self.symbol_str}: SL Target={sl_stop_price_adj:.{self.price_precision}f}, TP Target={tp_stop_price_adj:.{self.price_precision}f}")

                 sl_order_placed = False # Flag for API call success (order accepted)
                 tp_order_placed = False # Flag for API call success (order accepted)

                 # Place SL order (STOP_MARKET)
                 sl_client_order_id = generate_client_order_id("bot_FSL", self.symbol_str) # Include symbol and type in ID
                 self.logger.trading(f"TradingStrategy: Placing SL {order_side_for_sltp} order for {formatted_qty_to_close_str} {self.symbol_str} at stop price {sl_stop_price_adj:.{self.price_precision}f} with ClientOrderID: {sl_client_order_id}")
                 sl_order_response = place_trade_order(
                     self.client, self.symbol_str, order_side_for_sltp, float(formatted_qty_to_close_str), 'FUTURES', self.price_precision, self.quantity_precision,
                     order_type=config.FUTURE_ORDER_TYPE_STOP_MARKET, stop_price=sl_stop_price_adj, reduce_only=True, client_order_id=sl_client_order_id
                 )
                 if sl_order_response and sl_order_response.get('orderId'):
                     self.active_sl_order_id = sl_order_response['orderId'] # Update local state ID
                     sl_order_placed = True
                     self.logger.trading(f"TradingStrategy: Futures SL order placed: ID {self.active_sl_order_id}")
                 else:
                     # Log error already done in place_trade_order
                     self.logger.error(f"TradingStrategy: Failed to place FUTURES SL order for {self.symbol_str}.")

                 # Place TP order (TAKE_PROFIT_MARKET)
                 tp_client_order_id = generate_client_order_id("bot_FTP", self.symbol_str) # Include symbol and type in ID
                 self.logger.trading(f"TradingStrategy: Placing TP {order_side_for_sltp} order for {formatted_qty_to_close_str} {self.symbol_str} at stop price {tp_stop_price_adj:.{self.price_precision}f} with ClientOrderID: {tp_client_order_id}")
                 tp_order_response = place_trade_order(
                     self.client, self.symbol_str, order_side_for_sltp, float(formatted_qty_to_close_str), 'FUTURES', self.price_precision, self.quantity_precision,
                     order_type=config.FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET, stop_price=tp_stop_price_adj, reduce_only=True, client_order_id=tp_client_order_id
                 )
                 if tp_order_response and tp_order_response.get('orderId'):
                     self.active_tp_order_id = tp_order_response['orderId'] # Update local state ID
                     tp_order_placed = True
                     self.logger.trading(f"TradingStrategy: Futures TP order placed: ID {self.active_tp_order_id}")
                 else:
                     # Log error already done in place_trade_order
                     self.logger.error(f"TradingStrategy: Failed to place FUTURES TP order for {self.symbol_str}.")

                 # Consider placement successful only if BOTH orders were successfully requested via API
                 # If one fails, the position is unprotected!
                 placed_successfully = sl_order_placed and tp_order_placed

                 if not placed_successfully:
                      self.logger.error("TradingStrategy: One or both Futures SL/TP orders failed to place. Position is unprotected!")
                      # Clear local IDs if placement failed to prevent incorrect state
                      # This is crucial so the strategy *tries again* next cycle
                      self.clear_associated_order_state(reason="Futures SL/TP placement failed.")
                 else:
                      self.logger.trading(f"TradingStrategy: Futures SL/TP orders placement attempted successfully.")

            elif self.market_type == 'SPOT':
                 # SPOT uses an OCO (One-Cancels-the-Other) order type for combined SL/TP
                 # OCO SELL (to close LONG): TP is a LIMIT order, SL is a STOP_LOSS_LIMIT order.
                 # Need TP Limit price, SL Trigger price, SL Limit price
                 # Note: Pine Script uses 'profit=ticks' for strategy.exit, which implies a limit order at price + ticks.
                 # Binance OCO TP is a LIMIT order at the specified price.
                 tp_limit_price_val = tp_trigger_price_val # TP is the Limit order price in OCO
                 sl_limit_offset_ticks = Decimal('5') # Configurable small gap between SL trigger and SL limit
                 sl_limit_price_val = sl_trigger_price_val - (sl_limit_offset_ticks * tick_size_dec)
                 # Ensure SL limit price is not zero or negative
                 if sl_limit_price_val <= Decimal('0'):
                      # As a fallback, set limit price slightly below trigger price
                      sl_limit_price_val = sl_trigger_price_val * Decimal('0.995') # 0.5% below trigger
                 # Ensure it's at least one tick_size
                 sl_limit_price_val = max(sl_limit_price_val, tick_size_dec)

                 # Adjust Spot OCO prices to tick size and format as strings
                 # Pass float values to adjust_price_to_tick_size, then format string
                 tp_limit_price_adj = adjust_price_to_tick_size(float(tp_limit_price_val), tick_size_dec, self.price_precision, direction='UP') # TP Limit usually rounded up or nearest
                 sl_trigger_price_adj = adjust_price_to_tick_size(float(sl_trigger_price_val), tick_size_dec, self.price_precision, direction='DOWN') # SL Trigger usually rounded down or nearest
                 sl_limit_price_adj = adjust_price_to_tick_size(float(sl_limit_price_val), tick_size_dec, self.price_precision, direction='DOWN') # SL Limit usually rounded down or nearest

                 tp_limit_price_adj_str = format_price(tp_limit_price_adj, self.price_precision)
                 sl_trigger_price_adj_str = format_price(sl_trigger_price_adj, self.price_precision)
                 sl_limit_price_adj_str = format_price(sl_limit_price_adj, self.price_precision)

                 # Validate formatted prices are not None and > 0
                 if not (tp_limit_price_adj_str and float(tp_limit_price_adj_str) > 0 and
                         sl_trigger_price_adj_str and float(sl_trigger_price_adj_str) > 0 and
                         sl_limit_price_adj_str and float(sl_limit_price_adj_str) > 0):
                      self.logger.error(f"TradingStrategy: SPOT OCO: Invalid calculated/formatted prices. TP:{tp_limit_price_adj_str}, SLtrg:{sl_trigger_price_adj_str}, SLlim:{sl_limit_price_adj_str}. Cannot place order."); return False

                 # Validate logical price relationships (TP > SL trigger >= SL limit)
                 # Use Decimal for comparison of string prices
                 tp_limit_price_dec = Decimal(tp_limit_price_adj_str)
                 sl_trigger_price_dec = Decimal(sl_trigger_price_adj_str)
                 sl_limit_price_dec = Decimal(sl_limit_price_adj_str)

                 if not (tp_limit_price_dec > sl_trigger_price_dec): # TP must be > SL trigger for SELL OCO
                      self.logger.error(f"TradingStrategy: SPOT OCO: Logical price error: TP limit ({tp_limit_price_dec}) not > SL trigger ({sl_trigger_price_dec}). Cannot place order.")
                      return False
                 if not (sl_trigger_price_dec >= sl_limit_price_dec): # SL trigger must be >= SL limit
                      self.logger.error(f"TradingStrategy: SPOT OCO: Logical price error: SL trigger ({sl_trigger_price_dec}) not >= SL limit ({sl_limit_price_dec}). Cannot place order.")
                      return False

                 # Safety check against current market price - TP must be > current, SL trigger must be < current
                 if not (tp_limit_price_dec > current_price_dec):
                     self.logger.error(f"TradingStrategy: SPOT OCO SELL: Calculated TP limit price ({tp_limit_price_dec}) is not > current market price ({current_price_dec}). Cannot place order.")
                     return False
                 if not (sl_trigger_price_dec < current_price_dec):
                      self.logger.error(f"TradingStrategy: SPOT OCO SELL: Calculated SL trigger price ({sl_trigger_price_dec}) is not < current market price ({current_price_dec}). Cannot place order.")
                      # This check is critical. If SL trigger is not below current price, it triggers immediately.
                      # Similar to Futures, this should ideally be caught by exit logic first.
                      return False


                 self.logger.trading(f"TradingStrategy: Placing OCO SELL: Qty={formatted_qty_to_close_str}, TP_Limit={tp_limit_price_adj_str}, SL_Trigger={sl_trigger_price_adj_str}, SL_Limit={sl_limit_price_adj_str} for {self.symbol_str}")

                 # Place the OCO order using the dedicated helper function
                 # This helper uses safe_api_call internally
                 oco_response = place_spot_oco_sell_order(
                     self.client, self.symbol_str, formatted_qty_to_close_str,
                     tp_limit_price_adj_str, sl_trigger_price_adj_str, sl_limit_price_adj_str,
                     self.price_precision, self.quantity_precision, # Pass precisions for internal validation/logging
                     list_client_order_id=generate_client_order_id("bot_OCO", self.symbol_str) # Generate OCO list client ID
                 )

                 if oco_response and oco_response.get('orderListId'):
                     self.active_oco_order_list_id = oco_response['orderListId'] # Update local state ID
                     placed_successfully = True
                     self.logger.trading(f"TradingStrategy: SPOT OCO SELL order placed. ListID: {self.active_oco_order_list_id}")
                 else:
                     # Log error already done in place_spot_oco_sell_order
                     self.logger.error(f"TradingStrategy: Failed to place SPOT OCO SELL order for {self.symbol_str}. Response: {oco_response}")
                     # Clear local ID if placement failed
                     self.clear_associated_order_state(reason="SPOT OCO placement failed.")


            else: # Should not happen due to market_type validation in __init__
                 self.logger.error(f"TradingStrategy: Attempted to place associated orders for unknown market type {self.market_type}.")
                 placed_successfully = False

            # Return true if placement was successful (for all required orders)
            return placed_successfully

        except Exception as e_place:
            self.logger.error(f"TradingStrategy: Error calculating or placing associated orders for {self.symbol_str}: {e_place}\n{traceback.format_exc()}")
            # Ensure local IDs are cleared if an exception occurred during placement process
            self.clear_associated_order_state(reason="Exception during associated order placement.")
            return False # Indicate failure


    def _evaluate_and_execute_trades(self, closed_agg_candle: pd.Series, current_price_for_checks: float | Decimal):
        """
        Evaluates signals from the latest closed AGGREGATED candle and executes trade actions.
        Assumes state is synced and closed_agg_candle is the Series representing
        the last row of the aggregated DataFrame with indicators calculated.

        Args:
            closed_agg_candle (pd.Series): Series representing the latest closed
                                           aggregated candle with indicator and signal columns.
            current_price_for_checks (float | Decimal): The current market price
                                                         (e.g., close of the latest base candle)
                                                         used for safety checks (like implied SL/TP trigger).
        """
        if closed_agg_candle is None:
            self.logger.warning("TradingStrategy: Received empty closed_agg_candle. Cannot run strategy logic.")
            return

        last_closed_price_agg = closed_agg_candle['close'] # Close price of the aggregated candle
        current_price_dec = Decimal(str(current_price_for_checks)) # Ensure Decimal for checks

        # Check if critical prices are valid before proceeding
        if last_closed_price_agg is None or pd.isna(last_closed_price_agg) or current_price_dec.is_nan() or current_price_dec.is_infinite():
             self.logger.warning("TradingStrategy: Aggregated closed candle close price or current price for checks is invalid (NaN/None/Inf). Skipping strategy evaluation this cycle.")
             return

        # Extract signals from the *closed aggregated candle* (the Series provided)
        # Use .get(col, False) and bool() to handle potential missing columns or NaN values safely
        entry_long_signal = bool(closed_agg_candle.get('entry_long_cond', False))
        # Only evaluate short signal for FUTURES
        entry_short_signal = bool(closed_agg_candle.get('entry_short_cond', False)) if self.market_type == 'FUTURES' else False

        exit_long_rsi_condition = bool(closed_agg_candle.get('exit_long_rsi_cond', False))
        # Only evaluate short exit signal for FUTURES
        exit_short_rsi_condition = bool(closed_agg_candle.get('exit_short_rsi_cond', False)) if self.market_type == 'FUTURES' else False


        self.logger.strategy(f"--- TradingStrategy Cycle ({self.symbol_str}, {self.market_type}) ---")
        self.logger.strategy(f"Closed Aggregated Candle @ {closed_agg_candle.name.isoformat()}: Close={last_closed_price_agg:.{self.price_precision if self.price_precision is not None else 8}f}. Current Price (from latest base): {current_price_dec:.{self.price_precision if self.price_precision is not None else 8}f}")
        self.logger.strategy(f"Signals (from Closed Agg Candle): LongEntry={entry_long_signal}, ShortEntry={entry_short_signal}, ExitLongRSI={exit_long_rsi_condition}, ExitShortRSI={exit_short_rsi_condition}")
        self.log_current_state("Before Evaluation")

        action_taken_this_cycle = False # Flag to ensure only one major action (entry/exit) per cycle


        # --- Strategy Logic Flow ---

        # If in position (current_position_qty > 0 indicates an open position on Binance, synced state)
        if self.current_position_qty > 0 and self.current_position_side is not None and self.entry_price is not None:
             self.logger.data(f"TradingStrategy: In {self.current_position_side} position ({self.current_position_qty:.{self.quantity_precision if self.quantity_precision is not None else 8}f}) @ Entry {self.entry_price:.{self.price_precision if self.price_precision is not None else 8}f}")

             # Check if associated orders (SL/TP or OCO) are active locally (based on sync)
             is_associated_active_locally = False
             if self.market_type == 'FUTURES' and (self.active_sl_order_id is not None or self.active_tp_order_id is not None): is_associated_active_locally = True
             elif self.market_type == 'SPOT' and self.active_oco_order_list_id is not None: is_associated_active_locally = True


             # 1a. Check for Exit Signals (RSI or implied SL/TP trigger by *current* price)
             # Pine Script's strategy.exit with loss/profit ticks triggers based on the *market price*.
             # We simulate this by checking the current price against the calculated SL/TP levels from entry.
             rsi_exit_triggered = False
             if self.current_position_side == config.SIDE_LONG and exit_long_rsi_condition:
                 rsi_exit_triggered = True
                 self.logger.trading("TradingStrategy: RSI Long Exit signal triggered from closed aggregated candle.")
             elif self.current_position_side == config.SIDE_SHORT and exit_short_rsi_condition:
                 rsi_exit_triggered = True
                 self.logger.trading("TradingStrategy: RSI Short Exit signal triggered from closed aggregated candle.")

             # Check for implied SL/TP trigger based on CURRENT price (latest base candle close)
             implied_sltp_triggered_by_current_price = False
             sl_price_dec, tp_price_dec = None, None # For logging
             if self.use_sltp and self.entry_price is not None and self.tick_size is not None:
                  try:
                     entry_price_dec = Decimal(str(self.entry_price))
                     tick_size_dec = Decimal(str(self.tick_size))
                     stop_loss_ticks_dec = Decimal(str(self.stop_loss_ticks))
                     take_profit_ticks_dec = Decimal(str(self.take_profit_ticks))

                     if self.current_position_side == config.SIDE_LONG:
                         sl_price_dec = entry_price_dec - (stop_loss_ticks_dec * tick_size_dec)
                         tp_price_dec = entry_price_dec + (take_profit_ticks_dec * tick_size_dec)
                         # Check if current price crossed SL or TP level
                         if current_price_dec <= sl_price_dec or current_price_dec >= tp_price_dec: implied_sltp_triggered_by_current_price = True
                     elif self.current_position_side == config.SIDE_SHORT:
                         sl_price_dec = entry_price_dec + (stop_loss_ticks_dec * tick_size_dec)
                         tp_price_dec = entry_price_dec - (take_profit_ticks_dec * tick_size_dec)
                         # Check if current price crossed SL or TP level
                         if current_price_dec >= sl_price_dec or current_price_dec <= tp_price_dec: implied_sltp_triggered_by_current_price = True

                     if implied_sltp_triggered_by_current_price:
                         self.logger.trading(f"TradingStrategy: Implied SL/TP trigger check: current price {current_price_dec:.{self.price_precision}f} triggered vs (SL={sl_price_dec:.{self.price_precision}f}, TP={tp_price_dec:.{self.price_precision}f}) for {self.current_position_side} position.")
                     # else: # Too noisy for data log
                     #      self.logger.data(f"TradingStrategy: Implied SL/TP check: current price {current_price_dec:.{self.price_precision}f} not triggered vs (SL={sl_price_dec:.{self.price_precision}f}, TP={tp_price_dec:.{self.price_precision}f}).")

                  except Exception as e_implied:
                       self.logger.error(f"TradingStrategy: Error checking implied SL/TP trigger vs current price: {e_implied}\n{traceback.format_exc()}")
                       # Assume not triggered on error

             # Decide whether to trigger an exit action
             exit_triggered = rsi_exit_triggered or implied_sltp_triggered_by_current_price

             if exit_triggered:
                  self.logger.trading(f"TradingStrategy: Exit triggered for {self.current_position_side} position ({'RSI' if rsi_exit_triggered else 'Implied SL/TP'}).")

                  # Cancel any active associated orders first.
                  # This is important if SL/TP/OCO orders are used, as they should be cancelled before placing a market close.
                  # This also clears local state IDs if cancellation request is sent successfully.
                  # The sync on the *next* cycle will confirm their cancellation on Binance.
                  self.cancel_associated_orders()

                  # Close position via MARKET order.
                  self.logger.trading("TradingStrategy: Closing position via MARKET order...")
                  # close_current_position internally handles FUTURES (reduceOnly) or SPOT (SELL all base)
                  # It uses the currently held quantity (self.current_position_qty).
                  close_status = close_current_position(
                      self.client, self.symbol_str, self.market_type,
                      self.current_position_side, self.current_position_qty, # Pass current side and qty
                      self.price_precision, self.quantity_precision,
                      latest_kline_df=None # latest_kline_df might not be needed by close_current_position itself, pass None for now or the base DF from on_candle_close if needed for SPOT balance logging context
                  )

                  if close_status == 'PLACED':
                      self.logger.trading("TradingStrategy: Close market order placed. Verifying position closure...")
                      # Verify closure - This is a blocking poll in the current simplified setup
                      # A real bot would ideally listen to User Data Stream or poll asynchronously for balance changes/position updates.
                      # Pass current state context for verification helper.
                      if verify_position_closed(
                          self.client, self.symbol_str, self.market_type,
                          self.price_precision, self.quantity_precision,
                          expected_side_that_was_closed=self.current_position_side,
                          latest_kline_df=None # Pass None, verify_position_closed fetches balance directly
                      ):
                          self.logger.success("TradingStrategy: Position successfully verified closed on exchange.")
                          self.clear_state(f"Position exited via MARKET order and verified closed.") # Clear all local state after successful exit
                      else:
                          self.logger.error("TradingStrategy: Market close order placed, but position closure could NOT be verified on exchange.")
                          # If verification fails, keep local state as is. The sync on the next cycle
                          # will (hopefully) correct the state if the position eventually closes on Binance.
                  elif close_status == 'NO_POSITION': # Should not happen if we thought we were in a position, but good safeguard
                      self.logger.warning("TradingStrategy: close_current_position reported NO_POSITION, but strategy state thought we were in position. State was likely already outdated. Sync will correct it.")
                      # No need to clear state here, sync already ran and found no position.
                  else: # Close status is 'FAILED'
                       self.logger.error(f"TradingStrategy: Failed to place MARKET close order (status: {close_status}). Position may still be open.")
                       # Keep local state as is. Next sync will clarify.

                  action_taken_this_cycle = True # Action was attempted (even if verification or placement failed)

             # 1b. If in position and using SLTP, but associated orders are NOT active locally, try to place them.
             elif self.use_sltp and not is_associated_active_locally: # If using SLTP but orders not active locally
                  # Check if SL/TP ticks are configured and positive (if USE_SLTP is True, ideally they should be)
                  if (self.market_type == 'FUTURES' and (self.stop_loss_ticks <= 0 or self.take_profit_ticks <= 0)) or \
                     (self.market_type == 'SPOT' and (self.stop_loss_ticks <= 0 or self.take_profit_ticks <= 0)): # Simplified check
                      self.logger.warning(f"TradingStrategy: USE_SLTP is True, but STOP_LOSS_TICKS ({self.stop_loss_ticks}) or TAKE_PROFIT_TICKS ({self.take_profit_ticks}) are <= 0. Skipping associated order placement.")
                  else:
                       self.logger.trading("TradingStrategy: In position, USE_SLTP is True, but associated orders not active. Attempting to place them.")
                       # _place_associated_orders uses the current self.entry_price, self.current_position_side, self.current_position_qty
                       # Pass current_price_for_checks for safety checks within _place_associated_orders
                       if self._place_associated_orders(current_price_for_checks): # This updates local IDs on successful API request
                           self.logger.trading("TradingStrategy: Successfully attempted placing missing associated orders.")
                       else:
                           self.logger.error("TradingStrategy: Failed to place missing associated orders. Position is open without protection!")
                       action_taken_this_cycle = True # Action was attempted

             # 1c. If in position, USE_SLTP is true, and associated orders ARE active locally: Do nothing, let Binance handle exit.
             elif self.use_sltp and is_associated_active_locally:
                  self.logger.data("TradingStrategy: Position held and associated orders are active. Letting Binance manage exit.")
                  action_taken_this_cycle = True # Considered handled by Binance


             # 1d. If in position, USE_SLTP is false, and no exit signal: Hold position.
             elif not self.use_sltp and not exit_triggered: # Note: exit_triggered includes RSI and implied SLTP now
                  self.logger.data("TradingStrategy: Position held, USE_SLTP is False, and no exit signal. Holding position.")
                  action_taken_this_cycle = True # Considered holding


        # 2. If NOT in position, check for new entry signals
        elif not action_taken_this_cycle: # Only try to enter if no exit action was taken
             new_entry_side = None
             # Only LONG entries supported for SPOT in this strategy example
             if entry_long_signal: new_entry_side = config.SIDE_LONG
             # Only evaluate short entry signal for FUTURES
             elif self.market_type == 'FUTURES' and entry_short_signal: new_entry_side = config.SIDE_SHORT


             if new_entry_side:
                  self.logger.trading(f"TradingStrategy: No position held. Entry signal detected for {new_entry_side} at {closed_agg_candle.name.isoformat()}. Attempting entry order.")

                  # Ensure any lingering associated orders from a previous trade attempt (e.g., failed exit) are cancelled.
                  # The sync_state_with_binance at the start *should* have cleared these if they were no longer open,
                  # but attempt cancellation again as a safeguard.
                  # This also clears local state IDs if cancellation request is sent successfully.
                  self.cancel_associated_orders()

                  entry_order_side = config.SIDE_BUY if new_entry_side == config.SIDE_LONG else config.SIDE_SELL

                  # Calculate quantity - Pass current_price_for_checks (from latest base candle)
                  # for calculate_trade_qty's MIN_NOTIONAL or price estimation needs.
                  entry_qty = calculate_trade_qty(
                      self.client, self.symbol_str, entry_order_side, self.market_type,
                      self.config, # Pass full strategy config for QTY_PERCENT etc.
                      {'price_precision': self.price_precision, 'quantity_precision': self.quantity_precision, 'tick_size': self.tick_size}, # Pass relevant precisions
                      latest_kline_df=pd.DataFrame([{'close': current_price_for_checks}]) # Pass current price in a mock DF structure
                  )


                  if entry_qty > 0:
                      self.logger.trading(f"TradingStrategy: Calculated entry quantity: {entry_qty:.{self.quantity_precision if self.quantity_precision is not None else 8}f} for {new_entry_side} {self.symbol_str}")
                      formatted_entry_qty_str = format_quantity(entry_qty, self.quantity_precision)

                      if formatted_entry_qty_str is None or float(formatted_entry_qty_str) <= 0:
                          self.logger.warning(f"TradingStrategy: Formatted entry quantity is zero/invalid ({formatted_entry_qty_str}). Cannot place order.")
                      else:
                           # Place MARKET order for entry.
                           # Use place_trade_order helper, which uses safe_api_call.
                           # It returns the API response dictionary or None on failure.
                           entry_order_response = place_trade_order(
                               self.client, self.symbol_str, entry_order_side, float(formatted_entry_qty_str), self.market_type,
                               self.price_precision, self.quantity_precision, order_type=config.ORDER_TYPE_MARKET
                           )

                           # Check the API response to see if the market order was placed and accepted.
                           # Market orders typically fill immediately or near-immediately.
                           # We rely on the sync on the *next* cycle to update the position state definitively.
                           # However, we can attempt to update local state based on the *immediate* response if it reports as FILLED.
                           if entry_order_response and entry_order_response.get('orderId'):
                                order_id = entry_order_response['orderId']
                                order_status_from_response = entry_order_response.get('status')

                                if order_status_from_response == 'FILLED':
                                     # Attempt to extract filled details from the immediate response
                                     try:
                                          filled_qty_from_order = float(entry_order_response.get('executedQty', '0'))
                                          filled_price = None # Try to get average fill price
                                          if self.market_type == 'FUTURES':
                                              # Futures market orders often have 'avgPrice' in response
                                              # Note: This might not be the *final* avg price if partially filled initially
                                              avg_price_str = entry_order_response.get('avgPrice')
                                              if avg_price_str: filled_price = float(avg_price_str)
                                          elif self.market_type == 'SPOT':
                                               # Spot market BUY avg price = cummulativeQuoteQty / executedQty
                                               cummulative_quote_qty_str = entry_order_response.get('cummulativeQuoteQty')
                                               if cummulative_quote_qty_str and filled_qty_from_order > 0:
                                                   try: filled_price = float(cummulative_quote_qty_str) / filled_qty_from_order
                                                   except ZeroDivisionError: filled_price = 0.0 # Avoid division by zero

                                          if filled_price is not None and filled_price > 0 and filled_qty_from_order > 0:
                                              self.logger.trading(f"TradingStrategy: Entry {new_entry_side} order {order_id} FILLED immediately. Price: {filled_price:.{self.price_precision}f}, Qty: {filled_qty_from_order:.{self.quantity_precision}f}")

                                              # Update local state based on successful fill
                                              self.current_position_side = new_entry_side
                                              # IMPORTANT: Binance can partial fill MARKET orders (e.g., due to filters/liquidity).
                                              # The executedQty is the actual quantity filled by THIS order.
                                              # If this order fills the *entire* desired quantity (which is likely for MARKET)
                                              # then this is the new position qty. If it partially fills and there was
                                              # already a position (shouldn't happen with this strategy flow), this logic needs refinement.
                                              self.current_position_qty = filled_qty_from_order # The quantity *filled* by this order
                                              self.entry_price = filled_price # Use the actual fill price

                                              self.logger.trading(f"TradingStrategy: Local state updated after immediate fill: Side={self.current_position_side}, Qty={self.current_position_qty:.{self.quantity_precision}f}, EntryP={self.entry_price:.{self.price_precision}f}")

                                              # Place associated SL/TP/OCO if enabled, ONLY IF the entry was successful.
                                              if self.use_sltp and self.current_position_qty > 0: # Only place if actually in position
                                                   # Check if SL/TP ticks are configured and positive
                                                   if (self.market_type == 'FUTURES' and (self.stop_loss_ticks <= 0 or self.take_profit_ticks <= 0)) or \
                                                      (self.market_type == 'SPOT' and (self.stop_loss_ticks <= 0 or self.take_profit_ticks <= 0)):
                                                        self.logger.warning(f"TradingStrategy: USE_SLTP is True, but STOP_LOSS_TICKS ({self.stop_loss_ticks}) or TAKE_PROFIT_TICKS ({self.take_profit_ticks}) are <= 0. Skipping associated order placement after entry.")
                                                   else:
                                                        self.logger.trading("TradingStrategy: Placing associated SL/TP/OCO orders after successful entry.")
                                                        # Pass current_price_for_checks for safety checks
                                                        if not self._place_associated_orders(current_price_for_checks): # This updates local IDs on success
                                                            self.logger.error("TradingStrategy: Failed to place associated orders after entry. Position is open without protection!")
                                                        else:
                                                            self.logger.trading("TradingStrategy: Associated orders placement attempted successfully.")

                                              else: # Entry happened but USE_SLTP is False or entry_qty was 0
                                                   self.logger.trading("TradingStrategy: Not placing associated orders (USE_SLTP is False or entry qty was 0).")


                                          else: # Order reported FILLED but executedQty or price invalid
                                               self.logger.error(f"TradingStrategy: Entry order {order_id} reported FILLED but could not extract valid price/qty from response: {entry_order_response}")
                                               # Entry likely happened, but local state update failed.
                                               # Keep state clear and rely on next sync.
                                               self.clear_state(reason=f"Entry order {order_id} filled but price/qty extraction failed.")

                                elif order_status_from_response in ['NEW', 'PARTIALLY_FILLED', 'PENDING_CANCEL']:
                                     self.logger.warning(f"TradingStrategy: Entry {new_entry_side} order {order_id} placed, but not immediately FILLED (Status: {order_status_from_response}). Assuming entry failed for this cycle.")
                                     # If the market order isn't immediately filled, something might be wrong (e.g., minNotional issue on Binance side, or API lag).
                                     # For simplicity, assume entry failed for this cycle and clear state.
                                     # A more complex bot might track this order and poll, but that adds complexity.
                                     self.clear_state(reason=f"Entry order {order_id} not immediately filled (Status: {order_status_from_response}).")
                                else:
                                     self.logger.error(f"TradingStrategy: Entry {new_entry_side} order {order_id} placed, but returned unexpected status '{order_status_from_response}'. Response: {entry_order_response}. Assuming entry failed.")
                                     self.clear_state(reason=f"Entry order {order_id} placed, unexpected status.")


                           else: # place_trade_order returned None or response dictionary was invalid/no orderId
                                self.logger.error(f"TradingStrategy: Failed to place entry {new_entry_side} order for {self.symbol_str}. API call failed or invalid response.")
                                # Local state should already be clear.


                   # Note: if qty <= 0, a warning is logged by calculate_trade_qty and no order is placed.
                   # The strategy then proceeds to the 'no action' block.

                  action_taken_this_cycle = True # Action was attempted (place order or qty=0 warning)


        # 3. If no action was taken this cycle (not in position, no entry signal, and no attempt to place orders happened)
        if not action_taken_this_cycle:
            self.logger.data(f"TradingStrategy: No trade action taken this strategy cycle for {self.symbol_str}.")
            # Optional safeguard: If not in position, ensure no associated orders are active on exchange.
            # This is primarily handled by the sync at the start of the cycle.
            # The sync ensures local state reflects Binance. If local state says no position
            # and no associated orders, the state is clean.
            # Add a warning if local state indicates no position but associated orders are somehow still active
            if self.current_position_qty <= 0:
                 if self.market_type == 'FUTURES' and (self.active_sl_order_id is not None or self.active_tp_order_id is not None):
                     self.logger.warning(f"TradingStrategy: Local state mismatch: No position ({self.current_position_qty}) but Futures SL/TP orders ({self.active_sl_order_id}/{self.active_tp_order_id}) active locally. Sync should correct this or attempt cancellation.")
                 elif self.market_type == 'SPOT' and self.active_oco_order_list_id is not None:
                     self.logger.warning(f"TradingStrategy: Local state mismatch: No position ({self.current_position_qty}) but Spot OCO list ({self.active_oco_order_list_id}) active locally. Sync should correct this or attempt cancellation.")


        self.logger.strategy(f"--- TradingStrategy Cycle End ({self.symbol_str}) ---")

