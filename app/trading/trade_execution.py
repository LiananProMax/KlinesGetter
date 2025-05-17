#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import traceback
import pandas as pd
from datetime import datetime
from decimal import Decimal
import decimal
from app.core import config

# 从 binance_trading_api 导入所需的函数
from app.api_clients.binance_trading_api import (
    _get_futures_usdt_balance, _get_spot_asset_balance, 
    get_symbol_info_from_exchange, safe_api_call,
    format_price, format_quantity, split_symbol_for_coinapi,
    close_current_position, verify_position_closed, place_trade_order
)

# 获取日志器
logger = logging.getLogger(__name__)


def calculate_trade_qty(
    client,
    symbol_str: str,
    order_side: str, # BUY or SELL
    market_type: str, # FUTURES or SPOT
    strategy_config: dict, # Contains QTY_PERCENT, SPOT_QUOTE_ASSET (if SPOT)
    precisions: dict, # Must contain 'price_precision', 'quantity_precision', 'tick_size'
    latest_kline_df: pd.DataFrame | None = None # Optional, needed for SPOT MIN_NOTIONAL calc if using last close price
) -> float:
    """
    计算交易订单（进场或平仓）的数量，基于 QTY_PERCENT 和可用的余额/权益。

    参数:
        client: 已初始化的币安客户端实例。
        symbol_str: 交易符号（例如 "BTCUSDT"）。
        order_side: 订单方向（'BUY' 或 'SELL'）。
        market_type: 'FUTURES' 或 'SPOT'。
        strategy_config: 策略配置，包含 'QTY_PERCENT' 及可能的 'SPOT_QUOTE_ASSET'。
        precisions: 包含精度信息的字典: 'price_precision', 'quantity_precision', 'tick_size'。
        latest_kline_df: 可选的最新 K 线数据帧（用于获取最新价格上下文）。

    返回:
        float: 计算出的数量，如果计算失败或结果为零，返回 0.0。
    """
    if client is None:
        logger.error("calculate_trade_qty: Binance client is None.")
        return 0.0

    qty_percent = strategy_config.get('QTY_PERCENT')
    if qty_percent is None or not (0.0 < qty_percent <= 1.0):
        logger.error(f"calculate_trade_qty: Invalid QTY_PERCENT ({qty_percent}).")
        return 0.0

    price_precision = precisions.get('price_precision')
    quantity_precision = precisions.get('quantity_precision')
    tick_size = precisions.get('tick_size')

    if price_precision is None or quantity_precision is None or tick_size is None:
         logger.error("calculate_trade_qty: Missing required precision information.")
         return 0.0

    base_asset, quote_asset = None, None
    try:
         # 使用启发式分割符号。注意：对某些符号可能不准确。
         # 更稳健的方法是从交易所获取符号信息并解析 baseAsset/quoteAsset。
         # 让我们首先尝试获取符号信息以提高健壮性。
         symbol_info = get_symbol_info_from_exchange(client, symbol_str, market_type)
         if symbol_info and 'baseAsset' in symbol_info and 'quoteAsset' in symbol_info:
              base_asset = symbol_info['baseAsset']
              quote_asset = symbol_info['quoteAsset']
              logger.info(f"calculate_trade_qty: Fetched base/quote assets from exchange info: {base_asset}, {quote_asset}")
         else:
              logger.warning(f"calculate_trade_qty: Could not fetch base/quote from exchange info for {symbol_str}. Falling back to split_symbol_for_coinapi heuristic.")
              # 退回到启发式方法
              heuristic_split = split_symbol_for_coinapi(symbol_str)
              if '_' in heuristic_split:
                  base_asset, quote_asset = heuristic_split.split('_')
              else:
                  logger.error(f"calculate_trade_qty: Failed to split symbol '{symbol_str}' into base/quote using heuristic.")
                  return 0.0

    except Exception as e_symbol_split:
         logger.error(f"calculate_trade_qty: Error getting symbol info or splitting symbol: {e_symbol_split}\n{traceback.format_exc()}")
         return 0.0


    # --- 确定可用资金/使用数量 ---
    capital_to_use = Decimal('0')
    base_qty_to_sell = Decimal('0')
    trade_price = Decimal('0') # 用于计算的价格（估计）

    try:
        if market_type == 'FUTURES':
            # 对于期货的全仓保证金，资金是可用 USDT 余额
            available_usdt_balance = _get_futures_usdt_balance(client)
            if available_usdt_balance is None: # 助手在失败时返回 0，让我们明确检查它是否为 None
                 logger.error("calculate_trade_qty: Failed to get futures USDT balance.")
                 return 0.0
            available_usdt_balance_dec = Decimal(str(available_usdt_balance))
            capital_to_use = available_usdt_balance_dec * Decimal(str(qty_percent))
            logger.info(f"calculate_trade_qty: FUTURES: Available USDT={available_usdt_balance_dec:.8f}, Capital to use={capital_to_use:.8f}")

            # 需要估计交易价格来计算数量（用于市价订单）
            # 使用最新可用 K 线的收盘价进行估计。
            # 如果 latest_kline_df 可用且非空，使用其最后一个收盘价。
            if latest_kline_df is not None and not latest_kline_df.empty and 'close' in latest_kline_df.columns:
                 trade_price_float = latest_kline_df['close'].iloc[-1]
                 trade_price = Decimal(str(trade_price_float))
                 if trade_price <= Decimal('0') or trade_price.is_nan() or trade_price.is_infinite():
                     logger.warning(f"calculate_trade_qty: Latest kline close price {trade_price_float} is invalid. Attempting to fetch ticker price.")
                     trade_price = Decimal('0') # 如果无效则重置

            if trade_price <= Decimal('0'): # 如果 K 线价格无效或不可用
                # 作为备选，获取当前符号的行情价格
                try:
                    ticker = safe_api_call(client, client.get_symbol_ticker, symbol=symbol_str)
                    if ticker and 'price' in ticker:
                        trade_price = Decimal(ticker['price'])
                        logger.info(f"calculate_trade_qty: Fetched ticker price as fallback: {trade_price:.{price_precision}f}")
                    else:
                        logger.error("calculate_trade_qty: Failed to fetch ticker price for quantity calculation.")
                        return 0.0
                except Exception as e_ticker:
                    logger.error(f"calculate_trade_qty: Error fetching ticker price: {e_ticker}")
                    return 0.0

            if trade_price <= Decimal('0'):
                 logger.error("calculate_trade_qty: Trade price estimate is zero or negative. Cannot calculate quantity.")
                 return 0.0


        elif market_type == 'SPOT':
            # 对于现货买单，资金是可用的报价资产余额（如 USDT）
            if order_side == config.SIDE_BUY:
                 quote_asset_to_use = strategy_config.get('SPOT_QUOTE_ASSET', quote_asset) # 如果存在，使用配置值，否则使用派生的报价资产
                 if not quote_asset_to_use:
                     logger.error("calculate_trade_qty: SPOT BUY: SPOT_QUOTE_ASSET or derived quote asset not found.")
                     return 0.0

                 available_quote_balance = _get_spot_asset_balance(client, quote_asset_to_use)
                 if available_quote_balance is None: # 在 API 错误时助手返回 None
                      logger.error(f"calculate_trade_qty: SPOT BUY: Failed to get spot balance for {quote_asset_to_use}.")
                      return 0.0
                 available_quote_balance_dec = Decimal(str(available_quote_balance))
                 capital_to_use = available_quote_balance_dec * Decimal(str(qty_percent))
                 logger.info(f"calculate_trade_qty: SPOT BUY: Available {quote_asset_to_use}={available_quote_balance_dec:.8f}, Capital to use={capital_to_use:.8f}")

                 # 需要估计交易价格用于现货买入市价订单数量计算
                 # 币安市价买入使用 quoteOrderQty（如果提供），否则假设数量是基础资产数量。
                 # 我们当前的 place_trade_order 为现货市价买入实现使用 'quantity' 参数（基础资产数量）。
                 # 所以我们需要基于 capital_to_use 和估计价格计算基础资产数量。
                 # 从最新 K 线或行情获取最新价格进行估计
                 if latest_kline_df is not None and not latest_kline_df.empty and 'close' in latest_kline_df.columns:
                    trade_price_float = latest_kline_df['close'].iloc[-1]
                    trade_price = Decimal(str(trade_price_float))
                    if trade_price <= Decimal('0') or trade_price.is_nan() or trade_price.is_infinite():
                       logger.warning(f"calculate_trade_qty: Latest kline close price {trade_price_float} is invalid for SPOT BUY. Attempting to fetch ticker price.")
                       trade_price = Decimal('0') # 如果无效则重置

                 if trade_price <= Decimal('0'): # 如果 K 线价格无效或不可用
                     # 作为备选，获取当前符号的行情价格
                     try:
                         ticker = safe_api_call(client, client.get_symbol_ticker, symbol=symbol_str)
                         if ticker and 'price' in ticker:
                             trade_price = Decimal(ticker['price'])
                             logger.info(f"calculate_trade_qty: SPOT BUY: Fetched ticker price as fallback: {trade_price:.{price_precision}f}")
                         else:
                             logger.error("calculate_trade_qty: SPOT BUY: Failed to fetch ticker price for quantity calculation.")
                             return 0.0
                     except Exception as e_ticker:
                         logger.error(f"calculate_trade_qty: SPOT BUY: Error fetching ticker price: {e_ticker}")
                         return 0.0

                 if trade_price <= Decimal('0'):
                    logger.error("calculate_trade_qty: SPOT BUY: Trade price estimate is zero or negative. Cannot calculate quantity.")
                    return 0.0

            # 对于现货卖单，数量是可用的基础资产余额
            # （在平仓时）- QTY_PERCENT 通常不用于平仓。
            # 然而，策略可能调用此函数进行卖出入场（此处不适用）
            # 或计算平仓订单的卖出数量。
            # 在当前策略逻辑中，`close_current_position` 处理获取全部基础资产余额进行卖出。
            # 如果 calculate_trade_qty 以 SIDE_SELL 调用，它意味着平仓，
            # 所以我们应该返回当前基础资产余额。
            elif order_side == config.SIDE_SELL:
                 if not base_asset:
                     logger.error("calculate_trade_qty: SPOT SELL: Base asset not found.")
                     return 0.0
                 available_base_balance = _get_spot_asset_balance(client, base_asset)
                 if available_base_balance is None: # 在 API 错误时助手返回 None
                      logger.error(f"calculate_trade_qty: SPOT SELL: Failed to get spot balance for {base_asset}.")
                      return 0.0
                 base_qty_to_sell = Decimal(str(available_base_balance))
                 logger.info(f"calculate_trade_qty: SPOT SELL: Available {base_asset}={base_qty_to_sell:.8f}. Using this as quantity.")
                 # 这里不需要价格，因为数量基于余额，而非资金。

            else:
                logger.error(f"calculate_trade_qty: Unknown order side '{order_side}' for market type '{market_type}'.")
                return 0.0

        else:
            logger.error(f"calculate_trade_qty: Unknown market type '{market_type}'.")
            return 0.0

    except Exception as e_balance:
         logger.error(f"calculate_trade_qty: Error getting balance or estimating price: {e_balance}\n{traceback.format_exc()}")
         return 0.0

    # --- 基于资金和价格计算数量 ---
    calculated_qty_dec = Decimal('0')
    if market_type == 'FUTURES' or (market_type == 'SPOT' and order_side == config.SIDE_BUY):
        # 计算数量 = 资金 / 价格（对于买单）
        if trade_price > Decimal('0'):
            calculated_qty_dec = capital_to_use / trade_price
        else:
            logger.error("calculate_trade_qty: Estimated trade price is zero or negative after checks. Cannot calculate quantity.")
            return 0.0

    elif market_type == 'SPOT' and order_side == config.SIDE_SELL:
        # 对于现货卖出（平仓），数量就是上面计算的可用基础资产余额
        calculated_qty_dec = base_qty_to_sell


    # --- 应用过滤器/规则（如 MIN_NOTIONAL, LOT_SIZE, MIN_QTY, MAX_QTY） ---
    # 如果尚未完成/传递，获取交易所信息和过滤器
    # （Precisions 字典可能包含过滤器信息，但为了健壮性让我们重新获取）
    try:
        symbol_info = get_symbol_info_from_exchange(client, symbol_str, market_type)
        if symbol_info and 'filters' in symbol_info:
            filters = symbol_info['filters']

            for f in filters:
                if f.get('filterType') == 'LOT_SIZE' or f.get('filterType') == 'MARKET_LOT_SIZE':
                    try:
                        min_qty = Decimal(f.get('minQty', '0'))
                        max_qty = Decimal(f.get('maxQty', '99999999'))
                        step_size = Decimal(f.get('stepSize', '1'))

                        # 应用最小/最大数量约束
                        if calculated_qty_dec < min_qty:
                            logger.warning(f"calculate_trade_qty: Calculated quantity {calculated_qty_dec:.{quantity_precision}f} below minQty {min_qty}. Adjusting to minQty.")
                            calculated_qty_dec = min_qty
                        if calculated_qty_dec > max_qty:
                            logger.warning(f"calculate_trade_qty: Calculated quantity {calculated_qty_dec:.{quantity_precision}f} above maxQty {max_qty}. Adjusting to maxQty.")
                            calculated_qty_dec = max_qty

                        # 应用步长约束（截断到下方最近的步长）
                        if step_size > Decimal('0'):
                            # 计算有多少步长适合计算的数量
                            num_steps = (calculated_qty_dec / step_size).to_integral_value(rounding=decimal.ROUND_DOWN)
                            calculated_qty_dec = num_steps * step_size
                            logger.info(f"calculate_trade_qty: Adjusted quantity to stepSize {step_size}: {calculated_qty_dec:.{quantity_precision}f}")
                        else:
                             logger.warning(f"calculate_trade_qty: Invalid stepSize ({step_size}). Skipping step size adjustment.")


                    except Exception as e_lot_size:
                        logger.error(f"calculate_trade_qty: Error applying LOT_SIZE/MARKET_LOT_SIZE filter for {symbol_str}: {e_lot_size}")
                        # 继续下一个过滤器或完成

                elif market_type == 'SPOT' and order_side == config.SIDE_BUY and f.get('filterType') == 'MIN_NOTIONAL':
                     # 对于现货市价买单，确保总名义价值满足 MIN_NOTIONAL
                     # 名义价值 = 数量 * 价格
                     try:
                         min_notional = Decimal(f.get('minNotional', '0'))
                         # 使用估计的交易价格进行此检查
                         notional_value = calculated_qty_dec * trade_price
                         if notional_value < min_notional:
                             logger.warning(f"calculate_trade_qty: Calculated notional value {notional_value:.8f} below MIN_NOTIONAL {min_notional}. Cannot place order with this quantity.")
                             # 这个数量太小。我们可以尝试增加数量以满足 min_notional
                             # calculated_qty_dec = (min_notional / trade_price).quantize(step_size, rounding=decimal.ROUND_UP) # 这里需要 step_size
                             # 为简单起见，如果以计算的资金百分比低于最小名义价值，就返回 0.0。
                             logger.error("calculate_trade_qty: Cannot meet MIN_NOTIONAL with calculated quantity. Returning 0.0.")
                             return 0.0 # 如果不能满足 MIN_NOTIONAL，返回 0


                     except Exception as e_min_notional:
                         logger.error(f"calculate_trade_qty: Error applying MIN_NOTIONAL filter for {symbol_str}: {e_min_notional}")
                         # 继续

        else:
             logger.warning(f"calculate_trade_qty: Could not fetch filters for {symbol_str}. Filter limits will not be applied to quantity.")

    except Exception as e_filters:
        logger.error(f"calculate_trade_qty: Unexpected error fetching or applying filters for {symbol_str}: {e_filters}\n{traceback.format_exc()}")
        # 继续使用没有过滤器调整的计算数量


    # --- 最终格式化和验证 ---
    # 使用所需精度格式化 Decimal 数量为字符串（ROUND_DOWN）
    formatted_qty_str = format_quantity(calculated_qty_dec, quantity_precision)

    if formatted_qty_str is None:
        logger.error("calculate_trade_qty: Failed to format calculated quantity. Returning 0.0.")
        return 0.0

    final_qty_float = float(formatted_qty_str)

    # 最终检查：格式化的数量是否仍然为正且可交易？
    # 使用一个小阈值或从过滤器获取的实际 minQty（如果可用）。
    # 如果未获取过滤器，使用一个通用的小正阈值。
    min_tradeable_qty_threshold = Decimal('1e-8') # 默认小值
    # 尝试从之前获取的过滤器获取 minQty（如果可用）
    if symbol_info and 'filters' in symbol_info:
         for f_check in symbol_info.get('filters', []):
             if f_check.get('filterType') in ['LOT_SIZE', 'MARKET_LOT_SIZE']:
                  try:
                      min_qty_str = f_check.get('minQty', "1e-8")
                      if min_qty_str and float(min_qty_str) > 0:
                          min_tradeable_qty_threshold = Decimal(min_qty_str)
                      # else: 使用默认阈值
                  except Exception as e_minqty_check:
                       logger.warning(f"calculate_trade_qty: Could not parse minQty {f_check.get('minQty')} as Decimal: {e_minqty_check}. Using default threshold.")
                  break # 找到 LOT_SIZE/MARKET_LOT_SIZE 过滤器，退出循环

    if Decimal(str(final_qty_float)) < min_tradeable_qty_threshold:
        logger.warning(f"calculate_trade_qty: Final formatted quantity {final_qty_float:.{quantity_precision}f} is below minimum tradeable quantity threshold {float(min_tradeable_qty_threshold):.{quantity_precision}f}. Returning 0.0.")
        return 0.0 # 如果数量太小，返回 0

    logger.info(f"calculate_trade_qty: Final calculated and formatted quantity: {final_qty_float:.{quantity_precision}f}")
    return final_qty_float

class TradeExecution:
    """
    交易执行模块，处理信号评估和交易逻辑。
    """

    def __init__(self, strategy):
        """
        初始化交易执行模块。

        参数:
            strategy: TradingStrategy 实例的引用。
        """
        self.strategy = strategy  # 策略实例引用
        self.client = strategy.client  # Binance 客户端
        self.symbol_str = strategy.symbol_str  # 交易对符号
        self.market_type = strategy.market_type  # 市场类型
        self.price_precision = strategy.price_precision  # 价格精度
        self.quantity_precision = strategy.quantity_precision  # 数量精度
        self.tick_size = strategy.tick_size  # tick 大小
        self.logger = strategy.logger  # 使用策略的日志器

    def evaluate_and_execute_trades(self, closed_agg_candle: pd.Series, current_price_for_checks: float | Decimal):
        """
        评估最新关闭的聚合 K 线信号并执行交易。
        参数:
            closed_agg_candle: pd.Series，代表最新关闭的聚合 K 线数据。
            current_price_for_checks: 当前市场价格，用于检查和计算。
        """
        if closed_agg_candle is None:
            self.logger.warning("接收到空的关闭聚合 K 线数据。无法运行策略逻辑。")
            return

        last_closed_price_agg = closed_agg_candle['close']
        current_price_dec = Decimal(str(current_price_for_checks))

        if pd.isna(last_closed_price_agg) or current_price_dec.is_nan() or current_price_dec.is_infinite():
            self.logger.warning("聚合关闭 K 线收盘价或当前价格无效。跳过策略评估。")
            return

        # 提取信号
        entry_long_signal = bool(closed_agg_candle.get('entry_long_cond', False))
        entry_short_signal = bool(closed_agg_candle.get('entry_short_cond', False)) if self.market_type == 'FUTURES' else False
        exit_long_rsi_condition = bool(closed_agg_candle.get('exit_long_rsi_cond', False))
        exit_short_rsi_condition = bool(closed_agg_candle.get('exit_short_rsi_cond', False)) if self.market_type == 'FUTURES' else False

        self.logger.info(f"--- 交易策略周期 ({self.symbol_str}, {self.market_type}) ---")
        self.logger.info(f"关闭聚合 K 线 @ {closed_agg_candle.name.isoformat()}: 收盘价={last_closed_price_agg:.{self.price_precision}f}。当前价格 (从最新基础 K 线): {current_price_dec:.{self.price_precision}f}")
        self.logger.info(f"信号 (从关闭聚合 K 线): 多头入场={entry_long_signal}，空头入场={entry_short_signal}，多头 RSI 退出={exit_long_rsi_condition}，空头 RSI 退出={exit_short_rsi_condition}")
        self.strategy.log_current_state("评估前")

        action_taken_this_cycle = False  # 标记本周期是否执行了动作

        # 如果有仓位，检查退出信号
        if self.strategy.current_position_qty > 0 and self.strategy.current_position_side is not None and self.strategy.entry_price is not None:
            self.logger.info(f"有 {self.strategy.current_position_side} 仓位 ({self.strategy.current_position_qty:.{self.quantity_precision}f} @ 入场价 {self.strategy.entry_price:.{self.price_precision}f})")

            # 检查 RSI 退出信号或隐含 SL/TP 触发
            rsi_exit_triggered = False
            if self.strategy.current_position_side == config.SIDE_LONG and exit_long_rsi_condition:
                rsi_exit_triggered = True
                self.logger.info("RSI 多头退出信号触发。")
            elif self.strategy.current_position_side == config.SIDE_SHORT and exit_short_rsi_condition:
                rsi_exit_triggered = True
                self.logger.info("RSI 空头退出信号触发。")

            implied_sltp_triggered = False
            if self.strategy.use_sltp and self.strategy.entry_price is not None and self.strategy.tick_size is not None:
                entry_price_dec = Decimal(str(self.strategy.entry_price))
                tick_size_dec = Decimal(str(self.strategy.tick_size))
                stop_loss_ticks_dec = Decimal(str(self.strategy.stop_loss_ticks))
                take_profit_ticks_dec = Decimal(str(self.strategy.take_profit_ticks))

                sl_price_dec = entry_price_dec - (stop_loss_ticks_dec * tick_size_dec) if self.strategy.current_position_side == config.SIDE_LONG else entry_price_dec + (stop_loss_ticks_dec * tick_size_dec)
                tp_price_dec = entry_price_dec + (take_profit_ticks_dec * tick_size_dec) if self.strategy.current_position_side == config.SIDE_LONG else entry_price_dec - (take_profit_ticks_dec * tick_size_dec)

                if (self.strategy.current_position_side == config.SIDE_LONG and (current_price_dec <= sl_price_dec or current_price_dec >= tp_price_dec)) or \
                   (self.strategy.current_position_side == config.SIDE_SHORT and (current_price_dec >= sl_price_dec or current_price_dec <= tp_price_dec)):
                    implied_sltp_triggered = True
                    self.logger.info(f"隐含 SL/TP 触发: 当前价格 {current_price_dec:.{self.price_precision}f} 与 SL={sl_price_dec:.{self.price_precision}f} 或 TP={tp_price_dec:.{self.price_precision}f} 比较。")

            exit_triggered = rsi_exit_triggered or implied_sltp_triggered

            if exit_triggered:
                self.logger.info(f"触发退出 {self.strategy.current_position_side} 仓位 ({'RSI' if rsi_exit_triggered else '隐含 SL/TP'})。")

                # 取消活跃关联订单
                self.strategy.order_manager.cancel_associated_orders()

                # 通过市场订单关闭仓位
                self.logger.info("通过市场订单关闭仓位...")
                close_status = close_current_position(
                    self.client, self.symbol_str, self.market_type,
                    self.strategy.current_position_side, self.strategy.current_position_qty,
                    self.price_precision, self.quantity_precision
                )

                if close_status == 'PLACED':
                    self.logger.info("关闭市场订单放置成功。验证仓位关闭...")
                    if verify_position_closed(
                        self.client, self.symbol_str, self.market_type,
                        self.price_precision, self.quantity_precision,
                        expected_side_that_was_closed=self.strategy.current_position_side
                    ):
                        self.logger.info("仓位成功验证关闭。")
                        self.strategy.state_syncer.clear_state(f"通过市场订单退出仓位并验证关闭。")
                    else:
                        self.logger.error("市场关闭订单放置成功，但仓位关闭验证失败。")
                elif close_status == 'NO_POSITION':
                    self.logger.warning("close_current_position 报告无仓位，但策略状态认为有仓位。状态可能已过时，同步将修正。")
                else:
                    self.logger.error(f"放置市场关闭订单失败 (状态: {close_status})。仓位可能仍打开。")

                action_taken_this_cycle = True

            elif self.strategy.use_sltp and not self.has_associated_orders_active():
                if self.strategy.stop_loss_ticks <= 0 or self.strategy.take_profit_ticks <= 0:
                    self.logger.warning(f"USE_SLTP 为 True，但 STOP_LOSS_TICKS ({self.strategy.stop_loss_ticks}) 或 TAKE_PROFIT_TICKS ({self.strategy.take_profit_ticks}) <= 0。跳过关联订单放置。")
                else:
                    self.logger.info("有仓位，USE_SLTP 为 True，但关联订单未活跃。尝试放置。")
                    if not self.strategy.order_manager.place_associated_orders(current_price_for_checks):
                        self.logger.error("放置关联订单失败。仓位未受保护！")
                    else:
                        self.logger.info("关联订单放置尝试成功。")
                    action_taken_this_cycle = True

            elif self.strategy.use_sltp and self.has_associated_orders_active():
                self.logger.info("仓位持有，关联订单活跃。让 Binance 处理退出。")
                action_taken_this_cycle = True

            elif not self.strategy.use_sltp and not exit_triggered:
                self.logger.info("仓位持有，USE_SLTP 为 False，无退出信号。继续持有。")
                action_taken_this_cycle = True

        # 无仓位时，检查入场信号
        elif not action_taken_this_cycle:
            new_entry_side = None
            if entry_long_signal:
                new_entry_side = config.SIDE_LONG
            elif self.market_type == 'FUTURES' and entry_short_signal:
                new_entry_side = config.SIDE_SHORT

            if new_entry_side:
                self.logger.info(f"无仓位。检测到 {new_entry_side} 入场信号 @ {closed_agg_candle.name.isoformat()}。尝试入场订单。")

                # 取消任何残留关联订单
                self.strategy.order_manager.cancel_associated_orders()

                entry_order_side = config.SIDE_BUY if new_entry_side == config.SIDE_LONG else config.SIDE_SELL
                entry_qty = calculate_trade_qty(
                    self.client, self.symbol_str, entry_order_side, self.market_type,
                    self.strategy.config, {'price_precision': self.price_precision, 'quantity_precision': self.quantity_precision, 'tick_size': self.strategy.tick_size},
                    pd.DataFrame([{'close': current_price_for_checks}])
                )

                if entry_qty > 0:
                    formatted_entry_qty_str = format_quantity(entry_qty, self.quantity_precision)
                    if formatted_entry_qty_str is None or float(formatted_entry_qty_str) <= 0:
                        self.logger.warning(f"格式化入场数量无效 ({formatted_entry_qty_str})。无法放置订单。")
                    else:
                        entry_order_response = place_trade_order(
                            self.client, self.symbol_str, entry_order_side, entry_qty, self.market_type, self.price_precision, self.quantity_precision,
                            order_type=config.ORDER_TYPE_MARKET
                        )
                        if entry_order_response and entry_order_response.get('orderId'):
                            order_id = entry_order_response['orderId']
                            order_status = entry_order_response.get('status')
                            if order_status == 'FILLED':
                                filled_qty = float(entry_order_response.get('executedQty', '0'))
                                filled_price = float(entry_order_response.get('avgPrice', '0'))
                                if filled_price > 0 and filled_qty > 0:
                                    self.strategy.current_position_side = new_entry_side
                                    self.strategy.current_position_qty = filled_qty
                                    self.strategy.entry_price = filled_price
                                    self.logger.info(f"入场 {new_entry_side} 订单 {order_id} 立即成交。价格: {filled_price:.{self.price_precision}f}，数量: {filled_qty:.{self.quantity_precision}f}")
                                    if self.strategy.use_sltp and self.strategy.stop_loss_ticks > 0 and self.strategy.take_profit_ticks > 0:
                                        if self.strategy.order_manager.place_associated_orders(current_price_for_checks):
                                            self.logger.info("关联订单放置尝试成功。")
                                        else:
                                            self.logger.error("放置关联订单失败。仓位未受保护！")
                                    action_taken_this_cycle = True
                            else:
                                self.logger.warning(f"入场订单 {order_id} 放置但未立即成交 (状态: {order_status})。假设入场失败。")
                                self.strategy.clear_state(f"入场订单 {order_id} 未立即成交。")
                        else:
                            self.logger.error(f"放置入场订单失败。")
                else:
                    self.logger.warning("计算入场数量为零或无效。跳过入场。")

                action_taken_this_cycle = True

        if not action_taken_this_cycle:
            self.logger.info(f"本周期无交易动作: {self.symbol_str}。")

        self.logger.info(f"--- 交易策略周期结束 ({self.symbol_str}) ---")

    def has_associated_orders_active(self):
        """
        检查本地状态中是否有活跃关联订单。
        返回 True 如果有活跃订单。
        """
        if self.market_type == 'FUTURES':
            return self.strategy.active_sl_order_id is not None or self.strategy.active_tp_order_id is not None
        elif self.market_type == 'SPOT':
            return self.strategy.active_oco_order_list_id is not None
        return False
