#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
from datetime import datetime
from decimal import Decimal
from app.core import config
from app.utils.trading_utils import calculate_trade_qty

# 获取日志器
logger = logging.getLogger(__name__)

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
