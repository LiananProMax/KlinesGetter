#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from binance.client import Client
from app.core import config
from app.api_clients.binance_trading_api import (
    cancel_trade_order, cancel_spot_oco_order, place_spot_oco_sell_order,
    place_trade_order
)
from app.utils.trading_utils import (
    format_price, format_quantity, adjust_price_to_tick_size, generate_client_order_id
)

# 获取日志器
logger = logging.getLogger(__name__)

class OrderManagement:
    """
    订单管理模块，处理与订单相关的操作。
    """

    def __init__(self, strategy):
        """
        初始化订单管理模块。

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

    def cancel_associated_orders(self):
        """
        取消活跃的关联订单（止损/止盈或 OCO）。
        根据本地状态 ID 尝试取消订单。
        返回 True 如果至少一个取消请求成功发送。
        """
        cancelled_any = False
        attempted_cancellations = False

        if self.market_type == 'FUTURES':
            if self.strategy.active_sl_order_id is not None:
                attempted_cancellations = True
                self.logger.info(f"取消 Futures 止损订单 {self.strategy.active_sl_order_id} ...")
                if cancel_trade_order(self.client, self.symbol_str, self.strategy.active_sl_order_id, self.market_type):
                    cancelled_any = True
                else:
                    self.logger.warning(f"发送取消请求失败: Futures 止损订单 {self.strategy.active_sl_order_id}。")

            if self.strategy.active_tp_order_id is not None:
                attempted_cancellations = True
                self.logger.info(f"取消 Futures 止盈订单 {self.strategy.active_tp_order_id} ...")
                if cancel_trade_order(self.client, self.symbol_str, self.strategy.active_tp_order_id, self.market_type):
                    cancelled_any = True
                else:
                    self.logger.warning(f"发送取消请求失败: Futures 止盈订单 {self.strategy.active_tp_order_id}。")

        elif self.market_type == 'SPOT':
            if self.strategy.active_oco_order_list_id is not None:
                attempted_cancellations = True
                self.logger.info(f"取消 Spot OCO 订单列表 {self.strategy.active_oco_order_list_id} ...")
                if cancel_spot_oco_order(self.client, self.symbol_str, order_list_id=self.strategy.active_oco_order_list_id):
                    cancelled_any = True
                else:
                    self.logger.warning(f"发送取消请求失败: Spot OCO 订单列表 {self.strategy.active_oco_order_list_id}。")

        if attempted_cancellations:
            if cancelled_any:
                self.logger.info("成功发送部分关联订单取消请求。")
            else:
                self.logger.warning("所有关联订单取消请求发送失败。")
        else:
            self.logger.info("本地无活跃关联订单需要取消。")

        return cancelled_any

    def place_associated_orders(self, current_price_for_checks: float | Decimal):
        """
        为现有仓位放置关联订单（止损/止盈或 OCO），如果 USE_SLTP 为 True。
        更新本地状态 ID。
        返回 True 如果所有订单放置成功。
        """
        if not self.strategy.use_sltp:
            self.logger.info("USE_SLTP 为 False，不放置关联订单。")
            return False

        # 检查本地是否已有活跃订单
        is_associated_active_locally = False
        if self.market_type == 'FUTURES' and (self.strategy.active_sl_order_id is not None or self.strategy.active_tp_order_id is not None):
            is_associated_active_locally = True
        elif self.market_type == 'SPOT' and self.strategy.active_oco_order_list_id is not None:
            is_associated_active_locally = True

        if is_associated_active_locally:
            self.logger.info(f"关联订单已活跃，本地状态显示: {self.market_type} 订单 ID。跳过放置。")
            return True  # 认为已处理

        # 确保仓位和入场价格有效
        if not (self.strategy.current_position_side and self.strategy.entry_price is not None and self.strategy.current_position_qty > 0):
            self.logger.warning("无法放置关联订单: 无有效仓位状态。")
            return False

        # 计算目标价格
        try:
            entry_price_dec = Decimal(str(self.strategy.entry_price))
            tick_size_dec = Decimal(str(self.strategy.tick_size))
            stop_loss_ticks_dec = Decimal(str(self.strategy.stop_loss_ticks))
            take_profit_ticks_dec = Decimal(str(self.strategy.take_profit_ticks))
            current_price_dec = Decimal(str(current_price_for_checks))

            if self.strategy.current_position_side == config.SIDE_LONG:
                sl_trigger_price_val = entry_price_dec - (stop_loss_ticks_dec * tick_size_dec)
                tp_trigger_price_val = entry_price_dec + (take_profit_ticks_dec * tick_size_dec)
                sl_direction, tp_direction = 'DOWN', 'UP'
                order_side_for_sltp = config.SIDE_SELL
            elif self.strategy.current_position_side == config.SIDE_SHORT:
                sl_trigger_price_val = entry_price_dec + (stop_loss_ticks_dec * tick_size_dec)
                tp_trigger_price_val = entry_price_dec - (take_profit_ticks_dec * tick_size_dec)
                sl_direction, tp_direction = 'UP', 'DOWN'
                order_side_for_sltp = config.SIDE_BUY
            else:
                self.logger.error(f"无效的仓位方向 '{self.strategy.current_position_side}' 用于关联订单计算。无法放置订单。")
                return False

            qty_to_close_float = self.strategy.current_position_qty
            formatted_qty_to_close_str = format_quantity(qty_to_close_float, self.quantity_precision)
            if formatted_qty_to_close_str is None or float(formatted_qty_to_close_str) <= 0:
                self.logger.warning(f"计算的关闭数量 {qty_to_close_float} 格式化为无效值 '{formatted_qty_to_close_str}'。无法放置订单。")
                return False

            placed_successfully = False

            if self.market_type == 'FUTURES':
                # 调整 Futures SL/TP 触发价格到 tick 大小
                sl_stop_price_adj = adjust_price_to_tick_size(float(sl_trigger_price_val), tick_size_dec, self.price_precision, direction=sl_direction)
                tp_stop_price_adj = adjust_price_to_tick_size(float(tp_trigger_price_val), tick_size_dec, self.price_precision, direction=tp_direction)

                if sl_stop_price_adj is None or tp_stop_price_adj is None or sl_stop_price_adj <= 0 or tp_stop_price_adj <= 0:
                    self.logger.error(f"Futures SL/TP: 无效的调整后价格: SL={sl_stop_price_adj}，TP={tp_stop_price_adj}。无法放置订单。")
                    return False

                # 安全检查: 防止 SL/TP 立即触发
                if (self.strategy.current_position_side == config.SIDE_LONG and (Decimal(str(sl_stop_price_adj)) >= current_price_dec or Decimal(str(tp_stop_price_adj)) <= current_price_dec)) or \
                   (self.strategy.current_position_side == config.SIDE_SHORT and (Decimal(str(sl_stop_price_adj)) <= current_price_dec or Decimal(str(tp_stop_price_adj)) >= current_price_dec)):
                    self.logger.error(f"Futures SL/TP: 计算的调整后 SL/TP 价格 {sl_stop_price_adj}/{tp_stop_price_adj} 可能立即触发，当前价格 {current_price_dec}。无法放置订单。")
                    return False

                self.logger.info(f"计算的 Futures 目标价格: SL={sl_stop_price_adj:.{self.price_precision}f}，TP={tp_stop_price_adj:.{self.price_precision}f}")

                sl_order_placed = False
                tp_order_placed = False

                # 放置 SL 订单
                sl_client_order_id = generate_client_order_id("bot_FSL", self.symbol_str)
                self.logger.info(f"放置 SL {order_side_for_sltp} 订单，数量 {formatted_qty_to_close_str}，符号 {self.symbol_str}，止损价格 {sl_stop_price_adj:.{self.price_precision}f}，ClientOrderID: {sl_client_order_id}")
                sl_order_response = place_trade_order(
                    self.client, self.symbol_str, order_side_for_sltp, float(formatted_qty_to_close_str), 'FUTURES', self.price_precision, self.quantity_precision,
                    order_type=config.FUTURE_ORDER_TYPE_STOP_MARKET, stop_price=sl_stop_price_adj, reduce_only=True, client_order_id=sl_client_order_id
                )
                if sl_order_response and sl_order_response.get('orderId'):
                    self.strategy.active_sl_order_id = sl_order_response['orderId']
                    sl_order_placed = True
                    self.logger.info(f"Futures SL 订单放置成功: ID {self.strategy.active_sl_order_id}")
                else:
                    self.logger.error(f"放置 Futures SL 订单失败。")

                # 放置 TP 订单
                tp_client_order_id = generate_client_order_id("bot_FTP", self.symbol_str)
                self.logger.info(f"放置 TP {order_side_for_sltp} 订单，数量 {formatted_qty_to_close_str}，符号 {self.symbol_str}，止盈价格 {tp_stop_price_adj:.{self.price_precision}f}，ClientOrderID: {tp_client_order_id}")
                tp_order_response = place_trade_order(
                    self.client, self.symbol_str, order_side_for_sltp, float(formatted_qty_to_close_str), 'FUTURES', self.price_precision, self.quantity_precision,
                    order_type=config.FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET, stop_price=tp_stop_price_adj, reduce_only=True, client_order_id=tp_client_order_id
                )
                if tp_order_response and tp_order_response.get('orderId'):
                    self.strategy.active_tp_order_id = tp_order_response['orderId']
                    tp_order_placed = True
                    self.logger.info(f"Futures TP 订单放置成功: ID {self.strategy.active_tp_order_id}")
                else:
                    self.logger.error(f"放置 Futures TP 订单失败。")

                placed_successfully = sl_order_placed and tp_order_placed
                if not placed_successfully:
                    self.logger.error("一个或多个 Futures SL/TP 订单放置失败。仓位未受保护！")
                    self.strategy.clear_associated_order_state(reason="Futures SL/TP 放置失败。")
                else:
                    self.logger.info("Futures SL/TP 订单放置尝试成功。")

            elif self.market_type == 'SPOT':
                # Spot 使用 OCO 订单
                sl_limit_offset_ticks = Decimal('5')  # 小间隙，配置可调整
                sl_limit_price_val = sl_trigger_price_val - (sl_limit_offset_ticks * tick_size_dec)
                if sl_limit_price_val <= Decimal('0'):
                    sl_limit_price_val = sl_trigger_price_val * Decimal('0.995')  # 0.5% 以下，确保正数
                sl_limit_price_val = max(sl_limit_price_val, tick_size_dec)  # 确保至少一个 tick 大小

                # 调整 Spot OCO 价格到 tick 大小
                tp_limit_price_adj = adjust_price_to_tick_size(float(tp_trigger_price_val), tick_size_dec, self.price_precision, direction='UP')
                sl_trigger_price_adj = adjust_price_to_tick_size(float(sl_trigger_price_val), tick_size_dec, self.price_precision, direction='DOWN')
                sl_limit_price_adj = adjust_price_to_tick_size(float(sl_limit_price_val), tick_size_dec, self.price_precision, direction='DOWN')

                tp_limit_price_adj_str = format_price(tp_limit_price_adj, self.price_precision)
                sl_trigger_price_adj_str = format_price(sl_trigger_price_adj, self.price_precision)
                sl_limit_price_adj_str = format_price(sl_limit_price_adj, self.price_precision)

                if not (tp_limit_price_adj_str and float(tp_limit_price_adj_str) > 0 and
                        sl_trigger_price_adj_str and float(sl_trigger_price_adj_str) > 0 and
                        sl_limit_price_adj_str and float(sl_limit_price_adj_str) > 0):
                    self.logger.error(f"Spot OCO: 无效的计算或格式化价格。TP:{tp_limit_price_adj_str}，SL触发:{sl_trigger_price_adj_str}，SL限制:{sl_limit_price_adj_str}。无法放置订单。")
                    return False

                # 验证逻辑价格关系
                tp_limit_price_dec = Decimal(tp_limit_price_adj_str)
                sl_trigger_price_dec = Decimal(sl_trigger_price_adj_str)
                sl_limit_price_dec = Decimal(sl_limit_price_adj_str)

                if not (tp_limit_price_dec > sl_trigger_price_dec):
                    self.logger.error(f"Spot OCO: 逻辑价格错误: TP 限制 ({tp_limit_price_dec}) 未大于 SL 触发 ({sl_trigger_price_dec})。无法放置订单。")
                    return False
                if not (sl_trigger_price_dec >= sl_limit_price_dec):
                    self.logger.error(f"Spot OCO: 逻辑价格错误: SL 触发 ({sl_trigger_price_dec}) 未大于等于 SL 限制 ({sl_limit_price_dec})。无法放置订单。")
                    return False

                # 安全检查当前价格
                if not (tp_limit_price_dec > current_price_dec):
                    self.logger.error(f"Spot OCO SELL: 计算的 TP 限制价格 ({tp_limit_price_dec}) 未大于当前市场价格 ({current_price_dec})。无法放置订单。")
                    return False
                if not (sl_trigger_price_dec < current_price_dec):
                    self.logger.error(f"Spot OCO SELL: 计算的 SL 触发价格 ({sl_trigger_price_dec}) 未小于当前市场价格 ({current_price_dec})。无法放置订单。")
                    return False

                self.logger.info(f"放置 Spot OCO SELL 订单: 数量={formatted_qty_to_close_str}，TP限制={tp_limit_price_adj_str}，SL触发={sl_trigger_price_adj_str}，SL限制={sl_limit_price_adj_str}")

                oco_response = place_spot_oco_sell_order(
                    self.client, self.symbol_str, formatted_qty_to_close_str,
                    tp_limit_price_adj_str, sl_trigger_price_adj_str, sl_limit_price_adj_str,
                    self.price_precision, self.quantity_precision,
                    list_client_order_id=generate_client_order_id("bot_OCO", self.symbol_str)
                )

                if oco_response and oco_response.get('orderListId'):
                    self.strategy.active_oco_order_list_id = oco_response['orderListId']
                    placed_successfully = True
                    self.logger.info(f"Spot OCO SELL 订单放置成功。列表 ID: {self.strategy.active_oco_order_list_id}")
                else:
                    self.logger.error(f"放置 Spot OCO SELL 订单失败。响应: {oco_response}")
                    self.strategy.clear_associated_order_state(reason="Spot OCO 放置失败。")

            else:
                self.logger.error(f"无法放置关联订单: 未知市场类型 {self.market_type}。")
                placed_successfully = False

            return placed_successfully

        except Exception as e_place:
            self.logger.error(f"放置关联订单时出错: {e_place}\n{traceback.format_exc()}")
            self.strategy.clear_associated_order_state(reason="异常期间放置关联订单。")
            return False
