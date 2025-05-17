#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from binance.client import Client
from app.core import config
from app.api_clients.binance_trading_api import (
    get_current_market_status, get_open_orders_for_symbol_binance, get_open_oco_lists_binance
)
from app.utils.trading_utils import is_bot_order

# 获取日志器
logger = logging.getLogger(__name__)

class StateSync:
    """
    状态同步模块，处理与 Binance 同步状态的逻辑。
    """

    def __init__(self, strategy):
        """
        初始化状态同步模块。

        参数:
            strategy: TradingStrategy 实例的引用。
        """
        self.strategy = strategy  # 策略实例引用
        self.client = strategy.client  # Binance 客户端
        self.symbol_str = strategy.symbol_str  # 交易对符号
        self.market_type = strategy.market_type  # 市场类型
        self.price_precision = strategy.price_precision  # 价格精度
        self.quantity_precision = strategy.quantity_precision  # 数量精度
        self.logger = strategy.logger  # 使用策略的日志器

    def sync_state_with_binance(self, latest_base_kline_df: pd.DataFrame | None = None):
        """
        与 Binance 同步本地策略状态（仓位和关联订单）。
        参数:
            latest_base_kline_df: 可选，用于 SPOT 余额检查的日志上下文。
        """
        self.logger.info(f"同步本地状态与 Binance API，符号 {self.symbol_str} ({self.market_type})...")
        current_qty_before_sync = self.strategy.current_position_qty
        current_side_before_sync = self.strategy.current_position_side

        try:
            # 获取仓位状态
            position_data = get_current_market_status(self.client, self.symbol_str, self.market_type, latest_kline_df=latest_base_kline_df)
            position_api_status = position_data[-1] if position_data and isinstance(position_data, tuple) and len(position_data) > 0 else 'ERROR'

            synced_position_side = None
            synced_position_qty = 0.0
            synced_entry_price = self.strategy.entry_price  # 默认保留本地入场价格

            if position_api_status == 'OK':
                if self.market_type == 'FUTURES':
                    synced_position_side, synced_position_qty, synced_entry_price_from_api, _ = position_data
                    if abs(Decimal(str(synced_position_qty or 0))) > Decimal('0'):
                        if synced_entry_price_from_api is not None and synced_entry_price_from_api > 0:
                            synced_entry_price = synced_entry_price_from_api
                            self.logger.info(f"同步: 更新 Futures 入场价格为 API 值: {synced_entry_price}")
                        else:
                            self.logger.warning(f"同步: API 报告 Futures 仓位 ({synced_position_qty}) 但入场价格无效 ({synced_entry_price_from_api})。保留本地入场价格 {self.strategy.entry_price}。")
                    else:
                        synced_position_side = None
                        synced_position_qty = 0.0
                        synced_entry_price = None

                elif self.market_type == 'SPOT':
                    synced_position_side, synced_position_qty, synced_entry_price_from_api, _ = position_data
                    if synced_position_side == config.SIDE_LONG and synced_position_qty is not None and synced_position_qty > 0:
                        # SPOT API 没有可靠的平均入场价格，保留本地值
                        if self.strategy.entry_price is None:
                            self.logger.warning("同步: 检测到 SPOT 仓位，但本地入场价格为空。入场价格将保持为空。")
                            synced_entry_price = None
                    else:
                        synced_position_side = None
                        synced_position_qty = 0.0
                        synced_entry_price = None

            elif position_api_status == 'NO_POSITION':
                synced_position_side = None
                synced_position_qty = 0.0
                synced_entry_price = None

            else:
                self.logger.error(f"同步: 获取 Binance 仓位状态失败 ({position_api_status})。本地状态可能不准确。")
                return  # 保持本地状态不变

            # 更新本地仓位状态
            qty_changed = abs(Decimal(str(self.strategy.current_position_qty or 0))) != abs(Decimal(str(synced_position_qty or 0)))
            side_changed = self.strategy.current_position_side != synced_position_side

            if side_changed or qty_changed:
                self.logger.info(f"同步: 仓位状态更改: {current_side_before_sync} {current_qty_before_sync} -> {synced_position_side} {synced_position_qty}")
                self.strategy.current_position_side = synced_position_side
                self.strategy.current_position_qty = synced_position_qty
                if self.strategy.current_position_side:
                    if self.market_type == 'FUTURES' and synced_entry_price is not None and synced_entry_price > 0:
                        self.strategy.entry_price = synced_entry_price
                else:
                    self.strategy.entry_price = None
                    self.clear_associated_order_state(reason="仓位关闭。")

            # 同步关联订单状态
            open_orders_binance = get_open_orders_for_symbol_binance(self.client, self.symbol_str, self.market_type)
            if open_orders_binance is None:
                self.logger.warning(f"同步: 获取开放订单失败。本地关联订单状态可能不准确。")
            else:
                if self.market_type == 'FUTURES':
                    synced_sl_id, synced_tp_id = None, None
                    for order in open_orders_binance:
                        order_id = order.get('orderId')
                        order_type = order.get('type')
                        order_status = order.get('status')
                        is_reduce_only = order.get('reduceOnly', False)
                        client_order_id = order.get('clientOrderId')

                        if is_reduce_only and order_status in ['NEW', 'PARTIALLY_FILLED', 'PENDING_CANCEL'] and is_bot_order(client_order_id):
                            if order_type in [config.FUTURE_ORDER_TYPE_STOP_MARKET, config.FUTURE_ORDER_TYPE_STOP]:
                                if synced_sl_id is None:
                                    synced_sl_id = order_id
                            elif order_type in [config.FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET, config.FUTURE_ORDER_TYPE_TAKE_PROFIT]:
                                if synced_tp_id is None:
                                    synced_tp_id = order_id

                    if self.strategy.active_sl_order_id != synced_sl_id:
                        self.logger.info(f"同步: Futures 止损订单 ID 更新: {self.strategy.active_sl_order_id} -> {synced_sl_id}")
                        self.strategy.active_sl_order_id = synced_sl_id
                    if self.strategy.active_tp_order_id != synced_tp_id:
                        self.logger.info(f"同步: Futures 止盈订单 ID 更新: {self.strategy.active_tp_order_id} -> {synced_tp_id}")
                        self.strategy.active_tp_order_id = synced_tp_id

                elif self.market_type == 'SPOT':
                    synced_oco_list_id = None
                    open_oco_lists = get_open_oco_lists_binance(self.client, self.symbol_str)
                    if open_oco_lists is not None:
                        for oco_list in open_oco_lists:
                            list_id = oco_list.get('orderListId')
                            list_status_type = oco_list.get('listStatusType')
                            list_client_order_id = oco_list.get('listClientOrderId')
                            if is_bot_order(list_client_order_id) and list_status_type in ['EXECUTING', 'RESPONSE']:
                                if synced_oco_list_id is None:
                                    synced_oco_list_id = list_id
                                break
                    if self.strategy.active_oco_order_list_id != synced_oco_list_id:
                        self.logger.info(f"同步: Spot OCO 订单列表 ID 更新: {self.strategy.active_oco_order_list_id} -> {synced_oco_list_id}")
                        self.strategy.active_oco_order_list_id = synced_oco_list_id

        except Exception as sync_e:
            self.logger.error(f"状态同步出错: {sync_e}\n{traceback.format_exc()}")

    def clear_state(self, reason: str = ""):
        """
        清除所有本地策略状态变量。
        参数:
            reason: 清除状态的原因，用于日志。
        """
        self.logger.info(f"清除本地状态，原因: {reason}")
        self.strategy.current_position_side = None
        self.strategy.current_position_qty = 0.0
        self.strategy.entry_price = None
        self.clear_associated_order_state(reason)

    def clear_associated_order_state(self, reason: str = ""):
        """
        清除本地关联订单状态变量（不取消订单）。
        参数:
            reason: 清除状态的原因，用于日志。
        """
        self.logger.info(f"清除关联订单状态，原因: {reason}")
        if self.market_type == 'FUTURES':
            if self.strategy.active_sl_order_id is not None or self.strategy.active_tp_order_id is not None:
                self.logger.info(f"当前 Futures SL ID: {self.strategy.active_sl_order_id}，TP ID: {self.strategy.active_tp_order_id}")
                self.strategy.active_sl_order_id = None
                self.strategy.active_tp_order_id = None
                self.logger.info("已清除 Futures SL/TP ID。")
            else:
                self.logger.info("无 Futures SL/TP ID 需要清除。")
        elif self.market_type == 'SPOT':
            if self.strategy.active_oco_order_list_id is not None:
                self.logger.info(f"当前 Spot OCO 列表 ID: {self.strategy.active_oco_order_list_id}")
                self.strategy.active_oco_order_list_id = None
                self.logger.info("已清除 Spot OCO 列表 ID。")
            else:
                self.logger.info("无 Spot OCO 列表 ID 需要清除。")
        else:
            self.logger.warning(f"清除关联订单状态时，未知市场类型 {self.market_type}。")
