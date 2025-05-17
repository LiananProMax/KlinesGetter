#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import uuid
import traceback
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import decimal

# 使用app的日志
import logging
# 假设 app 的 logging 已经配置好，直接使用 logging.getLogger
logger = logging.getLogger(__name__)

# 导入 config
from app.core import config
# 导入 Binance Client 类型提示
from binance.client import Client


def split_symbol_for_coinapi(symbol: str) -> str:
    """
    将币安的交易对符号分割成基础资产和报价资产，符合CoinAPI格式（Base_Quote）。
    这主要是为了与原始策略的兼容性/命名一致性，
    尽管在新结构中不使用CoinAPI。
    示例："AVAXUSDT" -> "AVAX_USDT"
    """
    symbol_upper = symbol.upper()
    # 添加常见的报价资产
    quote_assets = ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB", "DAI", "PAX", "TUSD", "USDP"] # 添加更多常见的资产

    for quote in quote_assets:
        if symbol_upper.endswith(quote):
            base = symbol_upper[:-len(quote)]
            # Basic check to ensure base asset is not empty
            if base:
                return f"{base}_{quote}"

    # 用于不常见符号或非常规格式的后备方案
    # 这种启发式方法容易出错，但在标准后缀失败时尝试进行分割。
    # 有一个预定义列表或依赖交易所信息可能更好。
    # 更完善的方法是使用交易所信息来找到符号的基础/报价资产。
    # 目前，如果交易所信息不易获得，请保留原始启发式方法作为后备。
    # 如果从 calculate_trade_qty 调用，我们可能有可用的 symbol_info 或可以获取它。
    # 如果可能，让我们添加使用交易所信息的检查。

    try:
        # Attempt to get base/quote from exchange info if available
        # This might require passing client instance or fetching info here (less efficient)
        # For robustness, let's keep the heuristic but acknowledge it might be wrong.
        # Ideally, pass symbol_info from a place it's already fetched (like strategy init).
        # But for a utility function, making it self-sufficient or clearly stating dependencies is key.
        # Let's assume symbol_info isn't passed and rely on heuristic or add fetching here.
        # Adding fetching makes it slower. Best to rely on heuristic for this simple function.
        pass # Keep heuristic for now

    except Exception as e:
         logger.warning(f"Could not use exchange info to split symbol '{symbol}': {e}. Falling back to heuristic.")


    # Fallback heuristic: split in half
    if len(symbol_upper) > 3: # Assume common pattern, e.g., ABCXYZ
        split_point = len(symbol_upper) // 2
        base = symbol_upper[:split_point]
        quote = symbol_upper[split_point:]
        # Add another check: does the assumed quote asset look like a common one?
        if quote in quote_assets:
             return f"{base}_{quote}"
        # Otherwise, might be wrong split, log warning
        logger.warning(f"Symbol '{symbol}' heuristic split {base}_{quote} might be incorrect.")
        return f"{base}_{quote}" # Return the heuristic split anyway

    logger.warning(f"Could not reliably split symbol '{symbol}'. Returning original symbol.")
    # Final fallback: just return the original symbol if heuristic also fails or is too short
    return symbol_upper


def format_price(price: float | Decimal | str | None, precision: int | None):
    """
    根据指定的精度格式化价格值。
    处理价格的float、Decimal、字符串或None输入。
    返回字符串表示或在错误/无效输入时返回None。
    使用Decimal进行计算。
    """
    if price is None or precision is None:
        # logger.debug("format_price called with None price or precision.")
        return None

    try:
        # Convert to Decimal. Handle string 'inf', 'nan' etc. gracefully.
        price_dec = Decimal(str(price))

        if not isinstance(precision, int) or precision < 0:
            logger.error(f"format_price: Invalid price precision integer: {precision}. Must be >= 0.")
            return None

        if price_dec.is_infinite() or price_dec.is_nan():
             logger.warning(f"format_price: Received infinite or NaN price ({price}). Cannot format.")
             return None

        # Use quantize for accurate rounding to the specified decimal places
        # Create a Decimal representation of the target precision (e.g., '0.01' for precision 2)
        quantization_target = Decimal('1') / (Decimal('10') ** precision)
        # Using ROUND_HALF_UP as is common for prices, unless tickSize dictates otherwise
        # The adjust_price_to_tick_size should be used for final order prices.
        rounded_price_dec = price_dec.quantize(quantization_target, rounding=decimal.ROUND_HALF_UP)

        # Return as a string formatted to the exact number of decimal places
        # Using f-string formatting with .nf ensures correct decimal places
        return f"{rounded_price_dec:.{precision}f}"

    except Exception as e:
        logger.error(f"Error formatting price {price} with precision {precision}: {e}\n{traceback.format_exc()}")
        return None


def format_quantity(qty: float | Decimal | str | None, precision: int | None) -> str | None:
    """
    Format a quantity value according to the specified precision using ROUND_DOWN.
    Handles float, Decimal, string, or None input for quantity.
    Returns string representation or None on error/invalid input.
    Uses Decimal for calculations.
    """
    if qty is None or precision is None:
        # logger.debug("format_quantity called with None quantity or precision.")
        return None

    try:
        # Convert to Decimal. Handle string 'inf', 'nan' etc. gracefully.
        qty_dec = Decimal(str(qty))

        if not isinstance(precision, int) or precision < 0:
            logger.error(f"format_quantity: Invalid quantity precision integer: {precision}. Must be >= 0.")
            return None

        if qty_dec.is_infinite() or qty_dec.is_nan():
             logger.warning(f"format_quantity: Received infinite or NaN quantity ({qty}). Cannot format.")
             return None

        # Use quantize with ROUND_DOWN to effectively floor/truncate to the specified decimal places
        # This is standard practice for quantities to avoid ordering too much.
        # Create a Decimal representation of the target precision (e.g., '0.01' for precision 2)
        quantization_target = Decimal('1') / (Decimal('10') ** precision)
        floored_qty_dec = qty_dec.quantize(quantization_target, rounding=decimal.ROUND_DOWN)

        # Ensure the result is non-negative after flooring near zero
        if floored_qty_dec < Decimal('0'):
            floored_qty_dec = Decimal('0')

        # Return as a string formatted to the exact number of decimal places
        # Using f-string formatting with .nf ensures correct decimal places
        return f"{floored_qty_dec:.{precision}f}"

    except Exception as e:
        logger.error(f"Error formatting quantity {qty} with precision {precision}: {e}\n{traceback.format_exc()}")
        return None

def adjust_price_to_tick_size(price: float | Decimal, tick_size_decimal: Decimal | str | None, price_precision_int: int | None, direction: str = 'NONE') -> float | None:
    """
    Adjust a price to the nearest valid tick size.
    Uses Decimal for calculations and returns a float.
    Handles None for inputs.
    Converts tick_size_decimal from string if necessary.

    Args:
        price (float | Decimal): The price to adjust
        tick_size_decimal (Decimal | str | None): The tick size as a Decimal or string
        price_precision_int (int | None): The price precision (number of decimal places)
        direction (str): The rounding direction ('UP', 'DOWN', or 'NONE').
                         'UP': round up to the next tick.
                         'DOWN': round down to the previous tick.
                         'NONE': round to the nearest tick (HALF_UP).

    Returns:
        float: The adjusted price, or None if the adjustment failed or inputs are invalid.
    """
    if tick_size_decimal is None or price is None or price_precision_int is None:
        logger.warning(f"adjust_price_to_tick_size: Called with None tick_size ({tick_size_decimal}), price ({price}), or precision ({price_precision_int}).")
        return None

    try:
        if isinstance(tick_size_decimal, str):
             tick_size_dec = Decimal(tick_size_decimal)
        elif isinstance(tick_size_decimal, Decimal):
             tick_size_dec = tick_size_decimal
        else:
             logger.error(f"adjust_price_to_tick_size: Invalid tick_size_decimal type: {type(tick_size_decimal)}")
             return None


        if tick_size_dec <= Decimal(0):
             logger.error(f"adjust_price_to_tick_size: Invalid or non-positive tick_size ({tick_size_decimal}).")
             # Fallback: just format to precision if tick_size is invalid
             try:
                 price_dec_fallback = Decimal(str(price))
                 # Ensure precision is valid integer
                 if not isinstance(price_precision_int, int) or price_precision_int < 0:
                      logger.error(f"adjust_price_to_tick_size: Invalid price precision integer fallback: {price_precision_int}")
                      return None
                 quantization_target_fallback = Decimal('1') / (Decimal('10') ** price_precision_int)
                 return float(price_dec_fallback.quantize(quantization_target_fallback, rounding=decimal.ROUND_HALF_UP))
             except Exception as fe:
                  logger.error(f"adjust_price_to_tick_size: Fallback formatting failed for price {price} with precision {price_precision_int}: {fe}")
                  return None

        price_dec = Decimal(str(price))
        if price_dec.is_infinite() or price_dec.is_nan():
             logger.warning(f"adjust_price_to_tick_size: Received infinite or NaN price ({price}). Cannot adjust.")
             return None

        # Calculate how many ticks are in the price
        # Using quantize for division to handle potential floating point issues precisely
        num_ticks_dec = price_dec.quantize(Decimal('0.00000001') / tick_size_dec, rounding=decimal.ROUND_DOWN) / tick_size_dec
        # Example: 100.05 / 0.01 = 10005.0
        # Example: 100.055 / 0.01 = 10005.5

        # Round the number of ticks based on direction
        if direction == 'UP':
            rounded_num_ticks = num_ticks_dec.to_integral_value(rounding=decimal.ROUND_UP)
        elif direction == 'DOWN':
            rounded_num_ticks = num_ticks_dec.to_integral_value(rounding=decimal.ROUND_DOWN)
        else:  # Default to nearest tick
             rounded_num_ticks = num_ticks_dec.to_integral_value(rounding=decimal.ROUND_HALF_UP)


        # Multiply the rounded number of ticks by the tick size to get the adjusted price
        adjusted_price_dec = rounded_num_ticks * tick_size_dec

        # Final quantization to ensure it has no more decimal places than allowed by price_precision
        # Although tickSize is usually aligned with pricePrecision, this adds a safeguard.
        if isinstance(price_precision_int, int) and price_precision_int >= 0:
             quantization_target_precision = Decimal('1') / (Decimal('10') ** price_precision_int)
             # Use ROUND_DOWN for final precision formatting to be safe? Or HALF_UP?
             # Binance API expects price exactly matching tick size multiple, with #decimals <= pricePrecision.
             # Rounding to tick size multiple first is key. Then formatting string.
             # Let's just return the float value of the adjusted_price_dec here.
             pass # No extra quantization needed if adjust_price_to_tick_size is only for tick size

        return float(adjusted_price_dec)

    except Exception as e:
        logger.error(f"Error in adjust_price_to_tick_size: {e} with price={price}, tick_size={tick_size_decimal}, precision={price_precision_int}, direction={direction}\n{traceback.format_exc()}")
        return None


def generate_client_order_id(prefix: str = "bot", suffix: str = "") -> str:
    """
    Generates a unique clientOrderId for Binance orders.
    Binance clientOrderId max length is 36 characters.
    Structure: {prefix}_{timestamp_ms}_{uuid_short}{_suffix_if_any}
    Prioritizes keeping timestamp and UUID part, shortens prefix/suffix if needed.
    """
    timestamp_ms_str = str(int(datetime.now(timezone.utc).timestamp() * 1000))  # 13 digits
    unique_id_str = uuid.uuid4().hex[:8]  # Use first 8 hex chars (sufficiently unique for bot scale)

    # Core unique part: timestamp_ms + unique_id_str
    core_part = f"{timestamp_ms_str}_{unique_id_str}" # Length: 13 + 1 + 8 = 22

    max_len = 36

    # Calculate space needed for prefix and suffix including separators
    prefix_part = prefix if prefix else ""
    suffix_part = suffix if suffix else ""

    # Tentative structure: [prefix_]_core_[_suffix]
    # Length = len(prefix_part) + (1 if prefix_part else 0) + len(core_part) + (1 if suffix_part else 0) + len(suffix_part)
    # Let's use a simple joining strategy and then truncate if needed, warning if it happens.
    # A more complex approach could be to dynamically shorten prefix/suffix before joining.
    # Simpler: join, check length, truncate and warn.

    parts = [p for p in [prefix_part, core_part, suffix_part] if p] # Filter out empty strings
    generated_id = "_".join(parts)

    if len(generated_id) > max_len:
        # Shorten the generated ID from the end, prioritizing keeping the start (prefix, timestamp)
        truncated_id = generated_id[:max_len]
        logger.warning(f"clientOrderId: Generated ID '{generated_id}' ({len(generated_id)}) exceeds max length {max_len}. Truncating to '{truncated_id}'.")
        generated_id = truncated_id

    return generated_id


# calculate_trade_qty 函数已移动到 app/trading/trade_execution.py，以避免循环导入

# interval_to_timedelta and align_timestamp_to_interval were moved to app.utils.kline_utils
# Removing them from here to avoid duplication.
