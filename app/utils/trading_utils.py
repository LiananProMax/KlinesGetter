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

# 导入 config 和 api_clients，因为需要平衡信息和配置常量
from app.core import config
# 导入 Binance Client 类型提示，以及API helper 函数
from binance.client import Client
# 需要从 binance_trading_api 导入余额获取函数
from app.api_clients.binance_trading_api import _get_futures_usdt_balance, _get_spot_asset_balance, get_symbol_info_from_exchange, safe_api_call # safe_api_call needed for get_symbol_info/ticker


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


def calculate_trade_qty(
    client: Client,
    symbol_str: str,
    order_side: str, # BUY or SELL
    market_type: str, # FUTURES or SPOT
    strategy_config: dict, # Contains QTY_PERCENT, SPOT_QUOTE_ASSET (if SPOT)
    precisions: dict, # Must contain 'price_precision', 'quantity_precision', 'tick_size'
    latest_kline_df: pd.DataFrame | None = None # Optional, needed for SPOT MIN_NOTIONAL calc if using last close price
) -> float:
    """
    Calculates the quantity for a trade order (entry or closing) based on QTY_PERCENT
    and available balance/equity.

    Args:
        client (Client): Initialized Binance Client instance.
        symbol_str (str): Trading symbol (e.g., "BTCUSDT").
        order_side (str): The side of the order to place ('BUY' or 'SELL').
        market_type (str): 'FUTURES' or 'SPOT'.
        strategy_config (dict): Strategy configuration including 'QTY_PERCENT'
                                and potentially 'SPOT_QUOTE_ASSET'.
        precisions (dict): Dictionary with precision info: 'price_precision',
                           'quantity_precision', 'tick_size'.
        latest_kline_df (pd.DataFrame | None): Optional DataFrame of the latest
                                                klines (can be aggregated DF for
                                                getting latest price context).

    Returns:
        float: The calculated quantity as a float, or 0.0 if calculation fails
               or results in zero quantity.
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
         # Using the heuristic split symbol utility. Note: this might be wrong for some symbols.
         # A more robust way is to get symbol info from exchange and parse baseAsset/quoteAsset.
         # Let's try to fetch symbol info first for robustness.
         symbol_info = get_symbol_info_from_exchange(client, symbol_str, market_type)
         if symbol_info and 'baseAsset' in symbol_info and 'quoteAsset' in symbol_info:
              base_asset = symbol_info['baseAsset']
              quote_asset = symbol_info['quoteAsset']
              logger.data(f"calculate_trade_qty: Fetched base/quote assets from exchange info: {base_asset}, {quote_asset}")
         else:
              logger.warning(f"calculate_trade_qty: Could not fetch base/quote from exchange info for {symbol_str}. Falling back to split_symbol_for_coinapi heuristic.")
              # Fallback to heuristic
              heuristic_split = split_symbol_for_coinapi(symbol_str)
              if '_' in heuristic_split:
                  base_asset, quote_asset = heuristic_split.split('_')
              else:
                  logger.error(f"calculate_trade_qty: Failed to split symbol '{symbol_str}' into base/quote using heuristic.")
                  return 0.0

    except Exception as e_symbol_split:
         logger.error(f"calculate_trade_qty: Error getting symbol info or splitting symbol: {e_symbol_split}\n{traceback.format_exc()}")
         return 0.0


    # --- Determine Available Capital / Quantity to Use ---
    capital_to_use = Decimal('0')
    base_qty_to_sell = Decimal('0')
    trade_price = Decimal('0') # Price used for calculation (estimate)

    try:
        if market_type == 'FUTURES':
            # For Futures in cross margin, capital is available USDT balance
            available_usdt_balance = _get_futures_usdt_balance(client)
            if available_usdt_balance is None: # Helper returns 0 on failure, let's check explicitly if it could be None
                 logger.error("calculate_trade_qty: Failed to get futures USDT balance.")
                 return 0.0
            available_usdt_balance_dec = Decimal(str(available_usdt_balance))
            capital_to_use = available_usdt_balance_dec * Decimal(str(qty_percent))
            logger.data(f"calculate_trade_qty: FUTURES: Available USDT={available_usdt_balance_dec:.8f}, Capital to use={capital_to_use:.8f}")

            # Need an estimated trade price to calculate quantity (for MARKET orders)
            # Use the close price of the latest available kline for estimation.
            # If latest_kline_df is available and not empty, use its last close.
            if latest_kline_df is not None and not latest_kline_df.empty and 'close' in latest_kline_df.columns:
                 trade_price_float = latest_kline_df['close'].iloc[-1]
                 trade_price = Decimal(str(trade_price_float))
                 if trade_price <= Decimal('0') or trade_price.is_nan() or trade_price.is_infinite():
                     logger.warning(f"calculate_trade_qty: Latest kline close price {trade_price_float} is invalid. Attempting to fetch ticker price.")
                     trade_price = Decimal('0') # Reset if invalid

            if trade_price <= Decimal('0'): # If kline price was invalid or not available
                # Fetch current symbol ticker price as a fallback
                try:
                    ticker = safe_api_call(client, client.get_symbol_ticker, symbol=symbol_str)
                    if ticker and 'price' in ticker:
                        trade_price = Decimal(ticker['price'])
                        logger.data(f"calculate_trade_qty: Fetched ticker price as fallback: {trade_price:.{price_precision}f}")
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
            # For SPOT BUY orders, capital is available Quote Asset balance (e.g., USDT)
            if order_side == config.SIDE_BUY:
                 quote_asset_to_use = strategy_config.get('SPOT_QUOTE_ASSET', quote_asset) # Use config value if present, else derived quote asset
                 if not quote_asset_to_use:
                     logger.error("calculate_trade_qty: SPOT BUY: SPOT_QUOTE_ASSET or derived quote asset not found.")
                     return 0.0

                 available_quote_balance = _get_spot_asset_balance(client, quote_asset_to_use)
                 if available_quote_balance is None: # Helper returns None on API error
                      logger.error(f"calculate_trade_qty: SPOT BUY: Failed to get spot balance for {quote_asset_to_use}.")
                      return 0.0
                 available_quote_balance_dec = Decimal(str(available_quote_balance))
                 capital_to_use = available_quote_balance_dec * Decimal(str(qty_percent))
                 logger.data(f"calculate_trade_qty: SPOT BUY: Available {quote_asset_to_use}={available_quote_balance_dec:.8f}, Capital to use={capital_to_use:.8f}")

                 # Need an estimated trade price for SPOT BUY market order quantity calculation
                 # Binance MARKET BUY uses quoteOrderQty if provided, otherwise assumes quantity is base asset quantity.
                 # Our current implementation of place_trade_order for SPOT MARKET BUY uses the 'quantity' parameter (base asset quantity).
                 # So we need to calculate base asset quantity based on capital_to_use and estimated price.
                 # Get latest price for estimation (from latest kline or ticker)
                 if latest_kline_df is not None and not latest_kline_df.empty and 'close' in latest_kline_df.columns:
                    trade_price_float = latest_kline_df['close'].iloc[-1]
                    trade_price = Decimal(str(trade_price_float))
                    if trade_price <= Decimal('0') or trade_price.is_nan() or trade_price.is_infinite():
                       logger.warning(f"calculate_trade_qty: Latest kline close price {trade_price_float} is invalid for SPOT BUY. Attempting to fetch ticker price.")
                       trade_price = Decimal('0') # Reset if invalid

                 if trade_price <= Decimal('0'): # If kline price was invalid or not available
                     # Fetch current symbol ticker price as a fallback
                     try:
                         ticker = safe_api_call(client, client.get_symbol_ticker, symbol=symbol_str)
                         if ticker and 'price' in ticker:
                             trade_price = Decimal(ticker['price'])
                             logger.data(f"calculate_trade_qty: SPOT BUY: Fetched ticker price as fallback: {trade_price:.{price_precision}f}")
                         else:
                             logger.error("calculate_trade_qty: SPOT BUY: Failed to fetch ticker price for quantity calculation.")
                             return 0.0
                     except Exception as e_ticker:
                         logger.error(f"calculate_trade_qty: SPOT BUY: Error fetching ticker price: {e_ticker}")
                         return 0.0

                 if trade_price <= Decimal('0'):
                    logger.error("calculate_trade_qty: SPOT BUY: Trade price estimate is zero or negative. Cannot calculate quantity.")
                    return 0.0

            # For SPOT SELL orders, quantity is the *available base asset balance*
            # (when closing a position) - QTY_PERCENT isn't typically used for closing.
            # However, the strategy might call this function for a SELL *entry* (not applicable here)
            # or for calculating the *quantity to sell* for a close order.
            # In the current strategy logic, `close_current_position` handles getting the full base balance to sell.
            # If calculate_trade_qty is called with SIDE_SELL, it implies closing the position,
            # so we should return the current base asset balance.
            elif order_side == config.SIDE_SELL:
                 if not base_asset:
                     logger.error("calculate_trade_qty: SPOT SELL: Base asset not found.")
                     return 0.0
                 available_base_balance = _get_spot_asset_balance(client, base_asset)
                 if available_base_balance is None: # Helper returns None on API error
                      logger.error(f"calculate_trade_qty: SPOT SELL: Failed to get spot balance for {base_asset}.")
                      return 0.0
                 base_qty_to_sell = Decimal(str(available_base_balance))
                 logger.data(f"calculate_trade_qty: SPOT SELL: Available {base_asset}={base_qty_to_sell:.8f}. Using this as quantity.")
                 # No price needed here as quantity is based on balance, not capital.

            else:
                logger.error(f"calculate_trade_qty: Unknown order side '{order_side}' for market type '{market_type}'.")
                return 0.0

        else:
            logger.error(f"calculate_trade_qty: Unknown market type '{market_type}'.")
            return 0.0

    except Exception as e_balance:
         logger.error(f"calculate_trade_qty: Error getting balance or estimating price: {e_balance}\n{traceback.format_exc()}")
         return 0.0

    # --- Calculate Quantity based on Capital and Price ---
    calculated_qty_dec = Decimal('0')
    if market_type == 'FUTURES' or (market_type == 'SPOT' and order_side == config.SIDE_BUY):
        # Calculate quantity = Capital / Price (for BUY orders)
        if trade_price > Decimal('0'):
            calculated_qty_dec = capital_to_use / trade_price
        else:
            logger.error("calculate_trade_qty: Estimated trade price is zero or negative after checks. Cannot calculate quantity.")
            return 0.0

    elif market_type == 'SPOT' and order_side == config.SIDE_SELL:
        # For SPOT SELL (closing), the quantity is simply the available base asset balance calculated above
        calculated_qty_dec = base_qty_to_sell


    # --- Apply Filters/Rules (e.g., MIN_NOTIONAL, LOT_SIZE, MIN_QTY, MAX_QTY) ---
    # Fetch exchange info and filters if not already done/passed
    # (Precisions dict might contain filter info, but let's fetch fresh for robustness)
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

                        # Apply min/max quantity constraints
                        if calculated_qty_dec < min_qty:
                            logger.warning(f"calculate_trade_qty: Calculated quantity {calculated_qty_dec:.{quantity_precision}f} below minQty {min_qty}. Adjusting to minQty.")
                            calculated_qty_dec = min_qty
                        if calculated_qty_dec > max_qty:
                            logger.warning(f"calculate_trade_qty: Calculated quantity {calculated_qty_dec:.{quantity_precision}f} above maxQty {max_qty}. Adjusting to maxQty.")
                            calculated_qty_dec = max_qty

                        # Apply step size constraint (truncate to nearest step size below)
                        if step_size > Decimal('0'):
                            # Calculate how many steps fit into the calculated quantity
                            num_steps = (calculated_qty_dec / step_size).to_integral_value(rounding=decimal.ROUND_DOWN)
                            calculated_qty_dec = num_steps * step_size
                            logger.data(f"calculate_trade_qty: Adjusted quantity to stepSize {step_size}: {calculated_qty_dec:.{quantity_precision}f}")
                        else:
                             logger.warning(f"calculate_trade_qty: Invalid stepSize ({step_size}). Skipping step size adjustment.")


                    except Exception as e_lot_size:
                        logger.error(f"calculate_trade_qty: Error applying LOT_SIZE/MARKET_LOT_SIZE filter for {symbol_str}: {e_lot_size}")
                        # Continue to next filter or finish

                elif market_type == 'SPOT' and order_side == config.SIDE_BUY and f.get('filterType') == 'MIN_NOTIONAL':
                     # For SPOT MARKET BUY orders, ensure the total notional value meets MIN_NOTIONAL
                     # Notional value = quantity * price
                     try:
                         min_notional = Decimal(f.get('minNotional', '0'))
                         # Use the estimated trade price for this check
                         notional_value = calculated_qty_dec * trade_price
                         if notional_value < min_notional:
                             logger.warning(f"calculate_trade_qty: Calculated notional value {notional_value:.8f} below MIN_NOTIONAL {min_notional}. Cannot place order with this quantity.")
                             # This quantity is too small. We could try to increase quantity to meet min_notional
                             # calculated_qty_dec = (min_notional / trade_price).quantize(step_size, rounding=decimal.ROUND_UP) # Need step_size here
                             # For simplicity, just return 0.0 if it's below min notional with the calculated capital percentage.
                             logger.error("calculate_trade_qty: Cannot meet MIN_NOTIONAL with calculated quantity. Returning 0.0.")
                             return 0.0 # Return 0 if MIN_NOTIONAL cannot be met


                     except Exception as e_min_notional:
                         logger.error(f"calculate_trade_qty: Error applying MIN_NOTIONAL filter for {symbol_str}: {e_min_notional}")
                         # Continue

        else:
             logger.warning(f"calculate_trade_qty: Could not fetch filters for {symbol_str}. Filter limits will not be applied to quantity.")

    except Exception as e_filters:
        logger.error(f"calculate_trade_qty: Unexpected error fetching or applying filters for {symbol_str}: {e_filters}\n{traceback.format_exc()}")
        # Continue with calculated quantity without filter adjustment


    # --- Final Formatting and Validation ---
    # Format the Decimal quantity to string using the required precision (ROUND_DOWN)
    formatted_qty_str = format_quantity(calculated_qty_dec, quantity_precision)

    if formatted_qty_str is None:
        logger.error("calculate_trade_qty: Failed to format calculated quantity. Returning 0.0.")
        return 0.0

    final_qty_float = float(formatted_qty_str)

    # Final check: is the formatted quantity still positive and tradable?
    # Use a small threshold or the actual minQty from filters if available.
    # If filters weren't fetched, use a general small positive threshold.
    min_tradeable_qty_threshold = Decimal('1e-8') # Default small value
    # Attempt to get minQty from previously fetched filters if available
    if symbol_info and 'filters' in symbol_info:
         for f_check in symbol_info.get('filters', []):
             if f_check.get('filterType') in ['LOT_SIZE', 'MARKET_LOT_SIZE']:
                  try:
                      min_qty_str = f_check.get('minQty', "1e-8")
                      if min_qty_str and float(min_qty_str) > 0:
                          min_tradeable_qty_threshold = Decimal(min_qty_str)
                      # else: use default threshold
                  except Exception as e_minqty_check:
                       logger.warning(f"calculate_trade_qty: Could not parse minQty {f_check.get('minQty')} as Decimal: {e_minqty_check}. Using default threshold.")
                  break # Found LOT_SIZE/MARKET_LOT_SIZE filter, break

    if Decimal(str(final_qty_float)) < min_tradeable_qty_threshold:
        logger.warning(f"calculate_trade_qty: Final formatted quantity {final_qty_float:.{quantity_precision}f} is below minimum tradeable quantity threshold {float(min_tradeable_qty_threshold):.{quantity_precision}f}. Returning 0.0.")
        return 0.0 # Return 0 if quantity is too small

    logger.data(f"calculate_trade_qty: Final calculated and formatted quantity: {final_qty_float:.{quantity_precision}f}")
    return final_qty_float

# interval_to_timedelta and align_timestamp_to_interval were moved to app.utils.kline_utils
# Removing them from here to avoid duplication.
