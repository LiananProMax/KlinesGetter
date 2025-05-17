#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
import traceback
import requests
import pandas as pd # 用于处理K线数据格式
from decimal import Decimal # 用于高精度计算

# 使用app的日志和配置
import logging
logger = logging.getLogger(__name__) # 获取logger
from app.core import config # 导入配置，包括重试参数和Binance枚举值

# 导入Binance客户端（由main_app初始化并传递）
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

# 导入通用工具函数
from app.utils.trading_utils import (
    format_price, format_quantity, split_symbol_for_coinapi, generate_client_order_id, adjust_price_to_tick_size
)

# --- API 调用安全包装 ---
def safe_api_call(client: Client, func, *args, **kwargs):
    """
    Wrapper for Binance API calls with retry logic and error handling.
    Takes the client instance as the first argument.
    Uses retry logic from config.
    """
    max_retries = config.MAX_API_RETRIES
    retry_delay = config.API_RETRY_DELAY

    for attempt in range(max_retries):
        try:
            # 有些方法不接受 requests_params 参数，比如 futures_exchange_info
            # 根据函数名进行特殊处理
            if func.__name__ in ['futures_exchange_info', 'get_exchange_info']:
                # 这些方法不接受 requests_params，直接调用
                return func(*args, **kwargs)
            else:
                # 为其他方法添加超时参数
                if 'requests_params' not in kwargs:
                    kwargs['requests_params'] = {'timeout': 30}
                elif 'timeout' not in kwargs['requests_params']:
                    kwargs['requests_params']['timeout'] = 30
                
                return func(*args, **kwargs)
        except BinanceAPIException as e:
            delay = retry_delay * (2 ** attempt) # Basic exponential backoff
            if e.code in [-1003, -1021, -1001, -2015, -2019, -4003]: # Retryable errors (rate limit, timestamp, invalid signature, leverage not set, etc.)
                logger.warning(f"Binance API Error (Code {e.code}): {e.message}. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            elif e.code == -2011 and "Unknown order sent" in e.message:
                logger.warning(f"Binance: Attempted to cancel an order that might already be filled or cancelled (Code {e.code}): {e.message}")
                # Return a specific indicator for this common scenario where a cancel fails because the order is gone
                return {"status": "UNKNOWN_ORDER_OR_ALREADY_CANCELLED", "code": -2011, "message": e.message}
            else: # Non-retryable or unknown API error
                logger.error(f"Binance API Error (Code {e.code}): {e.message} for {func.__name__}")
                return None # Do not retry non-transient errors
        except (BinanceRequestException, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e: # Network errors
            delay = retry_delay * (2 ** attempt) # Basic exponential backoff
            logger.warning(f"Binance Network/Request Error: {e}. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except Exception as e: # Other unexpected errors
            logger.error(f"Unexpected error during Binance API call '{func.__name__}': {e}")
            logger.error(traceback.format_exc())
            return None # Do not retry unexpected errors

    logger.error(f"Binance API call '{func.__name__}' failed after {max_retries} attempts.")
    return None

# --- Symbol Info and Precision Helpers (Adapted to use Client) ---
def _get_precision_from_string(value_str):
    """Helper to count decimal places for precision."""
    if '.' in value_str:
        value_str = value_str.rstrip('0')
        if '.' in value_str:
             if value_str.endswith('.'): return 0
             return len(value_str.split('.')[1])
        else: return 0
    return 0

def get_symbol_info_from_exchange(client: Client, symbol_str: str, market_type: str):
    """Gets symbol info for futures or spot."""
    try:
        if market_type == 'FUTURES':
            # Use client.futures_exchange_info directly for Futures
            info = safe_api_call(client, client.futures_exchange_info)
        elif market_type == 'SPOT':
            # Use client.get_exchange_info for Spot
            info = safe_api_call(client, client.get_exchange_info)
        else:
            logger.error(f"get_symbol_info_from_exchange: Unknown MARKET_TYPE '{market_type}'")
            return None

        if info and 'symbols' in info:
            for item in info['symbols']:
                if item['symbol'] == symbol_str:
                    return item
        logger.error(f"Binance symbol info not found for {symbol_str} in {market_type} market.")
    except Exception as e:
        logger.error(f"Exception fetching Binance exchange info for {market_type}: {e}")
    return None

def get_precisions_and_tick_size(client: Client, symbol_str: str, market_type: str):
    """Fetches and returns precisions and tick size."""
    symbol_info = get_symbol_info_from_exchange(client, symbol_str, market_type)
    if not symbol_info:
        return None, None, None

    tick_size_dec = None
    price_prec = 0
    qty_prec = 0

    try:
        if market_type == 'FUTURES':
            price_prec = int(symbol_info.get('pricePrecision', 0))
            qty_prec = int(symbol_info.get('quantityPrecision', 0))
        elif market_type == 'SPOT':
            # For SPOT, extract from filters
            for f in symbol_info.get('filters', []):
                 if f.get('filterType') == 'PRICE_FILTER':
                      price_prec = _get_precision_from_string(f.get('tickSize', '1'))
                 elif f.get('filterType') in ['LOT_SIZE', 'MARKET_LOT_SIZE']:
                      qty_prec = _get_precision_from_string(f.get('stepSize', '1'))


        # Find tickSize from PRICE_FILTER (common for both FUTURES and SPOT)
        for f in symbol_info.get('filters', []):
            if f.get('filterType') == 'PRICE_FILTER':
                tick_size_dec = Decimal(f.get('tickSize', '0'))
                break # Found PRICE_FILTER

    except Exception as e:
        logger.error(f"Error parsing symbol info filters/precisions for {symbol_str} ({market_type}): {e}")
        return None, None, None

    if tick_size_dec is None or tick_size_dec <= Decimal(0):
         logger.error(f"Failed to determine valid tick size ({tick_size_dec}) for {symbol_str} ({market_type}).")
         return None, None, None

    return tick_size_dec, price_prec, qty_prec


# --- Balance Helpers (Adapted to use Client) ---
def _get_spot_asset_balance(client: Client, asset_str: str):
    """Get available balance for a spot asset."""
    try:
        # Use client.get_asset_balance for Spot
        balance_data = safe_api_call(client, client.get_asset_balance, asset=asset_str)
        if balance_data and 'free' in balance_data:
            return float(balance_data['free'])
        # logger.warning(f"Spot balance for {asset_str} not found or no 'free' amount.") # Keep less noisy
    except Exception as e:
        logger.error(f"Error getting spot asset balance for {asset_str}: {e}")
    return None # Use None to indicate API failure

def _get_futures_usdt_balance(client: Client):
    """Get available USDT balance in futures account."""
    try:
        # Use client.futures_account_balance for Futures
        balances_data = safe_api_call(client, client.futures_account_balance)
        if balances_data and isinstance(balances_data, list):
            for balance in balances_data:
                # Check for USDT asset
                if isinstance(balance, dict) and balance.get('asset') == 'USDT':
                    try: return float(balance.get('availableBalance', 0)) # Use availableBalance for trading
                    except (ValueError, TypeError): return 0
        # logger.warning("USDT asset not found in Binance Futures account balance.") # Keep less noisy
    except Exception as e:
        logger.error(f"Error getting futures USDT balance: {e}")
    return 0 # Return 0 on failure or not found

# --- Position Status (Adapted to use Client) ---
def get_current_market_status(client: Client, symbol_str: str, market_type: str, latest_kline_df: pd.DataFrame = None):
    """Get current market status (position, etc.)."""
    if client is None:
        logger.error("get_current_market_status: Binance client not available.")
        return None, 0.0, None, 'CLIENT_ERROR'

    if market_type == 'FUTURES':
        return _get_current_futures_status(client, symbol_str)
    elif market_type == 'SPOT':
        return _get_current_spot_status(client, symbol_str, latest_kline_df)
    else:
        logger.error(f"get_current_market_status: Unknown MARKET_TYPE '{market_type}'")
        return None, 0.0, None, 'CLIENT_ERROR'

def _get_current_futures_status(client: Client, symbol_str: str):
    """Get current futures status."""
    try:
        positions = safe_api_call(client, client.futures_position_information, symbol=symbol_str)
        if positions is None:
            logger.error(f"Futures Status ({symbol_str}): API call returned None for position_information.")
            return None, 0.0, None, 'API_ERROR'
        if not isinstance(positions, list):
             logger.error(f"Futures Status ({symbol_str}): Expected list from position_information, got {type(positions)}. Data: {positions}")
             return None, 0.0, None, 'PARSE_ERROR'
        if not positions:
             logger.info(f"Futures Status ({symbol_str}): Empty list from position_information. Assuming no position.")
             return None, 0.0, None, 'NO_POSITION'

        # In one-way mode, there's one element. In hedge mode, there could be two (LONG/SHORT side).
        # This logic needs to find the *relevant* position for the symbol.
        for position in positions:
            if isinstance(position, dict) and position.get('symbol') == symbol_str:
                # Check position amount for this symbol item
                position_amt_str = position.get('positionAmt', '0')
                entry_price_str = position.get('entryPrice', '0')
                try:
                    position_amt = float(position_amt_str)
                    entry_price = float(entry_price_str)
                except ValueError:
                    logger.error(f"Futures Status ({symbol_str}): Could not parse positionAmt '{position_amt_str}' or entryPrice '{entry_price_str}' to float.")
                    return None, 0.0, None, 'PARSE_ERROR'

                # Use a small threshold to check if position amount is effectively zero
                if abs(Decimal(str(position_amt))) > Decimal('0'): # Check if there's an actual position (non-zero amount)
                    # Determine side based on signed positionAmt
                    side = config.SIDE_LONG if position_amt > 0 else config.SIDE_SHORT
                    quantity = abs(position_amt)

                    # Binance Futures API entryPrice is generally reliable for the average entry price of the current position
                    # Only return a valid entry price (> 0)
                    valid_entry_price = entry_price if entry_price > 0 else None

                    logger.info(f"Futures Status ({symbol_str}): Found position - Side: {side}, Qty: {quantity}, EntryPrice: {valid_entry_price}")
                    return side, quantity, valid_entry_price, 'OK'
                else:
                     logger.info(f"Futures Status ({symbol_str}): Symbol found in positions but amount is zero ({position_amt}).")
                     # Continue checking other entries in case of Hedge Mode, but for one-way, this is the end.
                     # Assuming one-way mode for simplicity matching original bot. If hedge mode needed, loop needs to be more complex.

        # If loop finishes without returning, it means no non-zero position was found for the symbol
        logger.info(f"Futures Status ({symbol_str}): No non-zero position found.")
        return None, 0.0, None, 'NO_POSITION'

    except Exception as e:
        logger.error(f"Futures Status ({symbol_str}): Error fetching or parsing position information: {e}", exc_info=True)
        return None, 0.0, None, 'API_ERROR'

def _get_current_spot_status(client: Client, symbol_str: str, latest_kline_df: pd.DataFrame = None):
    """Get current spot status."""
    # Note: SPOT only supports LONG position (holding base asset) in this strategy structure
    # Get base and quote assets for the symbol
    try:
        base_asset, quote_asset = split_symbol_for_coinapi(symbol_str).split('_') # Assuming split works and provides 2 parts
    except ValueError:
        logger.error(f"Spot Status ({symbol_str}): Could not split symbol {symbol_str} into base and quote assets.")
        return None, 0.0, 0.0, None, 'PARSE_ERROR'

    # Get balances of base and quote assets
    base_asset_balance_float = _get_spot_asset_balance(client, base_asset)
    # quote_asset_balance_float = _get_spot_asset_balance(client, quote_asset) # Not directly used for position detection in this logic

    if base_asset_balance_float is None: # If base balance fetch failed, cannot determine position
        logger.error(f"Spot Status ({symbol_str}): Failed to fetch base asset balance for {base_asset}.")
        # Cannot determine status without balance
        return None, 0.0, 0.0, None, 'API_ERROR'

    # Determine minimum meaningful base quantity (needs symbol info from API)
    min_meaningful_base_qty_threshold = Decimal('1e-8') # Default small value
    symbol_info_spot_check = get_symbol_info_from_exchange(client, symbol_str, 'SPOT')
    if symbol_info_spot_check:
        for f_check in symbol_info_spot_check.get('filters', []):
            if f_check.get('filterType') in ['LOT_SIZE', 'MARKET_LOT_SIZE']:
                try:
                    min_qty_str = f_check.get('minQty', "1e-8")
                    # Ensure minQty interpretation is robust and use it if positive
                    if min_qty_str and float(min_qty_str) > 0: # Check if minQty_str is valid and positive
                        min_meaningful_base_qty_threshold = Decimal(min_qty_str)
                    # else: use default 1e-8 or higher fallback if needed
                except Exception as e_minqty_check:
                     logger.warning(f"Spot status: Could not parse minQty {f_check.get('minQty')} as Decimal: {e_minqty_check}. Using default threshold.")
                break # Found LOT_SIZE/MARKET_LOT_SIZE filter, break

    # Check if base asset balance is significant enough to be considered a 'position'
    # Use Decimal for robust comparison
    if Decimal(str(base_asset_balance_float)) >= min_meaningful_base_qty_threshold:
         # Holding sufficient base asset means we are in a LONG position
         synced_position_side = config.SIDE_LONG
         synced_position_qty = float(base_asset_balance_float) # The quantity is the base asset balance

         # Spot API does not provide a reliable average entry price per asset hold in standard endpoints.
         # Strategy must track this internally after a BUY fill, or attempt to derive from trades.
         # On API sync, the entry price is UNKNOWN from the API itself.
         synced_entry_price = None # Indicate unknown from API sync

         logger.info(f"Spot Status ({symbol_str}): Found position - Side: {synced_position_side}, Qty: {synced_position_qty}, EntryPrice: Unknown (from API)")
         return (
             synced_position_side, # 'LONG' or None
             synced_position_qty, # Base asset balance if in position, 0 otherwise
             # quote_asset_balance_float, # Not returned by this function
             synced_entry_price, # None
             'OK' # Status is OK if balances fetched successfully
         )
    else:
         # Base asset balance is below the minimum threshold, considered no position
         logger.info(f"Spot Status ({symbol_str}): Base asset balance {base_asset_balance_float} below threshold {float(min_meaningful_base_qty_threshold)}. No active position.")
         return None, 0.0, 0.0, None, 'NO_POSITION'


# --- Order Placement Functions (Adapted to use Client) ---
def place_trade_order(
    client: Client,
    symbol_str: str,
    side: str, # config.SIDE_BUY or config.SIDE_SELL
    qty: float,
    market_type: str, # config.MARKET_TYPE
    price_precision: int,
    quantity_precision: int,
    order_type: str | None = None, # config.ORDER_TYPE_MARKET etc.
    price: float | None = None,
    stop_price: float | None = None,
    reduce_only: bool = False, # Futures specific
    client_order_id: str | None = None, # Optional, will generate if None
):
    """Place an order on Binance (Futures or Spot)."""
    if client is None:
        logger.error("Binance client is None in place_trade_order.")
        return None

    # Generate clientOrderId if not provided
    if client_order_id is None:
        # Determine a suffix based on order type and reduceOnly
        order_purpose_suffix = order_type if order_type else "TRADE"
        if reduce_only:
            order_purpose_suffix += "_RO" # e.g., MARKET_RO, STOP_MARKET_RO
        # Add side to suffix for clarity
        order_purpose_suffix = f"{side}_{order_purpose_suffix}" # e.g., BUY_MARKET, SELL_STOP_MARKET_RO

        client_order_id = generate_client_order_id(prefix="bot", suffix=order_purpose_suffix)

    # Format quantity according to precision rules
    formatted_qty_str = format_quantity(qty, quantity_precision)
    if formatted_qty_str is None or float(formatted_qty_str) <= 0:
        logger.error(f"{market_type} Place Order: Invalid quantity {qty} formatted as '{formatted_qty_str}' for {symbol_str}. Cannot place order.")
        return None

    logger.trading(f"Attempting to place {market_type} {side} {order_type or 'default type'} order for {formatted_qty_str} {symbol_str} with clientOrderId: {client_order_id}")

    params = {
        'symbol': symbol_str,
        'side': side.upper(), # Ensure uppercase
        'type': order_type,
        'newClientOrderId': client_order_id
    }

    try:
        order_response = None

        if market_type == 'FUTURES':
            params['quantity'] = formatted_qty_str
            if reduce_only:
                 params['reduceOnly'] = True # Binance API expects boolean True/False for this parameter
            # Add price parameters for limit or stop orders
            if order_type in [config.FUTURE_ORDER_TYPE_LIMIT, config.FUTURE_ORDER_TYPE_STOP, config.FUTURE_ORDER_TYPE_TAKE_PROFIT,
                              config.FUTURE_ORDER_TYPE_STOP_MARKET, config.FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET]:
                if price is not None: # For LIMIT orders or Limit Price for Stop/Take Profit Limit
                    params['price'] = format_price(price, price_precision)
                if stop_price is not None: # For STOP, STOP_MARKET, TAKE_PROFIT, TAKE_PROFIT_MARKET
                    params['stopPrice'] = format_price(stop_price, price_precision)

            # Place the order using the futures client
            order_response = safe_api_call(client, client.futures_create_order, **params)


        elif market_type == 'SPOT':
             params['quantity'] = formatted_qty_str
             # Spot orders might need different parameters based on type (e.g., TimeInForce for LIMIT)
             if order_type == config.ORDER_TYPE_MARKET:
                 # Spot Market orders use quantity for both BUY (base asset amount) and SELL (base asset amount)
                 # or quoteOrderQty for BUY (quote asset amount to spend)
                 # This implementation uses 'quantity' (base asset amount)
                 if params['side'] == config.SIDE_BUY:
                     order_response = safe_api_call(client, client.order_market_buy, **params)
                 elif params['side'] == config.SIDE_SELL:
                     order_response = safe_api_call(client, client.order_market_sell, **params)
                 else:
                      logger.error(f"SPOT Place Order: Invalid side '{side}' for MARKET order on {symbol_str}."); return None

             elif order_type == config.ORDER_TYPE_LIMIT and price is not None:
                 params['price'] = format_price(price, price_precision)
                 params['timeInForce'] = config.TIME_IN_FORCE_GTC # Common for LIMIT orders
                 if params['side'] == config.SIDE_BUY:
                     order_response = safe_api_call(client, client.order_limit_buy, **params)
                 elif params['side'] == config.SIDE_SELL:
                     order_response = safe_api_call(client, client.order_limit_sell, **params)
                 else:
                      logger.error(f"SPOT Place Order: Invalid side '{side}' for LIMIT order on {symbol_str}."); return None

             else:
                 logger.error(f"SPOT Place Order: Unsupported/invalid order type '{order_type}' or missing required parameters (e.g., price for LIMIT) for {symbol_str}."); return None

        else:
            logger.error(f"place_trade_order: Unknown MARKET_TYPE '{market_type}'")
            return None

        # Check the response from safe_api_call
        if order_response and order_response.get('orderId'):
            # Log the full response for successful API interaction
            logger.info(f"{market_type} {side} {order_type or 'default'} order API Response for {symbol_str} (Qty: {formatted_qty_str}): {order_response}")
            # Check if order was accepted by Binance (has an orderId and status is not REJECTED)
            if order_response.get('status', '').upper() != 'REJECTED':
                logger.order(f"{market_type} {side} {order_type or 'default'} order PLACED for {symbol_str}. Order ID: {order_response['orderId']}")
                return order_response # Return the successful response dictionary
            else:
                # API call succeeded, but the order itself was rejected
                logger.error(f"{market_type} {side} {order_type or 'default'} order FAILED placement on Binance side for {symbol_str}. Response indicates rejection: {order_response}")
                return None # Indicate order was not successfully placed on Binance

        else:
            # safe_api_call returned None, meaning API call itself failed after retries
            logger.error(f"{market_type} {side} {order_type or 'default'} order API call FAILED for {symbol_str} after retries. safe_api_call returned None. See previous logs.")
            return None # Indicate API call failure

    except Exception as e:
        logger.error(f"{market_type} Place Order: Exception placing {side} {order_type} order for {qty} {symbol_str}: {e}", exc_info=True)
        return None


def cancel_trade_order(client: Client, symbol_str: str, order_id: int, market_type: str):
    """Cancel an order on Binance."""
    if client is None:
        logger.error("Binance client is None in cancel_trade_order.")
        return False
    logger.trading(f"Attempting to cancel {market_type} order {order_id} for {symbol_str}...")
    try:
        if market_type == 'FUTURES':
            # Futures cancel order by orderId or clientOrderId
            cancel_response = safe_api_call(client, client.futures_cancel_order, symbol=symbol_str, orderId=order_id)
        elif market_type == 'SPOT':
            # Spot cancel order by orderId or clientOrderId
            cancel_response = safe_api_call(client, client.cancel_order, symbol=symbol_str, orderId=order_id)
        else:
            logger.error(f"cancel_trade_order: Unknown MARKET_TYPE '{market_type}'")
            return False

        # Check if the response indicates success or that the order was already cancelled/filled
        if cancel_response is not None: # safe_api_call returning None means API call failed entirely
            # Check for the specific status returned by safe_api_call wrapper for 'Unknown order sent'
            if isinstance(cancel_response, dict) and cancel_response.get("status") == "UNKNOWN_ORDER_OR_ALREADY_CANCELLED":
                logger.trading(f"{market_type} order {order_id} was already cancelled/unknown (according to safe_api_call wrapper).")
                return True # Consider it cancelled for bot state purposes
            
            # Standard successful cancel response varies by market type
            if market_type == 'FUTURES':
                # Futures cancel returns a dict of the cancelled order details
                if isinstance(cancel_response, dict) and cancel_response.get('orderId') == order_id and cancel_response.get('status') in ['CANCELED', 'PENDING_CANCEL']:
                    logger.trading(f"FUTURES order {order_id} cancellation confirmed. Status: {cancel_response.get('status')}")
                    return True
                # Check for Binance specific error code for already cancelled/filled (-2011)
                if isinstance(cancel_response, dict) and cancel_response.get('code') == -2011:
                    logger.trading(f"FUTURES order {order_id} already cancelled/filled (Binance code -2011).")
                    return True
            
            elif market_type == 'SPOT':
                # Spot cancel can return a list of cancelled orders (usually one)
                if isinstance(cancel_response, list) and cancel_response and cancel_response[0].get('orderId') == order_id and cancel_response[0].get('status') in ['CANCELED', 'PENDING_CANCEL']:
                    logger.trading(f"SPOT order {order_id} cancellation confirmed. Status: {cancel_response[0].get('status')}")
                    return True
                # Check for Binance specific error code for already cancelled/filled (-2011) if the response is a dict error
                if isinstance(cancel_response, dict) and cancel_response.get('code') == -2011:
                    logger.trading(f"SPOT order {order_id} already cancelled/filled (Binance code -2011).")
                    return True
            
            # If we reached here, safe_api_call succeeded but response doesn't confirm cancellation or is ambiguous
            logger.warning(f"{market_type} order {order_id} cancellation attempted, but confirmation unclear. Response: {cancel_response}")
            # In this ambiguous case, assume failure to be safe. Sync will clarify on next run.
            return False
        else:
            # safe_api_call failed after retries
            logger.error(f"{market_type} order {order_id} cancellation API call FAILED.")
            return False
    except Exception as e:
        logger.error(f"Exception during {market_type} order cancellation {order_id}: {e}", exc_info=True)
        return False


def close_current_position(
    client: Client,
    symbol_str: str,
    market_type: str, # config.MARKET_TYPE
    current_side: str | None, # config.SIDE_LONG or config.SIDE_SHORT
    current_qty: float,
    price_precision: int,
    quantity_precision: int,
    latest_kline_df: pd.DataFrame | None = None # For SPOT market order price logging
):
    """Close a position on Binance (Futures or Spot) using a market order."""
    if client is None:
        logger.error("Binance client is None in close_current_position.")
        return 'FAILED'

    if market_type == 'FUTURES':
        if current_side is None or current_qty <= 0:
            logger.info(f"FUTURES Close Position: No position indicated for {symbol_str}.")
            return 'NO_POSITION'
        # Determine the side of the order needed to close the position
        close_order_side = config.SIDE_SELL if current_side == config.SIDE_LONG else config.SIDE_BUY

        logger.trading(f"Attempting to close FUTURES {current_side} position of {current_qty} {symbol_str} with a {close_order_side} MARKET order (reduceOnly).")
        # Place a MARKET order with reduceOnly=True
        close_order_response = place_trade_order(
            client, symbol_str, close_order_side, current_qty, 'FUTURES', price_precision, quantity_precision,
            order_type=config.FUTURE_ORDER_TYPE_MARKET, reduce_only=True
        )

        if close_order_response and close_order_response.get('orderId'):
            logger.order(f"FUTURES position close order PLACED for {symbol_str}. Order ID: {close_order_response['orderId']}")
            # For market close orders, they typically execute immediately or very quickly
            # 'PLACED' here means the API call was successful and the order was accepted by Binance
            # The caller is responsible for verifying position closure via verify_position_closed
            return 'PLACED'
        else:
            logger.error(f"Failed to place FUTURES position close order for {symbol_str}. Response: {close_order_response}")
            return 'FAILED'

    elif market_type == 'SPOT':
        # For SPOT, closing a position means selling the base asset holding (closing LONG)
        # This is done by placing a MARKET SELL order for the available quantity of the base asset.
        try:
            base_asset, _ = split_symbol_for_coinapi(symbol_str).split('_') # Assuming split works
        except ValueError:
             logger.error(f"Spot Close Position ({symbol_str}): Could not split symbol {symbol_str} into base/quote for balance check.")
             return 'FAILED'

        # Get the currently available balance of the base asset
        base_bal_to_sell_float = _get_spot_asset_balance(client, base_asset)

        if base_bal_to_sell_float is None:
             logger.error(f"Spot Close Position ({symbol_str}): Failed to get base asset balance for {base_asset}. Cannot close.")
             return 'FAILED'

        # Determine minimum meaningful base quantity (needs symbol info)
        min_tradeable_qty_base_threshold = Decimal('1e-8') # Default small value
        symbol_info = get_symbol_info_from_exchange(client, symbol_str, 'SPOT')
        if symbol_info and 'filters' in symbol_info:
            for f_filter in symbol_info.get('filters', []):
                if f_filter.get('filterType') in ['LOT_SIZE', 'MARKET_LOT_SIZE']:
                    try:
                        min_qty_str = f_filter.get('minQty', '0.00000001')
                        if min_qty_str and float(min_qty_str) > 0:
                             min_tradeable_qty_base_threshold = Decimal(min_qty_str)
                        # else: use default threshold
                    except Exception as e_min_qty:
                        logger.warning(f"Spot close: Could not parse minQty from filter {f_filter}: {e_min_qty}. Using default threshold.")
                    break # Found relevant filter

        # Check if the available balance is below the minimum tradeable quantity threshold
        if Decimal(str(base_bal_to_sell_float)) < min_tradeable_qty_base_threshold:
            # If balance is too small, there's effectively no position to close
            logger.trading(f"Spot Close Position ({symbol_str}): Base asset balance {base_bal_to_sell_float:.8f} below min tradeable threshold {float(min_tradeable_qty_base_threshold):.8f}. Assuming no significant position to close.")
            return 'NO_POSITION'

        # Format the quantity to sell using the symbol's quantity precision
        formatted_qty_to_sell_str = format_quantity(base_bal_to_sell_float, quantity_precision)
        if formatted_qty_to_sell_str is None or float(formatted_qty_to_sell_str) <= 0:
            logger.error(f"Spot Close Position ({symbol_str}): Calculated quantity to sell ({base_bal_to_sell_float}) resulted in invalid formatted quantity '{formatted_qty_to_sell_str}'. Cannot place sell order.")
            return 'FAILED'

        # Final check after formatting: is the formatted quantity still tradable?
        final_qty_to_sell_float = float(formatted_qty_to_sell_str)
        if Decimal(str(final_qty_to_sell_float)) < min_tradeable_qty_base_threshold:
            logger.warning(f"Spot Close Position ({symbol_str}): Final formatted quantity {final_qty_to_sell_float:.{quantity_precision}f} is below min tradeable quantity {float(min_tradeable_qty_base_threshold):.{quantity_precision}f}. Not attempting sell.")
            return 'NO_POSITION' # Treat as no position if quantity too small

        logger.trading(f"Attempting MARKET close of {final_qty_to_sell_float} {base_asset} for {symbol_str} on SPOT.")
        # Place a MARKET SELL order
        order = place_trade_order(
            client, symbol_str, config.SIDE_SELL, final_qty_to_sell_float, 'SPOT', price_precision, quantity_precision,
            order_type=config.ORDER_TYPE_MARKET, latest_kline_df=latest_kline_df # Pass kline for logging price estimate
        )

        if order and order.get('orderId'):
            logger.trading(f"Spot position close order placed. Order ID: {order['orderId']}. Status: {order.get('status')}")
            # For market orders, 'FILLED' might be immediate or require a very short poll.
            # The caller is responsible for verifying position closure via verify_position_closed
            return 'PLACED'
        else:
            logger.error(f"Failed to place Spot position close MARKET SELL order for {symbol_str}. API Response: {order}")
            return 'FAILED'
    else:
        logger.error(f"close_current_position: Unknown MARKET_TYPE '{market_type}'")
        return 'FAILED'


# --- Order Utility Functions (Adapted to use Client) ---
def is_bot_order(client_order_id: str | None) -> bool:
    """Checks if a clientOrderId matches the pattern generated by this bot."""
    if not client_order_id: return False
    # The pattern is prefix_timestamp_unique_id[_suffix]
    # Check if it starts with the expected prefix
    # Using "bot_" as the default prefix from generate_client_order_id
    expected_prefix = "bot_"
    return client_order_id.startswith(expected_prefix)

def get_open_oco_lists_binance(client: Client, symbol_str: str):
    """Fetches open OCO lists for a symbol (Spot only)."""
    if client is None:
        logger.error("get_open_oco_lists_binance: Binance client not initialized")
        return [] # Return empty list on client error

    try:
        # Get all open OCO orders for the user
        oco_lists = safe_api_call(client, client.get_open_oco_orders)
        if oco_lists is None:
            logger.error("get_open_oco_lists_binance: Failed to get open OCO lists (safe_api_call returned None)")
            return [] # Return empty list on API error

        # Filter by symbol
        # Note: Binance API get_open_oco_orders does not have a symbol parameter, it returns ALL open OCOs.
        # We must filter client-side.
        symbol_ocos = [oco for oco in oco_lists if oco.get('symbol') == symbol_str]
        logger.info(f"OCO Lists ({symbol_str}): Found {len(symbol_ocos)} open OCO list(s) matching symbol.")
        return symbol_ocos
    except Exception as e:
        logger.error(f"get_open_oco_lists_binance: Error fetching open OCO lists for {symbol_str}: {e}", exc_info=True)
        return [] # Return empty list on exception


def get_open_orders_for_symbol_binance(client: Client, symbol_str: str, market_type: str):
    """
    Get all open orders (regular and OCO legs) for a symbol from Binance.
    This function dispatches to the appropriate market-specific function.
    """
    if client is None:
        logger.error("get_open_orders_for_symbol_binance: Binance client not initialized")
        return [] # Return empty list on client error

    if market_type == 'FUTURES':
        return _get_open_futures_orders(client, symbol_str)
    elif market_type == 'SPOT':
        return _get_open_spot_orders(client, symbol_str)
    else:
        logger.error(f"get_open_orders_for_symbol_binance: Unknown MARKET_TYPE '{market_type}'")
        return []

def _get_open_futures_orders(client: Client, symbol_str: str):
    """
    Get all open futures orders for a symbol.
    """
    try:
        # safe_api_call will handle basic API errors and retries
        open_orders = safe_api_call(client, client.futures_get_open_orders, symbol=symbol_str)
        if open_orders is None:
            # safe_api_call will return None after multiple failed attempts
            logger.error(f"Futures Open Orders ({symbol_str}): Failed to get open orders after retries (safe_api_call returned None).")
            return [] # Return empty list on failure so callers can iterate

        # API call succeeded, even if no orders exist, it will return an empty list []
        logger.info(f"Futures Open Orders ({symbol_str}): Found {len(open_orders)} open order(s).")
        return open_orders
    except Exception as e:
        logger.error(f"Futures Open Orders ({symbol_str}): Exception fetching open orders: {e}", exc_info=True)
        return [] # Return empty list so callers can iterate


def _get_open_spot_orders(client: Client, symbol_str: str):
    """
    Get all open spot orders for a symbol (regular and OCO legs).
    """
    all_spot_open_orders = []
    try:
        # Get regular open orders
        regular_orders = safe_api_call(client, client.get_open_orders, symbol=symbol_str)
        if regular_orders is not None:
            all_spot_open_orders.extend(regular_orders)
        else:
            logger.warning(f"Spot Open Orders ({symbol_str}): Failed to fetch regular open orders (safe_api_call returned None).")

        # Get open OCO lists and add their legs
        oco_lists = get_open_oco_lists_binance(client, symbol_str) # Use the helper
        if oco_lists is not None:
            for oco_list in oco_lists:
                if oco_list and 'orders' in oco_list and oco_list['orders']:
                    # Add OCO leg orders to the results, marking them as part of an OCO
                    for order_leg in oco_list['orders']:
                         # Add context about the parent OCO list
                         order_leg['isOcoLeg'] = True
                         order_leg['orderListId'] = oco_list.get('orderListId')
                         order_leg['listClientOrderId'] = oco_list.get('listClientOrderId')
                         all_spot_open_orders.append(order_leg)
        else:
            logger.warning(f"Spot Open Orders ({symbol_str}): Failed to fetch OCO lists (returned None).")

        logger.info(f"Spot Open Orders ({symbol_str}): Found {len(all_spot_open_orders)} total open order(s) (regular + OCO legs).")
        return all_spot_open_orders
    except Exception as e:
        logger.error(f"Spot Open Orders ({symbol_str}): Exception fetching open orders: {e}", exc_info=True)
        return [] # Return empty list


def cancel_all_open_orders_for_symbol(client: Client, symbol_str: str, market_type: str):
    """
    Cancel all open orders for a symbol that were placed by this bot instance.
    Identified by clientOrderId pattern matching.
    This function dispatches to the appropriate market-specific function.
    """
    if client is None:
        logger.error("cancel_all_open_orders_for_symbol: Binance client not initialized")
        return False

    logger.trading(f"Cancelling all bot's open orders for {symbol_str} ({market_type})...")
    try:
        all_cancelled = True
        open_orders_list = get_open_orders_for_symbol_binance(client, symbol_str, market_type) # Get all open orders

        if open_orders_list is None:
             logger.error(f"Failed to get open orders to cancel for {symbol_str} ({market_type}).")
             return False # Cannot proceed if cannot get open orders

        # Iterate through all open orders and cancel those placed by the bot
        for order in open_orders_list:
            order_id = order.get('orderId')
            client_order_id = order.get('clientOrderId')
            order_list_id = order.get('orderListId') # For SPOT OCO legs

            if is_bot_order(client_order_id): # Check if it's a bot order based on clientOrderId pattern
                logger.trading(f"Cancel: Found bot order {order_id} (ClientOrderID: {client_order_id}) for {symbol_str}. Attempting cancellation...")
                
                # Check if it's an OCO leg. If so, we need to cancel the parent OCO list.
                if order.get('isOcoLeg') and order_list_id:
                    # Attempt to cancel the OCO list if we haven't already in this loop
                    # Need to ensure we only try to cancel each OCO list once.
                    # A set could track attempted OCO list cancellations.
                    # For simplicity now, just call cancel_spot_oco_order for this leg's list_id.
                    # The cancel_spot_oco_order function handles its own logic/retries.
                    # It's safe to call multiple times for the same list_id; Binance API handles it.
                    logger.trading(f"Cancel: Order {order_id} is part of OCO list {order_list_id}. Cancelling OCO list...")
                    if not cancel_spot_oco_order(client, symbol_str, order_list_id=order_list_id):
                         all_cancelled = False
                         logger.warning(f"Cancel: Failed to cancel bot OCO list {order_list_id} for {symbol_str}.")
                    # Note: After cancelling the OCO list, the individual legs will also be cancelled by Binance.
                    # We don't need to explicitly cancel the leg order_id.

                else: # Regular order (non-OCO leg) placed by the bot
                    if not cancel_trade_order(client, symbol_str, order_id, market_type):
                        all_cancelled = False
                        logger.warning(f"Cancel: Failed to cancel bot order {order_id} (ClientOrderID: {client_order_id}) for {symbol_str}")
            # Else: Ignore orders not placed by this bot instance

        if all_cancelled:
            logger.info(f"Successfully processed cancellation attempts for bot orders for {symbol_str} ({market_type}).")
        else:
            logger.warning(f"Some bot orders failed cancellation for {symbol_str} ({market_type}).")

        return all_cancelled

    except Exception as e:
        logger.error(f"Error cancelling all bot orders for {symbol_str} ({market_type}): {e}", exc_info=True)
        return False


def verify_position_closed(
    client: Client,
    symbol_str: str,
    market_type: str, # config.MARKET_TYPE
    price_precision: int, # Required for logging/threshold checks if needed
    quantity_precision: int, # Required for logging/threshold checks if needed
    expected_side_that_was_closed: str | None = None, # Context, not used for logic
    latest_kline_df: pd.DataFrame | None = None # Context, for SPOT balance check logging maybe
):
    """
    Verify that a position has been closed by checking the current market status on Binance.
    This polls the API for a limited number of attempts. Blocking call.
    """
    if client is None:
        logger.error("verify_position_closed: Binance client is None.")
        return False

    logger.info(f"Verifying position closure for {symbol_str} ({market_type})...")

    position_close_verify_delay = config.POSITION_CLOSE_VERIFY_DELAY
    position_close_verify_attempts = config.POSITION_CLOSE_VERIFY_ATTEMPTS

    for attempt in range(position_close_verify_attempts):
        # Use time.sleep for blocking delay
        time.sleep(position_close_verify_delay)
        logger.info(f"Verification attempt {attempt + 1}/{position_close_verify_attempts} for {symbol_str}...")

        # Get current position status using the helper function
        # Pass latest_kline_df for potential use in SPOT balance check if needed by _get_current_spot_status
        position_data = get_current_market_status(client, symbol_str, market_type, latest_kline_df)
        status_code = position_data[-1] if position_data and isinstance(position_data, tuple) and len(position_data) > 0 else 'ERROR'

        if status_code == 'CLIENT_ERROR' or status_code == 'API_ERROR' or status_code == 'PARSE_ERROR':
            logger.error(f"Failed to get market status during position close verification for {symbol_str} (Status: {status_code}). Cannot reliably verify closure.")
            # Continue retrying in case it was a transient API issue
            continue # Go to next attempt after sleep

        if market_type == 'FUTURES':
            # position_data for FUTURES: (side, quantity, entry_price, status_code)
            synced_side, synced_qty, _, _ = position_data

            if status_code == 'NO_POSITION' or (synced_qty is not None and synced_qty <= 0): # If API reports no position or zero quantity
                logger.success(f"FUTURES position for {symbol_str} successfully verified closed (API reports NO_POSITION or Qty<=0).")
                return True # Position confirmed closed

            elif status_code == 'OK': # API reports a non-zero position
                 logger.warning(f"FUTURES position for {symbol_str} NOT closed. Current Qty: {synced_qty}. Retrying verification...")
            # Note: Other status_codes handled by the initial check (API_ERROR etc)


        elif market_type == 'SPOT':
            # position_data for SPOT: (dominant_asset, base_bal, quote_bal, avg_entry_price_base, status_code)
            # Need to get base asset balance to confirm closure
            try:
                base_asset, _ = split_symbol_for_coinapi(symbol_str).split('_') # Assuming split works
            except ValueError:
                 logger.error(f"Spot Verify Close ({symbol_str}): Could not split symbol {symbol_str}. Cannot verify balance.")
                 # Cannot verify, return failure
                 return False

            base_bal_float = _get_spot_asset_balance(client, base_asset)

            if base_bal_float is None: # Failed to get balance
                 logger.error(f"Spot Verify Close ({symbol_str}): Failed to fetch base asset balance for {base_asset}. Cannot verify closure.")
                 # Continue retrying in case of transient API issue
                 continue # Go to next attempt after sleep

            # Determine minimum meaningful base quantity threshold
            min_tradeable_qty_base_threshold = Decimal('1e-8') # Default
            symbol_info = get_symbol_info_from_exchange(client, symbol_str, 'SPOT')
            if symbol_info and 'filters' in symbol_info:
                for f_filter in symbol_info.get('filters', []):
                    if f_filter.get('filterType') in ['LOT_SIZE', 'MARKET_LOT_SIZE']:
                        try:
                             min_qty_str = f_filter.get('minQty', '0.00000001')
                             if min_qty_str and float(min_qty_str) > 0:
                                  min_tradeable_qty_base_threshold = Decimal(min_qty_str)
                             # else: use default threshold
                        except Exception as e_min_qty:
                            logger.warning(f"Spot verify close: Could not parse minQty from filter {f_filter}: {e_min_qty}. Using default threshold.")
                        break # Found filter

            # Position is considered closed if base asset balance is below the threshold
            if Decimal(str(base_bal_float)) < min_tradeable_qty_base_threshold:
                logger.success(f"SPOT position for {symbol_str} successfully verified closed (base balance {base_bal_float:.{quantity_precision}f} below threshold {float(min_tradeable_qty_base_threshold):.{quantity_precision}f}).")
                return True # Position confirmed closed
            else:
                 logger.warning(f"SPOT position for {symbol_str} NOT fully closed. Current base asset balance: {base_bal_float:.{quantity_precision}f}. Retrying verification...")

        # If we reach here, position is not yet confirmed closed. Loop continues after sleep.

    # If the loop finishes without returning True, it means max attempts were reached without confirming closure.
    logger.error(f"Failed to verify position closure for {symbol_str} after {position_close_verify_attempts} attempts.")
    return False # Indicate failure to confirm closure


def cancel_spot_oco_order(client: Client, symbol_str: str, order_list_id: int | None = None, list_client_order_id: str | None = None):
    """
    Cancel a Spot OCO order list.
    Can cancel by orderListId or by listClientOrderId.
    Uses safe_api_call.
    """
    if client is None:
        logger.error(f"SPOT OCO Cancel: Binance client not available for {symbol_str}.")
        return False
    try:
        params = {'symbol': symbol_str}
        id_log = ""
        if order_list_id: # Prefer orderListId if available
            params['orderListId'] = order_list_id
            id_log = f"orderListId {order_list_id}"
        elif list_client_order_id:
            params['listClientOrderId'] = list_client_order_id
            id_log = f"listClientOrderId {list_client_order_id}"
        else:
            logger.error("SPOT OCO Cancel: Must provide either orderListId or listClientOrderId.")
            return False

        logger.trading(f"Attempting to cancel SPOT OCO for {symbol_str} using {id_log}.")
        # Use safe_api_call for cancellation
        cancel_response = safe_api_call(client, client.cancel_oco_order, **params)

        if cancel_response is not None: # safe_api_call succeeded (didn't return None after retries)
             # Check for the specific status returned by safe_api_call wrapper for 'Unknown order sent'
             if isinstance(cancel_response, dict) and cancel_response.get("status") == "UNKNOWN_ORDER_OR_ALREADY_CANCELLED":
                  logger.trading(f"SPOT OCO order {id_log} was already cancelled/unknown (according to safe_api_call wrapper).")
                  return True # Consider it cancelled for bot state purposes

             # Check Binance response structure for successful OCO cancel
             # Success response typically includes orderListId and listStatusType ('CANCELED' or 'ALL_DONE')
             if isinstance(cancel_response, dict) and cancel_response.get('orderListId') and cancel_response.get('listStatusType') in ['CANCELED', 'ALL_DONE']:
                 logger.order(f"SPOT OCO order cancellation confirmed for {id_log}. List ID: {cancel_response['orderListId']}")
                 return True
             # Check for Binance specific error code for already cancelled/filled (-2011)
             if isinstance(cancel_response, dict) and cancel_response.get('code') == -2011:
                  logger.trading(f"SPOT OCO order {id_log} already cancelled/filled (Binance code -2011).")
                  return True

             # If we reached here, API call was ok but response doesn't confirm cancellation clearly
             logger.warning(f"SPOT OCO order cancellation attempted for {id_log}, but confirmation unclear. Response: {cancel_response}")
             # Assume failure in this ambiguous case. State sync will clarify.
             return False
        else:
            # safe_api_call failed after retries
            logger.error(f"SPOT OCO order cancellation API call FAILED for {id_log}.")
            return False
    except Exception as e:
        logger.error(f"SPOT OCO Cancel: Exception cancelling OCO for {symbol_str} ({id_log}): {e}", exc_info=True)
        return False

def place_spot_oco_sell_order(
    client: Client,
    symbol_str: str,
    quantity: str,          # Base asset quantity to sell (as formatted string)
    price: str,             # Take profit limit price (as formatted string)
    stop_price: str,        # Stop loss trigger price (as formatted string)
    stop_limit_price: str,  # Stop loss limit price (as formatted string)
    price_precision: int,   # Required for price validation logging
    quantity_precision: int,# Required for quantity validation logging
    list_client_order_id: str | None = None, # Optional: custom ID for the OCO list
    force_tp_above_market: bool = True,  # Auto-adjust TP price if below market
    skip_price_validation: bool = False  # Skip market price validation (for testing)
):
    """
    Place a Spot OCO (One-Cancels-the-Other) SELL order.
    This typically involves a LIMIT_MAKER order (for take profit) and a STOP_LOSS_LIMIT order.
    Uses safe_api_call.
    Prices and quantity are expected to be formatted strings according to precision *before* calling this.
    Internal validation checks logical price relationships and against current market price (unless skipped).
    """
    if client is None:
        logger.error(f"SPOT OCO: Binance client not available for placing OCO order on {symbol_str}.")
        return None

    try:
        # Convert string inputs to Decimal for validation calculations
        try:
            qty_dec = Decimal(quantity)
            tp_price_dec = Decimal(price)
            sl_trigger_dec = Decimal(stop_price)
            sl_limit_dec = Decimal(stop_limit_price)
        except Exception as e_dec_conv:
            logger.error(f"SPOT OCO: Error converting string prices/qty to Decimal for validation: {e_dec_conv}. Qty:{quantity}, TP:{price}, SLtrg:{stop_price}, SLlim:{stop_limit_price}. Cannot place order.")
            return None

        if not skip_price_validation:
            # --- Price Validation against current market ---
            ticker_info = safe_api_call(client, client.get_symbol_ticker, symbol=symbol_str)
            if not ticker_info or 'price' not in ticker_info:
                logger.error(f"SPOT OCO ({symbol_str}): Could not fetch current market price for OCO validation. Cannot place order.")
                return None
            current_market_price = Decimal(ticker_info['price'])
            # Get tick size for auto-adjustment if needed
            symbol_info = get_symbol_info_from_exchange(client, symbol_str, 'SPOT')
            tick_size_dec = Decimal('0.01') # Default if not found
            if symbol_info and 'filters' in symbol_info:
                 for f_filter in symbol_info.get('filters', []):
                      if f_filter.get('filterType') == 'PRICE_FILTER':
                           try:
                                tick_size_dec = Decimal(f_filter.get('tickSize', '0.01'))
                           except Exception as e_ts_parse:
                                logger.warning(f"SPOT OCO: Could not parse tickSize from filter {f_filter}: {e_ts_parse}. Using default {tick_size_dec}.")
                           break # Found PRICE_FILTER


            if tp_price_dec <= current_market_price:
                if force_tp_above_market:
                    # Auto-adjust TP to be slightly above current market price (e.g., 5 ticks)
                    new_tp_price_dec = current_market_price + (Decimal('5') * tick_size_dec)
                    # Ensure it's formatted back to string using precision
                    price = format_price(float(new_tp_price_dec), price_precision) # Overwrite the input 'price' string
                    tp_price_dec = Decimal(price) # Update Decimal value for subsequent checks
                    logger.warning(f"SPOT OCO SELL ({symbol_str}): Auto-adjusted TP price to {price} (original was <= market price {current_market_price})")
                else:
                    logger.error(f"SPOT OCO SELL ({symbol_str}): Take profit price ({tp_price_dec}) must be above current market price ({current_market_price}). Cannot place order.")
                    return None

            # For a SELL OCO (closing LONG), SL trigger must be below market price
            if sl_trigger_dec >= current_market_price:
                logger.error(f"SPOT OCO SELL ({symbol_str}): Stop loss trigger price ({sl_trigger_dec}) must be below current market price ({current_market_price}). Cannot place order.")
                # Auto-adjusting SL trigger if above market is possible but risky (might trigger immediately)
                # For now, fail if it's above market.
                return None

            # For a SELL OCO, SL limit price must be less than or equal to SL trigger price
            if sl_limit_dec > sl_trigger_dec: # Note: Binance allows equal, but less is safer during volatility
                logger.error(f"SPOT OCO SELL ({symbol_str}): Stop loss limit price ({sl_limit_dec}) must be less than or equal to stop loss trigger price ({sl_trigger_dec}). Cannot place order.")
                # Auto-adjusting SL limit if above trigger is possible
                # Example: sl_limit_dec = sl_trigger_dec - (some_ticks * tick_size_dec)
                # Ensure SL limit is not zero or negative.
                return None

        else:
            # skip_price_validation is True
            logger.warning(f"SPOT OCO SELL ({symbol_str}): Skipping market price validation.")
        # --- END Price Validation ---

        # --- Logical Price Relationship Validation (after any adjustment) ---
        # For OCO SELL (closing LONG): TP_limit > SL_trigger > SL_limit (usually)
        # Also, all prices must be positive.
        if not (tp_price_dec > Decimal(0) and sl_trigger_dec > Decimal(0) and sl_limit_dec > Decimal(0) and qty_dec > Decimal(0)):
            logger.error(f"SPOT OCO SELL ({symbol_str}): Calculated prices/qty are zero or negative after adjustments/validation. TP:{tp_price_dec}, SLtrg:{sl_trigger_dec}, SLlim:{sl_limit_dec}, Qty:{qty_dec}. Cannot place order.")
            return None
        if not (tp_price_dec > sl_trigger_dec): # TP must be > SL trigger for SELL OCO
             logger.error(f"SPOT OCO SELL ({symbol_str}): Logical price error: TP price ({tp_price_dec}) not > SL trigger ({sl_trigger_dec}). Cannot place order.")
             return None
        if not (sl_trigger_dec >= sl_limit_dec): # SL trigger must be >= SL limit
             logger.error(f"SPOT OCO SELL ({symbol_str}): Logical price error: SL trigger ({sl_trigger_dec}) not >= SL limit ({sl_limit_dec}). Cannot place order.")
             return None
        # Optional: Check if SL trigger is significantly below entry, TP significantly above entry etc.


        # Generate listClientOrderId if not provided
        if list_client_order_id is None:
            # Example format: bot_OCO_SYMBOL_TIMESTAMP_UUIDshort
            list_client_order_id = generate_client_order_id("bot_OCO", symbol_str)

        # Build the parameters dictionary for the OCO order
        params = {
            'symbol': symbol_str,
            'side': config.SIDE_SELL,
            'quantity': quantity,              # Base asset quantity to sell (formatted string)
            'price': price,                    # Price for the LIMIT_MAKER (take profit) leg (formatted string)
            'stopPrice': stop_price,           # Trigger price for the STOP_LOSS_LIMIT leg (formatted string)
            'stopLimitPrice': stop_limit_price,# Limit price for the STOP_LOSS_LIMIT leg (formatted string)
            'stopLimitTimeInForce': config.TIME_IN_FORCE_GTC, # Typically GTC for OCO SL limit
            'listClientOrderId': list_client_order_id, # Unique ID for the OCO list
            # Optional: limitClientOrderId, stopClientOrderId for individual legs if needed for tracking
            # 'limitClientOrderId': generate_client_order_id("bot_TP", symbol_str),
            # 'stopClientOrderId': generate_client_order_id("bot_SL", symbol_str),
            # Additional parameters required by Binance OCO API
            'selfTradePreventionMode': 'NONE', # Required parameter
            'aboveType': 'ABOVE', # Required for OCO orders where TP is above entry (SELL OCO)
            # 'belowType': 'BELOW', # Required for OCO orders where SL is below entry (BUY OCO - not used here)
            # 'newOrderRespType': 'FULL' # Get full details in response, if needed
        }
        logger.trading(f"Attempting to place SPOT OCO SELL for {quantity} {symbol_str} with TP Limit={price}, SL Trig={stop_price}, SL Limit={stop_limit_price}. ListClientOID: {list_client_order_id}")
        logger.info(f"SPOT OCO Create Order Params: {params}")

        # Place the OCO order using safe_api_call
        oco_response = safe_api_call(client, client.create_oco_order, **params)

        if oco_response and oco_response.get('orderListId'):
            logger.order(f"SPOT OCO SELL order PLACED for {symbol_str}. List ID: {oco_response['orderListId']}, ClientListID: {list_client_order_id}")
            # Log details of the placed OCO legs if available in response
            for order_report in oco_response.get('orderReports', []):
                logger.info(f"OCO Leg: ID {order_report.get('orderId')}, ClientOID {order_report.get('clientOrderId')}, Type {order_report.get('type')}, Status {order_report.get('status')}")
            return oco_response
        else:
            logger.error(f"SPOT OCO SELL order FAILED for {symbol_str}. Response: {oco_response}")
            return None

    except Exception as e:
        logger.error(f"SPOT OCO: Exception placing OCO SELL order for {symbol_str}: {e}", exc_info=True)
        return None
