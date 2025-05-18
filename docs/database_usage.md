好的，这里有一份关于如何使用您的 K 线数据采集程序存储在 PostgreSQL 数据库中的数据的详细文档。这份文档主要面向需要在另一个程序中访问和利用这些数据的开发者。

-----

## 文档：访问与使用 K 线数据库数据

### 1\. 引言

本文档旨在指导开发者如何连接、查询和使用由 K 线数据采集程序存储在 PostgreSQL 数据库中的 K 线（OHLCV）数据。该程序会存储两种类型的 K 线数据：

  * **基础 K 线数据 (Base Kline Data)**：直接从 API 获取的原始时间间隔的 K 线数据（例如，每1分钟或每1小时）。
  * **聚合 K 线数据 (Aggregated Kline Data)**：由基础 K 线数据按预设规则聚合而成的 K 线数据（例如，每3分钟或每3小时）。

理解数据如何存储以及如何有效地查询这些数据，对于构建依赖这些数据的其他应用程序至关重要。

### 2\. 数据库连接

  * **数据库类型**：PostgreSQL
  * **连接参数**：要连接到数据库，您需要以下参数。这些参数通常是在 K 线数据采集程序的 `.env` 文件中配置的：
      * `DB_HOST`：数据库服务器的主机名或 IP 地址（例如：`localhost`）。
      * `DB_PORT`：数据库服务器的端口号（例如：`5432`）。
      * `DB_NAME`：数据库名称（例如：`postgres` 或您指定的数据库名）。
      * `DB_USER`：连接数据库的用户名（例如：`postgres` 或您指定的用户）。
      * `DB_PASSWORD`：连接数据库的密码。

### 3\. 数据表结构

数据存储在两个不同系列的表中，分别对应基础 K 线和聚合 K 线。

#### 3.1. 表命名约定

表名是根据交易对、K 线类型（基础/聚合）和时间间隔动态生成的。

  * **基础 K 线表 (Base Kline Table)**:

      * 格式: `klines_base_<symbol>_<interval>`
      * 示例: 如果交易对是 `BTCUSDT`，基础间隔是 `1m` (1分钟)，则表名为 `klines_base_btcusdt_1m`。
      * `<symbol>`: 交易对的小写形式，例如 `btcusdt`。
      * `<interval>`: 基础时间间隔的字符串表示，移除了特殊字符，例如 `1m`, `1h`。

  * **聚合 K 线表 (Aggregated Kline Table)**:

      * 格式: `klines_agg_<symbol>_<interval>`
      * 示例: 如果交易对是 `BTCUSDT`，聚合间隔是 `3m` (3分钟)，则表名为 `klines_agg_btcusdt_3m`。
      * `<symbol>`: 交易对的小写形式。
      * `<interval>`: 聚合时间间隔的字符串表示，移除了特殊字符。

#### 3.2. 列定义

基础 K 线表和聚合 K 线表具有相同的列结构：

| 列名           | 数据类型    | 描述                                                                 |
| :------------- | :---------- | :------------------------------------------------------------------- |
| `timestamp`    | `TIMESTAMPTZ` | K 线周期的**开始时间**，以 UTC（协调世界时）存储。该列是**主键**。 |
| `open`         | `NUMERIC`   | 开盘价                                                               |
| `high`         | `NUMERIC`   | 最高价                                                               |
| `low`          | `NUMERIC`   | 最低价                                                               |
| `close`        | `NUMERIC`   | 收盘价                                                               |
| `volume`       | `NUMERIC`   | 成交量 (以基础资产计)                                                |
| `quote_volume` | `NUMERIC`   | 成交额 (以计价资产计)                                                |

**注意**：`TIMESTAMPTZ` 类型在 PostgreSQL 中表示带时区的时间戳，并且在存储时通常会转换为 UTC。当您查询时，可以根据需要将其转换为其他时区。由于我们的程序确保了所有时间戳在存入时都是 UTC 的，因此您可以认为此列中的时间戳是明确的 UTC 时间。

### 4\. 数据特性

  * **时间戳 (Timestamps)**：
      * `timestamp` 列的值代表该 K 线周期的**开始时间**。例如，一个 `1m` K 线，如果其 `timestamp` 是 `2025-05-18 10:00:00+00`，则它代表的是从 `10:00:00` 到 `10:00:59` 这个时间段的数据。
      * 所有时间戳均以 **UTC** 存储。这是非常重要的一点，在进行任何基于时间的分析或显示时，都需要考虑到这一点。
  * **数据唯一性与更新 (Data Uniqueness and Updates)**：
      * `timestamp` 列是主键，这意味着在任何一个 K 线表中，每个时间戳都唯一对应一条记录。
      * 数据采集程序使用 `ON CONFLICT (timestamp) DO UPDATE` 机制写入数据。这意味着如果收到一个已存在时间戳的新 K 线数据（例如，WebSocket 更新了当前周期的最终数据），旧记录会被新数据覆盖，从而保证了数据的准确性和最新性。
  * **数据类型 `NUMERIC`**：
      * 价格和交易量使用 `NUMERIC` 类型存储，可以提供精确的数值，避免了浮点数可能带来的精度问题。在应用程序中读取这些值时，它们通常会被映射为 `Decimal` 类型（例如在 Python 中）或高精度浮点数。如果使用 `pandas` 读取，通常会自动转换为 `float64`。

### 5\. 数据访问示例

您可以使用任何支持 PostgreSQL 的编程语言或工具来访问这些数据。以下是一些常见的 SQL 查询示例和 Python 访问示例。

#### 5.1. SQL 查询示例

假设我们想查询交易对为 `BTCUSDT`，聚合间隔为 `3m` 的数据，其对应的表名为 `klines_agg_btcusdt_3m`。

  * **获取最新的 10 条聚合 K 线数据**：

    ```sql
    SELECT *
    FROM klines_agg_btcusdt_3m
    ORDER BY timestamp DESC
    LIMIT 10;
    ```

  * **获取特定时间范围内的聚合 K 线数据** (例如, 2025年5月17日全天 UTC 时间)：

    ```sql
    SELECT *
    FROM klines_agg_btcusdt_3m
    WHERE timestamp >= '2025-05-17 00:00:00+00'  -- 开始时间 (UTC)
      AND timestamp < '2025-05-18 00:00:00+00'   -- 结束时间 (UTC)，不包含
    ORDER BY timestamp ASC;
    ```

  * **获取特定聚合 K 线的收盘价和成交量**：

    ```sql
    SELECT timestamp, close, volume
    FROM klines_agg_btcusdt_3m
    WHERE timestamp = '2025-05-17 10:00:00+00'; -- 特定K线的开始时间 (UTC)
    ```

  * **将时间戳转换为特定时区** (例如，北京时间 UTC+8):

    ```sql
    SELECT
        timestamp AS timestamp_utc,
        timestamp AT TIME ZONE 'Asia/Shanghai' AS timestamp_beijing,
        open,
        high,
        low,
        close,
        volume
    FROM klines_agg_btcusdt_3m
    ORDER BY timestamp DESC
    LIMIT 5;
    ```

    *(注意: `AT TIME ZONE` 的行为取决于数据库服务器和客户端的配置，以及时间戳的原始存储方式。由于我们存的是 `TIMESTAMPTZ` 且为 UTC，这种转换是可靠的。)*

#### 5.2. Python 访问示例 (使用 `psycopg2` 和 `pandas`)

以下 Python 代码段演示了如何连接到数据库，并从指定的 K 线表中读取数据到 `pandas` DataFrame。

```python
import psycopg2
import pandas as pd
from decimal import Decimal # 用于处理 NUMERIC 类型

# --- 数据库连接参数 (根据您的配置修改) ---
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "postgres", # 您的数据库名
    "user": "postgres",   # 您的用户名
    "password": "your_password" # 您的密码
}

# --- K线表参数 ---
SYMBOL = "btcusdt" # 示例交易对
INTERVAL = "3m"    # 示例聚合间隔
TABLE_NAME = f"klines_agg_{SYMBOL}_{INTERVAL}" # 动态构建表名

def fetch_klines_to_dataframe(table_name: str, limit: int = 100, start_time_utc: str = None, end_time_utc: str = None):
    """
    从指定的K线表中获取数据并存入Pandas DataFrame。
    时间戳在DataFrame中将是UTC时区的datetime对象。
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = f"SELECT timestamp, open, high, low, close, volume, quote_volume FROM {table_name}"
        
        conditions = []
        params = []

        if start_time_utc:
            conditions.append("timestamp >= %s")
            params.append(start_time_utc)
        
        if end_time_utc:
            conditions.append("timestamp < %s") # 使用 < 以确保不包含end_time_utc本身对应的K线（如果它是一个周期的开始）
            params.append(end_time_utc)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY timestamp DESC" # 获取最新的数据在前
        
        if limit and not (start_time_utc or end_time_utc): # 仅当没有时间范围时应用limit获取最新数据
             query += f" LIMIT {limit}"


        print(f"Executing query: {cursor.mogrify(query, tuple(params) if params else None).decode('utf-8')}")
        cursor.execute(query, tuple(params) if params else None)
        
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        
        df = pd.DataFrame(rows, columns=colnames)

        if not df.empty:
            # Pandas to_datetime 会自动处理 TIMESTAMPTZ 为带有时区信息的 datetime64[ns, UTC]
            # df['timestamp'] = pd.to_datetime(df['timestamp']) # psycopg2 已经返回了带时区的datetime对象

            # 将 NUMERIC (Decimal) 转换为 float 进行分析，如果需要
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                if col in df.columns and df[col].apply(lambda x: isinstance(x, Decimal)).any():
                    df[col] = df[col].astype(float)
            
            # 如果获取时用了 DESC，这里反转回 ASC，方便时序分析
            df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

        return df

    except psycopg2.Error as e:
        print(f"数据库错误: {e}")
        return pd.DataFrame() # 返回空DataFrame
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # 示例：获取最新的10条数据
    print(f"--- 获取最新的10条 '{TABLE_NAME}' K线 ---")
    latest_klines_df = fetch_klines_to_dataframe(TABLE_NAME, limit=10)
    if not latest_klines_df.empty:
        print(latest_klines_df.head())
        print("\nDataFrame Info:")
        latest_klines_df.info()
        # 验证时间戳时区
        if 'timestamp' in latest_klines_df.columns and not latest_klines_df.empty:
            print(f"\n第一个时间戳: {latest_klines_df['timestamp'].iloc[0]}, 时区: {latest_klines_df['timestamp'].iloc[0].tzinfo}")

    print("-" * 50)

    # 示例：获取特定时间范围的数据
    print(f"--- 获取 '{TABLE_NAME}' K线从 2025-05-17 00:00:00 UTC 到 2025-05-17 01:00:00 UTC ---")
    # 注意：时间字符串应符合PostgreSQL对TIMESTAMPTZ的格式要求
    klines_in_range_df = fetch_klines_to_dataframe(
        TABLE_NAME,
        start_time_utc="2025-05-17 00:00:00+00",
        end_time_utc="2025-05-17 01:00:00+00"
    )
    if not klines_in_range_df.empty:
        print(klines_in_range_df)

    # 示例：处理其他表，如基础K线表
    BASE_TABLE_NAME = f"klines_base_{SYMBOL}_1m" # 假设基础间隔是1m
    print(f"\n--- 获取最新的5条 '{BASE_TABLE_NAME}' K线 ---")
    latest_base_klines_df = fetch_klines_to_dataframe(BASE_TABLE_NAME, limit=5)
    if not latest_base_klines_df.empty:
        print(latest_base_klines_df.head())
```

### 6\. 重要注意事项

  * **时区处理 (Timezone Handling)**：
      * **始终假设 `timestamp` 列是 UTC 的**。当您的应用程序需要以特定本地时间显示或处理数据时，请确保在从数据库读取数据后，将 UTC 时间戳正确转换为目标时区。
      * 在 Python 中，`pandas` 的 `Timestamp` 对象和 `datetime` 对象的 `astimezone()` 方法可用于时区转换。例如：`df['timestamp_local'] = df['timestamp'].dt.tz_convert('Asia/Shanghai')`。
  * **数据粒度选择 (Choosing Data Granularity)**：
      * 如果您需要高频率、详细的数据（例如，用于回测精细策略），请使用**基础 K 线表** (`klines_base_...`)。
      * 如果您需要概览数据、进行趋势分析或对性能要求较高且能接受较低粒度，请使用**聚合 K 线表** (`klines_agg_...`)。聚合表的数据量通常远小于基础表，查询速度更快。
  * **配置依赖 (Configuration Dependency)**：
      * 消费数据的应用程序需要知道 K 线数据采集程序使用的确切数据库连接参数、交易对 (`SYMBOL`) 以及基础间隔 (`BASE_INTERVAL`) 和聚合间隔 (`AGG_INTERVAL`)。这些信息用于正确构建表名。
  * **数据库负载 (Database Load)**：
      * 频繁或大规模的数据查询（例如，一次性拉取数百万行数据）可能会对数据库服务器造成显著负载。
      * **优化查询**：仅选择您需要的列，使用 `WHERE` 子句过滤数据，尤其是在 `timestamp` 列上。
      * **分页查询**：如果需要大量历史数据，考虑分批次查询。
      * **应用程序级缓存**：如果某些数据被频繁请求，可以在您的应用程序中实现缓存机制。
      * `timestamp` 列作为主键，已经具有高效的索引。
  * **数据完整性检查**：
      * 虽然不太可能发生，但如果需要，您可以检查 `timestamp` 列的连续性，以识别数据采集中可能存在的间隙（仅针对已关闭的K线）。例如，对于 `1m` K 线，连续记录的时间戳应该相差1分钟。

### 7\. 总结

通过本文档，您应该能够理解 K 线数据在 PostgreSQL 中的存储方式，并能有效地通过 SQL 查询或编程语言（如 Python）来访问这些数据。正确处理时间戳和选择合适的数据粒度是成功使用这些数据的关键。

-----