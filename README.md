# Binance K线数据工具

一个用于获取、存储、聚合和显示币安（Binance）K线（OHLCV）数据的Python应用程序。它支持从币安期货API获取历史和实时K线数据，并能将数据存储在PostgreSQL数据库或内存中，同时还能对K线数据进行聚合。

## 主要功能

- **历史数据获取**：从币安期货API批量获取指定交易对和时间间隔的历史K线数据。
- **实时数据流**：通过WebSocket连接接收实时的K线更新。
- **数据聚合**：将基础时间间隔的K线（例如1分钟）聚合成更大时间间隔的K线（例如3分钟）。
- **灵活存储**：
  - **PostgreSQL数据库**：将基础K线和聚合后的K线数据持久化存储到PostgreSQL数据库中。
  - **内存存储**：提供轻量级的内存存储选项，不依赖外部数据库。
- **动态配置**：通过 `.env` 文件轻松配置交易对、时间间隔、运行模式、API限制、日志级别和数据存储选项。
- **结构化日志**：使用 `structlog` 进行详细且易于阅读的日志记录，包括控制台输出和文件记录 (`logs/binance_kline_app.log`)。
- **K线显示**：在控制台实时显示最新的聚合K线数据。
- **运行模式**：
  - `TEST` 模式：使用较短的时间间隔（例如，基础1分钟 -> 聚合3分钟），便于快速测试和观察。
  - `PRODUCTION` 模式：使用较长的时间间隔（例如，基础1小时 -> 聚合3小时），适用于实际监控。

## 已完成改进

### 1. 日志文件组织

- 日志文件现在存储在 `logs/` 目录下，避免污染项目根目录。
- 在 `.gitignore` 中忽略 `logs/` 目录。

### 2. 网络请求重试机制

- 在获取历史K线数据时，添加了指数退避重试机制（默认重试3次）。
- 详细记录重试日志，提高网络异常情况下的健壮性。

### 3. 时间桶对齐修复

- 修复了聚合K线时间桶对齐问题，确保聚合桶从整点开始（例如3小时聚合从00:00, 03:00...开始），避免使用数据中第一个时间戳作为对齐基准。

### 4. 类型转换改进

- 解决了数据库返回的 `Decimal` 类型在日志中显示为 `Decimal('123.45')` 的问题，现在自动转换为浮点数。

## 目录结构

```text
├── .env                   # 实际运行配置文件 (需从.env.example复制创建)
├── .env.example           # 示例配置文件
├── ImprovedMinhPhan.pine  # 示例TradingView Pine脚本策略 (供参考)
├── app                    # 主应用程序代码目录
│   ├── __init__.py
│   ├── api_clients        # API客户端模块 (Binance)
│   │   ├── __init__.py
│   │   └── binance_api_manager.py
│   ├── core               # 核心配置和设置
│   │   ├── __init__.py
│   │   └── config.py
│   ├── data_handling      # 数据处理、存储和聚合模块
│   │   ├── __init__.py
│   │   ├── data_interfaces.py
│   │   ├── db_manager.py          # 数据库存储实现
│   │   ├── kline_aggregator.py    # K线聚合逻辑
│   │   └── kline_data_store.py    # 内存存储实现
│   ├── main_app.py        # 应用程序主逻辑和入口点
│   ├── tests              # 测试代码
│   │   └── test_kline_data_store.py
│   ├── ui                 # 用户界面模块 (目前为控制台显示)
│   │   ├── __init__.py
│   │   └── display_manager.py
│   └── utils              # 通用工具函数
│       ├── __init__.py
│       └── kline_utils.py
├── docs                   # 文档目录
│   └── database_usage.md  # 数据库使用详细文档
├── logs/                  # 日志文件目录（自动创建）
├── requirements.txt       # Python依赖包列表
└── run.py                 # 应用程序启动脚本
```

## 环境要求

- Python 3.8 或更高版本
- PostgreSQL 数据库服务器 (如果您选择 `DATA_STORE_TYPE="database"`)
- Git (用于克隆仓库)

## 安装与设置

1.  **克隆仓库**：

    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **创建并激活Python虚拟环境** (推荐)：

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安装依赖**：

    ```bash
    pip install -r requirements.txt
    ```

4.  **设置PostgreSQL数据库** (如果使用数据库存储)：

    - 确保您的PostgreSQL服务器正在运行。
    - 创建一个新的数据库（例如 `binance_data`）和用户（可选，您也可以使用现有用户如 `postgres`）。
    - 授予用户对该数据库的权限。
    - K 线数据表将由应用程序在首次运行时自动创建。

5.  **配置应用程序**：

    - 复制示例配置文件 `.env.example` 并重命名为 `.env`：
      ```bash
      cp .env.example .env
      ```
    - 编辑 `.env` 文件，根据您的需求修改配置参数。详见下一节“配置说明”。

## 配置说明 (`.env` 文件)

打开您的 `.env` 文件并根据需要修改以下变量：

- `API_BASE_URL_FUTURES`: 币安期货API的基础URL (通常不需要修改)。

  - 默认: `"https://fapi.binance.com"`

- `SYMBOL`: 您想要监控的交易对。

  - 示例: `"BTCUSDT"`, `"ETHUSDT"`

- `OPERATING_MODE`: 运行模式，决定了基础K线和聚合K线的时间间隔。

  - `"TEST"`: 基础间隔为 "1m" (1分钟)，聚合间隔为 "3m" (3分钟)。适合测试和快速观察。
  - `"PRODUCTION"`: 基础间隔为 "1h" (1小时)，聚合间隔为 "3h" (3小时)。适合长期稳定运行。

- `HISTORICAL_AGG_CANDLES_TO_DISPLAY`: 初始加载时，获取并显示多少条聚合后的K线。

  - 默认: `50`

- `MAX_KLINE_LIMIT_PER_REQUEST`: 单次请求币安API时获取K线的最大数量。

  - 默认: `1000` (币安的通常上限)

- `LOG_LEVEL`: 应用程序的日志级别。

  - 可选值: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`
  - 默认: `"INFO"`

- `DATA_STORE_TYPE`: K线数据的存储方式。

  - `"database"`: 将数据存储到PostgreSQL数据库中。
  - `"memory"`: 将数据存储在内存中 (程序关闭后数据会丢失)。
  - 默认 (在 `.env.example` 中): `"memory"`

- **PostgreSQL数据库配置** (仅当 `DATA_STORE_TYPE="database"` 时需要)：

  - `DB_HOST`: 数据库服务器主机名。 (默认: `"localhost"`)
  - `DB_PORT`: 数据库服务器端口。 (默认: `"5432"`)
  - `DB_NAME`: 数据库名称。 (默认: `"postgres"` 或您在 `.env.example` 中看到的 `binance_data`)
  - `DB_USER`: 数据库用户名。 (默认: `"postgres"`)
  - `DB_PASSWORD`: 数据库密码。 (请务必修改为您自己的安全密码)

**重要**: 首次运行时，如果选择数据库存储，应用程序会自动在指定的数据库中创建所需的表 (`klines_base_<symbol>_<interval>` 和 `klines_agg_<symbol>_<interval>`)。

## 运行应用程序

完成上述配置后，在项目根目录下运行：

```bash
python run.py
```

应用程序将开始获取数据、处理数据并通过控制台输出日志和最新的聚合K线信息。日志同时会写入到 `logs/binance_kline_app.log` 文件中。

## 输出说明

- **控制台**：
  - 程序启动和运行过程中的日志信息。
  - 实时更新的最新几条聚合K线数据 (开盘价, 最高价, 最低价, 收盘价, 时间戳, 状态)。
- **日志文件**：
  - `logs/binance_kline_app.log` 文件包含了更详细的运行日志，便于问题排查。
- **PostgreSQL数据库** (如果配置使用)：
  - 基础K线数据会存储在名为 `klines_base_<symbol>_<base_interval>` 的表中。
  - 聚合后的K线数据会存储在名为 `klines_agg_<symbol>_<agg_interval>` 的表中。

## 数据库使用

本应用程序可以将K线数据持久化到PostgreSQL数据库中，这使得其他应用程序或分析脚本可以方便地访问这些数据。

- **表结构**: 包含 `timestamp` (UTC, 主键), `open`, `high`, `low`, `close`, `volume`, `quote_volume`。
- **数据类型**: 基础K线和聚合K线分别存储在不同的表中，表名根据交易对和时间间隔动态生成。

关于如何连接数据库、查询特定表、理解数据特性以及如何在您自己的程序中使用这些数据的详细说明，请参阅：

**[详细数据库使用文档](./docs/database_usage.md)**

## 测试

项目中包含针对数据存储逻辑的单元测试。

- **测试文件**: `app/tests/test_kline_data_store.py`
- **运行测试**: 在项目根目录下，确保您的虚拟环境已激活，然后运行：
  ```bash
  python -m unittest discover app/tests
  ```
  或者更具体地：
  ```bash
  python -m unittest app.tests.test_kline_data_store

  ```
