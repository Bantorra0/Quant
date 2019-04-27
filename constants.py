TABLE, COLUMNS = "table", "columns"

DATE_FORMAT = "%Y-%m-%d"
TOKEN = 'ca7a0727b75dce94ad988adf953673340308f01bacf1a101d23f15fc'

STOCK_DAY = {
    TABLE: "stock_day",
    COLUMNS:
        ("code", "date", "open", "high", "low", "close", "vol", "amt", "adj_factor")
}

INDEX_DAY = {
    TABLE: "index_day",
    COLUMNS:
        ("code", "date", "open", "high", "low", "close", "vol")
}

STOCK_BASIC = {
    TABLE: "stock_basic",
    COLUMNS:
        ('code', 'symbol', 'name', 'area', 'industry', 'fullname', 'enname',
         'market', 'exchange', 'curr_type', 'list_status', 'list_date',
         'delist_date', 'is_hs')
}

STOCK_DAILY_BASIC = {
    TABLE: "stock_daily_basic",
    COLUMNS:
        ('code', 'date', 'close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb',
         'ps', 'ps_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv')
}

TIMEOUT=30

FEE_RATE = 2.5 / 10000
STAMP_TAX_RATE = (0,1/1000)

# BASE_DIR = r"C:\Users\dell-pc\DM\DM"

MODEL_DIR = "models"

FLOAT_DELTA = 1+1e-7

BUY_FLAG = 1
SELL_FLAG = -1


# Configs for trading pattern.
BUY_IN_PCT = 0.05
HOLDING_PCT_LIMIT = 0.5