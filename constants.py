TABLE, COLUMNS = "table", "columns"

DATE_FORMAT = "%Y-%m-%d"
TOKEN = 'ca7a0727b75dce94ad988adf953673340308f01bacf1a101d23f15fc'
STOCK_DAY = {TABLE: "stock_day", COLUMNS: (
"code", "date", "open", "high", "low", "close", "vol", "amt", "adj_factor")}
INDEX_DAY = {TABLE: "index_day",
             COLUMNS: ("code", "date", "open", "high", "low", "close", "vol")}

FEE_RATE = 2.5 / 10000
STAMP_TAX_RATE = (0,1/1000)

# BASE_DIR = r"C:\Users\dell-pc\Quant\Quant"

MODEL_DIR = "models"

FLOAT_DELTA = 1+1e-7

BUY_FLAG = 1
SELL_FLAG = -1