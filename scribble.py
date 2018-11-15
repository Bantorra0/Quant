import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# x = -np.arange(100)
# y = np.random.rand(100)
# line = plt.plot(x,y)
# plt.xticks([0,40,60,80])
# plt.legend([line],["line"])
# plt.show()


# import tushare as ts
# df = ts.get_k_data(code="sh",start="2000-01-01")
# print(df.iloc[0],df.iloc[-1])

import db_operations as dbop
from constants import TOKEN, STOCK_DAY, INDEX_DAY,TABLE,COLUMNS
import pandas as pd
import data_cleaning as dc
import collect

db_type = "sqlite3"
conn = dbop.connect_db(db_type)
cursor = conn.cursor()

codes = "600345", "600229", "002345", "002236", "002217", \
        "300345", "603799"
columns=["code","open","close"]
open = np.random.uniform(low=10, high=15, size=30)
close = np.random.uniform(low=10, high=15, size=30)
code = np.array([codes[0]] * 30)

array = np.vstack([code, open, close]).T
df = pd.DataFrame(array, columns=columns)

date_idx = sorted(["2018-09-{:02d}".format(i) for i in range(1,31)],reverse=True)
df.index=date_idx
print(df)


