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
import data_prepare as data_p

#


columns = ["code", "open", "close"]
codes = "600345", "600229", "002345", "002236", "002217", \
        "300345", "603799"

size = 30
open = np.arange(1,size+1)
close = open+1.5
code = np.array([codes[0]] * size)

array = np.vstack([code, open, close]).T
df = pd.DataFrame(array, columns=columns).astype(float)

date_idx = sorted(["2018-09-{:02d}".format(i) for i in range(1, 31)], reverse=True)
df.index = date_idx


i=2
n = len(date_idx)

# expected_df_mv = df.iloc[i-1:]
# expected_df_mv.index = date_idx[:n-i+1]
# expected_df_mv = expected_df_mv[columns[1:]]
# df_rolling = data_p.rolling(rolling_type="max",days=-i, df=df[columns[1:]], prefix=False)
# print(expected_df_mv==df_rolling)

k = int((i - 1) / 2)
if (i-1)%2 == 0:
    expected_df_mv = df.iloc[k:n-i+k+1]
else:
    expected_df_mv = df.iloc[k:n-i+k+1]+0.5
expected_df_mv.index = date_idx[:n - i + 1]
expected_df_mv = expected_df_mv[columns[1:]]
df_rolling = data_p.rolling(rolling_type="mean", days=-i, df=df[columns[1:]], prefix=False)
print(expected_df_mv)
print(df_rolling)
print(expected_df_mv==df_rolling)
