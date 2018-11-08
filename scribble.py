import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# x = -np.arange(100)
# y = np.random.rand(100)
# line = plt.plot(x,y)
# plt.xticks([0,40,60,80])
# plt.legend([line],["line"])
# plt.show()


import tushare as ts
df = ts.get_k_data(code="sh",start="2000-01-01")
print(df.iloc[0],df.iloc[-1])