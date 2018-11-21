import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

figs, ax = plt.subplots()
x = np.arange(100)
y = np.random.rand(100)
line = ax.plot(x,y,label="line")
ax.legend()
plt.xticks([0,40,60,80])
plt.show()


# import tushare as ts
# df = ts.get_k_data(code="sh",start="2000-01-01")
# print(df.iloc[0],df.iloc[-1])

