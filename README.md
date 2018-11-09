# Quant

## 流程：
### 获取数据
- 下载数据    :df
- 数据检查、清洗、补全 (df)
- 写入数据库

### 模型训练
- 从数据库获取数据    :df_raw
- 原始数据初步加工    (df_raw):df
- 生成衍生变量并进行拼接，生成大宽表   (df)：df_wide
- 对于大宽表，根据需要过滤含Null行(df_wide)     :df_wide_filtered
- 生成X与y   (df_wide):X,y
- 切分数据集   (X,y):X_train,y_train,X_test,y_test
- 模型定义与训练
- 模型评估，包括定义模型评估方法与画图


### 单日交易完整流程：采用每个交易日收盘后生成次日的交易计划，次日根据行情执行该交易计划的模式
- 收盘后，读取模型并利用当日及之前数据进行预测，返回信号 (df):df_signal
- 根据当日数据与模型信号，生成次日交易计划    (df_signal):plan
- 下一个交易日内，根据日内行情与交易计划，在开盘、盘中、尾盘执行交易   (df_signal, account):transaction
    - 具体执行时，先(统一)生成交易委托   (df_signal):orders
    - 然后根据交易委托，完成交易(更新账户)       (orders,df_signal,account):transactions
- 根据执行的交易，更新账户相关信息        (transactions,account):null


### 回测流程：
- 从指定起始时间点开始，逐个交易日模拟交易，并返回交易委托、成交委托、持仓情况、资产总额、基准线(如沪深300)对应资产总额 :(orders,transactions, my_model_value)
    - 非交易日：直接跳过
    - 每个交易日内：
        - 执行前一个交易日生成的交易计划（具体流程见上文），并返回交易委托、成交委托、持仓情况、资产总额、基准线(如沪深300)对应资产总额 :(orders,transactions, my_model_value, )
        - 生成下一个交易日的交易计划（具体流程见上文）  :(plan)
- 打印交易委托、成交委托、（每次交易后）持仓情况、资产总额与基准线对应资产总额，并作图。