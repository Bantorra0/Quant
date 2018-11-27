#
#
#
# class Trader:
#     @classmethod
#     def strategy_for_stck_in_pos(cls, code, account:Account,day_signal):
#         signal = day_signal[day_signal["code"] == code]
#
#         if account.records[code][1]==-1:
#             raise ValueError("Highest price of {} is -1".format(code))
#         init_buy_price = account.records[code][0]
#         init_buy_cnt = account.records[code][2]
#         qfq_close = signal["qfq_close"].iloc[0]
#         # f1mv_qfq_open = signal["f1mv_qfq_open"].iloc[0]
#
#         retracement = (account.records[code][1]-qfq_close)
#         sell_cond0 = retracement >= max(
#             (account.records[code][1]-account.records[code][0])*0.25,
#             account.records[code][1]*0.1)
#
#         sell_cond1 = (signal["y_l_rise"].iloc[0] <= 0.3) \
#                      & (signal["y_s_decline"].iloc[0] <=-0.1)
#
#         # Generate order based on various conditions.
#         if sell_cond0 or sell_cond1:
#             return cls.order_sell_by_stck_pct(code, percent=1,
#                                               price=qfq_close,account=account)
#         elif sum(account.stocks[code].values())==init_buy_cnt:
#             # Stage after buying first commitment.
#             if qfq_close/init_buy_price <= 0.95:
#                 return cls.order_sell_by_stck_pct(code, percent=1,
#                                            price=qfq_close,
#                                            account=account)
#             elif qfq_close/init_buy_price >= 1.05:
#                 return [BUY_FLAG, code,qfq_close, init_buy_cnt]
#         elif sum(account.stocks[code].values())==2*init_buy_cnt:
#             # Stage after buying second commitment.
#             if qfq_close/init_buy_price <= 1:
#                 return cls.order_sell_by_stck_pct(code, percent=1,
#                                            price=qfq_close,
#                                            account=account)
#             elif qfq_close/init_buy_price >= 1.1:
#                 return [BUY_FLAG, code, qfq_close, init_buy_cnt]
#         elif sum(account.stocks[code].values()) == 3*init_buy_cnt:
#             # Stage after buying third commitment.
#             if qfq_close/init_buy_price <= 1.034:
#                 return cls.order_sell_by_stck_pct(code, percent=1,
#                                            price=qfq_close,
#                                            account=account)
#         else:
#             return None
#
#         @classmethod
#         def strategy_for_stck_not_in_pos(cls, code, account: Account,
#                                          day_signal):
#             signal = day_signal[day_signal["code"] == code]
#             init_buy_cond = (signal["y_l_rise"] >= 0.55) & (
#                         signal["y_s_decline"] >= -0.03) & (
#                                         signal["y_s_rise"] >= 0.1)
#             if init_buy_cond.iloc[0]:
#                 pct = 0.2
#                 prices = {
#                 code: day_signal[day_signal["code"] == code]["qfq_close"].iloc[
#                     0] for code in day_signal["code"]}
#                 price = prices[code]
#                 return [cls.order_buy_pct(code, percent=pct, price=price,
#                                           account=account, prices=prices)]
#             else:
#                 return None
#
#         @classmethod
#         def gen_orders(cls, day_signal, account: Account):
#             orders = []
#             # Generate orders.
#             for code in day_signal["code"]:
#                 if code not in account.stocks:
#                     o = cls.strategy_for_stck_not_in_pos(code, account,
#                                                          day_signal)
#                 else:
#                     o = cls.strategy_for_stck_in_pos(code, account, day_signal)
#                 if o:
#                     orders.append(o)
#             return orders
#
#       @classmethod
#       def exe_orders_next_morning(cls, orders, day_signal, account:Account):
#         # Execute orders.
#         prices = {code: day_signal[day_signal["code"] == code]["f1mv_qfq_open"].iloc[0]
#                   for code in day_signal["code"]}
#
#         transactions = []
#         for o in orders:
#             flag, code, price, cnt = o[:4]
#             # Update price in order to f1mv_qfq_open.
#             price = prices[code]
#             stock_signal = day_signal[day_signal["code"] == code]
#             result=None
#             if flag==BUY_FLAG:
#                 if stock_signal["f1mv_qfq_low"].iloc[0] \
#                         == stock_signal["f1mv_qfq_high"].iloc[0]:
#                     print("一字板涨停：买入失败")
#                     continue
#                 result = cls.buy_by_cnt(code,price,cnt,account)
#             elif flag == SELL_FLAG:
#                 if stock_signal["f1mv_qfq_low"].iloc[0] \
#                         == stock_signal["f1mv_qfq_high"].iloc[0]:
#                     print("一字板跌停：卖出失败")
#                     continue
#                 result = cls.sell_by_cnt(code,cnt,price,account)
#
#             if result:
#                 transactions.append([day_signal.index[0]]+list(result[1:]))
#         return transactions


#
# # Try to cover all cases in if clause with order same as source codes.
# account.records[codes[0]] = [1, 1, 20000]
# account.records[codes[1]] = [1, 1.05, 20000]
# account.records[codes[2]] = [1, 1.1, 20000]
# account.records[codes[3]] = [1, 2, 20000]
# account.records[codes[4]] = [1, 1.5, 20000]
# account.records[codes[5]] = [1, 1, 20000]
# account.records[codes[6]] = [1, 1.05, 20000]
# account.records[codes[7]] = [1, 1.1, 20000]
# account.stocks[codes[0]] = {1: 20000}
# account.stocks[codes[1]] = {1: 20000, 1.05: 20000}
# account.stocks[codes[2]] = {1: 20000, 1.05: 20000, 1.1: 20000}
# account.stocks[codes[3]] = {1: 20000, 1.05: 20000, 1.1: 20000}
# account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
# account.stocks[codes[5]] = {1: 20000}
# account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
# account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
# day_signal0 = pd.DataFrame(
#     [[codes[0], 0.95, 0.95, 0.6, 0.1, -0.03],  # Sell all
#      [codes[1], 1, 1, 0.49, 0.12, -0.02],  # Sell all
#      [codes[2], 1.034, 1.034, 0.61, 0.09, -0.01],  # Sell all
#      [codes[3], 1.74, 1.74, 0.61, 0.11, -0.04],  # Sell all
#      [codes[4], 1.34, 1.34, 0.61, 0.11, -0.04],  # Sell all
#      [codes[5], 1.05, 1.05, 0.3, 0.11, -0.04],  # Second buy
#      [codes[6], 1.1, 1.1, 0.61, 0.11, -0.1],  # Third buy
#      [codes[7], 1.1, 1.1, 0.29, 0.11, -0.11]  # Sell all
#      ], columns=columns)


# orders = [("600345",300,24.5),("002345",500,11.2),("600229",1200,9.62),("600229",100,9.32)]
# for code,cnt,price in orders:
#     trader.buy_by_cnt(code, price, cnt,account)
#     amt -= cnt * price
#     self.assertEqual(amt, account.cash)
#
# pos = {"600345":{24.5:300},"002345":{11.2:500},"600229":{9.62:1200,9.32:100}}
# self.assertEqual(pos, account.stocks)
#
# def test_sell_by_cnt(self):
#     trader = self.trader
#     amt = 100000
#     account = trading.Account(init_amt=amt)
#
#     orders_buy = [("600345", 300, 24.5), ("002345", 500, 11.2), ("600229", 1200, 9.62), ("600229", 100, 9.32)]
#     for code, cnt, price in orders_buy:
#         trader.buy_by_cnt(code, price,cnt, account)
#         amt -= cnt * price
#
#     # Include a case that buys 600345 once and sells all, the code should be del from stocks's key.
#     # Include a case that buys and sells the same amount with the same price,
#     #       the price should be del from stocks[code]'s key.
#     orders_sell = [("600345", 300, 26.5), ("002345", 300, 12.2),("002345", 100, 12.5), ("600229", 600, 9.72), ("600229", 100, 9.32)]
#     for code, cnt, price in orders_sell:
#         trader.sell_by_cnt(code, -cnt, price, account)
#         amt += cnt * price
#     pos = {"002345":{11.2:500,12.2:-300,12.5:-100},"600229":{9.62:1200,9.72:-600}}
#     self.assertEqual(pos,account.stocks)
#     self.assertEqual(amt,account.cash)
#
#     # Include a case that total amount of buy and sell is 0, the code should be del from stocks.
#     orders_sell = [("600229", 600, 9.52)]
#     for code, cnt, price in orders_sell:
#         trader.sell_by_cnt(code, -cnt, price, account)
#         amt += cnt * price
#     pos = {"002345":{11.2:500,12.2:-300,12.5:-100}}
#     self.assertEqual(pos,account.stocks)
#     self.assertEqual(amt,account.cash)
#
#
#     # Include a case that buys 600345 once and sells all, the code should be del from stocks's key.
#     # Include a case that buys and sells the same amount with the same price,
#     #       the price should be del from stocks[code]'s key.
#     orders_sell = [("600345", 300, 26.5), ("002345", 300, 12.2),
#                    ("002345", 100, 12.5), ("600229", 600, 9.72),
#                    ("600229", 100, 9.32)]
#     for code, cnt, price in orders_sell:
#         trader.sell_by_cnt(code, -cnt, price, account)
#         amt += cnt * price
#     pos = {"002345": {11.2: 500, 12.2: -300, 12.5: -100},
#            "600229": {9.62: 1200, 9.72: -600}}
#     current_pos = {"002345": 100, "600229": 600}
#     prices = {"600345": 25, "002345": 12.1, "600229": 9.5}
#     self.assertEqual(
#         amt+sum([prices[code] * current_pos[code] for code in
#                  current_pos]),
#         trader.tot_amt(account, prices))
#
#     # Include a case that total amount of buy and sell is 0, the code should be del from stocks.
#     orders_sell = [("600229", 600, 9.52)]
#     for code, cnt, price in orders_sell:
#         trader.sell_by_cnt(code, -cnt, price, account)
#         amt += cnt * price
#     current_pos = {"002345": 100}
#     prices = {"600345": 25.5, "002345": 12.3, "600229": 9.7}
#     self.assertEqual(
#         amt+sum([prices[code] * current_pos[code] for code in
#                  current_pos]),
#         trader.tot_amt(account, prices))
#
#
# def test_strategy_for_stck_not_in_pos(self):
#     trader = self.trader
#     amt = 100000
#     account = trading.Account(init_amt=amt)
#     columns = ["code", "qfq_close", "f1mv_qfq_open",
#                "y_l_rise", "y_s_rise", "y_s_decline"]
#
#     codes = "600345", "600229", "002345", "002335"
#
#     # Initial buy conditions combine three simple conditions.
#     # Test data below cover a case satisfying three conditions,
#     # and three cases each failed with one condition.
#     day_signal0 = pd.DataFrame([[codes[0], 1, 1.01, 0.6, 0.1, -0.03],
#                                 [codes[1], 1, 0.99, 0.49, 0.12,-0.02],
#                                 [codes[2], 1, 1, 0.61, 0.09, -0.01],
#                                 [codes[3], 1, 1, 0.61, 0.11, -0.04]],
#                                columns=columns)
#     # print(day_signal0)
#     prices0 = {
#     code: day_signal0[day_signal0["code"] == code]["qfq_close"].iloc[0] for
#     code in day_signal0["code"]}
#     # print(prices0)
#     orders = []
#     for code in day_signal0["code"]:
#         if code not in account.stocks:
#             o = trader.strategy_for_stck_not_in_pos(code, account,
#                                                     day_signal0)
#             if o:
#                 orders.append(o)
#     expected_orders = [
#         [trading.BUY_FLAG, codes[0], prices0[codes[0]], 20000]]
#     self.assertEqual(expected_orders, orders)
#
# def test_strategy_for_stck_in_pos(self):
#     trader = self.trader
#     amt = 100000
#     account = trading.Account(init_amt=amt)
#     columns = ["code", "qfq_close", "f1mv_qfq_open",
#                "y_l_rise", "y_s_rise", "y_s_decline"]
#
#     codes="600345","600229","002345","002335","002236","002217",\
#           "300345","603799"
#
#     # Try to cover all cases in if clause with order same as source codes.
#     account.records[codes[0]] = [1, 1, 20000]
#     account.records[codes[1]] = [1, 1.05, 20000]
#     account.records[codes[2]] = [1, 1.1, 20000]
#     account.records[codes[3]] = [1, 2, 20000]
#     account.records[codes[4]] = [1, 1.5, 20000]
#     account.records[codes[5]] = [1, 1, 20000]
#     account.records[codes[6]] = [1, 1.05, 20000]
#     account.records[codes[7]] = [1, 1.1, 20000]
#     account.stocks[codes[0]] = {1: 20000}
#     account.stocks[codes[1]] = {1: 20000, 1.05: 20000}
#     account.stocks[codes[2]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     account.stocks[codes[3]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     account.stocks[codes[5]] = {1: 20000}
#     account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
#     account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     day_signal0 = pd.DataFrame(
#         [[codes[0], 0.95, 0.95, 0.6, 0.1, -0.03],  # Sell all
#          [codes[1], 1, 1, 0.49, 0.12, -0.02],  # Sell all
#          [codes[2], 1.034, 1.034, 0.61, 0.09, -0.01],  # Sell all
#          [codes[3], 1.74, 1.74, 0.61, 0.11, -0.04],  # Sell all
#          [codes[4], 1.34, 1.34, 0.61, 0.11, -0.04],  # Sell all
#          [codes[5], 1.05, 1.05, 0.3, 0.11, -0.04],  # Second buy
#          [codes[6], 1.1, 1.1, 0.61, 0.11, -0.1],  # Third buy
#          [codes[7], 1.1, 1.1, 0.29, 0.11, -0.11]  # Sell all
#          ], columns=columns)
#
#     print("Include any nan in day_signal:",day_signal0.isna().any().any())
#
#     # print(day_signal0)
#     prices0 = {code: day_signal0[day_signal0["code"] == code]["qfq_close"].iloc[0]
#                for code in day_signal0["code"]}
#     # print(prices0)
#     orders = []
#     for code in day_signal0["code"]:
#         if code in account.stocks:
#             o = trader.strategy_for_stck_in_pos(code, account,
#                                                 day_signal0)
#             if o:
#                 orders.append(o)
#     expected_orders = [[trading.SELL_FLAG,codes[0],prices0[codes[0]],-20000],
#                        [trading.SELL_FLAG, codes[1], prices0[codes[1]],
#                         -40000],
#                        [trading.SELL_FLAG, codes[2], prices0[codes[2]],
#                         -60000],
#                        [trading.SELL_FLAG, codes[3], prices0[codes[3]],
#                        -60000],
#                        [trading.SELL_FLAG, codes[4], prices0[codes[4]],
#                        -60000],
#                        [trading.BUY_FLAG, codes[5], prices0[codes[5]],
#                         20000],
#                        [trading.BUY_FLAG, codes[6], prices0[codes[6]],
#                         20000],
#                        [trading.SELL_FLAG, codes[7], prices0[codes[7]],
#                        -60000],
#                        ]
#     self.assertEqual(expected_orders,orders)
#
# def test_gen_orders(self):
#     trader = self.trader
#     amt = 100000
#     account = trading.Account(init_amt=amt)
#     columns = ["code", "qfq_close", "f1mv_qfq_open",
#                "y_l_rise", "y_s_rise", "y_s_decline"]
#
#     codes = "600345", "600229", "002345", "002335", "002236", "002217", "300345", "603799"
#
#     account.records[codes[4]] = [1, 1.5, 20000]
#     account.records[codes[5]] = [1, 1, 20000]
#     account.records[codes[6]] = [1, 1.05, 20000]
#     account.records[codes[7]] = [1, 1.1, 20000]
#     account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     account.stocks[codes[5]] = {1: 20000}
#     account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
#     account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     day_signal0 = pd.DataFrame([
#         [codes[0], 1, 1.01, 0.6, 0.1, -0.03],
#         [codes[1], 1, 0.99, 0.49, 0.12, -0.02],
#         [codes[2], 1, 1.01, 0.61, 0.09, -0.01],
#         [codes[3], 1, 1.02, 0.61, 0.11, -0.04],
#         [codes[4], 1.34, 1.35, 0.61, 0.11, -0.04],  # Sell all
#         [codes[5], 1.05, 1.053, 0.3, 0.11, -0.04],   # Second buy
#         [codes[6], 1.1, 1.101, 0.61, 0.11, -0.1],     # Third buy
#         [codes[7], 1.1, 1.09, 0.29, 0.11, -0.11]     # Sell all
#         ], columns=columns)
#
#     prices0 = {
#         code: day_signal0[day_signal0["code"] == code][
#             "qfq_close"].iloc[0] for code in day_signal0["code"]}
#     # print(prices0)
#
#     orders = trader.gen_orders(day_signal0,account)
#     expected_orders = [
#         [trading.BUY_FLAG, codes[0], prices0[codes[0]], int(622.8) * 100],
#         [trading.SELL_FLAG, codes[4], prices0[codes[4]], -60000],
#         [trading.BUY_FLAG, codes[5], prices0[codes[5]], 20000],
#         [trading.BUY_FLAG, codes[6], prices0[codes[6]], 20000],
#         [trading.SELL_FLAG, codes[7], prices0[codes[7]], -60000],
#     ]
#     self.assertEqual(expected_orders,orders)
#
# def test_exe_orders(self):
#     trader = self.trader
#     amt = 100000
#     account = trading.Account(init_amt=amt)
#     columns = ["code", "qfq_close", "f1mv_qfq_open", "f1mv_qfq_high",
#                "f1mv_qfq_low", "y_l_rise", "y_s_rise", "y_s_decline"]
#
#     codes = "600345", "600229", "002345", "002335", "002236", "002217", "300345", "603799"
#
#     account.records[codes[4]] = [1, 1.5, 20000]
#     account.records[codes[5]] = [1, 1, 20000]
#     account.records[codes[6]] = [1, 1.05, 20000]
#     account.records[codes[7]] = [1, 1.1, 20000]
#     account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     account.stocks[codes[5]] = {1: 20000}
#     account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
#     account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     day_signal0 = pd.DataFrame(
#         [[codes[0], 1, 1.01, 1.02, 0.98, 0.6, 0.1, -0.03],
#             [codes[1], 1, 0.99, 1.02, 0.98, 0.49, 0.12, -0.02],
#             [codes[2], 1, 1.01, 1.03, 0.99, 0.61, 0.09, -0.01],
#             [codes[3], 1, 1.02, 1.04, 1.01, 0.61, 0.11, -0.04],
#             [codes[4], 1.34, 1.35, 1.36, 1.33, 0.61, 0.11, -0.04],
#             # Sell all
#             [codes[5], 1.05, 1.053, 1.054, 1.04, 0.3, 0.11, -0.04],
#             # Second buy
#             [codes[6], 1.1, 1.101, 1.102, 1.09, 0.61, 0.11, -0.1],
#             # Third buy
#             [codes[7], 1.1, 1.09, 1.101, 1.08, 0.29, 0.11, -0.11]
#             # Sell all
#         ], columns=columns)
#     date_idx = "2018-10-01"
#     day_signal0.index = [date_idx]*len(day_signal0)
#
#     orders = trader.gen_orders(day_signal0,account)
#     transactions = trader.exe_orders_next_morning(orders=orders, day_signal=day_signal0,
#                                                   account=account)
#
#     expected_stocks = {codes[0]:{1.01:62200},
#                        codes[5]:{1:20000,1.053:20000},
#                        codes[6]:{1:20000,1.05:20000,1.101:20000}
#                        }
#     expected_transactions = [
#         [date_idx,codes[0], 1.01, 62200],
#         [date_idx, codes[4], 1.35, -60000],
#         [date_idx,codes[5], 1.053, 20000],
#         [date_idx,codes[6], 1.101, 20000],
#         [date_idx, codes[7], 1.09, -60000],
#     ]
#     self.assertEqual(expected_stocks,account.stocks)
#     self.assertEqual(expected_transactions,transactions)
#
# def test_update_records(self):
#     trader = self.trader
#     amt = 100000
#     account = trading.Account(init_amt=amt)
#     columns = ["code", "qfq_close",
#                "f1mv_qfq_open", "f1mv_qfq_high","f1mv_qfq_low",
#                "y_l_rise","y_s_rise", "y_s_decline"]
#
#     codes = "600345", "600229", "002345", "002335", "002236", "002217", "300345", "603799"
#
#     account.records[codes[4]] = [1, 1.5, 20000]
#     account.records[codes[5]] = [1, 1, 20000]
#     account.records[codes[6]] = [1, 1.05, 20000]
#     account.records[codes[7]] = [1, 1.1, 20000]
#     account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     account.stocks[codes[5]] = {1: 20000}
#     account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
#     account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     day_signal0 = pd.DataFrame([
#         [codes[0], 1, 1.01,1.02,0.98, 0.6, 0.1, -0.03],
#         [codes[1], 1, 0.99,1.02,0.98, 0.49, 0.12, -0.02],
#         [codes[2], 1, 1.01,1.03,0.99, 0.61, 0.09, -0.01],
#         [codes[3], 1, 1.02,1.04,1.01, 0.61, 0.11, -0.04],
#         [codes[4], 1.34, 1.35,1.36,1.33, 0.61, 0.11, -0.04],  # Sell all
#         [codes[5], 1.05, 1.053,1.054,1.04, 0.3, 0.11, -0.04],  # Second buy
#         [codes[6], 1.1, 1.101,1.102,1.09, 0.61, 0.11, -0.1],  # Third buy
#         [codes[7], 1.1, 1.09,1.101,1.08, 0.29, 0.11, -0.11]  # Sell all
#     ], columns=columns)
#
#     # Use columns.difference to make sure not use unnecessary or future
#     # information.
#     orders = trader.gen_orders(day_signal0[
#                                    day_signal0.columns.difference([
#                                        "f1mv_qfq_open","f1mv_qfq_high",
#                                        "f1mv_qfq_low"
#                                    ])], account)
#     trader.exe_orders_next_morning(orders=orders,
#                                    day_signal=day_signal0,
#                                    account=account)
#     trader.update_records(day_signal=day_signal0[
#                           day_signal0.columns.difference([
#                               "f1mv_qfq_open","f1mv_qfq_low"])],account=account)
#
#     expected_stocks = {codes[0]: {1.01: 62200},
#                        codes[5]: {1: 20000, 1.053: 20000},
#                        codes[6]: {1: 20000, 1.05: 20000, 1.101: 20000}}
#
#     expected_records = {codes[0]: [1.01,1.02, 62200],
#                         codes[5]: [1,1.054,20000],
#                         codes[6]: [1,1.102,20000]}
#     self.assertEqual(expected_records,account.records)
#
# def test_exe_orders_hit_limit(self):
#     trader = self.trader
#     amt = 100000
#     account = trading.Account(init_amt=amt)
#     columns = ["code", "qfq_close", "f1mv_qfq_open", "f1mv_qfq_high",
#                "f1mv_qfq_low",
#                "y_l_rise", "y_s_rise", "y_s_decline"]
#
#     codes = ["600345", "600229", "002345", "002335", "002236", "002217",
#              "300345", "603799"]
#     account.stocks[codes[1]] = {1:20000}
#     orders = [(trading.BUY_FLAG,codes[0],1,20000),
#               (trading.SELL_FLAG,codes[1],1,20000)]
#     day_signals = pd.DataFrame([
#         [codes[0], 1, 1.1,1.1,1.1, 0.6, 0.1, -0.03],
#         [codes[1], 1, 0.9,0.9,0.9, 0.49, 0.12, -0.02],
#     ], columns=columns)
#     trader.exe_orders_next_morning(orders, day_signal=day_signals, account=account)
#     expected_stocks = {codes[1]:{1:20000}}
#     self.assertEqual(expected_stocks,account.stocks)
#
# def test_trade(self):
#     trader = self.trader
#     amt = 100000
#     account = trading.Account(init_amt=amt)
#     columns = ["code", "qfq_close", "f1mv_qfq_open", "f1mv_qfq_high",
#                "f1mv_qfq_low", "y_l_rise", "y_s_rise", "y_s_decline"]
#
#     codes = "600345", "600229", "002345", "002335", "002236", "002217", "300345", "603799"
#
#     account.records[codes[4]] = [1, 1.5, 20000]
#     account.records[codes[5]] = [1, 1, 20000]
#     account.records[codes[6]] = [1, 1.05, 20000]
#     account.records[codes[7]] = [1, 1.1, 20000]
#     account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     account.stocks[codes[5]] = {1: 20000}
#     account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
#     account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
#     day_signal0 = pd.DataFrame(
#         [[codes[0], 1, 1.01, 1.02, 0.98, 0.6, 0.1, -0.03],
#             [codes[1], 1, 0.99, 1.02, 0.98, 0.49, 0.12, -0.02],
#             [codes[2], 1, 1.01, 1.03, 0.99, 0.61, 0.09, -0.01],
#             [codes[3], 1, 1.02, 1.04, 1.01, 0.61, 0.11, -0.04],
#             [codes[4], 1.34, 1.35, 1.36, 1.33, 0.61, 0.11, -0.04],
#             # Sell all
#             [codes[5], 1.05, 1.053, 1.054, 1.04, 0.3, 0.11, -0.04],
#             # Second buy
#             [codes[6], 1.1, 1.101, 1.102, 1.09, 0.61, 0.11, -0.1],
#             # Third buy
#             [codes[7], 1.1, 1.09, 1.101, 1.08, 0.29, 0.11, -0.11]
#             # Sell all
#         ], columns=columns)
#
#     # Use columns.difference to make sure not use unnecessary or future
#     # information.
#     trader.trade(day_signal=day_signal0,account=account)
#
#     expected_stocks = {codes[0]: {1.01: 62200},
#                        codes[5]: {1: 20000, 1.053: 20000},
#                        codes[6]: {1: 20000, 1.05: 20000, 1.101: 20000}}
#
#     expected_records = {codes[0]: [1.01, 1.02, 62200],
#                         codes[5]: [1, 1.054, 20000],
#                         codes[6]: [1, 1.102, 20000]}
#     self.assertEqual(expected_stocks,account.stocks)
#     self.assertEqual(expected_records, account.records)


# def test_whole(self):
    # test
    # df_test = df_all[df_all["code"]=="600887.SH"]
    # basic_cols = ["open", "high", "low", "close", "amt", "adj_factor"]
    # derived_cols = ['change_rate_p1mv_open', 'change_rate_p1mv_high',
    #                 'change_rate_p1mv_low', 'change_rate_p1mv_close',
    #                 'change_rate_p1mv_amt', 'change_rate_p3mv_open',
    #                 'change_rate_p3mv_high', 'change_rate_p3mv_low',
    #                 'change_rate_p3mv_close', 'change_rate_p3mv_amt',
    #                 'change_rate_p5mv_open', 'change_rate_p5mv_high',
    #                 'change_rate_p5mv_low', 'change_rate_p5mv_close',
    #                 'change_rate_p5mv_amt', 'change_rate_p5max_open',
    #                 'change_rate_p5max_high', 'change_rate_p5max_low',
    #                 'change_rate_p5max_close', 'change_rate_p5max_amt',
    #                 'change_rate_p5min_open', 'change_rate_p5min_high',
    #                 'change_rate_p5min_low', 'change_rate_p5min_close',
    #                 'change_rate_p5min_amt', 'change_rate_p5mean_open',
    #                 'change_rate_p5mean_high', 'change_rate_p5mean_low',
    #                 'change_rate_p5mean_close', 'change_rate_p5mean_amt',
    #                 'change_rate_p20max_open', 'change_rate_p20max_high',
    #                 'change_rate_p20max_low', 'change_rate_p20max_close',
    #                 'change_rate_p20max_amt', 'change_rate_p20min_open',
    #                 'change_rate_p20min_high', 'change_rate_p20min_low',
    #                 'change_rate_p20min_close', 'change_rate_p20min_amt',
    #                 'change_rate_p20mean_open', 'change_rate_p20mean_high',
    #                 'change_rate_p20mean_low', 'change_rate_p20mean_close',
    #                 'change_rate_p20mean_amt', 'f1mv_open', 'f1mv_high',
    #                 'f1mv_low', 'f1mv_close', 'f20max_f1mv_high',
    #                 'sz50_open', 'sz50_high', 'sz50_low', 'sz50_close',
    #                 'sz50_vol', 'sz50_change_rate_p1mv_open',
    #                 'sz50_change_rate_p1mv_high',
    #                 'sz50_change_rate_p1mv_low',
    #                 'sz50_change_rate_p1mv_close',
    #                 'sz50_change_rate_p1mv_vol']
    #
    # test_cols = basic_cols + derived_cols
    # print(test_cols)
    # df_test[test_cols].sort_index(ascending=False).iloc[:100].to_excel(
    #     "test_data.xlsx",header=True,index=True)

    # # test
    # df_test_list = []
    # for code in df_all["code"].unique()[:3]:
    #     df = df_all[df_all["code"]==code].sort_index(
    #         ascending=False).iloc[:50]
    #     print(df)
    #     df_test_list.append(df)
    # pd.concat(df_test_list).to_excel("test_data.xlsx",header=True,index=True)
    # pass



# def rolling(rolling_type, days, df: pd.DataFrame, cols, move=0,
#             has_prefix=True):
#     _check_int(days)
#     cols = _make_iterable(cols)
#
#     period = abs(days)
#     if rolling_type == "max":
#         df_rolling = df[cols].rolling(window=abs(days)).max()
#     elif rolling_type == "min":
#         df_rolling = df[cols].rolling(window=abs(days)).min()
#     elif rolling_type == "mean":
#         df_rolling = df[cols].rolling(window=abs(days)).max()
#     else:
#         raise ValueError(
#             "rolling_type='{}' is not supported.".format(rolling_type))
#
#     if move != 0:
#         df_rolling = _move(move, df_rolling)
#     n = len(df_rolling)
#     idxes = df_rolling.index
#     if days > 0:
#         pre = "f" + str(abs(days)) + rolling_type
#         df_rolling = df_rolling.iloc[period - 1:n]
#         df_rolling.index = idxes[period - 1:n]
#     else:
#         pre = "p" + str(abs(days)) + rolling_type
#         df_rolling = df_rolling.iloc[period - 1:n]
#         if n - period + 1 >= 0:
#             df_rolling.index = idxes[:n - period + 1]
#
#     if has_prefix:
#         return _prefix(pre, df_rolling)
#     else:
#         return df_rolling





#
# def _rolling_max(days, df: pd.DataFrame, cols, move=0, has_prefix=True):
#     _check_int(days)
#     cols = _make_iterable(cols)
#
#     period = abs(days)
#     df_rolling = df[cols].rolling(window=abs(days)).max()
#     if move != 0:
#         # print("--------",move)
#         # print(df_rolling[df["code"] == "600887.SH"]["high"].iloc[:30])
#         df_rolling = _move(move,df_rolling)  # print(df_rolling[df["code"] == "600887.SH"]["f1mv_high"].iloc[:30])
#     n = len(df_rolling)
#     idxes = df_rolling.index
#     if days > 0:
#         pre = "f" + str(abs(days)) + "max"
#         df_rolling = df_rolling.iloc[period - 1:n]
#         df_rolling.index = idxes[
#                            period - 1:n]  # df_rolling = df_rolling.iloc[period-1:n+move]  # df_rolling.index = df.index[period-1-move:n]
#     else:
#         pre = "p" + str(abs(days)) + "max"
#         df_rolling = df_rolling.iloc[period - 1:n]
#         if n - period + 1 >= 0:
#             df_rolling.index = idxes[:n - period + 1]
#
#         # df_rolling = df_rolling.iloc[period-1+move:n]  # df_rolling.index = df.index[:n-period+1-move]
#
#     if has_prefix:
#         return _prefix(pre, df_rolling)
#     else:
#         return df_rolling
#
#
# def _rolling_min(days, df: pd.DataFrame, cols, move=0, has_prefix=True):
#     _check_int(days)
#     cols = _make_iterable(cols)
#
#     period = abs(days)
#     df_rolling = df[cols].rolling(window=abs(days)).min()
#     if move != 0:
#         # print("--------",move)
#         # print(df_rolling[df["code"] == "600887.SH"]["high"].iloc[:30])
#         df_rolling = _move(move,
#                           df_rolling)  # print(df_rolling[df["code"] == "600887.SH"]["f1mv_high"].iloc[:30])
#     n = len(df_rolling)
#     idxes = df_rolling.index
#     if days > 0:
#         pre = "f" + str(abs(days)) + "min"
#         df_rolling = df_rolling.iloc[period - 1:n]
#         df_rolling.index = idxes[period - 1:n]
#     else:
#         pre = "p" + str(abs(days)) + "min"
#         df_rolling = df_rolling.iloc[period - 1:n]
#         if n - period + 1 >= 0:
#             df_rolling.index = idxes[:n - period + 1]
#
#     if has_prefix:
#         return _prefix(pre, df_rolling)
#     else:
#         return df_rolling
#
#
# def _rolling_mean(days, df: pd.DataFrame, cols, move=0, has_prefix=True):
#     _check_int(days)
#     cols = _make_iterable(cols)
#
#     period = abs(days)
#     df_rolling = df[cols].rolling(window=abs(days)).mean()
#     if move != 0:
#         df_rolling = _move(move, df_rolling)
#     n = len(df_rolling)
#     idxes = df_rolling.index
#     if days > 0:
#         pre = "f" + str(abs(days)) + "mean"
#         df_rolling = df_rolling.iloc[period - 1:n]
#         df_rolling.index = idxes[period - 1:n]
#     else:
#         pre = "p" + str(abs(days)) + "mean"
#         df_rolling = df_rolling.iloc[period - 1:n]
#         if n - period + 1 >= 0:
#             df_rolling.index = idxes[:n - period + 1]
#
#     if has_prefix:
#         return _prefix(pre, df_rolling)
#     else:
#         return df_rolling




#     db_type = "sqlite3"
#     #
#     # conn = dbop.connect_db(db_type)
#     # cursor = conn.cursor()
#     #
#     # pred_period=20
#     # df_all,cols_future = prepare_data(cursor,pred_period=pred_period,start="2011-01-01")
#     #
#     # # test
#     # # df_test = df_all[df_all["code"]=="600887.SH"]
#     # # basic_cols = ["open", "high", "low", "close", "amt", "adj_factor"]
#     # # derived_cols = ['change_rate_p1mv_open', 'change_rate_p1mv_high',
#     # #                 'change_rate_p1mv_low', 'change_rate_p1mv_close',
#     # #                 'change_rate_p1mv_amt', 'change_rate_p3mv_open',
#     # #                 'change_rate_p3mv_high', 'change_rate_p3mv_low',
#     # #                 'change_rate_p3mv_close', 'change_rate_p3mv_amt',
#     # #                 'change_rate_p5mv_open', 'change_rate_p5mv_high',
#     # #                 'change_rate_p5mv_low', 'change_rate_p5mv_close',
#     # #                 'change_rate_p5mv_amt', 'change_rate_p5max_open',
#     # #                 'change_rate_p5max_high', 'change_rate_p5max_low',
#     # #                 'change_rate_p5max_close', 'change_rate_p5max_amt',
#     # #                 'change_rate_p5min_open', 'change_rate_p5min_high',
#     # #                 'change_rate_p5min_low', 'change_rate_p5min_close',
#     # #                 'change_rate_p5min_amt', 'change_rate_p5mean_open',
#     # #                 'change_rate_p5mean_high', 'change_rate_p5mean_low',
#     # #                 'change_rate_p5mean_close', 'change_rate_p5mean_amt',
#     # #                 'change_rate_p20max_open', 'change_rate_p20max_high',
#     # #                 'change_rate_p20max_low', 'change_rate_p20max_close',
#     # #                 'change_rate_p20max_amt', 'change_rate_p20min_open',
#     # #                 'change_rate_p20min_high', 'change_rate_p20min_low',
#     # #                 'change_rate_p20min_close', 'change_rate_p20min_amt',
#     # #                 'change_rate_p20mean_open', 'change_rate_p20mean_high',
#     # #                 'change_rate_p20mean_low', 'change_rate_p20mean_close',
#     # #                 'change_rate_p20mean_amt', 'f1mv_open', 'f1mv_high',
#     # #                 'f1mv_low', 'f1mv_close', 'f20max_f1mv_high',
#     # #                 'sz50_open', 'sz50_high', 'sz50_low', 'sz50_close',
#     # #                 'sz50_vol', 'sz50_change_rate_p1mv_open',
#     # #                 'sz50_change_rate_p1mv_high',
#     # #                 'sz50_change_rate_p1mv_low',
#     # #                 'sz50_change_rate_p1mv_close',
#     # #                 'sz50_change_rate_p1mv_vol']
#     # #
#     # # test_cols = basic_cols + derived_cols
#     # # print(test_cols)
#     # # df_test[test_cols].sort_index(ascending=False).iloc[:100].to_excel(
#     # #     "test_data.xlsx",header=True,index=True)
#     #
#     #
#     #
#     #
#     # # # test
#     # # df_test_list = []
#     # # for code in df_all["code"].unique()[:3]:
#     # #     df = df_all[df_all["code"]==code].sort_index(
#     # #         ascending=False).iloc[:50]
#     # #     print(df)
#     # #     df_test_list.append(df)
#     # # pd.concat(df_test_list).to_excel("test_data.xlsx",header=True,index=True)
#     # #
#     # #
#     # import xgboost.sklearn as xgb
#     # import lightgbm.sklearn as lgbm
#     # import sklearn.metrics as metrics
#     # import matplotlib.pyplot as plt
#     # import time
#     # import sklearn.preprocessing as preproc
#     #
#     # period = (df_all.index >= "2014-01-01")
#     # df_all = df_all[period]
#     #
#     # df_all = df_all[df_all["amt"]!=0]
#     #
#     # y = gen_y(df_all, threshold=0.15, pred_period=pred_period)
#     # print("null:",sum(y.isnull()))
#     #
#     # features = df_all.columns.difference(cols_future+["code"])
#     #
#     #
#     # X = df_all[features]
#     #
#     #
#     # # X,y = drop_null(X,y)
#     # X = X[y.notnull()]
#     # X_full = df_all[y.notnull()]
#     # print("full and X",X.shape,X_full.shape)
#     # y = y.dropna()
#     # print(X.shape,y.shape)
#     # print("total positive", sum(y))
#     #
#     # condition = (X.index >= "2018-01-01")
#     # X_train, y_train = X[~condition], y[~condition]
#     # X_test, y_test = X[condition], y[condition]
#     #
#     # print(X_test.shape,y_test.shape)
#     # print("test positive:", sum(y_test))
#     #
#     # X_train_full = X_full.loc[condition]
#     # X_test_full = X_full.loc[condition]
#     #
#     # print(X_test_full.shape,X_test.shape)
#     # print(X_test_full[(X_test_full.index == "2018-01-02") & (X_test_full["code"]=="002217.SZ")].shape)
#     #
#     #
#     # # print(X_test_full["code"].iloc[np.array(y_test == 1)])
#     # # print(X_test_full[X_test_full["code"]=="002217.SZ"])
#     #
#     # # # scaler = preproc.StandardScaler()
#     # # # X_train = scaler.fit_transform(X_train)
#     # # # X_test = scaler.transform(X_test)
#     # #
#     # # # X_train,selector = feature_select(X_train,y_train)
#     # # # X_test = selector.transform(X_test)
#     # #
#     # #
#     # scale_pos_weight = sum(y==0)/sum(y==1)
#     #
#     # clfs = [
#     #     lgbm.LGBMClassifier(n_estimators=300, scale_pos_weight=0.1,
#     #                         num_leaves=100, max_depth=8, random_state=0),
#     #     xgb.XGBClassifier(n_estimators=300, scale_pos_weight=0.1,
#     #                       max_depth=5,
#     #                       random_state=0,),
#     # ]
#     #
#     # y_prd_list = []
#     # colors = ["r", "b"]
#     # for clf, c in zip(clfs, colors):
#     #     t1 = time.time()
#     #     clf.fit(X_train, y_train)
#     #     t2 = time.time()
#     #     y_prd_list.append([clf, t2 - t1, clf.predict_proba(X_test), c])
#     #
#     # for clf, t, y_prd_prob, c in y_prd_list:
#     #     y_prd = np.where(y_prd_prob[:, 0] < 0.25, 1, 0)
#     #     print(clf.classes_)
#     #     print(y_prd.shape, sum(y_prd))
#     #
#     #     print(X_test_full["code"].iloc[y_prd==1])
#     #     # print(X_test_full["code"])
#     #
#     #     print("accuracy", metrics.accuracy_score(y_test, y_prd))
#     #     print("precison", metrics.precision_score(y_test, y_prd))
#     #     print("recall", metrics.recall_score(y_test, y_prd))
#     #     precision, recall, _ = metrics.precision_recall_curve(y_test, y_prd_prob[:, 1])
#     #
#     #     plt.figure()
#     #     plt.title(clf.__class__)
#     #     plt.xlim(0, 1)
#     #     plt.ylim(0, 1)
#     #     plt.xlabel("recall")
#     #     plt.ylabel("precision")
#     #     plt.plot(recall, precision, color=c)
#     #     print(clf, t)
#     #
#     # plt.show()


# f_name1 = "XGBRegressor_20high"
# f_name2 = "XGBRegressor_5low"
# f_name3 = "XGBRegressor_5high"
# model_type = "XGBRegressor"

# models["model_l_high"] = ml_model.load_model(model_type,pred_period=20,is_high=True)
# models["model_s_low"] = ml_model.load_model(model_type,pred_period=5,is_high=False)
# models["model_s_high"] = ml_model.load_model(model_type,pred_period=5,is_high=True)
