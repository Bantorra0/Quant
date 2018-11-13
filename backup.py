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