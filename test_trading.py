import unittest

import pandas as pd

import trading


class TraderTestCase(unittest.TestCase):
    def setUp(self):
        self.trader = trading.Trader()

    def test_buy_by_cnt(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        orders = [("600345",300,24.5),("002345",500,11.2),("600229",1200,9.62),("600229",100,9.32)]
        for code,cnt,price in orders:
            trader.buy_by_cnt(code, cnt, price, account)
            amt -= cnt * price
            self.assertEqual(amt, account.cash)

        pos = {"600345":{24.5:300},"002345":{11.2:500},"600229":{9.62:1200,9.32:100}}
        self.assertEqual(pos, account.stocks)

    def test_sell_by_cnt(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        orders_buy = [("600345", 300, 24.5), ("002345", 500, 11.2), ("600229", 1200, 9.62), ("600229", 100, 9.32)]
        for code, cnt, price in orders_buy:
            trader.buy_by_cnt(code, cnt, price, account)
            amt -= cnt * price

        # Include a case that buys 600345 once and sells all, the code should be del from stocks's key.
        # Include a case that buys and sells the same amount with the same price,
        #       the price should be del from stocks[code]'s key.
        orders_sell = [("600345", 300, 26.5), ("002345", 300, 12.2),("002345", 100, 12.5), ("600229", 600, 9.72), ("600229", 100, 9.32)]
        for code, cnt, price in orders_sell:
            trader.sell_by_cnt(code, -cnt, price, account)
            amt += cnt * price
        pos = {"002345":{11.2:500,12.2:-300,12.5:-100},"600229":{9.62:1200,9.72:-600}}
        self.assertEqual(pos,account.stocks)
        self.assertEqual(amt,account.cash)

        # Include a case that total amount of buy and sell is 0, the code should be del from stocks.
        orders_sell = [("600229", 600, 9.52)]
        for code, cnt, price in orders_sell:
            trader.sell_by_cnt(code, -cnt, price, account)
            amt += cnt * price
        pos = {"002345":{11.2:500,12.2:-300,12.5:-100}}
        self.assertEqual(pos,account.stocks)
        self.assertEqual(amt,account.cash)

    def test_tot_amt(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        orders_buy = [("600345", 300, 24.5), ("002345", 500, 11.2),
                      ("600229", 1200, 9.62), ("600229", 100, 9.32)]
        for code, cnt, price in orders_buy:
            trader.buy_by_cnt(code, cnt, price, account)
            amt -= cnt * price
        current_pos = {"600345":300, "002345":500,"600229":1300}
        prices = {"600345":25, "002345":12.1,"600229":9.5}
        self.assertEqual(amt+sum([prices[code]*current_pos[code] for code in
                              current_pos]),
                         trader.tot_amt(account,prices))

        # Include a case that buys 600345 once and sells all, the code should be del from stocks's key.
        # Include a case that buys and sells the same amount with the same price,
        #       the price should be del from stocks[code]'s key.
        orders_sell = [("600345", 300, 26.5), ("002345", 300, 12.2),
                       ("002345", 100, 12.5), ("600229", 600, 9.72),
                       ("600229", 100, 9.32)]
        for code, cnt, price in orders_sell:
            trader.sell_by_cnt(code, -cnt, price, account)
            amt += cnt * price
        pos = {"002345": {11.2: 500, 12.2: -300, 12.5: -100},
               "600229": {9.62: 1200, 9.72: -600}}
        current_pos = {"002345": 100, "600229": 600}
        prices = {"600345": 25, "002345": 12.1, "600229": 9.5}
        self.assertEqual(
            amt+sum([prices[code] * current_pos[code] for code in
                     current_pos]),
            trader.tot_amt(account, prices))

        # Include a case that total amount of buy and sell is 0, the code should be del from stocks.
        orders_sell = [("600229", 600, 9.52)]
        for code, cnt, price in orders_sell:
            trader.sell_by_cnt(code, -cnt, price, account)
            amt += cnt * price
        current_pos = {"002345": 100}
        prices = {"600345": 25.5, "002345": 12.3, "600229": 9.7}
        self.assertEqual(
            amt+sum([prices[code] * current_pos[code] for code in
                     current_pos]),
            trader.tot_amt(account, prices))

    def test_get_cnt_from_percent(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        cnt = trader.get_cnt_from_percent(0.2, 3.5, account,{})
        self.assertEqual(5700,cnt)

    def test_strategy_for_stck_not_in_pos(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)
        columns = ["code", "qfq_close", "f1mv_qfq_open",
                   "y_l_rise", "y_s_rise", "y_s_decline"]

        codes = "600345", "600229", "002345", "002335"

        # Initial buy conditions combine three simple conditions.
        # Test data below cover a case satisfying three conditions,
        # and three cases each failed with one condition.
        day_signal0 = pd.DataFrame([[codes[0], 1, 1.01, 0.6, 0.1, -0.03],
                                    [codes[1], 1, 0.99, 0.49, 0.12,-0.02],
                                    [codes[2], 1, 1, 0.61, 0.09, -0.01],
                                    [codes[3], 1, 1, 0.61, 0.11, -0.04]],
                                   columns=columns)
        # print(day_signal0)
        prices0 = {
        code: day_signal0[day_signal0["code"] == code]["qfq_close"].iloc[0] for
        code in day_signal0["code"]}
        # print(prices0)
        orders = []
        for code in day_signal0["code"]:
            if code not in account.stocks:
                o = trader.strategy_for_stck_not_in_pos(code, account,
                                                        day_signal0)
                if o:
                    orders.append(o)
        expected_orders = [
            (trading.BUY_FLAG, codes[0], prices0[codes[0]], 20000)]
        self.assertEqual(expected_orders, orders)

    def test_strategy_for_stck_in_pos(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)
        columns = ["code", "qfq_close", "f1mv_qfq_open",
                   "y_l_rise", "y_s_rise", "y_s_decline"]

        codes="600345","600229","002345","002335","002236","002217",\
              "300345","603799"

        # Try to cover all cases in if clause with order same as source codes.
        account.records[codes[0]] = [1, 1, 20000]
        account.records[codes[1]] = [1, 1.05, 20000]
        account.records[codes[2]] = [1, 1.1, 20000]
        account.records[codes[3]] = [1, 2, 20000]
        account.records[codes[4]] = [1, 1.5, 20000]
        account.records[codes[5]] = [1, 1, 20000]
        account.records[codes[6]] = [1, 1.05, 20000]
        account.records[codes[7]] = [1, 1.1, 20000]
        account.stocks[codes[0]] = {1: 20000}
        account.stocks[codes[1]] = {1: 20000, 1.05: 20000}
        account.stocks[codes[2]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        account.stocks[codes[3]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        account.stocks[codes[5]] = {1: 20000}
        account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
        account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        day_signal0 = pd.DataFrame(
            [[codes[0], 0.95, 0.95, 0.6, 0.1, -0.03],  # Sell all
             [codes[1], 1, 1, 0.49, 0.12, -0.02],  # Sell all
             [codes[2], 1.034, 1.034, 0.61, 0.09, -0.01],  # Sell all
             [codes[3], 1.74, 1.74, 0.61, 0.11, -0.04],  # Sell all
             [codes[4], 1.34, 1.34, 0.61, 0.11, -0.04],  # Sell all
             [codes[5], 1.05, 1.05, 0.3, 0.11, -0.04],  # Second buy
             [codes[6], 1.1, 1.1, 0.61, 0.11, -0.1],  # Third buy
             [codes[7], 1.1, 1.1, 0.29, 0.11, -0.11]  # Sell all
             ], columns=columns)

        print("Include any nan in day_signal:",day_signal0.isna().any().any())

        # print(day_signal0)
        prices0 = {code: day_signal0[day_signal0["code"] == code]["qfq_close"].iloc[0]
                   for code in day_signal0["code"]}
        # print(prices0)
        orders = []
        for code in day_signal0["code"]:
            if code in account.stocks:
                o = trader.strategy_for_stck_in_pos(code, account,
                                                    day_signal0)
                if o:
                    orders.append(o)
        expected_orders = [(trading.SELL_FLAG,codes[0],prices0[codes[0]],-20000),
                           (trading.SELL_FLAG, codes[1], prices0[codes[1]],
                            -40000),
                           (trading.SELL_FLAG, codes[2], prices0[codes[2]],
                            -60000),
                           (trading.SELL_FLAG, codes[3], prices0[codes[3]],
                           -60000),
                           (trading.SELL_FLAG, codes[4], prices0[codes[4]],
                           -60000),
                           (trading.BUY_FLAG, codes[5], prices0[codes[5]],
                            20000),
                           (trading.BUY_FLAG, codes[6], prices0[codes[6]],
                            20000),
                           (trading.SELL_FLAG, codes[7], prices0[codes[7]],
                           -60000),
                           ]
        self.assertEqual(expected_orders,orders)

    def test_gen_orders(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)
        columns = ["code", "qfq_close", "f1mv_qfq_open",
                   "y_l_rise", "y_s_rise", "y_s_decline"]

        codes = "600345", "600229", "002345", "002335", "002236", "002217", "300345", "603799"

        account.records[codes[4]] = [1, 1.5, 20000]
        account.records[codes[5]] = [1, 1, 20000]
        account.records[codes[6]] = [1, 1.05, 20000]
        account.records[codes[7]] = [1, 1.1, 20000]
        account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        account.stocks[codes[5]] = {1: 20000}
        account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
        account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        day_signal0 = pd.DataFrame([
            [codes[0], 1, 1.01, 0.6, 0.1, -0.03],
            [codes[1], 1, 0.99, 0.49, 0.12, -0.02],
            [codes[2], 1, 1.01, 0.61, 0.09, -0.01],
            [codes[3], 1, 1.02, 0.61, 0.11, -0.04],
            [codes[4], 1.34, 1.35, 0.61, 0.11, -0.04],  # Sell all
            [codes[5], 1.05, 1.053, 0.3, 0.11, -0.04],   # Second buy
            [codes[6], 1.1, 1.101, 0.61, 0.11, -0.1],     # Third buy
            [codes[7], 1.1, 1.09, 0.29, 0.11, -0.11]     # Sell all
            ], columns=columns)

        prices0 = {
            code: day_signal0[day_signal0["code"] == code][
                "qfq_close"].iloc[0] for code in day_signal0["code"]}
        # print(prices0)

        orders = trader.gen_orders(day_signal0,account)
        expected_orders = [
            (trading.BUY_FLAG, codes[0], prices0[codes[0]], int(622.8)*100),
            (trading.SELL_FLAG, codes[4], prices0[codes[4]], -60000),
            (trading.BUY_FLAG, codes[5], prices0[codes[5]], 20000),
            (trading.BUY_FLAG, codes[6], prices0[codes[6]], 20000),
            (trading.SELL_FLAG, codes[7], prices0[codes[7]], -60000),
        ]
        self.assertEqual(expected_orders,orders)

    def test_exe_orders(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)
        columns = ["code", "qfq_close", "f1mv_qfq_open", "f1mv_qfq_high",
                   "f1mv_qfq_low", "y_l_rise", "y_s_rise", "y_s_decline"]

        codes = "600345", "600229", "002345", "002335", "002236", "002217", "300345", "603799"

        account.records[codes[4]] = [1, 1.5, 20000]
        account.records[codes[5]] = [1, 1, 20000]
        account.records[codes[6]] = [1, 1.05, 20000]
        account.records[codes[7]] = [1, 1.1, 20000]
        account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        account.stocks[codes[5]] = {1: 20000}
        account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
        account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        day_signal0 = pd.DataFrame(
            [[codes[0], 1, 1.01, 1.02, 0.98, 0.6, 0.1, -0.03],
                [codes[1], 1, 0.99, 1.02, 0.98, 0.49, 0.12, -0.02],
                [codes[2], 1, 1.01, 1.03, 0.99, 0.61, 0.09, -0.01],
                [codes[3], 1, 1.02, 1.04, 1.01, 0.61, 0.11, -0.04],
                [codes[4], 1.34, 1.35, 1.36, 1.33, 0.61, 0.11, -0.04],
                # Sell all
                [codes[5], 1.05, 1.053, 1.054, 1.04, 0.3, 0.11, -0.04],
                # Second buy
                [codes[6], 1.1, 1.101, 1.102, 1.09, 0.61, 0.11, -0.1],
                # Third buy
                [codes[7], 1.1, 1.09, 1.101, 1.08, 0.29, 0.11, -0.11]
                # Sell all
            ], columns=columns)
        date_idx = "2018-10-01"
        day_signal0.index = [date_idx]*len(day_signal0)

        orders = trader.gen_orders(day_signal0,account)
        transactions = trader.exe_orders(orders=orders,day_signal=day_signal0,
                           account=account)

        expected_stocks = {codes[0]:{1.01:62200},
                           codes[5]:{1:20000,1.053:20000},
                           codes[6]:{1:20000,1.05:20000,1.101:20000}
                           }
        expected_transactions = [
            [date_idx,codes[0], 1.01, 62200],
            [date_idx, codes[4], 1.35, -60000],
            [date_idx,codes[5], 1.053, 20000],
            [date_idx,codes[6], 1.101, 20000],
            [date_idx, codes[7], 1.09, -60000],
        ]
        self.assertEqual(expected_stocks,account.stocks)
        self.assertEqual(expected_transactions,transactions)

    def test_update_records(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)
        columns = ["code", "qfq_close",
                   "f1mv_qfq_open", "f1mv_qfq_high","f1mv_qfq_low",
                   "y_l_rise","y_s_rise", "y_s_decline"]

        codes = "600345", "600229", "002345", "002335", "002236", "002217", "300345", "603799"

        account.records[codes[4]] = [1, 1.5, 20000]
        account.records[codes[5]] = [1, 1, 20000]
        account.records[codes[6]] = [1, 1.05, 20000]
        account.records[codes[7]] = [1, 1.1, 20000]
        account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        account.stocks[codes[5]] = {1: 20000}
        account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
        account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        day_signal0 = pd.DataFrame([
            [codes[0], 1, 1.01,1.02,0.98, 0.6, 0.1, -0.03],
            [codes[1], 1, 0.99,1.02,0.98, 0.49, 0.12, -0.02],
            [codes[2], 1, 1.01,1.03,0.99, 0.61, 0.09, -0.01],
            [codes[3], 1, 1.02,1.04,1.01, 0.61, 0.11, -0.04],
            [codes[4], 1.34, 1.35,1.36,1.33, 0.61, 0.11, -0.04],  # Sell all
            [codes[5], 1.05, 1.053,1.054,1.04, 0.3, 0.11, -0.04],  # Second buy
            [codes[6], 1.1, 1.101,1.102,1.09, 0.61, 0.11, -0.1],  # Third buy
            [codes[7], 1.1, 1.09,1.101,1.08, 0.29, 0.11, -0.11]  # Sell all
        ], columns=columns)

        # Use columns.difference to make sure not use unnecessary or future
        # information.
        orders = trader.gen_orders(day_signal0[
                                       day_signal0.columns.difference([
                                           "f1mv_qfq_open","f1mv_qfq_high",
                                           "f1mv_qfq_low"
                                       ])], account)
        trader.exe_orders(orders=orders,
                          day_signal=day_signal0,
                          account=account)
        trader.update_records(day_signal=day_signal0[
                              day_signal0.columns.difference([
                                  "f1mv_qfq_open","f1mv_qfq_low"])],account=account)

        expected_stocks = {codes[0]: {1.01: 62200},
                           codes[5]: {1: 20000, 1.053: 20000},
                           codes[6]: {1: 20000, 1.05: 20000, 1.101: 20000}}

        expected_records = {codes[0]: [1.01,1.02, 62200],
                            codes[5]: [1,1.054,20000],
                            codes[6]: [1,1.102,20000]}
        self.assertEqual(expected_records,account.records)

    def test_exe_orders_hit_limit(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)
        columns = ["code", "qfq_close", "f1mv_qfq_open", "f1mv_qfq_high",
                   "f1mv_qfq_low",
                   "y_l_rise", "y_s_rise", "y_s_decline"]

        codes = ["600345", "600229", "002345", "002335", "002236", "002217",
                 "300345", "603799"]
        account.stocks[codes[1]] = {1:20000}
        orders = [(trading.BUY_FLAG,codes[0],1,20000),
                  (trading.SELL_FLAG,codes[1],1,20000)]
        day_signals = pd.DataFrame([
            [codes[0], 1, 1.1,1.1,1.1, 0.6, 0.1, -0.03],
            [codes[1], 1, 0.9,0.9,0.9, 0.49, 0.12, -0.02],
        ], columns=columns)
        trader.exe_orders(orders,day_signal=day_signals,account=account)
        expected_stocks = {codes[1]:{1:20000}}
        self.assertEqual(expected_stocks,account.stocks)

    def test_trade(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)
        columns = ["code", "qfq_close", "f1mv_qfq_open", "f1mv_qfq_high",
                   "f1mv_qfq_low", "y_l_rise", "y_s_rise", "y_s_decline"]

        codes = "600345", "600229", "002345", "002335", "002236", "002217", "300345", "603799"

        account.records[codes[4]] = [1, 1.5, 20000]
        account.records[codes[5]] = [1, 1, 20000]
        account.records[codes[6]] = [1, 1.05, 20000]
        account.records[codes[7]] = [1, 1.1, 20000]
        account.stocks[codes[4]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        account.stocks[codes[5]] = {1: 20000}
        account.stocks[codes[6]] = {1: 20000, 1.05: 20000}
        account.stocks[codes[7]] = {1: 20000, 1.05: 20000, 1.1: 20000}
        day_signal0 = pd.DataFrame(
            [[codes[0], 1, 1.01, 1.02, 0.98, 0.6, 0.1, -0.03],
                [codes[1], 1, 0.99, 1.02, 0.98, 0.49, 0.12, -0.02],
                [codes[2], 1, 1.01, 1.03, 0.99, 0.61, 0.09, -0.01],
                [codes[3], 1, 1.02, 1.04, 1.01, 0.61, 0.11, -0.04],
                [codes[4], 1.34, 1.35, 1.36, 1.33, 0.61, 0.11, -0.04],
                # Sell all
                [codes[5], 1.05, 1.053, 1.054, 1.04, 0.3, 0.11, -0.04],
                # Second buy
                [codes[6], 1.1, 1.101, 1.102, 1.09, 0.61, 0.11, -0.1],
                # Third buy
                [codes[7], 1.1, 1.09, 1.101, 1.08, 0.29, 0.11, -0.11]
                # Sell all
            ], columns=columns)

        # Use columns.difference to make sure not use unnecessary or future
        # information.
        trader.trade(day_signal=day_signal0,account=account)

        expected_stocks = {codes[0]: {1.01: 62200},
                           codes[5]: {1: 20000, 1.053: 20000},
                           codes[6]: {1: 20000, 1.05: 20000, 1.101: 20000}}

        expected_records = {codes[0]: [1.01, 1.02, 62200],
                            codes[5]: [1, 1.054, 20000],
                            codes[6]: [1, 1.102, 20000]}
        self.assertEqual(expected_stocks,account.stocks)
        self.assertEqual(expected_records, account.records)


if __name__ == '__main__':
    unittest.main()
