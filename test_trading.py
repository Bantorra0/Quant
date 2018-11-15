import copy
import unittest

import pandas as pd

import constants as const
import trading


class TraderTestCase(unittest.TestCase):
    def setUp(self):
        self.trader = trading.Trader()

    def test_exe_single_order(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        # Case that cnt=0.
        order = ["600345",1.5,0]
        trader.exe_single_order(*order,account=account)
        expected_stocks = {}
        expected_cash = amt
        self.assertEqual(expected_cash,account.cash)
        self.assertEqual(expected_stocks, account.stocks)

        # Case that the buying stock is not in account.stocks.
        order = ["600345", 1.5, 10000]
        trader.exe_single_order(*order, account=account)
        expected_stocks = {"600345":{1.5:10000}}
        expected_cash -= order[1]*order[2]
        self.assertEqual(expected_cash, account.cash)
        self.assertEqual(expected_stocks, account.stocks)

        # Case that the buying or selling stock is in account.stocks.keys(), but price not in stocks[code].keys()
        order = ["600345", 1.6, 10000]
        trader.exe_single_order(*order, account=account)
        expected_stocks = {"600345":{1.5:10000, 1.6:10000}}
        expected_cash -= order[1] * order[2]
        self.assertEqual(expected_cash, account.cash)
        self.assertEqual(expected_stocks, account.stocks)

        # Case that the buying or selling stock is in account.stocks.keys(), and price in stocks[code].keys().
        order = ["600345", 1.6, 5000]
        trader.exe_single_order(*order, account=account)
        expected_stocks = {"600345": {1.5: 10000, 1.6: 15000}}
        expected_cash -= order[1] * order[2]
        self.assertEqual(expected_cash, account.cash)
        self.assertEqual(expected_stocks, account.stocks)

        # Case that selling stock is in account.stocks.keys(),
        # and price in stocks[code].keys()
        # and sells exact the same value, so the price is no longer a key.
        order = ["600345", 1.5, -10000]
        trader.exe_single_order(*order, account=account)
        expected_stocks = {"600345": {1.6: 15000}}
        expected_cash -= order[1] * order[2]
        self.assertEqual(expected_cash, account.cash)
        self.assertEqual(expected_stocks, account.stocks)

        # Case that sells a stock, so that no price is in stocks[code].keys().
        order = ["600345", 1.6, -15000]
        trader.exe_single_order(*order, account=account)
        expected_stocks = {}
        expected_cash -= order[1] * order[2]
        self.assertEqual(expected_cash, account.cash)
        self.assertEqual(expected_stocks, account.stocks)

        # Case that buys and sells a stock, so that the holding shares is 0.
        # Integrated test.
        orders = [["600345", 1.4, 6000],
                  ["600345", 1.4, 0],
                  ["600345", 1.6, 4000],
                  ["600345", 1.5, 5000],
                  ["600345", 1.5, -5000],
                  ["600345", 1.7, -10000]]
        for o in orders:
            trader.exe_single_order(*o, account=account)
            expected_cash -= o[1] * o[2]
        expected_stocks = {}
        self.assertEqual(expected_cash, account.cash)
        self.assertEqual(expected_stocks, account.stocks)


    def test_buy_by_cnt(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)
        expected_account = trading.Account(init_amt=amt)

        # Case that cnt<0.
        order = ["600345", 1.6, -15000]
        paras = *order,account
        self.assertRaises(ValueError,trader.buy_by_cnt,*paras)
        self.assertEqual(amt,account.cash)
        self.assertEqual({},account.stocks)

        # Case that cnt=0.
        order = ["600345", 1.6, 0]
        self.assertIsNone(trader.buy_by_cnt(*order,account))
        self.assertEqual(amt,account.cash)
        self.assertEqual({},account.stocks)

        # Case that buy_max is False and cash is not enough.
        order = ["600345", 1.6, 100000]
        self.assertIsNone(trader.buy_by_cnt(*order, account, buy_max=False))
        self.assertEqual(amt, account.cash)
        self.assertEqual({}, account.stocks)

        # Case that buy_max is True and cash is not enough.
        order = ["600345", 1.6, 100000]
        expected_t = [const.BUY_FLAG,"600345", 1.6, 62500]
        t = trader.buy_by_cnt(*order, account, buy_max=True)
        trader.buy_by_cnt(*expected_t[1:],account=expected_account)
        self.assertEqual(expected_t,t)
        self.assertEqual(expected_account.cash, account.cash)
        self.assertEqual(expected_account.stocks, account.stocks)


    def test_sell_by_cnt(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)
        expected_account = trading.Account(init_amt=amt)

        # Case that cnt>0.
        order = ["600345", 1.6, 15000]
        paras = *order, account
        self.assertRaises(ValueError, trader.sell_by_cnt, *paras)
        self.assertEqual(amt, account.cash)
        self.assertEqual({}, account.stocks)

        # Case that cnt=0.
        order = ["600345", 1.6, 0]
        self.assertIsNone(trader.sell_by_cnt(*order, account))
        self.assertEqual(amt, account.cash)
        self.assertEqual({}, account.stocks)

        # Case that selling a stock not in holding stocks.
        order = ["600345", 1.6, -10000]
        paras = *order, account
        self.assertRaises(ValueError, trader.sell_by_cnt, *paras)
        self.assertEqual(amt, account.cash)
        self.assertEqual({}, account.stocks)

        # Case that selling shares of a stock is more than holding shares.
        order = ["600345", 1.6, -10000]
        stocks = {"600345":{1.5:2000,1.4:3000}}
        account.stocks = stocks.copy()
        expected_account.stocks = stocks.copy()
        paras = *order, account
        self.assertRaises(ValueError, trader.sell_by_cnt, *paras)
        self.assertEqual(expected_account.cash, account.cash)
        self.assertEqual(expected_account.stocks, account.stocks)

        # Case of normal selling.
        order = ["600345", 1.6, -4000]
        stocks = {"600345": {1.5: 2000, 1.4: 3000}}
        account.stocks, account.cash = copy.deepcopy(stocks), amt
        expected_account.stocks, expected_account.cash = copy.deepcopy(stocks), amt
        paras = *order, account
        t = trader.sell_by_cnt(*paras)
        expected_t = [const.SELL_FLAG,"600345", 1.6, -4000]
        trader.sell_by_cnt(*expected_t[1:],expected_account)
        self.assertEqual(expected_t,t)
        self.assertEqual(expected_account.cash, account.cash)
        self.assertEqual(expected_account.stocks, account.stocks)

    def test_tot_amt(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        stocks = {"600345":{24.5:300},
                  "002345":{11.2:500},
                  "600229":{9.62:1200,9.32:100}}
        account.stocks = stocks
        current_pos = {"600345":300, "002345":500,"600229":1300}
        prices = {"600345":25, "002345":12.1,"600229":9.5}
        self.assertEqual(amt+sum([prices[code]*current_pos[code] for code in
                              current_pos]),
                         trader.tot_amt(account,prices))

    def test_get_cnt_from_percent(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        stocks = {"600345": {24.5: 300},
                  "002345": {11.2: 500},
                  "600229": {9.62: 1200, 9.32: 100}}
        account.stocks = stocks
        current_pos = {"600345": 300, "002345": 500, "600229": 1300}
        prices = {"600345": 25, "002345": 12.1, "600229": 9.5}

        cnt = trader.get_cnt_from_percent(0.2, 3.5, account,prices)
        self.assertEqual(7100,cnt)

    def test_order_buy_pct(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        stocks = {"600345": {24.5: 300},
                  "002345": {11.2: 500},
                  "600229": {9.62: 1200, 9.32: 100}}
        account.stocks = stocks
        current_pos = {"600345": 300, "002345": 500, "600229": 1300}
        prices = {"600345": 25, "002345": 12.1, "600229": 9.5}

        # Case that percent>100%.
        self.assertRaises(ValueError, trader.order_buy_pct,*("600345",1.1,25,account,prices))

        # Normal case.
        expected_o = [const.BUY_FLAG,"002359",1.5,16700]
        self.assertEqual(expected_o, trader.order_buy_pct("002359",0.2, 1.5,account,prices))

    def test_order_sell_by_stck_pct(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        codes = "600345", "600229", "002345"
        stocks = {"600345": {24.5: 300}, "002345": {11.2: 500},
                  "600229": {9.62: 1200, 9.32: 100}}
        account.stocks = stocks

        # Case that percent>100%.
        self.assertRaises(ValueError,trader.order_sell_by_stck_pct,
                          *(codes[0],1.1,25,account))

        # Case that percent = 100%.
        order = trader.order_sell_by_stck_pct(codes[0],1,25,account)
        expected_order = [const.SELL_FLAG, codes[0],25,-300]
        self.assertEqual(expected_order,order)

        # Normal case,
        order = trader.order_sell_by_stck_pct(codes[1], 0.5, 9.5, account)
        expected_order = [const.SELL_FLAG, codes[1], 9.5, -600]
        self.assertEqual(expected_order, order)

    def test_plan_for_stck_not_in_pos(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        columns = ["code", "qfq_close","y_l_rise", "y_s_rise", "y_s_decline"]

        codes = "600345", "600229", "002345", "002335"

        # Initial buy conditions combine three simple conditions.
        # Test data below cover a case satisfying three conditions,
        # and three cases each failed with one condition.
        day_signal0 = pd.DataFrame([[codes[0], 1, 0.6, 0.1, -0.03],
                                    [codes[1], 1.1, 0.49, 0.12,-0.02],
                                    [codes[2], 0.9, 0.61, 0.07, -0.01],
                                    [codes[3], 1.5, 0.61, 0.11, -0.05]],
                                   columns=columns)

        expected_plans = [[[const.BUY_FLAG,codes[0], "open", 20000]]]
        plans = []
        for c in codes:
            p = trader.plan_for_stck_not_in_pos(c,account,day_signal0)
            if p:
                plans.append(p)
        self.assertEqual(expected_plans,plans)

    def test_plan_for_stck_in_pos(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        columns = ["code", "qfq_close", "y_l_rise", "y_s_rise", "y_s_decline"]
        codes = "600345", "600229", "002345", "002236", "002217", \
                "300345", "603799"

        # Case when account.records is not updated properly.
        account.records = {codes[0]: [1, -1, 20000]}
        day_signal0 = pd.DataFrame(
            [[codes[0], 1.34, 0.6, 0.11, -0.01]
             ], columns=columns)
        self.assertRaises(ValueError, trader.plan_for_stck_in_pos, *(codes[0], account, day_signal0))

        # Case 0: the stock decline significantly, over 10% from the peak.
        # Case 1: the stock decline significantly, over 25% rise lost.
        # Case 2: the model signal for the given stock is bad, close out position.
        # Case 3: first commitment.
        # Case 4: second long commitment.
        # Case 5: third long commitment.
        records = {
            codes[0]: [1, 1.5, 20000],
            codes[1]: [1, 2, 20000],
            codes[2]: [1, 2, 20000],
            codes[3]: [1, 1.04, 20000],
            codes[4]: [3.5, 3.7, 20000],
            codes[5]: [3.5, 3.9, 20000]
        }
        stocks = {
            codes[0]: {1: 20000},
            codes[1]: {1: 20000},
            codes[2]: {1: 20000},
            codes[3]: {1: 20000},
            codes[4]: {3.5: 20000, 3.68: 20000},
            codes[5]: {3.5: 20000, 3.68: 20000, 3.88: 20000}
        }
        day_signal0 = pd.DataFrame(
            [[codes[0], 1.34, 0.6, 0.11, -0.01],
             [codes[1], 1.74, 0.6, 0.11, -0.01],
             [codes[2], 1.98, 0.29, 0.11, -0.11],
             [codes[3], 1.03, 0.6, 0.11, -0.11],
             [codes[4], 3.69, 0.28, 0.11, -0.01],
             [codes[5], 3.66, 0.6, 0.11, -0.01]
             ], columns=columns)
        expected_plan = [
            [[const.SELL_FLAG, codes[0], "open", -20000]],
            [[const.SELL_FLAG, codes[1], "open", -20000]],
            [[const.SELL_FLAG, codes[2], "open", -20000]],
            [[const.SELL_FLAG, codes[3], 0.95, -20000],
             [const.BUY_FLAG, codes[3], 1.05, 20000]],
            [[const.SELL_FLAG, codes[4], 3.5, -40000],
             [const.BUY_FLAG, codes[4], 3.85, 20000]],
            [[const.SELL_FLAG, codes[5], 3.63, -60000]]
        ]

        for i in range(len(day_signal0)):
            account.records = {codes[i]:records[codes[i]]}
            account.stocks = {codes[i]:stocks[codes[i]]}
            plan = trader.plan_for_stck_in_pos(codes[i],account,day_signal0.iloc[[i]])
            self.assertEqual(expected_plan[i],plan)

    def test_gen_trading_plan(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        columns = ["code", "qfq_close", "y_l_rise", "y_s_rise", "y_s_decline"]
        codes = "600345", "600229", "002345", "002236", "002217", "300345", "603799", "002359"

        # Integrated test.
        account.records = {codes[0]: [1, 1.5, 20000],
                           codes[1]: [1, 2, 20000],
                           codes[2]: [1, 1.04, 20000],
                           codes[3]: [3.5, 3.7, 20000],
                           codes[4]: [3.5, 3.9, 20000]}

        account.stocks = {codes[0]: {1: 20000},
                          codes[1]: {1: 20000},
                          codes[2]: {1: 20000},
                          codes[3]: {3.5: 20000, 3.68: 20000},
                          codes[4]: {3.5: 20000, 3.68: 20000, 3.88: 20000}}

        day_signal0 = pd.DataFrame([
            [codes[0], 1.34, 0.6, 0.11, -0.01],
            [codes[1], 1.74, 0.6, 0.11, -0.01],
            [codes[2], 1.03, 0.6, 0.11, -0.01],
            [codes[3], 3.69, 0.6, 0.11, -0.01],
            [codes[4], 3.66, 0.6, 0.11, -0.01],
            [codes[5], 1.1, 0.49, 0.12, -0.02],
            [codes[6], 0.9, 0.61, 0.07, -0.01],
            [codes[7], 1.5, 0.61, 0.11, -0.05]
        ], columns=columns)

        expected_plan = [
            [[const.SELL_FLAG, codes[0], "open", -20000]],
            [[const.SELL_FLAG, codes[1], "open", -20000]],
            [[const.SELL_FLAG, codes[2], 0.95, -20000],
             [const.BUY_FLAG, codes[2], 1.05, 20000]],
            [[const.SELL_FLAG, codes[3], 3.5, -40000],
             [const.BUY_FLAG, codes[3], 3.85, 20000]],
            [[const.SELL_FLAG, codes[4], 3.63, -60000]]
        ]

        plan = trader.gen_trading_plan(day_signal0,account)
        self.assertEqual(expected_plan, plan)

    def test_gen_orders_from_plan(self):
        trader = self.trader
        amt = 100000

        columns = ["code", "qfq_open","qfq_high","qfq_low","qfq_close","amt","change_rate_p1mv_close"]
        codes = "600345", "600229", "002345", "002236", "002217", "300345", \
                "603799","600703","002359"

        # Case 0: trading amount = 0.
        # Case 1: trading qfq_high= qfq_low.
        # Case 2: normal selling at qfq_open.
        # Case 3: normal buying at qfq_open.
        # Case 4: buying fails due to reaching rise limit.
        # Case 5: selling fails due to reaching decline limit.
        # Case 6: normal selling.
        # Case 7: normal buying.
        # Case 8: empty plan
        day_signal0 = pd.DataFrame([[codes[0], 1.34, 1.39, 1.33, 1.35, 0, 0.05],
                                    [codes[1], 1.74, 1.74, 1.74, 1.74, 100, 0.05],
                                    [codes[2], 3.8, 3.89, 3.79, 3.87, 3600,
                                     0.05],
                                    [codes[3], 3.78, 3.89, 3.79, 3.87, 3600,
                                     -0.05],
                                    [codes[4], 1.03, 1.02, 1.09, 1.09, 1000,
                                     -0.09],
                                    [codes[5], 3.6, 3.6, 3.4, 3.4, 2000, 0.108],
                                    [codes[6], 3.66, 3.67, 3.59, 3.60, 2600,
                                     0.03],
                                    [codes[7], 3.8, 3.89, 3.79, 3.87, 3600,
                                     -0.05],
                                    [codes[8], 3.8, 3.89, 3.79, 3.87, 3600,
                                     -0.05],
                                    ],
                                   columns=columns)
        plan = [[[const.SELL_FLAG, codes[0], 1.35, -20000],
                 [const.BUY_FLAG, codes[0], 1.35, -20000]],
                [[const.SELL_FLAG, codes[1], 1.74, -20000],
                 [const.BUY_FLAG, codes[1], 1.69, -20000]],
                [[const.SELL_FLAG, codes[2], "open", -40000],
                 [const.BUY_FLAG, codes[2], 3.85, 20000]],
                [[const.SELL_FLAG, codes[3], 3.6, -40000],
                 [const.BUY_FLAG, codes[3], "open", 20000]],
                [[const.SELL_FLAG, codes[4], 0.95, -20000],
                 [const.BUY_FLAG, codes[4], 1.05, 20000]],
                [[const.SELL_FLAG, codes[5], 3.5, -40000],
                 [const.BUY_FLAG, codes[5], 3.85, 20000]],
                [[const.SELL_FLAG, codes[6], 3.63, -60000]],
                [[const.SELL_FLAG, codes[7], 3.5, -40000],
                 [const.BUY_FLAG, codes[7], 3.85, 20000]],
                [],
                ]
        expected_orders = [
            [],
            [],
            [const.SELL_FLAG, codes[2], 3.8, -40000],
            [const.BUY_FLAG, codes[3], 3.78, 20000],
            [],
            [],
            [const.SELL_FLAG, codes[6], 3.60, -60000],
            [const.BUY_FLAG, codes[7], 3.87, 20000],
            []
        ]

        for i in range(len(plan)):
            orders = trader.gen_orders_from_plan([plan[i]], day_signal0.iloc[[i]])
            self.assertEqual([expected_orders[i]] if expected_orders[i] else [], orders)

        # Integrated testcase.
        orders = trader.gen_orders_from_plan(plan,day_signal0)
        self.assertEqual([o for o in expected_orders if o], orders)

    def test_exe_orders(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        columns = ["code", "qfq_open", "qfq_high", "qfq_low", "qfq_close",
                   "amt", "change_rate_p1mv_close"]
        codes = "600345", "600229", "002345", "002236", "002217", "300345", "603799", "600703"

        # Case 0: buy cnt=0, no transaction executed.
        # Case 1: no enough cash, no transaction executed.
        # Case 2: sell cnt=0, no transaction executed.
        # Case 3: normal sell.
        # Case 4: normal buy.
        # Case 5: normal sell.
        # Case 6: normal buy.
        orders = [
            [],
            [const.BUY_FLAG, codes[0], 1.5, 0],
            [const.BUY_FLAG, codes[1], 1.5, 100000],
            [const.SELL_FLAG, codes[2], 1.5, 0],
            [const.SELL_FLAG, codes[3], 3.8, -40000],
            [const.BUY_FLAG, codes[4], 3.78, 20000],
            [const.SELL_FLAG, codes[5], 3.60, -60000],
            [const.BUY_FLAG, codes[6], 3.87, 20000],
        ]
        account.stocks = {codes[0]: {1: 20000},
                          codes[1]: {1: 20000},
                          codes[2]: {1: 20000},
                          codes[3]: {3.5: 20000, 3.68: 20000},
                          codes[4]: {3.5: 20000},
                          codes[5]: {3.5: 20000, 3.68: 20000, 3.88: 20000},
                          codes[6]: {3.5: 20000},
                          }
        day_signal0 = pd.DataFrame([[codes[0], 1.34, 1.39, 1.33, 1.35, 0, 0.05],
                                    [codes[1], 1.74, 1.74, 1.74, 1.74, 100, 0.05],
                                    [codes[2], 3.8, 3.89, 3.79, 3.87, 3600,
                                     0.05],
                                    [codes[3], 3.78, 3.89, 3.79, 3.87, 3600,
                                     -0.05],
                                    [codes[4], 1.03, 1.02, 1.09, 1.09, 1000,
                                     -0.09],
                                    [codes[5], 3.6, 3.6, 3.4, 3.4, 2000, 0.108],
                                    [codes[6], 3.66, 3.67, 3.59, 3.60, 2600,
                                     0.03],
                                    ],
                                    columns=columns)
        date = "2018-11-11"
        day_signal0.index = [date]*len(day_signal0)
        expected_transactions = [
            [date, codes[3], 3.8, -40000],
            [date, codes[4], 3.78, 20000],
            [date, codes[5], 3.60, -60000],
            [date, codes[6], 3.87, 20000],
        ]

        transactions = trader.exe_orders(orders,day_signal0,account)
        self.assertEqual(expected_transactions,transactions)

    def test_update_records(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        columns = ["code", "qfq_open", "qfq_high", "qfq_low", "qfq_close",
                   "amt", "change_rate_p1mv_close"]
        codes = "600345", "600229", "002345", "002236", "002217", "300345", "603799", "600703"

        day_signal0 = pd.DataFrame([[codes[0], 1.34, 1.39, 1.33, 1.35, 0, 0.05],
                                    [codes[1], 1.34, 1.39, 1.33, 1.35, 0, 0.05],
                                    [codes[2], 1.34, 1.39, 1.33, 1.35, 0, 0.05],
                                    [codes[3], 3.8, 3.89, 3.79, 3.87, 3600,
                                     0.05],
                                    [codes[4], 3.78, 3.89, 3.79, 3.87, 3600,
                                     -0.05],
                                    [codes[5], 1.03, 1.09, 1.01, 1.05, 1000,
                                     -0.09]],columns = columns)

        # Case 0: records[code] == {} and stocks[code] is abnormal.
        stocks_list = [
            {codes[0]: None},
            {codes[0]: {}},
            {codes[0]: {1: 20000, 1.05: 20000}},
        ]
        records_list = [{}]*len(stocks_list)
        for stocks, records in zip(stocks_list,records_list):
            account.stocks = stocks
            account.records = records
            self.assertRaises(ValueError, trader.update_records, *(day_signal0.iloc[[0]],account))

        # Case 1: code in stocks but not in records after buying a new stock,
        #   add to records.
        # Case 2: code in records but not in stocks after selling all shares of that stock,
        #   delete this record.
        # Case 3: code in records and stocks, but stocks[code] is empty,
        #   delete this record.
        # Case 4: code in records and stocks, but holding shares in stocks[code] is 0.
        #   delete this record.
        # Case 5: code in both records and stocks,
        #   update info.
        stocks_list = [
            [codes[1],{1.34: 20000}],
            [],
            [codes[3],{}],
            [codes[4],{3.5:20000,3.7:20000,3.85:-40000}],
            [codes[5],{0.9: 20000, 0.95: 20000, 1.0: 20000}],
        ]
        records_list = [
            [],
            [codes[2], [1.1, 1.4,20000]],
            [codes[3], [3.5, 3.9, 20000]],
            [codes[4], [3.2, 3.9, 20000]],
            [codes[5], [0.9, 1.02, 20000]]
        ]
        expected_records_list = [
            [codes[1], [1.34, 1.39, 20000]],
            [],
            [],
            [],
            [codes[5], [0.9, 1.09, 20000]],
        ]
        for stocks,records,expected_records in zip(stocks_list,records_list,expected_records_list):
            account.stocks = {} if not stocks else {stocks[0]:stocks[1]}
            account.records = {} if not records else {records[0]:records[1]}
            expected_records = {} if not expected_records else {expected_records[0]:expected_records[1]}
            trader.update_records(day_signal0,account)
            self.assertEqual(expected_records, account.records)

        # Integrated test.
        account.stocks = {stocks[0]:stocks[1] for stocks in stocks_list if stocks}
        account.records = {records[0]:records[1] for records in records_list if records}
        expected_records = {expected_records[0]:expected_records[1]
                            for expected_records in expected_records_list if expected_records}
        trader.update_records(day_signal0,account)
        self.assertEqual(expected_records,account.records)


if __name__ == '__main__':
    unittest.main()
