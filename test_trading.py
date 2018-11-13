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


    def test_plan_for_stck_in_pos(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        columns = ["code", "qfq_close", "y_l_rise", "y_s_rise", "y_s_decline"]
        codes = "600345", "600229", "002345", "002236", "002217", \
                "300345", "603799"

        day_signal0 = pd.DataFrame([[codes[0], 1, 0.6, 0.1, -0.03],
                                    [codes[1], 1.1, 0.49, 0.12, -0.02],
                                    [codes[2], 0.9, 0.61, 0.07, -0.01],
                                    [codes[3], 1.5, 0.61, 0.11, -0.05]],
                                   columns=columns)

        # Case when account.records is not updated properly.
        account.records = {codes[0]:[1,-1,20000]}
        self.assertRaises(ValueError, trader.plan_for_stck_in_pos,*(codes[0],account,day_signal0))

        # Case that the stock decline significantly, over 10% from the peak.
        account.records = {codes[0]:[1,1.5,20000]}
        account.stocks = {codes[0]:{1:20000}}
        day_signal0 = pd.DataFrame([[codes[0], 1.34, 0.6, 0.11, -0.01]],
                                   columns=columns)
        expected_plan = [[const.SELL_FLAG,codes[0],"open",-20000]]
        plan = trader.plan_for_stck_in_pos(codes[0],account,day_signal0)
        self.assertEqual(expected_plan,plan)

        # Case that the stock decline significantly, over 25% rise lost.
        account.records = {codes[1]: [1, 2, 20000]}
        account.stocks = {codes[1]: {1: 20000}}
        day_signal0 = pd.DataFrame([[codes[1], 1.74, 0.6, 0.11, -0.01]],
                                   columns=columns)
        expected_plan = [[const.SELL_FLAG, codes[1], "open", -20000]]
        plan = trader.plan_for_stck_in_pos(codes[1], account, day_signal0)
        self.assertEqual(expected_plan, plan)

        # Case after first commitment.
        account.records = {codes[2]: [1, 1.04, 20000]}
        account.stocks = {codes[2]: {1: 20000}}
        day_signal0 = pd.DataFrame([[codes[2], 1.03, 0.6, 0.11, -0.01]],
                                   columns=columns)
        expected_plan = [[const.SELL_FLAG,codes[2], 0.95,-20000],
                         [const.BUY_FLAG, codes[2],1.05,20000]]
        plan = trader.plan_for_stck_in_pos(codes[2],account,day_signal0)
        self.assertEqual(expected_plan,plan)

        # Case after second long commitment.
        account.records = {codes[3]: [3.5, 3.7, 20000]}
        account.stocks = {codes[3]: {3.5: 20000, 3.68:20000}}
        day_signal0 = pd.DataFrame([[codes[3], 3.69, 0.6, 0.11, -0.01]],
                                   columns=columns)
        expected_plan = [[const.SELL_FLAG, codes[3],3.5 , -40000],
                         [const.BUY_FLAG, codes[3], 3.85, 20000]]
        plan = trader.plan_for_stck_in_pos(codes[3], account, day_signal0)
        self.assertEqual(expected_plan, plan)

        # Case after third long commitment.
        account.records = {codes[4]: [3.5, 3.9, 20000]}
        account.stocks = {codes[4]: {3.5: 20000, 3.68: 20000, 3.88:20000}}
        day_signal0 = pd.DataFrame([[codes[4], 3.66, 0.6, 0.11, -0.01]],
                                   columns=columns)
        expected_plan = [[const.SELL_FLAG, codes[4], 3.63, -60000]]
        plan = trader.plan_for_stck_in_pos(codes[4], account, day_signal0)
        self.assertEqual(expected_plan, plan)


    def test_gen_trading_plan(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        columns = ["code", "qfq_close", "y_l_rise", "y_s_rise", "y_s_decline"]
        codes = "600345", "600229", "002345", "002236", "002217", "300345", "603799"

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

        day_signal0 = pd.DataFrame([[codes[0], 1.34, 0.6, 0.11, -0.01],
            [codes[1], 1.74, 0.6, 0.11, -0.01],
            [codes[2], 1.03, 0.6, 0.11, -0.01],
            [codes[3], 3.69, 0.6, 0.11, -0.01],
            [codes[4], 3.66, 0.6, 0.11, -0.01]], columns=columns)

        expected_plan = [[[const.SELL_FLAG, codes[0], "open", -20000]],
            [[const.SELL_FLAG, codes[1], "open", -20000]],
            [[const.SELL_FLAG, codes[2], 0.95, -20000],
             [const.BUY_FLAG, codes[2], 1.05, 20000]],
            [[const.SELL_FLAG, codes[3], 3.5, -40000],
             [const.BUY_FLAG, codes[3], 3.85, 20000]],
            [[const.SELL_FLAG, codes[4], 3.63, -60000]]]

        plan = trader.gen_trading_plan(day_signal0,account)
        self.assertEqual(expected_plan, plan)

    def test_gen_orders_from_plan(self):
        trader = self.trader
        amt = 100000

        columns = ["code", "qfq_open","qfq_high","qfq_low","qfq_close","amt","change_rate_p1mv_close"]
        codes = "600345", "600229", "002345", "002236", "002217", "300345", \
                "603799","600703"

        plan = [[[const.SELL_FLAG, codes[0], 1.35, -20000],
                 [const.BUY_FLAG, codes[0], 1.35, -20000]],
                [[const.SELL_FLAG, codes[1], 1.74, -20000],
                 [const.BUY_FLAG, codes[1], 1.69, -20000]],
                [[const.SELL_FLAG, codes[2], 0.95, -20000],
                 [const.BUY_FLAG, codes[2], 1.05, 20000]],
                [[const.SELL_FLAG, codes[3], 3.5, -40000],
                 [const.BUY_FLAG, codes[3], 3.85, 20000]],
                [[const.SELL_FLAG, codes[4], 3.63, -60000]],
                [[const.SELL_FLAG, codes[5], 3.5, -40000],
                 [const.BUY_FLAG, codes[5], 3.85, 20000]],
                [[const.SELL_FLAG, codes[6], "open", -40000],
                 [const.BUY_FLAG, codes[6], 3.85, 20000]],
                [[const.SELL_FLAG, codes[7], 3.6, -40000],
                 [const.BUY_FLAG, codes[7], "open", 20000]],
                ]

        day_signal0 = pd.DataFrame([[codes[0], 1.34, 1.39,1.33,1.35,0,0.05],
                                    [codes[1], 1.74, 1.74, 1.74, 1.74, 100,0.05],
                                    [codes[2], 1.03, 1.02, 1.09, 1.09, 1000,
                                     -0.09],
                                    [codes[3], 3.6, 3.6,3.4,3.4,2000,0.108],
                                    [codes[4], 3.66, 3.67,3.59,3.60, 2600,
                                     0.03],
                                    [codes[5], 3.8, 3.89, 3.79, 3.87, 3600,
                                     -0.05],
                                    [codes[6], 3.8, 3.89, 3.79, 3.87, 3600,
                                     0.05],
                                    [codes[7], 3.78, 3.89, 3.79, 3.87, 3600,
                                     -0.05],
                                    []
                                    ],
                                   columns=columns)

        # Case that trading amount = 0.
        orders = trader.gen_orders_from_plan([plan[0]],day_signal0.iloc[[0]])
        expected_orders = []
        self.assertEqual(expected_orders,orders)

        # Case that trading qfq_high= qfq_low.
        orders = trader.gen_orders_from_plan([plan[1]], day_signal0.iloc[[1]])
        expected_orders = []
        self.assertEqual(expected_orders, orders)

        # Case of normal selling at qfq_open.
        orders = trader.gen_orders_from_plan([plan[6]], day_signal0.iloc[[6]])
        expected_orders = [[const.SELL_FLAG, codes[6], 3.8, -40000]]
        self.assertEqual(expected_orders, orders)

        # Case of normal selling at qfq_open.
        orders = trader.gen_orders_from_plan([plan[7]], day_signal0.iloc[[7]])
        expected_orders = [[const.BUY_FLAG, codes[7], 3.78, 20000]]
        self.assertEqual(expected_orders, orders)

        # Case that buying fails due to reaching rise limit.
        orders = trader.gen_orders_from_plan([plan[2]], day_signal0.iloc[[2]])
        expected_orders = []
        self.assertEqual(expected_orders, orders)

        # Case that selling fails due to reaching decline limit.
        orders = trader.gen_orders_from_plan([plan[3]], day_signal0.iloc[[3]])
        expected_orders = []
        self.assertEqual(expected_orders, orders)

        # Case of normal selling.
        orders = trader.gen_orders_from_plan([plan[4]], day_signal0.iloc[[4]])
        expected_orders = [[const.SELL_FLAG,codes[4],3.60,-60000]]
        self.assertEqual(expected_orders, orders)

        # Case of normal buying.
        orders = trader.gen_orders_from_plan([plan[5]], day_signal0.iloc[[5]])
        expected_orders = [[const.BUY_FLAG, codes[5], 3.87, 20000]]
        self.assertEqual(expected_orders, orders)


        # Integrated testcase.
        orders = trader.gen_orders_from_plan(plan,day_signal0)
        expected_orders = [[const.SELL_FLAG,codes[4],3.60,-60000],
                           [const.BUY_FLAG, codes[5], 3.87, 20000],
                           [const.SELL_FLAG, codes[6], 3.8, -40000],
                           [const.BUY_FLAG, codes[7], 3.78, 20000]]
        self.assertEqual(expected_orders, orders)

    def test_exe_orders(self):
        trader = self.trader
        amt = 100000
        account = trading.Account(init_amt=amt)

        columns = ["code", "qfq_open", "qfq_high", "qfq_low", "qfq_close",
                   "amt", "change_rate_p1mv_close"]
        codes = "600345", "600229", "002345", "002236", "002217", "300345", "603799", "600703"

        orders = [[const.SELL_FLAG,codes[4],3.60,-60000],
                           [const.BUY_FLAG, codes[5], 3.87, 20000],
                           [const.SELL_FLAG, codes[6], 3.8, -40000],
                           [const.BUY_FLAG, codes[7], 3.78, 20000]]








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

if __name__ == '__main__':
    unittest.main()
