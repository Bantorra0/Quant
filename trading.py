from collect import stck_pools
from constants import DATE_FORMAT,FEE_RATE, BUY_FLAG,SELL_FLAG
import ml_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import xgboost as xgb
import lightgbm as lgbm


class Account:
    def __init__(self, init_amt=1000000, fee_rate=FEE_RATE):
        self.fee_rate = fee_rate
        self.cash = init_amt
        self.stocks = {}
        self.records = {}


class Trader:
    @staticmethod
    def exe_single_order(code, price, cnt, account: Account):
        """
        Execute order and update account.
        Assume buying when calculating and cnt<0 indicates selling.
        :param code:
        :param cnt:
        :param price:
        :param account:
        :return:
        """
        if cnt == 0:
            return

        account.cash -= cnt * price
        if code not in account.stocks:
            account.stocks[code] = {price: cnt}
        elif price not in account.stocks[code]:
            account.stocks[code][price] = cnt
        else:
            account.stocks[code][price] += cnt
            if account.stocks[code][price] == 0:
                del account.stocks[code][price]

        # Delete a stock key when holding shares of the stock is 0.
        if not account.stocks[code].keys() \
                or sum(account.stocks[code].values()) == 0:
            del account.stocks[code]

    @classmethod
    def buy_by_cnt(cls, code, price, cnt, account: Account, buy_max=False):
        if cnt < 0:
            raise ValueError("Buying cnt {}<0".format(cnt))
        elif cnt == 0:
            return None
        elif not buy_max and cnt * price > account.cash:
            print("Cash {0} yuan is not enough for buying {1} {2} shares with "
                  "price {3}!".format(account.cash, code, cnt, price))
            return None
        elif buy_max and cnt * price > account.cash:
            cnt = int(account.cash / price / 100) * 100

        cls.exe_single_order(code, price, cnt, account)
        return [BUY_FLAG, code, price, cnt]

    @classmethod
    def sell_by_cnt(cls, code, price, cnt, account: Account):
        if cnt > 0:
            raise ValueError("Selling cnt {}>0".format(cnt))
        elif cnt == 0:
            return None
        elif code not in account.stocks or not account.stocks[code].keys():
            raise ValueError("Selling {0} while not having any".format(code))
        else:
            long_pos = sum(account.stocks[code].values())
            if abs(cnt) > long_pos:
                raise ValueError("Selling {0} {1} shares while having only {2}".format(code, abs(cnt), long_pos))
            cls.exe_single_order(code, price, cnt, account)
            return [SELL_FLAG, code, price, cnt]

    @staticmethod
    def tot_amt(account: Account, prices):
        amt = account.cash
        for code, pos in account.stocks.items():
            amt += sum(pos.values()) * prices[code]
        return amt

    @classmethod
    def get_cnt_from_percent(cls, percent, price, account: Account, prices):
        tot_value = cls.tot_amt(account, prices)
        return int(tot_value * percent / 100 / price) * 100

    @classmethod
    def order_buy_pct(cls, code, percent, price, account: Account,
                      prices):
        if percent > 1:
            raise ValueError("Percent {}>100%".format(percent))
        else:
            cnt = cls.get_cnt_from_percent(percent, price, account,
                                           prices)
        return [BUY_FLAG, code, price, cnt]

    @classmethod
    def order_sell_by_stck_pct(cls, code, percent, price, account: Account):
        if percent > 1:
            raise ValueError("Percent {}>100%".format(percent))
        elif percent == 1:
            cnt = int(sum(account.stocks[code].values()))
        else:
            cnt = int(sum(account.stocks[code].values()) * percent / 100) * 100
        return [SELL_FLAG, code, price, -cnt]

    @classmethod
    def plan_for_stck_not_in_pos(cls, code, account: Account, day_signal):
        stock_signal = day_signal[day_signal["code"] == code]
        init_buy_cond = (stock_signal["y_l_rise"] >= 0.2) \
                        & (stock_signal["y_s_decline"] >= -0.04) \
                        & (stock_signal["y_s_rise"] >= 0.06)
        if init_buy_cond.iloc[0]:
            pct = 0.1
            prices = {code: day_signal[day_signal["code"] == code][
                "qfq_close"].iloc[0] for code in day_signal["code"]}

            p = cls.order_buy_pct(code, percent=pct, price=prices[code],
                                     account=account, prices=prices)
            p[2]="open"
            return [p]
        else:
            return None

    @classmethod
    def plan_for_stck_in_pos(cls, code, account: Account, day_signal):
        stock_signal = day_signal[day_signal["code"] == code]

        if account.records[code][1] == -1:
            raise ValueError("Highest price of {} is -1".format(code))
        init_buy_price = account.records[code][0]
        init_buy_cnt = account.records[code][2]
        qfq_close = stock_signal["qfq_close"].iloc[0]

        retracement = (account.records[code][1] - qfq_close)
        sell_cond0 = (retracement >= max(
            (account.records[code][1] - account.records[code][0]) * 0.3,
            account.records[code][1] * 0.15))
        # sell_cond0 = (retracement >= account.records[code][1] * 0.1)

        sell_cond1 = (stock_signal["y_l_rise"].iloc[0] <= 0.1) \
                     & (stock_signal["y_s_decline"].iloc[0] <= -0.08)

        # Generate order based on various conditions.
        plan=None
        if sell_cond0 or sell_cond1:
            o = cls.order_sell_by_stck_pct(code, percent=1,
                                              price=qfq_close, account=account)
            o[2] = "open"
            plan = [o]
        elif sum(account.stocks[code].values()) == init_buy_cnt:
            # Stage after buying first commitment.
            sell_price = round(init_buy_price*0.95,2)
            buy_price = round(init_buy_price*1.05,2)
            plan=[cls.order_sell_by_stck_pct(code, percent=1,
                                             price=sell_price,
                                             account=account)]
            plan.append([BUY_FLAG, code, buy_price, init_buy_cnt])
        elif sum(account.stocks[code].values()) == 2 * init_buy_cnt:
            # Stage after buying second commitment.
            holding_shares = sum([v for v in account.stocks[code].values()])
            cost = sum([k*v for k,v in account.stocks[code].items()])
            sell_price = round(cost * (1-0.025)/holding_shares,2)
            buy_price = round(init_buy_price * 1.1, 2)
            plan=[cls.order_sell_by_stck_pct(code, percent=1,
                                                  price=sell_price,
                                                  account=account)]
            plan.append([BUY_FLAG, code, buy_price, init_buy_cnt])
        elif sum(account.stocks[code].values()) == 3 * init_buy_cnt:
            # Stage after buying third commitment.
            holding_shares = sum([v for v in account.stocks[code].values()])
            cost = sum([k * v for k, v in account.stocks[code].items()])
            sell_price = round(cost * (1 - 0.0166) / holding_shares, 2)
            plan = [cls.order_sell_by_stck_pct(code, percent=1,
                                                  price=sell_price,
                                                  account=account)]
        return plan

    @classmethod
    def gen_trading_plan(cls,day_signal:pd.DataFrame,account:Account):
        plan = []
        # Generate plan.
        for code in day_signal.sort_values(by="y_l_rise",ascending=False)["code"]:
            if code not in account.stocks:
                p = cls.plan_for_stck_not_in_pos(code, account, day_signal)
            else:
                p = cls.plan_for_stck_in_pos(code, account, day_signal)
            if p:
                plan.append(p)
        return plan

    @classmethod
    def gen_orders_from_plan(cls,plan,day_signal):
        orders = []
        for stock_plan in plan:
            if stock_plan:
                code = stock_plan[0][1]
                stock_signal = day_signal[day_signal["code"]==code]
                # 成交量为0，停牌中，
                # 或者一字涨跌停板，无法进行交易。
                if stock_signal["amt"].iloc[0] == 0 or\
                    stock_signal["qfq_high"].iloc[0]==stock_signal[
                    "qfq_low"].iloc[0]:
                    continue
            else:
                continue
            for flag, code, price,cnt in stock_plan:
                if price == "open":
                    price = stock_signal["qfq_open"].iloc[0]
                    orders.append([flag, code, price,cnt])
                    break
                else:
                    qfq_high = stock_signal["qfq_high"].iloc[0]
                    qfq_low = stock_signal["qfq_low"].iloc[0]
                    qfq_close = stock_signal["qfq_close"].iloc[0]

                    if flag == BUY_FLAG and qfq_high > price:
                        if stock_signal[
                                   "change_rate_p1mv_close"].iloc[
                                   0]<-0.089:
                            print("收盘涨停板，加仓失败！")
                        else:
                            orders.append([flag,code,qfq_close,cnt])
                        break
                    elif flag==SELL_FLAG and qfq_low < price:
                        if stock_signal["change_rate_p1mv_close"].iloc[0]>0.107:
                            print("收盘跌停板，平仓失败")
                        else:
                            orders.append([flag, code, qfq_close, cnt])
                        break
        return orders

    @classmethod
    def exe_orders(cls, orders, day_signal, account:Account):
        # Execute orders.
        transactions = []
        date = day_signal.index[0]
        for o in orders:
            if not o:
                continue

            flag, code, price, cnt = o[:4]
            result=None
            if flag==BUY_FLAG:
                result = cls.buy_by_cnt(code,price,cnt,account)
            elif flag == SELL_FLAG:
                result = cls.sell_by_cnt(code,price,cnt,account)

            if result:
                transactions.append([date]+list(result[1:]))
        return transactions

    @classmethod
    def update_records(cls,day_signal,account:Account):
        # Update account.records.
        for code in day_signal["code"]:
            # Add record first if necessary after buying.
            if code not in account.records and code in account.stocks:
                if type(account.stocks[code])==dict and len(account.stocks[code].keys()) == 1:
                    price, cnt = list(account.stocks[code].items())[0]
                    account.records[code] = [price, -1, cnt]
                else:
                    raise ValueError("Inconsistency:{0} not in "
                                     "account.records has "
                                     "multiple transactions in "
                                     "account.stocks".format(code))

            # Delete or update records.
            if code in account.records:
                if code not in account.stocks\
                        or not account.stocks[code].keys()\
                        or sum(account.stocks[code].values()) == 0:
                    # Delete account.records according to account.stocks.
                    del account.records[code]
                else:
                    # Update remaining records.
                    qfq_close = day_signal[day_signal["code"] == code][
                        "qfq_high"].iloc[0]
                    if qfq_close > account.records[code][1]:
                        account.records[code][1] = qfq_close


    @classmethod
    def order_target_percent(cls,code, percent, price, account:Account,
                             prices):
        pass

    # @classmethod
    # def trade(cls,day_signal,account:Account):
    #     orders = cls.gen_orders(day_signal=day_signal,account=account)
    #     transactions = cls.exe_orders_next_morning(orders, day_signal=day_signal,
    #                                                account=account)
    #     orders = [[day_signal.index[0], o[1], o[2],
    #                o[3] if o[0] == BUY_FLAG else -o[3]] for o in orders]
    #     cls.update_records(day_signal=day_signal,account=account)
    #     return orders,transactions
    #
    # @classmethod
    # def trade_with_plan(cls, day_signal, account: Account):
    #     plan = cls.gen_trading_plan(day_signal=day_signal,account=account)
    #     orders = cls.gen_orders_from_plan(plan,day_signal=day_signal)
    #     transactions = cls.exe_orders(orders,day_signal=day_signal,account=account)
    #     orders = [[day_signal.index[0], o[1], o[2],
    #                o[3] if o[0] == BUY_FLAG else -o[3]] for o in orders]
    #     cls.update_records(day_signal=day_signal, account=account)
    #     return orders, transactions


class BackTest:
    def __init__(self, start="2018-01-01", end=None, benchmark='hs300',
                 universe=stck_pools(), capital_base=1000000, freq='d'):
        self.start = start
        self.end = end if end else datetime.datetime.now().strftime(DATE_FORMAT)
        self.universe = universe
        self.capital_base = capital_base
        self.freq = freq  # Time unit of trading frequency
        self.refresh_rate = 1  # Rate of adjusting positions
        self.time_delta = datetime.timedelta(days=self.refresh_rate)
        self.benchmark = benchmark

    def init_account(self, fee_rate=FEE_RATE):
        self.account = Account(self.capital_base, fee_rate)

    def init_trader(self):
        self.trader = Trader()

    def backtest_batch_pred(self, models):
        self.init_account()
        self.init_trader()

        date = datetime.datetime.strptime(self.start,DATE_FORMAT)
        end = datetime.datetime.strptime(self.end,DATE_FORMAT)

        lower_bound = datetime.datetime.strptime(self.start,
                                                 DATE_FORMAT)-750*self.time_delta
        lower_bound = datetime.datetime.strftime(lower_bound,DATE_FORMAT)
        # print(lower_bound, self.start)

        df_all, cols_future = ml_model.gen_data(pred_period=20,
                                                lower_bound= lower_bound,
                                                start=self.start)
        print("df_all:",df_all.shape)

        signals = df_all

        X = ml_model.gen_X(df_all, cols_future)
        signals["y_l_rise"] = models["model_l_high"].predict(X)
        signals["y_s_rise"] = models["model_s_high"].predict(X)
        signals["y_s_decline"] = models["model_s_low"].predict(X)

        day_delta = datetime.timedelta(days=1)

        df_asset_values = pd.DataFrame(columns = ["my_model",self.benchmark])

        orders,transactions, stocks=[],[],{}
        day_plan, day_orders, day_transactions,pos=[],[],[],{}
        while date <= end:
            date_idx = datetime.datetime.strftime(date, DATE_FORMAT)
            # If next day is not a trading date, continue.
            if date_idx not in signals.index:
                date = date + day_delta
                continue

            day_signal = signals.loc[date_idx]
            main_cols = ["qfq_open", "qfq_high", "qfq_low", "qfq_close"]

            # Skip the trading date when there is null in data.
            # May be removed in future, because it is not reasonable.
            if day_signal[main_cols].isna().any().any():
                date = date + day_delta
                continue

            # Execute the trading plan made on previous trading day,
            # including generating and executing orders, updating account information.
            day_orders = self.trader.gen_orders_from_plan(day_plan,day_signal=day_signal)
            day_transactions = self.trader.exe_orders(day_orders, day_signal=day_signal, account=self.account)
            self.trader.update_records(day_signal=day_signal,account=self.account)

            # Save all orders and transactions after adding dates to them.
            if day_orders:
                day_orders = [[date_idx, o[1], o[2], o[3]] for o in day_orders]
                orders.extend(day_orders)
            if day_transactions:
                # v.copy() is necessary, because v is also a dict and will be modified when backtest goes on.
                pos = {code: (day_signal[day_signal["code"] == code][
                                  "qfq_close"].iloc[0], v.copy())
                       for code, v in self.account.stocks.items()}
                stocks[date_idx] = pos
                day_transactions = [[date_idx, t[1], t[2], t[3]] for t in
                                    day_transactions]
                transactions.extend(day_transactions)

            # Calculate the total asset amount of my model and baseline on current trading date,
            # based on qfq closing price.
            prices = {
                code: day_signal[day_signal["code"] == code][
                    "qfq_close"].iloc[0] for code in day_signal["code"]}
            my_model_value = self.trader.tot_amt(account=self.account,
                                                 prices=prices)
            if len(df_asset_values.index) == 0:
                baseline_value = self.capital_base
            else:
                baseline_value = day_signal[self.benchmark+"_close"].iloc[0] / signals.loc[
                    df_asset_values.index.min(), self.benchmark+"_close"].iloc[
                    0] * self.capital_base
            df_asset_values.loc[date_idx] = [my_model_value, baseline_value]


            # Make a plan for the next trading day.
            day_plan = self.trader.gen_trading_plan(day_signal=day_signal,account=self.account)

            date = date + day_delta
        return df_asset_values,orders,transactions,stocks


    def backtest(self, models):
        self.init_account()
        self.init_trader()

        date = datetime.datetime.strptime(self.start,DATE_FORMAT)
        end = datetime.datetime.strptime(self.end,DATE_FORMAT)

        lower_bound = datetime.datetime.strptime(self.start,
                                                 DATE_FORMAT)-750*self.time_delta
        lower_bound = datetime.datetime.strftime(lower_bound,DATE_FORMAT)
        # print(lower_bound, self.start)

        df_all, cols_future = ml_model.gen_data(pred_period=20,
                                                lower_bound= lower_bound,
                                                start=self.start)
        print("df_all:",df_all.shape)

        X = ml_model.gen_X(df_all, cols_future)

        day_delta = datetime.timedelta(days=1)

        df_asset_values = pd.DataFrame(columns = ["my_model",self.benchmark])

        orders,transactions, stocks=[],[],{}
        day_plan, day_orders, day_transactions,pos=[],[],[],{}
        while date <= end:
            date_idx = datetime.datetime.strftime(date, DATE_FORMAT)
            # If next day is not a trading date, continue.
            if date_idx not in df_all.index:
                date = date + day_delta
                continue

            day_signal = df_all.loc[date_idx].copy()
            X_day_slice = X.loc[date_idx]
            day_signal["y_l_rise"] = models["model_l_high"].predict(
                X_day_slice)
            day_signal["y_s_rise"] = models["model_s_high"].predict(
                X_day_slice)
            day_signal["y_s_decline"] = models["model_s_low"].predict(
                X_day_slice)
            main_cols = ["qfq_open", "qfq_high", "qfq_low", "qfq_close"]

            # Skip the trading date when there is null in data.
            # May be removed in future, because it is not reasonable.
            if day_signal[main_cols].isna().any().any():
                date = date + day_delta
                continue

            # Execute the trading plan made on previous trading day,
            # including generating and executing orders, updating account information.
            day_orders = self.trader.gen_orders_from_plan(day_plan,day_signal=day_signal)
            day_transactions = self.trader.exe_orders(day_orders, day_signal=day_signal, account=self.account)
            self.trader.update_records(day_signal=day_signal,account=self.account)

            # Save all orders and transactions after adding dates to them.
            if day_orders:
                day_orders = [[date_idx, o[1], o[2], o[3]] for o in day_orders]
                orders.extend(day_orders)
            if day_transactions:
                # v.copy() is necessary, because v is also a dict and will be modified when backtest goes on.
                pos = {code: (day_signal[day_signal["code"] == code][
                                  "qfq_close"].iloc[0], v.copy())
                       for code, v in self.account.stocks.items()}
                stocks[date_idx] = pos
                day_transactions = [[date_idx, t[1], t[2], t[3]] for t in
                                    day_transactions]
                transactions.extend(day_transactions)

            # Calculate the total asset amount of my model and baseline on current trading date,
            # based on qfq closing price.
            prices = {
                code: day_signal[day_signal["code"] == code][
                    "qfq_close"].iloc[0] for code in day_signal["code"]}
            my_model_value = self.trader.tot_amt(account=self.account,
                                                 prices=prices)
            if len(df_asset_values.index) == 0:
                baseline_value = self.capital_base
            else:
                baseline_value = day_signal[self.benchmark+"_close"].iloc[0] / df_all.loc[
                    df_asset_values.index.min(), self.benchmark+"_close"].iloc[
                    0] * self.capital_base
            df_asset_values.loc[date_idx] = [my_model_value, baseline_value]
            print(date_idx, dict(df_asset_values.loc[date_idx]))


            # Make a plan for the next trading day.
            day_plan = self.trader.gen_trading_plan(day_signal=day_signal,account=self.account)

            date = date + day_delta
        return df_asset_values,orders,transactions,stocks


    def backtest_with_updating_model(self, models, update_frequency=60):
        self.init_account()
        self.init_trader()

        date = datetime.datetime.strptime(self.start,DATE_FORMAT)
        end = datetime.datetime.strptime(self.end,DATE_FORMAT)

        training_bound = datetime.datetime.strptime(self.start,
                                                 DATE_FORMAT) - 1825 * \
                         self.time_delta
        training_bound = datetime.datetime.strftime(training_bound, DATE_FORMAT)

        lower_bound = datetime.datetime.strptime(training_bound,
                                                 DATE_FORMAT)-750*self.time_delta
        lower_bound = datetime.datetime.strftime(lower_bound,DATE_FORMAT)
        # print(lower_bound, self.start)

        df_all, cols_future = ml_model.gen_data(pred_period=20,
                                                lower_bound= lower_bound,
                                                start=training_bound)
        df_all2, cols_future2 = ml_model.gen_data(pred_period=5,
                                                lower_bound=lower_bound,
                                                start=training_bound)
        print("df_all:",df_all.shape)
        trading_date_idxes = df_all.index.unique().sort_values(ascending=False)

        X = ml_model.gen_X(df_all, cols_future)
        df_backtest =df_all[df_all.index>=self.start]
        paras = [("y_l_rise",{"pred_period":20,"is_high":True,"is_clf":False},df_all),
                 ("y_s_rise",{"pred_period":5,"is_high":True,"is_clf":False},df_all2),
                 ("y_s_decline",{"pred_period":5,"is_high":False,"is_clf":False},df_all2),]
        Y = pd.concat([ml_model.gen_y(v2,**v1) for k,v1,v2 in paras],axis=1)
        Y.columns = [k for k,_,_ in paras]
        Y.index = X.index
        ycol_model_dict = {ycol:ycol.replace("y","model")
                                    .replace("rise","high")
                                    .replace("decline","low")
                           for ycol in Y.columns}

        day_delta = datetime.timedelta(days=1)

        df_asset_values = pd.DataFrame(columns = ["my_model",self.benchmark])

        orders,transactions, stocks=[],[],{}
        day_plan, day_orders, day_transactions,pos=[],[],[],{}
        cnt = 0
        while date <= end:
            date_idx = datetime.datetime.strftime(date, DATE_FORMAT)
            # If next day is not a trading date, continue.
            if date_idx not in trading_date_idxes:
                date = date + day_delta
                continue

            day_signal = df_all.loc[date_idx].copy()
            main_cols = ["qfq_open", "qfq_high", "qfq_low", "qfq_close"]
            day_signal = day_signal[(day_signal[main_cols].notnull()).all(
                axis=1)]

            if cnt%update_frequency == 0:
                print("count:",cnt)
                t=time.time()
                for ycol in Y.columns:
                    train_date_idx = trading_date_idxes[
                                         trading_date_idxes<=date_idx][cnt:-25]
                    X_train = X.loc[train_date_idx]
                    y_train = Y.loc[train_date_idx,ycol]
                    print(train_date_idx.shape, X_train.shape, y_train.shape)
                    models[ycol_model_dict[ycol]].fit(X_train,y_train)
                    print("time:",time.time()-t)

            X_day_slice = X.loc[date_idx]
            day_signal["y_l_rise"] = models["model_l_high"].predict(X_day_slice)
            day_signal["y_s_rise"] = models["model_s_high"].predict(X_day_slice)
            day_signal["y_s_decline"] = models["model_s_low"].predict(X_day_slice)
            print(day_signal["y_l_rise"].max(), day_signal["y_s_rise"].max(),
                  day_signal["y_s_decline"].min())

            # Execute the trading plan made on previous trading day,
            # including generating and executing orders, updating account information.
            day_orders = self.trader.gen_orders_from_plan(day_plan,day_signal=day_signal)
            day_transactions = self.trader.exe_orders(day_orders, day_signal=day_signal, account=self.account)
            self.trader.update_records(day_signal=day_signal,account=self.account)

            # Save all orders and transactions after adding dates to them.
            if day_orders:
                day_orders = [[date_idx, o[1], o[2], o[3]] for o in day_orders]
                orders.extend(day_orders)
            if day_transactions:
                # v.copy() is necessary, because v is also a dict and will be modified when backtest goes on.
                pos = {code: (day_signal[day_signal["code"] == code][
                                  "qfq_close"].iloc[0], v.copy())
                       for code, v in self.account.stocks.items()}
                stocks[date_idx] = pos
                day_transactions = [[date_idx, t[1], t[2], t[3]] for t in
                                    day_transactions]
                transactions.extend(day_transactions)

            # Calculate the total asset amount of my model and baseline on current trading date,
            # based on qfq closing price.
            prices = {
                code: day_signal[day_signal["code"] == code][
                    "qfq_close"].iloc[0] for code in day_signal["code"]}
            my_model_value = self.trader.tot_amt(account=self.account,
                                                 prices=prices)
            if len(df_asset_values)==0:
                baseline_value = self.capital_base
            else:
                # baseline_value = day_signal[self.benchmark+"_close"].iloc[0] / df_all.loc[
                #     df_asset_values.index.min(), self.benchmark+"_close"].iloc[
                #     0] * self.capital_base

                baseline_val_list = []
                for code in day_signal["code"]:
                    day_stock_signal = day_signal[day_signal["code"]==code]
                    df_backtest_stock = df_backtest[df_backtest["code"]==code]
                    start_idx = df_backtest_stock[
                        "qfq_close"].first_valid_index()
                    if start_idx < date_idx:
                        current_qfq_close = day_stock_signal["qfq_close"]
                        start_qfq_close = df_backtest_stock.loc[start_idx,
                                                           "qfq_close"]
                        stock_relative_value = current_qfq_close/start_qfq_close\
                                           * df_asset_values.loc[start_idx,
                                                                 self.benchmark]
                        baseline_val_list.append(stock_relative_value)
                baseline_value = np.mean(baseline_val_list) if \
                    baseline_val_list else df_asset_values[
                    self.benchmark].iloc[-1]
            df_asset_values.loc[date_idx] = [my_model_value, baseline_value]
            print(date_idx, dict(df_asset_values.loc[date_idx]))

            # Make a plan for the next trading day.
            day_plan = self.trader.gen_trading_plan(day_signal=day_signal,account=self.account)

            date = date + day_delta
            cnt +=1
        return df_asset_values,orders,transactions,stocks


def main():
    # f_name1 = "XGBRegressor_20high"
    # f_name2 = "XGBRegressor_5low"
    # f_name3 = "XGBRegressor_5high"
    model_type = "XGBRegressor"

    models = {}
    # models["model_l_high"] = ml_model.load_model(model_type,pred_period=20,is_high=True)
    # models["model_s_low"] = ml_model.load_model(model_type,pred_period=5,is_high=False)
    # models["model_s_high"] = ml_model.load_model(model_type,pred_period=5,is_high=True)

    # models["model_l_high"] = lgbm.LGBMRegressor(n_estimators=100,
    #                                             num_leaves=128, max_depth=10,
    #                    random_state=0, min_child_weight=5)
    # models["model_s_low"] = lgbm.LGBMRegressor(n_estimators=100,
    #                                            num_leaves=128, max_depth=10,
    #                    random_state=0, min_child_weight=5)
    # models["model_s_high"] = lgbm.LGBMRegressor(n_estimators=100,
    #                                             num_leaves=128, max_depth=10,
    #                    random_state=0, min_child_weight=5)

    models["model_l_high"] = xgb.XGBRegressor(n_estimators=100,max_depth=8,
                                                random_state=0,
                                                min_child_weight=5)
    models["model_s_low"] = xgb.XGBRegressor(n_estimators=100,max_depth=8,
                                               random_state=0,
                                               min_child_weight=5)
    models["model_s_high"] = xgb.XGBRegressor(n_estimators=100,max_depth=8,
                                                random_state=0,
                                                min_child_weight=5)

    backtester = BackTest(start="2014-01-01")
    df_asset_values,orders,transactions,stocks = backtester.backtest_with_updating_model(models)
    # for date,row in df_asset_values.iterrows():
    #     print(date,dict(row))
    print("Transactions:",len(transactions))
    for e in sorted(transactions,key=lambda x:(x[1],x[0])):
        print(e)
    for k,v in sorted(stocks.items()):
        print(k,v)

    dates = df_asset_values.index
    figs,axes = plt.subplots()
    for col in df_asset_values.columns:
        axes.plot(dates, df_asset_values[col],label=col)
    axes.legend(loc="upper left")

    df_asset_values = df_asset_values.sort_index()
    prev = df_asset_values.index[0][5:7]
    seasons = ["01","04","07","10"]
    ticks = [df_asset_values.index[0]]
    for date_idx in df_asset_values.index[1:]:
        if date_idx[5:7] in seasons and date_idx[5:7]!=prev:
            ticks.append(date_idx)
            prev = date_idx[5:7]
    ticks.append(df_asset_values.index[-1])
    labels = [ticks[0].replace("-","")[2:]]\
             +[date_idx[2:4]+date_idx[5:7] if date_idx[5:7]=="01" else
               date_idx[5:7] for date_idx in ticks[1:-1]]\
             + [ticks[-1].replace("-","")[2:]]
    plt.xticks(ticks, labels)
    plt.show()


if __name__ == '__main__':
    main()
