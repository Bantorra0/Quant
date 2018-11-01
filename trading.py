import datetime
from collect import stck_pools
from constants import DATE_FORMAT,FEE_RATE
import ml_model
import os,pickle


class Account:
    def __init__(self, init_amt=1000000, fee_rate=FEE_RATE):
        self.fee_rate = fee_rate
        self.cash = init_amt
        self.stocks = {}

    def day_trade(self,order):
        # TODO: add trade logic.
        pass


class Trader:

    @staticmethod
    def gen_trade_order(day_signal, account:Account, date,
                        threshold=0.2):
        # print(day_signal)
        order = {}
        positions = {}
        stck_amt = 0
        for code, num in account.stocks.items():
            stck_amt = day_signal[code]["qfq_close"] * num
            positions[code] = stck_amt


        threshold_l_rise_buy = 0.5
        threshold_l_rise_sell = 0.3
        threshold_s_decline_buy = -0.03
        threshold_s_decline_sell = -0.08
        threshold_s_rise_sell = 0.04

        indicators = day_signal[["code","y_l_rise", "y_s_decline", "y_s_rise"]]
        print(indicators)

        sell_cond1 = (indicators["y_l_rise"]<threshold_l_rise_sell) & (indicators["y_s_rise"]<threshold_s_rise_sell)
        sell_cond2 = indicators["y_s_decline"]<threshold_s_decline_sell
        for code in account.stocks:
            if code in indicators[sell_cond1 | sell_cond2]["code"]:
                positions[code] =0

        buy_cond = (indicators["y_l_rise"]>threshold_l_rise_buy) & (indicators["y_s_decline"]>threshold_s_decline_buy)
        for code in indicators[buy_cond]["code"]:
            positions[code] = 0.2

        for code in positions:
            pass

        print(positions)

        return order

    @staticmethod
    def tot_amt(account: Account, prices):
        amt = account.cash
        for code, pos in account.stocks.items():
            amt += sum(pos.values()) * prices[code]
        return amt

    @staticmethod
    def order_target_percent(code, percent, price, account:Account):
        pass

    @staticmethod
    def order(code,cnt,price,account:Account):
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

        if not account.stocks[code].values() or sum(account.stocks[code].values()) == 0:
            del account.stocks[code]


    @classmethod
    def order_buy(cls,code, cnt, price, account:Account):
        if cnt*price > account.cash:
            raise ValueError("Cash {0} yuan is not enough for buying {1} {2} shares with price {3}!".format(account.cash, code, cnt, price))

        cls.order(code,cnt,price,account)


    @classmethod
    def order_sell(cls,code,cnt,price,account:Account):
        if code not in account.stocks:
            raise ValueError("Selling {0} while not having any".format(code))
        else:
            long_pos = sum(account.stocks[code].values())
            if cnt > long_pos:
                raise ValueError("Selling {0} {1} shares while having only {2}".format(code,cnt,long_pos))

        cls.order(code, -cnt, price, account)




class BackTest:
    def __init__(self, start="2018-01-01", end=None, benchmark='HS300',
                 universe=stck_pools(), capital_base=1000000, freq='d'):
        self.start = start
        self.end = end if end else datetime.datetime.now().strftime(DATE_FORMAT)
        self.universe = universe
        self.capital_base = capital_base
        self.freq = freq  # Time unit of trading frequency
        self.refresh_rate = 1  # Rate of adjusting positions
        self.time_delta = datetime.timedelta(days=self.refresh_rate)

    def init_account(self, fee_rate=FEE_RATE):
        self.account = Account(self.capital_base, fee_rate)


    def init_trader(self):
        self.trader = Trader()


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

        signals = df_all


        X = ml_model.gen_X(df_all, cols_future)
        signals["y_l_rise"] = models["model_l_high"].predict(X)
        signals["y_s_rise"] = models["model_s_high"].predict(X)
        signals["y_s_decline"] = models["model_s_low"].predict(X)

        day_delta = datetime.timedelta(days=1)

        while date <= end:
            date_idx = datetime.datetime.strftime(date, DATE_FORMAT)
            if date_idx not in signals.index:
                date = date + day_delta
                continue

            day_signal = signals.loc[date_idx]
            order = self.trader.gen_trade_order(day_signal,self.account,date)
            self.account.day_trade(order)

            date = date + self.time_delta


def trade():
    # f_name1 = "XGBRegressor_20high"
    # f_name2 = "XGBRegressor_5low"
    # f_name3 = "XGBRegressor_5high"
    model_type = "XGBRegressor"

    models = {}
    models["model_l_high"] = ml_model.load_model(model_type,pred_period=20,is_high=True)
    models["model_s_low"] = ml_model.load_model(model_type,pred_period=5,is_high=False)
    models["model_s_high"] = ml_model.load_model(model_type,pred_period=5,is_high=True)
    # with open(os.path.join(os.getcwd(), f_name1), "rb") as f:
    #     models["model_l_high"] = pickle.load(f)
    # with open(os.path.join(os.getcwd(), f_name2), "rb") as f:
    #     models["model_s_low"] = pickle.load(f)
    # with open(os.path.join(os.getcwd(), f_name3), "rb") as f:
    #     models["model_s_high"] = pickle.load(f)

    backtester = BackTest(start="2018-08-15")

    backtester.backtest(models)


if __name__ == '__main__':
    trade()
