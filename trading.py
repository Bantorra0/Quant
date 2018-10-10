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
    def gen_trade_order(self, day_signal, account:Account, date,
                        threshold=0.2):
        # print(day_signal)
        order = {}
        positions = {}
        stck_amt = 0
        for code, num in account.stocks.items():
            stck_amt = day_signal[code]["qfq_close"] * num
            positions[code] = stck_amt


        threshold_l_rise_buy = 0.4
        threshold_l_rise_sell = 0.3
        threshold_s_decline_buy = -0.04
        threshold_s_decline_sell = -0.08
        threshold_s_rise_sell = 0.04

        indicators = day_signal[["y_l_rise", "y_s_decline", "y_s_rise"]]

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

        fq_cols = ["open", "high", "low", "close"]
        qfq_cols = ["qfq_"+col for col in fq_cols]
        tomorrow_qfq_cols = ["f1mv_"+col for col in qfq_cols]
        X = ml_model.gen_X(df_all, cols_future)
        X = X[X.columns.difference(qfq_cols+tomorrow_qfq_cols)]
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
    f_name1 = "XGBRegressor_20high"
    f_name2 = "XGBRegressor_5low"
    f_name3 = "XGBRegressor_5high"

    models = {}
    print("models:", f_name1, f_name2, f_name3)
    with open(os.path.join(os.getcwd(), f_name1), "rb") as f:
        models["model_l_high"] = pickle.load(f)
    with open(os.path.join(os.getcwd(), f_name2), "rb") as f:
        models["model_s_low"] = pickle.load(f)
    with open(os.path.join(os.getcwd(), f_name3), "rb") as f:
        models["model_s_high"] = pickle.load(f)

    backtester = BackTest(start="2018-08-15")

    backtester.backtest(models)


if __name__ == '__main__':
    trade()
