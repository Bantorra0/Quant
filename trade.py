import datetime
from collect import stck_pools
from constants import DATE_FORMAT,FEE_RATE,BASE_DIR
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

        for code in account.stocks:
            if day_signal[day_signal["code"]==code]["y_pred"]<threshold:
                order[code] == 0

        buying_stcks = day_signal[day_signal["y_pred"]>threshold][["code",
                                                               "y_pred"]]
        print(buying_stcks)

        # day_data= self.get_day_data(date)
        # TODO: generate trade order
        return order

    def get_day_data(self,date):
        day_data = None
        # TODO: get day data
        return day_data


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


    def backtest(self, model):
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
        X_full = df_all
        y_pred = model.predict(X)
        signals = X_full.copy()
        signals["y_pred"] = y_pred

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


def main():
    f_name = "XGBRegressor_20high"
    print("model:", f_name)
    with open(os.path.join(BASE_DIR, f_name), "rb") as f:
        model = pickle.load(f)
    backtester = BackTest(start="2018-08-15")

    backtester.backtest(model)


if __name__ == '__main__':
    main()
