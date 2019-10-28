import unittest

import numpy as np
import pandas as pd

import feature_engineering as FE
import db_operations as dbop
from constants import *
import data_process as dp


class FETestCase(unittest.TestCase):
    def setUp(self):
        cursor = dbop.connect_db("sqlite3").cursor()
        start = 20190101
        df = dbop.create_df(cursor, STOCK_DAY[TABLE],
                            start=start,
                            where_clause="code in ('002349.SZ','600352.SH','600350.SH','600001.SH')",
                            # where_clause="code='600350.SH'",
                            )
        self.df = dp.proc_stock_d(dp.prepare_stock_d(df))


    def test_move_batch(self):
        # Use normal move as expected.
        for n in [-10,-5,0,5,10]:
            cols = ["open","high","low","close"]
            expected = pd.concat([FE.move(n,group) for _,group in self.df[cols].groupby(level="code")])\
                .dropna().sort_index()
            actual = FE.move_batch(n,self.df[cols],sort=False)\
                .dropna().sort_index()

            self.assertTrue((actual == expected).all().all())


    def test_rolling_batch(self):
        cols = ["open", "high", "low", "close"]
        delta = 1e-10

        # Test multiple cols single ops.
        for n in [-10,-5,5,10]:
            expected = pd.concat([FE.rolling("max",-n,group) for _,group in self.df[cols].groupby(level="code")])\
                    .dropna().sort_index()
            actual = FE.rolling_batch("max",n,self.df[cols],sort=False).dropna().sort_index()
            self.assertTrue((actual == expected).all().all())

        # Test multiple cols and ops.
        # Precision from high to low: max=min>mean>std, thus delta=1e-10 is used.x`
        ops = ["max","min","mean","std"]
        for n in [-10,-5,5,10]:
            expected = pd.concat(
                [pd.concat([FE.rolling(op,-n,group) for _,group in self.df[cols].groupby(level="code")]) for op in ops],
                axis=1
            ).dropna().sort_index()
            actual = FE.rolling_batch(ops,n,self.df[cols],sort=False).dropna().sort_index()[expected.columns]
            self.assertTrue(((expected-actual).abs()<delta).all().all())


    def test_chg_rate_batch(self):
        cols = ["open", "high", "low", "close"]
        df_mv = FE.move_batch(5,self.df[cols],sort=False)

        expected = FE.chg_rate(df_mv, self.df[cols]).dropna().sort_index()
        actual = FE.chg_rate_batch(df_mv,self.df[cols],sort=False).dropna().sort_index()
        print((expected!=actual).any(axis=1).sum())
        self.assertTrue((expected==actual).all().all())


    def test_candle_stick_batch(self):
        cols = ["open", "high", "low", "close","avg"]
        delta = 1e-12

        expected = FE.candle_stick(self.df[cols]).dropna().sort_index()
        actual = FE.candle_stick_batch(self.df[cols]).dropna().sort_index()
        # Use equal instead of abs < delta because results are rounded to 2 decimals and dtype is float16.
        self.assertTrue((expected==actual).all().all())
        # self.assertTrue(((expected-actual).abs()<delta).all().all())


    def test_k_MA_batch(self):
        delta = 1e-10
        cols = ["close", "vol", "amt"]

        for k in [3,5,10,20,60]:
            actual = FE.k_MA_batch(k, self.df[cols], sort=False).dropna().sort_index()
            expected = pd.concat([FE.k_MA(k, group) for _, group in self.df[cols].groupby(level="code")]).dropna().sort_index()
            self.assertTrue(((expected - actual).abs() < delta).all().all())


    def test_k_line(self):
        delta = 1e-10
        cols = ["open", "high", "low", "close", "vol", "amt"]

        for k in [3,5,10,20,60]:
            actual = FE.k_line_batch(k, self.df[cols], sort=False)\
                .dropna().sort_index()
            expected = pd.concat([FE.k_line(k, group) for _, group in self.df[cols].groupby(level="code")])\
                .dropna().sort_index()
            self.assertTrue(((expected - actual).abs() < delta).all().all())

    def test_stock_d_FE_batch(self):
        epsilon = 1e-4
        delta = 1e-4

        targets = [{"period": 20, "func": "max", "col": "high"},
                   {"period": 20, "func": "min", "col": "low"},
                   {"period": 20, "func": "avg", "col": ""},
                   {"period": 5, "func": "max", "col": "high"},
                   {"period": 5, "func": "min", "col": "low"},
                   {"period": 5, "func": "avg", "col": ""},
                   ]

        actual, _ = FE.stock_d_FE_batch(self.df, targets=targets)
        actual = actual.sort_index().fillna(0)

        expected = pd.concat([FE.stock_d_FE(group, targets=targets)[0] for _, group in self.df.groupby(level="code")]) \
            .sort_index().fillna(0)
        error = (expected - actual).abs() / (expected + epsilon)
        test_cond = error<delta

        self.assertTrue(test_cond.all().all())


if __name__ == '__main__':
    unittest.main()
