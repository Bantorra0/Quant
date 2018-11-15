import unittest
import pandas as pd
import numpy as np
import data_prepare as data_p


class MoveTestCase(unittest.TestCase):
    def setUp(self):
        columns = ["code", "open", "close"]
        codes = "600345", "600229", "002345", "002236", "002217", \
                "300345", "603799"

        open = np.random.uniform(low=10, high=15, size=30)
        close = np.random.uniform(low=10, high=15, size=30)
        code = np.array([codes[0]] * 30)

        array = np.vstack([code, open, close]).T
        df = pd.DataFrame(array, columns=columns)

        date_idx = sorted(["2018-09-{:02d}".format(i) for i in range(1, 31)], reverse=True)
        df.index = date_idx

        self.columns = columns
        self.codes = codes
        self.df = df
        self.date_idx = date_idx


    def test_move_p(self):
        columns = self.columns
        df = self.df.copy(deep=True)
        date_idx = df.index

        # Move past data, cols=None,prefix=False.
        for i in range(1,10):
            expected_df_mv = df.iloc[i:]
            expected_df_mv.index = date_idx[:-i]
            df_mv = data_p._move(days=i,df=df, prefix=False)
            self.assertTrue((expected_df_mv==df_mv).all().all())

        # Move past data, cols=["code","open"],prefix=False.
        for i in range(1, 10):
            expected_df_mv = df.iloc[i:]
            expected_df_mv.index = date_idx[:-i]
            expected_df_mv = expected_df_mv[columns[:2]]
            expected_df_mv.columns = ["p{}mv_".format(i)+col for col in expected_df_mv.columns]
            df_mv = data_p._move(days=i, df=df,cols=columns[:2], prefix=True)
            self.assertTrue((expected_df_mv == df_mv).all().all())

    def test_move_f(self):
        columns = self.columns
        df = self.df.copy(deep=True)
        date_idx = df.index

        # Move future data, cols=None,prefix=False.
        for i in range(1, 10):
            expected_df_mv = df.iloc[:-i]
            expected_df_mv.index = date_idx[i:]
            df_mv = data_p._move(days=-i, df=df, prefix=False)
            self.assertTrue((expected_df_mv == df_mv).all().all())

        # Move future data, cols=["code","open"],prefix=False.
        for i in range(1, 10):
            expected_df_mv = df.iloc[:-i]
            expected_df_mv.index = date_idx[i:]
            expected_df_mv = expected_df_mv[columns[:2]]
            expected_df_mv.columns = ["f{}mv_".format(i) + col for col in expected_df_mv.columns]
            df_mv = data_p._move(days=-i, df=df, cols=columns[:2], prefix=True)
            self.assertTrue((expected_df_mv == df_mv).all().all())


class RollingTestCase(unittest.TestCase):
    def test_rolling(self):
        self.assertEqual(True, False)


#
#
# class ChangeRateTestCase(unittest.TestCase):
#     def test_change_rate(self):
#         self.assertEqual(True, False)






if __name__ == '__main__':
    unittest.main()
