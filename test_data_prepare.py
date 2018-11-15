import unittest
import pandas as pd
import numpy as np
import data_prepare as data_p


class MoveTestCase(unittest.TestCase):
    def setUp(self):
        columns = ["code", "open", "close"]
        codes = "600345", "600229", "002345", "002236", "002217", \
                "300345", "603799"

        size=30
        open = np.random.uniform(low=10, high=15, size=size)
        close = np.random.uniform(low=10, high=15, size=size)
        code = np.array([codes[0]] * size)

        array = np.vstack([code, open, close]).T
        df = pd.DataFrame(array, columns=columns)

        date_idx = sorted(["2018-09-{:02d}".format(i) for i in range(1, 31)], reverse=True)
        df.index = date_idx

        self.columns = columns
        self.codes = codes
        self.df = df.astype(float)
        self.date_idx = date_idx

    def test_move_p(self):
        columns = self.columns
        df = self.df.copy(deep=True)
        date_idx = df.index

        # Case 0: Move past data, cols=None,prefix=False.
        for i in range(1,10):
            expected_df_mv = df.iloc[i:]
            expected_df_mv.index = date_idx[:-i]
            expected_df_mv = expected_df_mv[columns[1:]]
            df_mv = data_p.move(days=i, df=df[columns[1:]], prefix=False)
            self.assertTrue((expected_df_mv==df_mv).all().all())

        # Case 1: Move past data, cols=["code","open"],prefix=False.
        for i in range(1, 10):
            expected_df_mv = df.iloc[i:]
            expected_df_mv.index = date_idx[:-i]
            expected_df_mv = expected_df_mv[columns[1:]]
            expected_df_mv.columns = ["p{}mv_".format(i)+col for col in expected_df_mv.columns]
            df_mv = data_p.move(days=i, df=df, cols=columns[1:], prefix=True)
            self.assertTrue((expected_df_mv == df_mv).all().all())

        # Make sure the input df is not modified.
        self.assertTrue((self.df==df).all().all())

    def test_move_f(self):
        columns = self.columns
        df = self.df.copy(deep=True)
        date_idx = df.index

        # Move future data, cols=None,prefix=False.
        for i in range(1, 10):
            expected_df_mv = df.iloc[:-i]
            expected_df_mv.index = date_idx[i:]
            expected_df_mv = expected_df_mv[columns[1:]]
            df_mv = data_p.move(days=-i, df=df[columns[1:]], prefix=False)
            self.assertTrue((expected_df_mv == df_mv).all().all())

        # Move future data, cols=["code","open"],prefix=False.
        for i in range(1, 10):
            expected_df_mv = df.iloc[:-i]
            expected_df_mv.index = date_idx[i:]
            expected_df_mv = expected_df_mv[columns[1:]]
            expected_df_mv.columns = ["f{}mv_".format(i) + col for col in expected_df_mv.columns]
            df_mv = data_p.move(days=-i, df=df, cols=columns[1:], prefix=True)
            self.assertTrue((expected_df_mv == df_mv).all().all())

        # Make sure the input df is not modified.
        self.assertTrue((self.df==df).all().all())


class RollingTestCase(unittest.TestCase):
    def setUp(self):
        columns = ["code", "open", "close"]
        codes = "600345", "600229", "002345", "002236", "002217", \
                "300345", "603799"

        size = 30
        open = np.arange(1,size+1)
        close = open+1.5
        code = np.array([codes[0]] * size)

        array = np.vstack([code, open, close]).T
        df = pd.DataFrame(array, columns=columns)

        date_idx = sorted(["2018-09-{:02d}".format(i) for i in range(1, 31)], reverse=True)
        df.index = date_idx

        self.columns = columns
        self.codes = codes
        self.df = df.astype(float)
        self.date_idx = date_idx

    def test_rolling_type_error(self):
        columns = self.columns
        df = self.df.copy(deep=True)
        date_idx = df.index
        n = len(date_idx)

        # Use past data and rolling max, cols=None,prefix=False.
        for i in range(2, 10):
            self.assertRaises(ValueError,data_p.rolling,
                              *("median",-i,df[columns[1:]],False))

    def test_rolling_p(self):
        columns = self.columns
        df = self.df.copy(deep=True)
        date_idx = df.index
        n = len(date_idx)

        # Use past data and rolling max, cols=None,prefix=False.
        for i in range(2, 10):
            expected_df_rolling = df.iloc[i-1:]
            expected_df_rolling.index = date_idx[:n-i+1]
            expected_df_rolling = expected_df_rolling[columns[1:]]
            df_rolling = data_p.rolling(rolling_type="max",days=-i, df=df[columns[1:]], prefix=False)
            self.assertTrue((expected_df_rolling == df_rolling).all().all())

        # Use past data and rolling min, cols=None,prefix=False.
        for i in range(2, 10):
            expected_df_rolling = df.iloc[:n-i+1]
            expected_df_rolling = expected_df_rolling[columns[1:]]
            df_rolling = data_p.rolling(rolling_type="min", days=-i, df=df[columns[1:]], prefix=False)
            self.assertTrue((expected_df_rolling == df_rolling).all().all())

        # Use past data and rolling mean, cols=None,prefix=False.
        for i in range(2,10):
            k = int((i - 1) / 2)
            if (i-1)%2 == 0:
                expected_df_rolling = df.iloc[k:n-i+k+1]
            else:
                expected_df_rolling = df.iloc[k:n-i+k+1]+0.5
            expected_df_rolling.index = date_idx[:n - i + 1]
            expected_df_rolling = expected_df_rolling[columns[1:]]
            df_rolling = data_p.rolling(rolling_type="mean", days=-i, df=df[columns[1:]], prefix=False)
            self.assertTrue((expected_df_rolling == df_rolling).all().all())

        # Use past data and rolling max, cols=["open","close"],prefix=True.
        for i in range(2, 10):
            expected_df_rolling = df.iloc[i - 1:]
            expected_df_rolling.index = date_idx[:n - i + 1]
            expected_df_rolling = expected_df_rolling[columns[1:]]
            expected_df_rolling.columns = ["p{}max_".format(i)+col for col in expected_df_rolling.columns]
            df_rolling = data_p.rolling(rolling_type="max", days=-i, df=df, cols=columns[1:], prefix=True)
            self.assertTrue((expected_df_rolling == df_rolling).all().all())

        # Make sure the input df is not modified.
        self.assertTrue((self.df==df).all().all())

    def test_rolling_f(self):
        columns = self.columns
        df = self.df.copy(deep=True)
        date_idx = df.index
        n = len(date_idx)

        # Use future data and rolling max, cols=None,prefix=False.
        for i in range(2, 10):
            expected_df_mv = df.iloc[i-1:]
            expected_df_mv.index = date_idx[i-1:]
            expected_df_mv = expected_df_mv[columns[1:]]
            df_rolling = data_p.rolling(rolling_type="max",days=i, df=df[columns[1:]], prefix=False)
            self.assertTrue((expected_df_mv == df_rolling).all().all())

        # Use future data and rolling min, cols=None,prefix=False.
        for i in range(2, 10):
            expected_df_mv = df.iloc[:n-i+1]
            expected_df_mv.index = date_idx[i-1:]
            expected_df_mv = expected_df_mv[columns[1:]]
            df_rolling = data_p.rolling(rolling_type="min", days=i, df=df[columns[1:]], prefix=False)
            self.assertTrue((expected_df_mv == df_rolling).all().all())

        # Use future data and rolling mean, cols=None,prefix=False.
        for i in range(2,10):
            k = int((i - 1) / 2)
            if (i-1)%2 == 0:
                expected_df_mv = df.iloc[k:n-i+k+1]
            else:
                expected_df_mv = df.iloc[k:n-i+k+1]+0.5
            expected_df_mv.index = date_idx[i-1:]
            expected_df_mv = expected_df_mv[columns[1:]]
            df_rolling = data_p.rolling(rolling_type="mean", days=i, df=df[columns[1:]], prefix=False)
            self.assertTrue((expected_df_mv == df_rolling).all().all())

        # Use future data and rolling max, cols=None,prefix=False.
        for i in range(2, 10):
            expected_df_mv = df.iloc[i - 1:]
            expected_df_mv.index = date_idx[i-1:]
            expected_df_mv = expected_df_mv[columns[1:]]
            expected_df_mv.columns = ["f{}max_".format(i)+col for col in expected_df_mv.columns]
            df_rolling = data_p.rolling(rolling_type="max", days=i, df=df, cols=columns[1:], prefix=True)
            self.assertTrue((expected_df_mv == df_rolling).all().all())

        # Make sure the input df is not modified.
        self.assertTrue((self.df==df).all().all())


class ChangeRateTestCase(unittest.TestCase):
    def setUp(self):
        columns = ["code", "open", "close"]
        codes = "600345", "600229", "002345", "002236", "002217", \
                "300345", "603799"

        size = 30
        open = np.arange(1,size+1)
        close = open+1.5
        code = np.array([codes[0]] * size)

        array = np.vstack([code, open, close]).T
        df = pd.DataFrame(array, columns=columns)

        date_idx = sorted(["2018-09-{:02d}".format(i) for i in range(1, 31)], reverse=True)
        df.index = date_idx

        self.columns = columns
        self.codes = codes
        self.df = df.astype(float)
        self.date_idx = date_idx

    def test_change_rate(self):
        columns = self.columns
        df = self.df.copy(deep=True)
        date_idx = df.index
        n = len(date_idx)
        target_cols = columns[1:]

        # Column lengths are not the same.
        for i in range(2, 5):
            df2 = df.copy()
            df2.loc[:, target_cols] = df2[target_cols] * i
            self.assertRaises(ValueError,data_p.change_rate,*(df,df2,
                                                              target_cols,
                                                              columns))

        # Both cols1 and cols2 are given.
        for i in range(2,5):
            df2 = df.copy()
            df2.loc[:,target_cols] = df2[target_cols]*i
            df_change_rate = data_p.change_rate(df,df2,cols1=target_cols,
                                                cols2=target_cols)
            self.assertTrue((i-1==df_change_rate).all().all())

        # cols1 is given but cols2 is not.
        for i in range(2, 5):
            df2 = df.copy()
            df2.loc[:, target_cols] = df2[target_cols] * i
            df_change_rate = data_p.change_rate(df, df2[target_cols], cols1=target_cols)
            self.assertTrue((i - 1 == df_change_rate).all().all())

        # cols2 is given but cols1 is not.
        for i in range(2, 5):
            df2 = df.copy()
            df2.loc[:, target_cols] = df2[target_cols] * i
            df_change_rate = data_p.change_rate(df[target_cols], df2,
                                                cols2=target_cols)
            self.assertTrue((i - 1 == df_change_rate).all().all())

        # Both cols1 and cols2 are not given.
        for i in range(2, 5):
            df2 = df.copy()
            df2.loc[:, target_cols] = df2[target_cols] * i
            df_change_rate = data_p.change_rate(df[target_cols], df2[target_cols])
            self.assertTrue((i - 1 == df_change_rate).all().all())

        # Make sure the input df is not modified.
        self.assertTrue((self.df==df).all().all())


if __name__ == '__main__':
    unittest.main()
