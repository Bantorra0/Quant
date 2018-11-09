import unittest
import data_cleaning as dc
import pandas as pd
import db_operations as dbop


class DataCleaningTestCase(unittest.TestCase):
    def test_fillna_single_stock_day(self):
        dates = ["2018-10-{:02d}".format(i) for i in range(1,14)]
        columns = ["date","code","open", "high", "low", "close","vol", "amt","adj_factor"]

        # Cover cases that all rows are not null and change nothing.
        normal_rows = [["2018-10-01", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1],
                       ["2018-10-02", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1],
                       ["2018-10-03", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1]
                       ]
        error_rows = [["2018-10-04", "002217.SZ", None, 5.2, 5.0, 5.05, 1000, 5100, 5.1],
                      ["2018-10-05", "002217.SZ", 5.1, None, 5.0, 5.05, 1000, 5100, 5.1],
                      ["2018-10-06", "002217.SZ", 5.1, 5.2, None, 5.05, 1000, 5100, 5.1],
                      ["2018-10-07", "002217.SZ", 5.1, 5.2, 5.0, None, 1000, 5100, 5.1],
                      ["2018-10-08", "002217.SZ", 5.1, 5.2, 5.0, 5.05, None, 5100, 5.1],
                      ["2018-10-09", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, None, 5.1],
                      ["2018-10-10", "002217.SZ", None, 5.2, 5.0, 5.05, None, 5100, 5.1],
                      ["2018-10-11", "002217.SZ", 5.1, None, None, 5.05, 1000, 5100, 5.1],
                      ["2018-10-12", "002217.SZ", 5.1, 5.2, 5.0, 5.05, None, None, 5.1],
                      ["2018-10-13", "002217.SZ", None, None, None, None, None, 5100, 5.1]
                      ]
        filling_rows =[["2018-10-04", "002217.SZ", None, None, None, None, None, None, 5.2],
                       ["2018-10-04", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, None],
                       ["2018-10-04", "002217.SZ", None, None, None, None, None, None, None]
                       ]

        # Cover the case that a valid start_date doesn't exist.
        df_single_stock = pd.DataFrame(normal_rows,columns=columns)
        args = df_single_stock, dates[3:]
        self.assertRaises(ValueError,dc.fillna_single_stock_day,*args)

        # Cover the case that df contains data not in the given trading dates.
        df_single_stock = pd.DataFrame(normal_rows, columns=columns)
        args = df_single_stock, [dates[0],dates[2]]
        self.assertRaises(ValueError, dc.fillna_single_stock_day, *args)

        # Cover the case that all rows of df are not null.
        df_single_stock = pd.DataFrame(normal_rows, columns=columns)
        df_changed = dc.fillna_single_stock_day(df_single_stock, dates[:3])
        self.assertEqual(0, len(df_changed))

        # Cover cases of rows of irregular partial missing raises ValueError.
        for row in error_rows:
            df_single_stock = pd.DataFrame(
                [row],columns=columns
            )
            args = df_single_stock, dates
            self.assertRaises(ValueError, dc.fillna_single_stock_day, *args)

        # Cover the case that fills ["open","high","low","close","vol","amt"].
        df_single_stock = pd.DataFrame(normal_rows+[filling_rows[0]],columns=columns)
        df_changed = dc.fillna_single_stock_day(df_single_stock,dates[:4])
        expected_df_changed = pd.DataFrame([["2018-10-04", "002217.SZ", 5.05, 5.05, 5.05, 5.05, 0, 0, 5.2]],columns=columns)
        self.assertEqual(True,(expected_df_changed==df_changed).all().all())

        # Cover the case that fills ["adj_factor"]
        df_single_stock = pd.DataFrame(normal_rows + [filling_rows[1]], columns=columns)
        df_changed = dc.fillna_single_stock_day(df_single_stock, dates[:4])
        expected_df_changed = pd.DataFrame([["2018-10-04", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1]],
                                           columns=columns)
        self.assertEqual(True, (expected_df_changed == df_changed).all().all())

        # Cover the case that fills ["open","high","low","close","vol","amt","adj_factor"].
        df_single_stock = pd.DataFrame(normal_rows + [filling_rows[2]], columns=columns)
        df_changed = dc.fillna_single_stock_day(df_single_stock, dates[:4])
        expected_df_changed = pd.DataFrame([["2018-10-04", "002217.SZ", 5.05, 5.05, 5.05, 5.05, 0, 0, 5.1]],
                                           columns=columns)
        self.assertEqual(True, (expected_df_changed == df_changed).all().all())

        # Cover the case that the whole role, including index, is missing.
        df_single_stock = pd.DataFrame(normal_rows, columns=columns)
        df_changed = dc.fillna_single_stock_day(df_single_stock, dates[:4])
        expected_df_changed = pd.DataFrame([["2018-10-04", "002217.SZ", 5.05, 5.05, 5.05, 5.05, 0, 0, 5.1]],
                                           columns=columns)
        self.assertEqual(True, (expected_df_changed == df_changed).all().all())

        # Cover the case that the whole role, including index, is missing.
        df_single_stock = pd.DataFrame(normal_rows, columns=columns)
        df_changed = dc.fillna_single_stock_day(df_single_stock, dates[:4])
        expected_df_changed = pd.DataFrame([["2018-10-04", "002217.SZ", 5.05, 5.05, 5.05, 5.05, 0, 0, 5.1]],
                                           columns=columns)
        self.assertEqual(True, (expected_df_changed == df_changed).all().all())

        # Integrated test.
        rows = [["2018-10-01", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1],
                ["2018-10-02", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, None],
                ["2018-10-03", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1],
                ["2018-10-04", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, None],
                ["2018-10-05", "002217.SZ", None, None, None, None, None, None, 5.1],
                ["2018-10-06", "002217.SZ", 5.2, 5.3, 5.1, 5.15, 1100, 5500, 5.2],
                ["2018-10-07", "002217.SZ", None, None, None, None, None, None, None],
                ]
        changed_rows = [
                ["2018-10-04", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1],
                ["2018-10-05", "002217.SZ", 5.05, 5.05, 5.05, 5.05, 0, 0, 5.1],
                ["2018-10-07", "002217.SZ", 5.15, 5.15, 5.15, 5.15, 0, 0, 5.2],
                ["2018-10-08", "002217.SZ", 5.15, 5.15, 5.15, 5.15, 0, 0, 5.2],
                ]
        df_single_stock = pd.DataFrame(rows, columns=columns)
        df_changed = dc.fillna_single_stock_day(df_single_stock, dates[2:8])
        expected_df_changed = pd.DataFrame(changed_rows,
                                           columns=columns)
        self.assertEqual(True, (expected_df_changed == df_changed).all().all())


    def test_fillna_stock_day(self):
        dates = ["2018-10-{:02d}".format(i) for i in range(1, 14)]
        columns = ["date", "code", "open", "high", "low", "close", "vol", "amt", "adj_factor"]

        # Integrated test.
        rows = [
            # 002217.SZ
            ["2018-10-01", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1],
            ["2018-10-02", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, None],
            ["2018-10-03", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1],
            ["2018-10-04", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, None],
            ["2018-10-05", "002217.SZ", None, None, None, None, None, None, 5.1],
            ["2018-10-06", "002217.SZ", 5.2, 5.3, 5.1, 5.15, 1100, 5500, 5.2],
            ["2018-10-07", "002217.SZ", None, None, None, None, None, None, None],
            # 002345.SZ
            ["2018-10-01", "002345.SZ", 5.1, 5.2, None, 5.05, 1000, 5100, 5.1],
            ["2018-10-02", "002345.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1],
            ["2018-10-03", "002345.SZ", 5.1, 5.2, 5.0, 5.05, None, 5100, 5.1],
            ["2018-10-04", "002345.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 6],
            ["2018-10-05", "002345.SZ", None, None, None, None, None, None, 6],
            ["2018-10-06", "002345.SZ", 5.2, 5.3, 5.1, 5.15, 1100, 5500, None],
            ["2018-10-07", "002345.SZ", None, None, None, None, None, None, None],
        ]
        changed_rows = [
            # 002217.SZ
            ["2018-10-04", "002217.SZ", 5.1, 5.2, 5.0, 5.05, 1000, 5100, 5.1],
            ["2018-10-05", "002217.SZ", 5.05, 5.05, 5.05, 5.05, 0, 0, 5.1],
            ["2018-10-07", "002217.SZ", 5.15, 5.15, 5.15, 5.15, 0, 0, 5.2],
            ["2018-10-08", "002217.SZ", 5.15, 5.15, 5.15, 5.15, 0, 0, 5.2],
            # 002345.SZ
            ["2018-10-05", "002345.SZ", 5.05, 5.05, 5.05, 5.05, 0, 0, 6],
            ["2018-10-06", "002345.SZ", 5.2, 5.3, 5.1, 5.15, 1100, 5500, 6],
            ["2018-10-07", "002345.SZ", 5.15, 5.15, 5.15, 5.15, 0, 0, 6],
            ["2018-10-08", "002345.SZ", 5.15, 5.15, 5.15, 5.15, 0, 0, 6],
        ]
        df_stock_day = pd.DataFrame(rows, columns=columns)
        df_changed = pd.concat(dc.fillna_stock_day(df_stock_day=df_stock_day, dates=dates[2:8]),sort=False).set_index(["date","code"]).sort_index()
        expected_df_changed = pd.DataFrame(changed_rows,
                                           columns=columns).set_index(["date","code"]).sort_index()
        self.assertEqual(True, (expected_df_changed == df_changed).all().all())


if __name__ == '__main__':
    unittest.main()
