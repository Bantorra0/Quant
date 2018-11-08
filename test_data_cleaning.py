import unittest
import data_cleaning as dc
import pandas as pd
import db_operations as dbop


class DataCleaningTestCase(unittest.TestCase):
    def fillna_stock_day(self):
        dates = dbop.get_trading_dates(db_type="sqlite3")
        columns = ["code","open", "high", "low", "close","vol", "amt","adj_factor"]
        df_single_stock = pd.DataFrame(
            [[],
             []
             ],
            columns=columns
        )
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
