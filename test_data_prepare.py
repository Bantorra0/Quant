import unittest


class MyTestCase(unittest.TestCase):
    def test_move(self):
        self.assertEqual(True, False)

    def test_rolling(self):
        self.assertEqual(True, False)


    def test_whold(self):
        # test
        # df_test = df_all[df_all["code"]=="600887.SH"]
        # basic_cols = ["open", "high", "low", "close", "amt", "adj_factor"]
        # derived_cols = ['change_rate_p1mv_open', 'change_rate_p1mv_high',
        #                 'change_rate_p1mv_low', 'change_rate_p1mv_close',
        #                 'change_rate_p1mv_amt', 'change_rate_p3mv_open',
        #                 'change_rate_p3mv_high', 'change_rate_p3mv_low',
        #                 'change_rate_p3mv_close', 'change_rate_p3mv_amt',
        #                 'change_rate_p5mv_open', 'change_rate_p5mv_high',
        #                 'change_rate_p5mv_low', 'change_rate_p5mv_close',
        #                 'change_rate_p5mv_amt', 'change_rate_p5max_open',
        #                 'change_rate_p5max_high', 'change_rate_p5max_low',
        #                 'change_rate_p5max_close', 'change_rate_p5max_amt',
        #                 'change_rate_p5min_open', 'change_rate_p5min_high',
        #                 'change_rate_p5min_low', 'change_rate_p5min_close',
        #                 'change_rate_p5min_amt', 'change_rate_p5mean_open',
        #                 'change_rate_p5mean_high', 'change_rate_p5mean_low',
        #                 'change_rate_p5mean_close', 'change_rate_p5mean_amt',
        #                 'change_rate_p20max_open', 'change_rate_p20max_high',
        #                 'change_rate_p20max_low', 'change_rate_p20max_close',
        #                 'change_rate_p20max_amt', 'change_rate_p20min_open',
        #                 'change_rate_p20min_high', 'change_rate_p20min_low',
        #                 'change_rate_p20min_close', 'change_rate_p20min_amt',
        #                 'change_rate_p20mean_open', 'change_rate_p20mean_high',
        #                 'change_rate_p20mean_low', 'change_rate_p20mean_close',
        #                 'change_rate_p20mean_amt', 'f1mv_open', 'f1mv_high',
        #                 'f1mv_low', 'f1mv_close', 'f20max_f1mv_high',
        #                 'sz50_open', 'sz50_high', 'sz50_low', 'sz50_close',
        #                 'sz50_vol', 'sz50_change_rate_p1mv_open',
        #                 'sz50_change_rate_p1mv_high',
        #                 'sz50_change_rate_p1mv_low',
        #                 'sz50_change_rate_p1mv_close',
        #                 'sz50_change_rate_p1mv_vol']
        #
        # test_cols = basic_cols + derived_cols
        # print(test_cols)
        # df_test[test_cols].sort_index(ascending=False).iloc[:100].to_excel(
        #     "test_data.xlsx",header=True,index=True)


        # # test
        # df_test_list = []
        # for code in df_all["code"].unique()[:3]:
        #     df = df_all[df_all["code"]==code].sort_index(
        #         ascending=False).iloc[:50]
        #     print(df)
        #     df_test_list.append(df)
        # pd.concat(df_test_list).to_excel("test_data.xlsx",header=True,index=True)
        pass




if __name__ == '__main__':
    unittest.main()
