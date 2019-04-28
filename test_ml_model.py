import unittest
import pandas as pd
import numpy as np
import ml_model as ml



class MLModelTestCase(unittest.TestCase):
    def test_corr(self):
        n = 10000
        X = pd.DataFrame(np.random.normal(2, 1, size=(n, 10)), columns=list("abcdefghij"))
        Y = pd.DataFrame(np.random.normal(-1, 3, (n, 3)), columns=["y1", "y2", "y3"])

        corr_actual = ml.corr(X, Y)
        corr_expected = pd.concat([X, Y], axis=1).corr().loc[X.columns, Y.columns]

        delta = 1e-9
        self.assertTrue(((corr_actual-corr_expected).abs()/corr_expected.abs()<delta).all().all())



if __name__ == '__main__':
    unittest.main()
