''' Collection of tests that verify whether the
    classes work as expected.
'''

import unittest
import numpy as np
import pandas as pd

from model.generic_ml_inference import BlpEstimator


class TestBLP(unittest.TestCase):
    '''Test whether the BLP estimator works correctly

    For heterogeneity we expect a coefficient close to 1.
    A value close to 0 suggest no heterogeneity detected.
    '''

    def setUp(self) -> None:
        self._hetero_data = (
            np.full(1000, 3),
            np.arange(1,1001),
            np.concatenate(
                [
                np.full(500, 3),
                np.arange(501,1001)
                ]
            ),
            np.concatenate(
                [
                np.full(500,0),
                np.full(500,1)
                ]
            )
        )

        self._homo_data = (
            np.random.normal(loc=0,scale=1,size=1000),
            np.random.normal(loc=0,scale=1,size=1000),
            np.random.normal(loc=0,scale=1,size=1000),
            np.random.randint(0,2,1000)
        )


    def test_no_heterogeneity(self):
        ''' Test whether the coefficient for heterogeneity is accurate
            on synthetic data, which has no heterogeneity.
        '''
        base, treat, y, t = self._homo_data

        est = BlpEstimator()
        est.fit(
            base,
            treat,
            t,
            y
        )
        pvalue = est.summary['P>|t|']['ß_2']
        print('pvalue no hetero: ', pvalue)
        print('param value no hetero: ', est.blp_params[-1])
        self.assertTrue(pvalue > 0.1)
        self.assertTrue(round(est.blp_params[-1])==0)


    def test_heterogeneity(self):
        ''' Test whether BLP detects
            linear synthetic heterogeneity.
        '''
        base, treat, y, t = self._hetero_data

        est = BlpEstimator()
        est.fit(
            base,
            treat,
            t,
            y
        )
        pvalue = est.summary['P>|t|']['ß_2']
        print('pvalue with hetero: ', pvalue)
        print('param value with hetero: ', est.blp_params[-1])
        self.assertTrue(pvalue < 0.05)
        self.assertTrue(round(est.blp_params[-1])==1)
    

if __name__ == '__main__':
    unittest.main()
