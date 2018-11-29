#!/usr/bin/env python3

import unittest
import sys
import pickle

import numpy as np

from testutils import DpkTestCase, slowtest
from ...metric import BrillLindquistMetric
from ...exprs.trig import CosineSeries
from .starshapedcurve import StarShapedCurve
from .refparamcurve import RefParamCurve


class TestRefparamcurve(DpkTestCase):
    def setUp(self):
        self.__metric = BrillLindquistMetric(d=1.4)
        self.__c_ref = StarShapedCurve.create_sphere(
            radius=2.0, metric=self.__metric, num=1
        )
        self.__c = RefParamCurve.from_curve(
            self.__c_ref, offset_coeffs=[-0.1, 0, 0.1]
        )

    def tearDown(self):
        pass

    def test_general(self):
        c, c_ref = self.__c, self.__c_ref
        self.assertIs(c.metric, c_ref.metric)

    def test_pickle(self):
        c = self.__c
        s = pickle.dumps(c)
        c_copy = pickle.loads(s)
        self.assertListEqual(c(0.789).tolist(), c(0.789).tolist())
        # now that evaluators have been created, try pickling again
        s = pickle.dumps(c)
        c_copy = pickle.loads(s)
        self.assertListEqual(c(0.789).tolist(), c(0.789).tolist())

    def test_linearized_equation(self):
        metric = BrillLindquistMetric(m1=1.4, m2=1.1, d=1.4, axis='z')
        h = CosineSeries(a_n=[6.0, 0.07, 0., 0., -0.03, -0.01])
        c_ref = StarShapedCurve(h, metric=metric)
        c = RefParamCurve.from_curve(c_ref)
        params = np.linspace(0.01, np.pi-0.01, 10)
        (dhH, dhpH, dhppH), inhom = c.linearized_equation(params)
        H = -inhom
        np.testing.assert_allclose(
            H,
            [0.11832, 0.137624, 0.15999, 0.153356, 0.140799, 0.148376,
             0.157814, 0.151442, 0.140906, 0.13725],
            atol=1e-6
        )
        np.testing.assert_allclose(
            dhH,
            [-0.004245, -0.013611, -0.038131, -0.033697, -0.032082, -0.036771,
             -0.042249, -0.033088, -0.021925, -0.018058],
            atol=1e-6
        )
        np.testing.assert_allclose(
            dhpH,
            [-11.269645,  -0.303452,  -0.12973 ,  -0.057644,  -0.015391,
             0.019615,   0.065263,   0.136898,   0.310508,  11.528608],
            atol=1e-6
        )
        np.testing.assert_allclose(
            dhppH,
            [-0.112699, -0.112406, -0.112435, -0.113337, -0.114513, -0.114832,
             -0.114682, -0.114787, -0.115116, -0.115289],
            atol=1e-6
        )

    def test_ricci_scalar(self):
        c = self.__c
        R = c.ricci_scalar(0.5)
        # This just records the currently produced value to detect if changes
        # to the code affect its value. Correctness of this particular value
        # is not yet checked.
        self.assertAlmostEqual(R, 0.1289847156947791)
        # TODO: Add analytically known cases (e.g. spheres in Schwarzschild metric)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
