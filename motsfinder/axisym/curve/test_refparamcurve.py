#!/usr/bin/env python3

import unittest
import sys
import pickle

import numpy as np

from testutils import DpkTestCase, slowtest
from ...metric import BrillLindquistMetric
from ...exprs.trig import CosineSeries
from .basecurve import BaseCurve
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

    @slowtest
    def test_loaded_MOTS_basics(self):
        c = BaseCurve.load('testdata/BL_inner_d0_65_ratio4.npy')
        self.assertAlmostEqual(c.area(), 50.26868688267552)
        self.assertAlmostEqual(c.arc_length(), 7.644951949268928)
        self.assertAlmostEqual(c.inner_z_distance(self.__c, where='top'),
                               -3.3182427382084243)
        self.assertAlmostEqual(c.inner_z_distance(self.__c, where='bottom'),
                               -2.673169190394275)
        self.assertAlmostEqual(c.inner_x_distance(self.__c, where='zero'),
                               3.7988046780100144)
        self.assertAlmostEqual(c.inner_x_distance(self.__c, where='max'),
                               3.260430220907522)

    @slowtest
    def test_loaded_MOTS_expansions(self):
        c = BaseCurve.load('testdata/BL_inner_d0_65_ratio4.npy')
        self.assertAlmostEqual(c.expansion(np.pi/3), 0., places=9)
        N = 19
        space = np.linspace(0., np.pi, N+1, endpoint=False)[1:]
        self.assertListAlmostEqual(
            c.expansions(space), [0.]*N, delta=1e-9,
        )
        self.assertAlmostEqual(c.average_expansion(), 0., places=9)
        with c.temp_metric(self.__metric):
            self.assertAlmostEqual(
                c.average_expansion(), -0.1369950947087454,
            )

    @slowtest
    def test_loaded_MOTS_ricci(self):
        c = BaseCurve.load('testdata/BL_inner_d0_65_ratio4.npy')
        N = 19
        space = np.linspace(0., np.pi, N+1, endpoint=False)[1:]
        self.assertListAlmostEqual(
            [c.ricci_scalar(l) for l in space],
            [ 4.62121874,  4.39457175,  3.73114053,  1.41863392, -4.09206428, -3.79545421,
             -0.89220898,  0.12821144,  0.37170241,  0.4604185,   0.50859362,  0.54079685,
              0.5642595,   0.58189917,  0.59524478,  0.60523398,  0.61248341,  0.61740384,
              0.62025835],
            delta=1e-6,
        )

    @slowtest
    def test_loaded_MOTS_extrinsic(self):
        c = BaseCurve.load('testdata/BL_inner_d0_65_ratio4.npy')
        N = 19
        space = np.linspace(0., np.pi, N+1, endpoint=False)[1:]
        self.assertListAlmostEqual(
            [c.extrinsic_surface_curvature(l, trace=True) for l in space],
            [0.]*N,
            delta=1e-9,
        )
        self.assertListAlmostEqual(
            [c.extrinsic_surface_curvature(l, square=True) for l in space],
            [0.0004650227072452532, 0.011084157441843656, 0.10886137493380305,
             0.8522911953494112, 3.6707108339649634, 3.0908212552520284,
             0.8503288361300836, 0.1562792581464134, 0.033619451315009125,
             0.008369436950131984, 0.002264349179561774, 0.0006415246831802435,
             0.0001851092886020002, 5.299909468126335e-05, 1.4598750364301464e-05,
             3.693967744092343e-06, 7.877942093287853e-07, 1.1580823992854409e-07,
             6.022442900852292e-09],
            delta=1e-8,
        )

    @slowtest
    def test_loaded_MOTS_stability(self):
        c = BaseCurve.load('testdata/BL_inner_d0_65_ratio4.npy')
        s, spectrum = c.stability_parameter(m_max=0, full_output=True)
        vals = spectrum.get(l='all', m=0)
        self.assertAlmostEqual(s, -0.700531266359229)
        self.assertAlmostEqual(sorted(vals.real)[0], -0.700531266359229)
        self.assertAlmostEqual(sorted(vals.real)[1], 0.4060364507055123)

    @slowtest
    def test_loaded_MOTS_multipoles(self):
        c = BaseCurve.load('testdata/BL_inner_d0_65_ratio4.npy')
        self.assertListAlmostEqual(
            c.multipoles(max_n=3),
            [np.sqrt(np.pi), 0., 1.0346628126506796, 1.9559062623823318],
            delta=1e-6,
        )


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
