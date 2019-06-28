#!/usr/bin/env python3

import unittest
import sys

from testutils import DpkTestCase, slowtest
from .numutils import IntegrationResults


class TestNumutils(DpkTestCase):
    def test_integrationresults(self):
        res = IntegrationResults(
            (1.2, 2e-2, dict()),
            (1.8, 3e-2, dict(), "Something went wrong."),
        )
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res.value, 3.0)
        self.assertAlmostEqual(res.error, 5e-2)
        self.assertFalse(res.all_ok())
        self.assertAlmostEqual(res[0].value, 1.2)
        self.assertAlmostEqual(res[0].error, 2e-2)
        self.assertAlmostEqual(res[1].value, 1.8)
        res = IntegrationResults(
            (1.2, 2e-2, dict()),
            (1.8, 3e-2, dict()),
        )
        self.assertTrue(res.all_ok())


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
