#!/usr/bin/env python3

import unittest
import os
import sys

import os.path as op
sys.path.append(op.realpath(op.join(__file__, op.pardir, op.pardir)))

import cythonize_all
from testutils import TestSettings


def run_tests():
    try:
        cythonize_all.main()
    except ImportError:
        # Cythonize is not mandatory for this project. If the call fails, the
        # slower pure Python implementations will be used.
        print("NOTE: Cython not installed. Not producing optimized modules.")
    failfast = '-f' in sys.argv or '--failfast' in sys.argv
    buffering = '-b' in sys.argv or '--buffer' in sys.argv
    timing = '-t' in sys.argv or '--timing' in sys.argv
    runSlow = '-s' in sys.argv or '--run-slow-tests' in sys.argv
    TestSettings.failfast = failfast
    TestSettings.buffering = buffering
    TestSettings.timing = timing
    TestSettings.skipslow = not runSlow
    suite = unittest.TestLoader().discover(os.path.dirname(os.path.realpath(__file__)), pattern="test_*.py")
    return len(unittest.TextTestRunner(verbosity=2, failfast=failfast, buffer=buffering).run(suite).failures)


if __name__ == '__main__':
    run_tests()
