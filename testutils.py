r"""@package testutils

Utilities to add more features to `unittest` classes.

This module provides a custom subclass of `unittest.TestCase`, namely
DpkTestCase, which obeys the global configuration settings in TestSettings.
The latter can be configured by the script invoking the test run.

This module also introduces a new decorator slowtest, which, when applied,
leads to the test being skipped on normal runs. The script starting the test
must set `TestSettings.skipslow` to `False` for the slow tests to be run.
"""

from __future__ import print_function
from builtins import range
import sys
import functools
import unittest
import time


__all__ = [
    "DpkTestCase",
    "TestSettings",
    "slowtest",
]


def slowtest(func):
    """Decorator for skipping a test if TestSettings.skipslow is true."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if TestSettings.skipslow:
            raise unittest.SkipTest("skipping slow tests")
        return func(*args, **kwargs)
    return wrapper


class DpkTestCase(unittest.TestCase):
    """Tweaked baseclass for unit tests.

    By deriving from this class, you:
        * Get timed individual tests (needs `verbosity=2`) if
          TestSettings.timing is true.
        * Can implement a hook (failureHook()) that is called after a test has
          failed (or errored). It is called before tearDown(), allowing you
          to, for example, collect generated files for inspection before they
          are deleted.
    """
    @classmethod
    def setUpClass(cls):
        if cls is not DpkTestCase:
            if cls.setUp is not DpkTestCase.setUp:
                setUp = cls.setUp
                @functools.wraps(setUp)
                def setUpWrapper(self, *args, **kwargs):
                    DpkTestCase.setUp(self)
                    return setUp(self, *args, **kwargs)
                cls.setUp = setUpWrapper
            if cls.tearDown is not DpkTestCase.tearDown:
                tearDown = cls.tearDown
                @functools.wraps(tearDown)
                def tearDownWrapper(self, *args, **kwargs):
                    DpkTestCase.tearDown(self)
                    return tearDown(self, *args, **kwargs)
                cls.tearDown = tearDownWrapper

    def __lastTestSkipped(self):
        r"""Return whether the previous test was skipped."""
        if self.__result is None:
            return False
        return len(self.__result.skipped) > self.__prevSkipped

    def __lastTestOK(self):
        r"""Return whether the previous test result was success."""
        if self.__result is None:
            return True
        if len(self.__result.errors) > self.__prevErrors or len(self.__result.failures) > self.__prevFailures:
            return False
        return True

    def __shouldPrintTiming(self):
        r"""Return whether timing information should be printed."""
        if not TestSettings.timing or self.__lastTestSkipped() or not self.__lastTestOK():
            return False
        if self.__result is None:
            return True
        return not self.__result.dots and self.__result.showAll

    def run(self, result=None):
        self.__result = result
        self.__prevErrors = 0
        self.__prevFailures = 0
        self.__prevSkipped = 0
        if result is not None:
            self.__prevErrors = len(result.errors)
            self.__prevFailures = len(result.failures)
            self.__prevSkipped = len(result.skipped)
        unittest.TestCase.run(self, result)

    def setUp(self):
        self.startTime = time.time()
        self.__tornDown = False

    def tearDown(self):
        if self.__tornDown: return
        self.__tornDown = True
        if not self.__lastTestOK():
            self.failureHook(self.__result)
        if self.__shouldPrintTiming():
            duration = time.time() - self.startTime
            print("(%.4f seconds) ... " % (duration), file=sys.stderr, end='')

    def failureHook(self, result):
        r"""Custom function called just after a fail/error occurred.

        Subclasses may implement this function to e.g. collect result data
        before tearDown() gets called.
        """
        pass

    def assertIsType(self, obj, cls):
        r"""Assert that an object is exactly of a certain type."""
        self.assertIs(type(obj), cls)

    def assertListAlmostEqual(self, a, b, places=None, delta=None):
        r"""Assert that two iterables contain (almost) the same values."""
        if places is not None and delta is not None:
            raise TypeError("Cannot use delta and places at the same time")
        if places is None and delta is None:
            places = 7
        if len(a) != len(b):
            raise self.failureException("Lists have different lengths (%d != %d)" % (len(a), len(b)))
        fails = []
        for i in range(len(a)):
            if a[i] == b[i]:
                continue
            if delta is not None:
                if abs(a[i]-b[i]) > delta:
                    fails.append(i)
            else:
                if round(abs(a[i]-b[i]), places) != 0:
                    fails.append(i)
        if fails:
            msg = "%d elements differ.\n" % len(fails)
            maxN = 9
            if len(fails) <= maxN:
                msg += "Differing elements:\n"
            else:
                msg += "First few differing elements:\n"
            msg += "\n".join(["  [{i}] {a} != {b}    (difference: {d})".format(i=i, a=a[i], b=b[i], d=(b[i]-a[i]))
                              for i in fails[:maxN]])
            raise self.failureException(msg)


class TestSettings(object):
    """Global settings for tests."""
    ## Stop test run on first fail/error.
    failfast = False
    ## Whether output is buffered.\ Just information, cannot be used to toggle output buffering.
    buffering = False
    ## Whether the timing for each test case should be printed.
    timing = False
    ## Control whether tests marked as slowtest should be skipped.
    skipslow = True
