r"""@package motsfinder.pickle_helpers

Helper functions to (un)pickle problematic objects.

The main problem are mpmath matrices and constants such as `mp.pi`. Running
all your values to pickle through prepare_value() replaces the problematic
ones with placeholders or picklable versions. To restore the values to their
original form, run them through restore_value().

For convenience, prepare_dict() and restore_dict() do this on the values of
dictionaries, which is suitable for use in a `__getstate__()` and
`__setstate__()` implementation, respectively.

Example implementations might look like:

\code
def __getstate__(self):
    return prepare_dict(self.__dict__)

def __setstate__(self, state):
    self.__dict__.update(restore_dict(state))
\endcode
"""

from __future__ import print_function
from six import iteritems

from mpmath import mp


__all__ = [
    "prepare_value",
    "prepare_dict",
    "restore_value",
    "restore_dict",
]


class _MpMatrix(object):
    r"""Picklable version of an mpmath matrix."""
    # pylint: disable=too-few-public-methods
    def __init__(self, mat):
        r"""Init function.

        @param mat
            mpmath matrix which will be stored converted to a list. This list
            is picklable, since `mp.mpf` floats are.
        """
        self._mat = mat.tolist()

    @property
    def matrix(self):
        r"""Convert the stored representation back to a real mpmath matrix."""
        return mp.matrix(self._mat)


class _MpPi(object):
    r"""Dummy class representing the mpmath constant for pi."""
    # pylint: disable=too-few-public-methods
    @property
    def value(self):
        r"""The actual mpmath constant `mp.pi`."""
        return mp.pi


def prepare_value(value):
    r"""Prepare a value for being pickled.

    Most values are left untouched, only problematic ones are replaced by
    placeholders that can be pickled.

    There is no guarantee that all kinds of (e.g. nested) types are prepared
    in such a way that pickling succeeds. This function (and restore_value())
    should be extended on a case-by-case basis if such functionality is
    required.
    """
    if type(value) is tuple:
        value = tuple(prepare_value(v) for v in value)
    if type(value) is list:
        value = [prepare_value(v) for v in value]
    if type(value) is mp.matrix:
        value = _MpMatrix(value)
    if value is mp.pi:
        value = _MpPi()
    return value


def restore_value(value):
    r"""Restore an unpickled value to its original form."""
    if type(value) is tuple:
        value = tuple(restore_value(v) for v in value)
    if type(value) is list:
        value = [restore_value(v) for v in value]
    if isinstance(value, _MpMatrix):
        value = value.matrix
    if isinstance(value, _MpPi):
        value = value.value
    return value


def prepare_dict(data):
    r"""Convenience method to run prepare_value() on a dict."""
    return dict((k, prepare_value(v)) for k, v in iteritems(data))


def restore_dict(data):
    r"""Convenience method to run restore_dict() on a dict."""
    return dict((k, restore_value(v)) for k, v in iteritems(data))
