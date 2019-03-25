r"""@package motsfinder.utils

General utilities for simplifying certain tasks in Python.
"""

from __future__ import print_function
import functools
import importlib.util
from builtins import range, map

from glob import glob
import os
import os.path as op
import re
import subprocess
import time
from timeit import default_timer
import datetime
from contextlib import contextmanager

import numpy as np


__all__ = [
    "get_git_commit",
    "import_file_as_module",
    "lmap",
    "lrange",
    "isiterable",
    "get_chunks",
    "parallel_compute",
    "process_pool",
    "merge_dicts",
    "update_dict",
    "insert_missing",
    "print_indented",
    "timethis",
    "cache_method_results",
    "find_file",
    "find_files",
]


def get_git_commit():
    r"""Return the output of `git describe` to identify the current version.

    Note that `git` has to be installed and the project source needs to be in
    a git repository at runtime.
    """
    cwd = op.dirname(op.realpath(__file__))
    result = subprocess.check_output(
        ["git", "describe", "--always", "--dirty"],
        cwd=cwd
    )
    return result.decode('utf-8').strip()


def import_file_as_module(fname, mname='loaded_module'):
    r"""Given a filename, load it as a Python module.

    Any classes or functions defined in the file can then be retrieved as
    attributes from the returned module. This allows e.g. filenames to be
    provided as arguments to functions in case a function should be loaded
    from that file during runtime.
    """
    spec = importlib.util.spec_from_file_location(mname, fname)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return cfg_module


def lmap(func, *iterables):
    r"""Implementation of `map` that returns a list instead of a generator."""
    # This is efficient even in Python 2 due to the builtins.map import.
    return list(map(func, *iterables))


def lrange(*args):
    r"""Convenience wrapper to call `list(range(...))`."""
    return list(range(*args))


def isiterable(obj):
    """Check whether an object is iterable.

    Note that this returns `True` for strings, which you may or may not intend
    to check for.
    """
    try:
        _ = iter(obj)
    except TypeError:
        return False
    return True


def get_chunks(items, chunksize):
    """Generator for partitioning items into chunks."""
    if not isinstance(items, (list, tuple)):
        items = list(items)
    for i in range(0, len(items), chunksize):
        yield items[i:i+chunksize]


def parallel_compute(func, arg_list, args=(), kwargs=None, processes=None,
                     callstyle='plain', pool=None):
    r"""Perform a task on a list of arguments in parallel.

    This uses multiple processes to perform the computation of `func` once for
    each element of `arg_list` in parallel. This is *not* multithreading but
    multiprocessing, which means that all objects (including any `self`
    attached to `func` in case that is a bound object method) must be
    picklable for cross-process transfer.

    This function assumes that copying the necessary objects is expensive and
    hence it does not use a processing pool to optimally distribute the tasks
    (as that would copy the object bound to `func` on each call). Instead, the
    argument list is chopped up into `N` tasks, where `N==processes` (if
    given) is the number of parallel processes to invoke. Once all processes
    have finished, the results are collected and returned in the order they
    were passed in `arg_list`.

    @param func
        Callable to invoke for each element of `arg_list`. May be a bound
        method in which case `self` must be picklable.
    @param arg_list
        Iterable of arguments to let `func` work on. Each element may be the
        argument itself, or a list/tuple of arguments, or a dictionary with
        keyword arguments. The interpretation is configured using `callstyle`.
    @param args
        Additional positional arguments to pass to `func` in each call.
    @param kwargs
        Additional keyword arguments to pass to `func` in each call.
    @param processes
        Number of processes to run in parallel. If not given, uses the number
        of available threads of the current architecture.
    @param callstyle
        How to pass the individual arguments in `arg_list` to `func`.
        Default is `plain`. The values mean: `plain` to pass the argument as
        first positional argument, `list` to expand the argument into a list
        and pass the elements as individual positional arguments, or `dict` to
        treat the elements as dictionaries to pass as keyword arguments to
        `func`.
    @param pool
        Optional pool to re-use. If not given, a new pool is created for the
        given number of parallel tasks. Note that the child processes are
        terminated only if the pool was created within this function. Any
        supplied `pool` will be left as-is.
    """
    styles = ['plain', 'list', 'dict']
    if callstyle not in styles:
        raise ValueError("Unknown callstyle '%s'. Valid styles: %s"
                         % (callstyle, ", ".join(styles)))
    f = _FuncWrap(func, callstyle, args=args, kwargs=kwargs or dict())
    if processes is None:
        processes = len(os.sched_getaffinity(0))
    if processes <= 1:
        return list(map(f, arg_list))
    chunks = [[] for i in range(processes)]
    for items in list(get_chunks(arg_list, processes)):
        for i, a in enumerate(items):
            chunks[i].append(a)
    runners = [_Runner(f, chunk) for chunk in chunks]
    with process_pool(processes=processes, pool=pool) as p:
        workers = [p.apply_async(runner, ()) for runner in runners]
        results = [None] * len(arg_list)
        for i, worker in enumerate(workers):
            worker_results = worker.get()
            for j, result in enumerate(worker_results):
                results[i+j*processes] = result
    return results


@contextmanager
def process_pool(processes=None, pool=None):
    r"""Context to create a pool and terminate cleanly after usage.

    @param processes
        Number of processes to run in parallel. If not given, uses the number
        of available threads of the current architecture.
    @param pool
        If given, simply returns that pool without terminating it afterwards.
    """
    if pool is not None:
        yield pool
    else:
        if processes is None:
            processes = len(os.sched_getaffinity(0))
        from multiprocessing import Pool
        with Pool(processes=processes) as pool:
            yield pool


class _Runner():
    r"""Class to store a function and argument list chunk."""

    __slots__ = ("_f", "_arg_chunk",)

    def __init__(self, func, arg_chunk):
        r"""Create a runner object for a given chunk of arguments."""
        self._f = func
        self._arg_chunk = arg_chunk

    def __call__(self):
        r"""Call the function on each of the arguments and return the result."""
        return [self._f(arg) for arg in self._arg_chunk]


class _FuncWrap():
    r"""Helper class to interpret arguments and call a function."""

    __slots__ = ("_f", "_callstyle", "_args", "_kwargs")

    def __init__(self, func, callstyle, args, kwargs):
        r"""Create a function wrapper.

        @param func
            Function to use.
        @param callstyle
            Callstyle for invoking `func` (see parallel_compute()).
        @param args
            Positional arguments for `func` (see parallel_compute()).
        @param kwargs
            Keyword arguments for `func` (see parallel_compute()).
        """
        self._f = func
        self._callstyle = callstyle
        self._args = args
        self._kwargs = kwargs

    def __call__(self, args):
        r"""Call the function with the given args interpreted as configured."""
        if self._callstyle == 'plain':
            return self._f(args, *self._args, **self._kwargs)
        if self._callstyle == 'list':
            return self._f(*args, *self._args, **self._kwargs)
        return self._f(*self._args, **args, **self._kwargs)


def merge_dicts(*dicts):
    """Merge two or more dicts, later ones replacing values of earlier ones.

    Note that only shallow copies are made of the dicts.

    Example:
        a = dict(a=1, b=2, c=3)
        b = dict(c=-3, d=-4)
        c = merge_dicts(a, b)
        # Result: dict(a=1, b=2, c=-3, d=-4)
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def update_dict(dict_arg, **kwargs):
    r"""Merge items into a dict non-destructively.

    A new dict is returned, the original dict is not changed.
    """
    return merge_dicts(dict_arg, kwargs)


def insert_missing(dict_arg, **kwargs):
    """Insert all missing kwargs into the given dict and return it.

    Note that the original dict `dict_arg` is not altered.
    """
    return merge_dicts(kwargs, dict_arg)


def print_indented(prefix, obj):
    r"""Print a prefix followed by an object with correct indenting of multiple lines.

    Printing an object that has a multi-line string representation looks ugly
    when prefixed by another string.

    @b Examples

    \code
        >>> A = np.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> print("A = %s" % A)
        A = [[1 2 3]
         [4 5 6]
         [7 8 9]]

    \endcode

    With this function, we can do instead:

    \code
        >>> A = np.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> print_indented("A = ", A)
        A = [[1 2 3]
             [4 5 6]
             [7 8 9]]

    \endcode
    """
    if isinstance(prefix, int):
        prefix = " " * prefix
    m = len(prefix)
    lines = (prefix + str(obj)).splitlines()
    result = [lines[0]]
    result += [(" "*m)+l for l in lines[1:]]
    print("\n".join(result))


@contextmanager
def timethis(start_msg=None, end_msg="Elapsed time: {}", silent=False, eol=True):
    r"""Context manager for timing code execution.

    Args:
        start_msg: String to print at the beginning. May contain the
            placeholder ``'{now}'``, which will be replaced by the current
            date and time. A value of `True` will be taken to mean
            ``"Started: {now}``.
        end_msg: String to print after execution. Default is
            ``"Elapsed time: {}"``.
        silent: Whether to print anything at all. May be useful when a
            function has a verbosity setting to conditionally time its results.
        eol: Whether to print a newline after each message. May be useful to
            print execution time in line with the starting message.
    """
    if silent:
        yield
        return
    if start_msg is True:
        start_msg = "Started: {now}"
    if start_msg is not None:
        print(start_msg.format(now=time.strftime('%Y-%m-%d %H:%M:%S')),
              end='\n' if eol else '', flush=not eol)
    start = default_timer()
    try:
        yield
    finally:
        if end_msg is not None:
            time_str = datetime.timedelta(seconds=default_timer()-start)
            print(end_msg.format(time_str))


def cache_method_results(key=None):
    r"""Create a decorator to cache instance method results.

    The decorator is rather simplistic and the overhead is comparable to a few
    extra arithmetic operations.

    It will create a new attribute ``'_method_cache'`` on the instance
    containing a dictionary with keys for each cached method. These will also
    be dictionaries containing the results for the different arguments.

    @b Limitations

    The following limitations are due to its simplicity:
        * does not work for classes with slots unless the ``'_method_cache'``
          slot is added
        * cached methods cannot be called with keyword args (you may use a
          non-cached wrapper method) or non-hashable types
        * the cache is not limited to a certain size
        * if a cached method of a subclasses has the same name as a cached
          method of a super class, you need to specify the `key` argument to
          set a unique key for the method

    @param key
        Unique key to store the method's results in. By default, the method
        name is used (obtained via ``method.__name__``).

    @b Examples
    ```
        class MyClass():
            @cache_method_results()
            def some_lengthy_computation(self, a, b, c):
                result = a**2 + b**3 + c**4
                return result
    ```
    """
    def cache_decorator(method):
        fn_key = method.__name__ if key is None else key
        @functools.wraps(method)
        def wrapper(self, *args):
            method_cache = getattr(self, '_method_cache', dict())
            self._method_cache = method_cache
            cache = method_cache[fn_key] = method_cache.get(fn_key, dict())
            try:
                return cache[args]
            except KeyError:
                result = method(self, *args)
                cache[args] = result
                return result
        return wrapper
    return cache_decorator


def save_to_file(filename, data, overwrite=False, verbose=True,
                 showname='data', mkpath=True):
    r"""Save an object to disk.

    This uses `numpy.save()` to store an object in a file. Use
    load_from_file() to restore the data afterwards.

    @param filename
        The file name to store the data in. An extension ``'.npy'`` will be
        added if not already there.
    @param overwrite
        Whether to overwrite an existing file with the same name. If `False`
        (default) and such a file exists, a `RuntimeError` is raised.
    @param verbose
        Whether to print when the file was written. Default is `True`.
    @param showname
        Name to print in the confirmation message in case `verbose==True`.

    @b Notes

    The data will be put into a 1-element list to avoid creating 0-dimensional
    numpy arrays.
    """
    filename = op.expanduser(filename)
    if not filename.endswith('.npy'):
        filename += '.npy'
    if mkpath:
        os.makedirs(op.normpath(op.dirname(filename)), exist_ok=True)
    if op.exists(filename) and not overwrite:
        raise RuntimeError("File already exists.")
    np.save(filename, [data])
    if verbose:
        print("%s saved to: %s" % (showname, filename))


def load_from_file(filename):
    r"""Load an object from disk.

    If the object had been stored using save_to_file(), the result should be a
    perfect copy of the object.

    @b Notes

    This assumes the object is the only element of a list stored in the file,
    which will be the case if the file was created using save_to_file(). If
    the data is not a single-element list, it is returned as is.
    """
    filename = op.expanduser(filename)
    result = np.load(filename)
    if result.shape == (1,):
        return result[0]
    # Not a single value. Return as is.
    return result


def find_file(pattern, recursive=False, skip_regex=None, regex=None,
              load=False, full_output=False, verbose=False):
    r"""Find a file based on a glob pattern and optional regex patterns.

    Optionally, a matching file can be loaded. In this case, it is ignored if
    the file exists but contains no data (i.e. `None` was saved).

    @param pattern
        Shell glob pattern. May contain ``'**'`` if `recursive=True`.
    @param recursive
        Activate recursive wildcard ``'**'`` in `pattern`. Default is `False`.
    @param skip_regex
        Files matching this regex will be ignored.
    @param regex
        If given, only files matching this regex will be considered.
    @param load
        If `True`, load the file and return the data. Default is `False`.
    @param full_output
        If `load=True`, return both the data and filename. Default is `False`,
        i.e. only return the data. Ignored if `load=False`.
    @param verbose
        Print a note in case data was loaded from a file.
    """
    result = find_files(pattern, recursive=recursive, skip_regex=skip_regex,
                        regex=regex, load=load, full_output=full_output,
                        max_num=1, verbose=verbose)
    if len(result) == 1:
        return result[0]
    raise FileNotFoundError("No data found for pattern: %s" % pattern)


def find_files(pattern, recursive=False, skip_regex=None, regex=None,
               load=False, full_output=False, max_num=None, verbose=False):
    r"""Find files based on a glob pattern and optional regex patterns.

    Optionally, all matching files can be loaded. In this case, files are
    ignored if they contains no data (i.e. `None` was saved).

    @param pattern
        Shell glob pattern. May contain ``'**'`` if `recursive=True`.
    @param recursive
        Activate recursive wildcard ``'**'`` in `pattern`. Default is `False`.
    @param skip_regex
        Files matching this regex will be ignored.
    @param regex
        If given, only files matching this regex will be considered.
    @param load
        If `True`, load the files and return the data. Default is `False`.
    @param full_output
        If `load=True`, return both the data and filenames. Default is
        `False`, i.e. only return the data. Ignored if `load=False`.
    @param max_num
        Maximum number of files to collect. Default is to collect all
        matching files.
    @param verbose
        Print a note in case data was loaded from a file.
    """
    result = []
    for fn in glob(pattern, recursive=recursive):
        if skip_regex and re.search(skip_regex, fn):
            continue
        if regex and not re.search(regex, fn):
            continue
        if load:
            obj = load_from_file(fn)
            if obj:
                if verbose:
                    print("Data loaded: %s" % fn)
                result.append((obj, fn) if full_output else obj)
        else:
            result.append(fn)
        if max_num and len(result) == max_num:
            break
    return result
