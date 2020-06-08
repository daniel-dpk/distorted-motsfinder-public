r"""@package motsfinder.axisym.trackmots.findslices

Find simulation slices in a folder hierarchy.
"""

from glob import glob
import os.path as op
import re


__all__ = [
    "find_slices",
]


def find_slices(sim_dir, backwards=False, start_slice=None, end_slice=None,
                max_slices=None, backwards_skip_first=True,
                skip_checkpoints=True, disp=True):
    r"""Gather all simulation data files.

    @param sim_dir
        Folder to start searching recursively for simulation data files.
    @param backwards
        Whether to reverse the order of the slices found. Default is `False`.
    @param start_slice
        First slice to consider. When ``backwards==True``, this should be the
        *latest* slice in the series. Note that if a start slice is specified
        in the ``backwards==True`` case, the slice itself is skipped to allow
        simultaneously starting forward and backwards runs.
        Default is to find all slices (including the last one in the backwards
        case).
    @param end_slice
        Last slice to consider. Take all (except for `start_slice`) if not
        given (default).
    @param max_slices
        Truncate the number of slices returned after this many if given.
    @param backwards_skip_first
        If `True` (default), skip the starting slice when going backwards as
        described above. If `False`, include the starting slice in all cases.
    @param skip_checkpoints
        Whether to ignore checkpoints in the simulation folders. Default is
        `True`.
    @param disp
        Raise a `ValueError` if no slices are found.
    """
    files = glob("%s/**/*.it*.s5" % sim_dir, recursive=True)
    # remove duplicates (caused by symlinks) by constructing a dict
    files = dict([(op.basename(fn), fn) for fn in files
                  if re.search(r"\.it\d{10}\.s5", fn)
                  and not re.search(r"output-\d{4}-active", fn)
                  and (not skip_checkpoints or not re.search(r"checkpoint\.", fn))])
    files = list(files.items())
    files.sort(key=lambda f: f[0]) # sort based on base name
    files = [fn for _, fn in files] # return the relative names
    if backwards:
        files.reverse()
    active = start_slice is None
    result = []
    for fn in files:
        if (start_slice is not None
                and re.search(r"\.it%010d\.s5" % start_slice, fn)):
            active = True
            if backwards and backwards_skip_first:
                # Starting a forward and backward search simultaneously
                # requires one of them to ignore the start slice.
                continue
        if active:
            result.append(fn)
        if (end_slice is not None
                and re.search(r"\.it%010d\.s5" % end_slice, fn)):
            break
    if disp and not result:
        raise ValueError("No simulation data found in: %s" % sim_dir)
    if max_slices and len(result) > max_slices:
        result = result[:max_slices]
    return result
