#!/usr/bin/env python3
r"""@package fix_doxypypy

Script to fix a few of the issues with doxypypy.

To have Doxygen tags be interpreted inside Python docstrings, we currently use
doxypypy (https://pypi.org/project/doxypypy/). This translates the docstrings
and converts them to "proper" Doxygen comments above the documented entity.
However, there are several problems with input being manipulated in seemingly
inconsistent ways.

The problems encounters thus far are:
    * Code blocks being created from regular text, sometimes depending on how
      many empty lines precede the text or how long the preceding non-empty
      line was.
    * Code blocks not being closed at the appropriate point.
    * Parameter descriptions having a short last line end up appearing as
      several parameters.
    * Words and punctuation being removed from the output. This also seems to
      depend on line lengths, so restructuring sentences/lines can fix this
      sometimes.
    * LaTeX formulas ending in punctuation sometimes get truncated.
    * LaTeX formulas containing primes `'` to indicate derivatives sometimes
      lead to the primes and a few more characters disappearing. This can
      sometimes be fixed by not putting the LaTeX code on its own line.

An example for the last mentioned (LaTeX) problem is the following.
This line gets rendered incorrectly:
\f[
    f''(x) - a b \sin(b x) f'(x) = a b^2 \cos(b x) f(x)
\f]
This following gets rendered correctly:
\f[ f''(x) - a b \sin(b x) f'(x) = a b^2 \cos(b x) f(x) \f]


## What this Script does

This script loads the `doxypypy` module and patches it to remove some of the
features which lead to the above problems. The produced output is then
transformed again to fix code-block detections. Finally, the generated code is
printed to `stdout`.

For example, consider the following docstring

    class MyClass():
        def foo(self):
            r'''Brief description.

            Some equation:
            \f[
                f''(x) - a b \sin(b x) f'(x) = a b^2 \cos(b x) f(x)
            \f]

            Code example:

                def my_foo(x):
                    return 1 - 2*x

            A sentence that ends with two words separated by a comma on a single
            line, etc.
            '''
            pass

Doxypypy on its own converts this to:

    class MyClass():
        ## @brief Brief description.
        #
        #       Some equation:
        #       \f[
        # f a b \sin(b x) f'(x) = a b^2 \cos(b x) f(x)
        #       \f]
        #
        #       Code example:
        #
        #           def my_foo(x):
        # return    2*x
        #
        #       A sentence that ends with two words separated by a comma on a single
        # line
        # etc.
        #
        def foo(self):
            pass

This script instead produces:

    class MyClass():
        ## @brief Brief description.
        #
        # Some equation:
        # \f[
        #     f''(x) - a b \sin(b x) f'(x) = a b^2 \cos(b x) f(x)
        # \f]
        #
        # Code example:
        #
        #     def my_foo(x):
        #         return 1 - 2*x
        #
        # A sentence that ends with two words separated by a comma on a single
        # line, etc.
        #
        def foo(self):
            pass
"""

from contextlib import redirect_stdout
from io import StringIO
import sys
import re

from doxypypy import doxypypy


class DoxyCommentBlock():
    r"""Class representing a single Doxygen comment block.

    This is used to keep track of where a Doxygen comment block starts and
    ends and performs the de-indentation for printing.
    """
    def __init__(self):
        self._brief = None
        self._long = []
        self._start_pat = re.compile(r"\s*(?:## @brief\s|##[^#\s])")
        self._end_pat = re.compile(r"\s*(?:[^#\s]|## @brief\s)")
        self._indent_pat = re.compile(r"\s*#(\s*)")
        self._empty_pat = re.compile(r"\s*#(?:\s*| @\w+)?$")
        self._nonempty_pat = re.compile(r"(\s*#\s*)(\S.*)")
        self._invalid_brief_pat = re.compile(r"(\s*##\s*)@brief\s+(@package\s+.*)")
        self._invalid_brief = False

    def reset(self):
        r"""Reset this object to be ready to find the next block."""
        self._brief = None
        self._long = []

    @property
    def in_block(self):
        r"""Whether we're currently in a Doxygen comment block."""
        return self._brief is not None

    def starts_block(self, line):
        r"""Check whether a given line starts a Doxygen comment block."""
        return self._start_pat.match(line)

    def ends_block(self, line):
        r"""Check whether a given line is outside the comment block."""
        return self._end_pat.match(line)

    def start_block(self, brief):
        r"""Start a new comment block with a given brief description."""
        if self.in_block:
            raise RuntimeError(
                "Cannot start Doxygen comment inside Doxygen comment."
            )
        m = self._invalid_brief_pat.match(brief)
        if m:
            self._brief = "%s%s" % (m[1], m[2])
            self._invalid_brief = True
        else:
            self._brief = brief
            self._invalid_brief = False

    def add(self, line):
        r"""Add a line to the long description of this comment block."""
        self._long.append(line)

    def _get_indent(self, line):
        r"""Find the indentation of a given line.

        This is not the indentation of the line itself but the indentation of
        the description text after the first comment hash `#`. Empty lines and
        those with only single-word properties usually put by doxypypy
        directly above the functions (and ignoring any indentation) are not
        counted, i.e. this function returns `None` for these lines.
        """
        if self._empty_pat.match(line):
            return None
        m = self._indent_pat.match(line)
        if m:
            return len(m[1])
        return None

    def long_desc(self):
        r"""Return a generator for the fixed long description block."""
        lines = self._long
        if self._invalid_brief:
            lines = lines.copy()
            for i, line in enumerate(lines):
                m = self._nonempty_pat.match(line)
                if m:
                    lines[i] = "%s@brief %s" % (m[1], m[2])
                    break
        indents = map(self._get_indent, lines)
        indents = [i for i in indents if i is not None]
        if not indents:
            for line in lines:
                yield line
            return
        indent = min(indents)
        pat = re.compile(r"(^\s*#)\s{%d}" % indent)
        for line in lines:
            yield pat.sub(r"\1 ", line)

    def __str__(self):
        r"""Fix the comment block and return it as a string."""
        return "%s\n%s" % (self._brief, "\n".join(self.long_desc()))


def main():
    r"""Perform a (patched) doxypypy run and then fix comment blocks."""

    # Patch the AstWalker class to remove heuristically detecting "args"
    # constructs (which often misbehave at unexpected places) and the
    # automatic "list" detection (which misbehaves equally). Ideally, doxypypy
    # should have flags to disable these two features.
    never_match = re.compile(r"(?!a)a")
    doxypypy.AstWalker._AstWalker__argsRE = never_match
    doxypypy.AstWalker._AstWalker__listRE = never_match

    strout = StringIO()
    error = None
    with redirect_stdout(strout):
        try:
            doxypypy.main()
        except SystemExit:
            error = True
        except:
            error = "Error: %s" % sys.exc_info()[0]
    out = strout.getvalue()
    if error:
        print(out)
        if error is not True:
            print(error)
        sys.exit()
    lines = out.rstrip('\r\n').split('\n')

    # Perform additional clean-up after doxypypy had its go.
    doxy_block = DoxyCommentBlock()
    for line in lines:
        line = line.rstrip('\r\n')
        if doxy_block.in_block:
            if doxy_block.ends_block(line):
                print(doxy_block)
                doxy_block.reset()
                print(line)
            else:
                doxy_block.add(line)
        else:
            if doxy_block.starts_block(line):
                doxy_block.start_block(line)
            else:
                print(line)
    if doxy_block.in_block:
        print(doxy_block)


if __name__ == '__main__':
    main()
