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

This script reads from `stdin` and prints to `stdout`, i.e. you'd pipe this
after `doxypypy` in your filter command.

It does *not* fix all the problems mentioned above (a re-implementation of
`doxypypy` would probably be the easiest solution). However, a few of the
layout related and code-block problems are mitigated by removing the excessive
indentation that `doxypypy` keeps in its output. This is done by finding the
common minimum indentation of a Doxygen comment block created by `doxypypy`
and replacing it by one space only.

For example, consider the following docstring

    class MyClass():
        def foo(self):
            r'''Brief description.

            Long description.
            '''

Doxypypy converts this to:

    class MyClass():
        ## @brief Brief description.
        #
        #        Long description.
        #
        def foo(self):
            pass

This script then translates this to:

    class MyClass():
        ## @brief Brief description.
        #
        # Long description.
        #
        def foo(self):
            pass
"""

import sys
import re


class DoxyCommentBlock():
    r"""Class representing a single Doxygen comment block.

    This is used to keep track of where a Doxygen comment block starts and
    ends and performs the de-indentation for printing.
    """
    def __init__(self):
        self._brief = None
        self._long = []
        self._start_pat = re.compile(r"\s*(?:## @brief\s|##[^#\s])")
        self._end_pat = re.compile(r"\s*[^#\s]")
        self._indent_pat = re.compile(r"\s*#(\s*)")
        self._empty_pat = re.compile(r"\s*#(?:\s*| @\w+)?$")

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
        self._brief = brief

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
        indents = map(self._get_indent, self._long)
        indents = [i for i in indents if i is not None]
        if not indents:
            for line in self._long:
                yield line
            return
        indent = min(indents)
        pat = re.compile(r"(^\s*#)\s{%d}" % indent)
        for line in self._long:
            yield pat.sub(r"\1 ", line)

    def __str__(self):
        r"""Fix the comment block and return it as a string."""
        return "%s\n%s" % (self._brief, "\n".join(self.long_desc()))


def main():
    r"""Read from stdin, fix comment block, and print to stdout."""
    doxy_block = DoxyCommentBlock()
    for line in sys.stdin:
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
