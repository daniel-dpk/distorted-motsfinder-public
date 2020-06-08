#!/usr/bin/env python3
r"""@package md_filter

Script to prepare markdown files meant for Github browsing for Doxypy.
"""

import sys
import os.path as op
import re


class Parser():
    def __init__(self, fname):
        self.fname = fname
        self.root = op.dirname(op.dirname(op.realpath(__file__)))
        self.path = op.dirname(op.realpath(fname))
        if not self.path.startswith(self.root):
            raise RuntimeError("Could not find relative path of .md file.")
        self.relpath = self.path[len(self.root)+1:]

    def process_line(self, line):
        m = re.search(r"(.*)\[([^\]]+)\]\(([^\s\)]+)\)(.*)", line)
        if m:
            line = "%s[%s](%s)%s" % (m[1], m[2], self.process_link(m[3]), m[4])
        return line

    def process_link(self, link):
        link = self._process_link(link)
        if link == "README.md":
            return "index.html"
        return link

    def _process_link(self, link):
        if link.startswith("docs_input/") and (link.endswith(".html") or
                                               link.endswith(".ipynb")):
            # Links to static HTML pages in docs_input/ should be preserved.
            return "../../%s" % link
        comps = link.split("/")
        if not comps[0].startswith("."):
            return link
        if comps[0] == ".":
            comps[0] = self.relpath
        up = len([1 for c in comps if c == ".."])
        if up:
            path = self.relpath
            for _ in range(up):
                path = op.dirname(path)
            comps = [path] + [c for c in comps if c != ".."]
        comps = [c for c in comps if c]
        link = "/".join(comps)
        return link

    def doit(self):
        with open(self.fname, 'r') as f:
            for line in f:
                line = line.rstrip('\r\n')
                # The Zenodo link doesn't work in Doxygen generated docs.
                if re.match(r"\[!\[DOI.*zenodo\.org.*\.svg.*\)$", line):
                    continue
                line = self.process_line(line)
                print(line)


def main(fname):
    Parser(fname).doit()


if __name__ == '__main__':
    main(fname=sys.argv[1])
