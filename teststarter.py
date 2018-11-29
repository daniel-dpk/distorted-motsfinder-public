#!/usr/bin/env python

import sys
import os.path as op
sys.path.append(op.dirname(op.realpath(__file__)))
import call_python
call_python.Main("tests.py", *sys.argv[1:]).main()
