# Preparation to running the examples

See the [README.md](../README.md) for basic installation instructions.

All the examples require the following *init cell* to work:

```.py
%matplotlib inline
import sys
import os.path as op

# Change the path to where the code is located.
sys.path.append(op.expanduser('~/src/distorted-motsfinder-public'))

import numpy as np
import matplotlib.pyplot as plt
from motsfinder.ipyutils import reload_all
exec(reload_all())
```

Note that you can put the line `exec(reload_all())` at the beginning of any
cell to immediately see the results of changed source files without restarting
the notebook's kernel.


## Optional: Adding the source folder to your `PYTHONPATH`

You can avoid having to modify `sys.path` by including the
``'~/src/distorted-motsfinder-public'`` directory in your `PYTHONPATH`
environment variable. If you use the `bash` shell, you may e.g. add
the following to your ``~/.bashrc`` file:

```.sh
export "PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}$HOME/src/distorted-motsfinder-public"
```
