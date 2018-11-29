#!/usr/bin/env python3

import sys
import os
import os.path as op
import logging


def _skip_dir(d):
    if d.startswith('.'):
        return True
    return False


def _process_files(files, module_base='', annotate=True, numpy_includes=True):
    from setuptools import setup, Extension
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = annotate

    old_argv = sys.argv
    try:
        # Needed for distutils.
        sys.argv = [sys.argv[0], 'build_ext', '--inplace']

        if sys.platform.startswith("win"):
            cp_args = ["/O2"]
        else:
            cp_args = ["-O2"]
            cp_args.append("-march=native")

        def _make_extension(f):
            name = op.splitext(f)[0]
            module = op.basename(name)
            if module_base:
                module = "%s.%s" % (module_base, module)
            return Extension(module, [f], extra_compile_args=cp_args)

        ext_modules = [_make_extension(f) for f in files]

        include_dirs = []
        if numpy_includes:
            import numpy
            include_dirs.append(numpy.get_include())

        setup(
            cmdclass={'build_ext': build_ext},
            include_dirs=include_dirs,
            ext_modules=cythonize(ext_modules),
        )
    finally:
        sys.argv = old_argv


def main():
    if '-v' in sys.argv or '-verbose' in sys.argv:
        logging.getLogger().setLevel(logging.INFO)
    root_dir = op.dirname(op.realpath(__file__))
    module_root = op.basename(root_dir)
    cwd = os.getcwd()
    try:
        os.chdir(op.realpath(op.join(root_dir, op.pardir)))
        for cur_dir, dirs, files in os.walk(root_dir, topdown=True):
            dirs[:] = [d for d in dirs if not _skip_dir(d)]
            pyx_files = [op.join(cur_dir, f) for f in files if f.endswith(".pyx")]
            if pyx_files:
                logging.info("Cythonizing in dir: %s", cur_dir)
                rel_path = [p for p in cur_dir[len(root_dir):].split(op.sep) if p]
                module_base = ".".join([module_root] + rel_path)
                _process_files(pyx_files, module_base=module_base)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
