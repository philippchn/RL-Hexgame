"""
Build the hex_mcts C++ extension.

Usage:
    pip install pybind11
    python setup_mcts.py build_ext --inplace

The resulting hex_mcts.cpython-*.so is imported automatically by
facade_alphazero.py when present; falls back to pure Python if absent.
"""

import sys
from setuptools import Extension, setup

try:
    import pybind11
except ImportError:
    print("pybind11 not found — run: pip install pybind11", file=sys.stderr)
    sys.exit(1)

extra_compile = ["-O3", "-std=c++17", "-ffast-math"]
if sys.platform == "darwin":
    extra_compile += ["-stdlib=libc++"]

ext = Extension(
    name="hex_mcts",
    sources=["hex_mcts.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=extra_compile,
)

setup(name="hex_mcts", ext_modules=[ext])
