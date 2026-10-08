#core/build_cython/build_cython.py
import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

BUILD_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR  = os.path.dirname(BUILD_DIR)
os.chdir(CORE_DIR)   # sources and in-place outputs are relative to core/, whatever the working directory

extensions = [
    Extension("backtesters.ZX_compute_BT_NPY", ["backtesters/ZX_compute_BT_NPY.pyx"]),
    Extension("backtesters.ZX_compute_BT_YPY", ["backtesters/ZX_compute_BT_YPY.pyx"]),
    Extension("utils.metrics_core", ["utils/metrics_core.pyx"]),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
        annotate=False,
    ),
    include_dirs=[np.get_include()],
    options={"build": {"build_base": os.path.join(BUILD_DIR, "build")}},
)