from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("ZX_compute_BT_NPY", ["ZX_compute_BT_NPY.pyx"]),
    Extension("ZX_compute_BT_YPY", ["ZX_compute_BT_YPY.pyx"]),
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
)