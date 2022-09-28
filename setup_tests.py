import sysconfig
from setuptools import Extension, setup
from typing import List, Optional, Tuple

from Cython.Build import cythonize
import numpy as np

cython_macros: List[Tuple[str, Optional[str]]] = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
]


cflags_list = sysconfig.get_config_var('CFLAGS')
if cflags_list is None:
    cflags_list = []

extra_compile_args = set(cflags_list.split())
extra_compile_args.discard('-Wstrict-prototypes')
extra_compile_args.add('-Wno-misleading-indentation')
extra_compile_args.add("-fno-var-tracking-assignments")
extra_compile_args.add("-std=c++14")

extensions = [
    Extension(
        "*",
        ["test/unit/*.pyx"],
        define_macros=cython_macros,
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=list(extra_compile_args),
    )
]

compiler_directives = {
    "language_level": 3,
    "binding": True,
    "embedsignature": True,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "nonecheck": False,
    "linetrace": True,
    "annotation_typing": False
}

extensions = cythonize(extensions, compiler_directives=compiler_directives)
setup(ext_modules=extensions, package_dir={"": "test/unit"})
