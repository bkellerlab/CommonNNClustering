import os
import sysconfig
import tempfile
from distutils.errors import CompileError
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from typing import List, Optional, Tuple

from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np


def has_flag(compiler, flagname):
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except CompileError:
            return False
    return True


def flag_filter(compiler, *flags):
    result = []
    for flag in flags:
        if has_flag(compiler, flag):
            result.append(flag)
    return result


class BuildExt(build_ext):
    compile_flags = {"msvc": ['/std:c++14'], "unix": ["-std=c++14"]}

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.compile_flags.get(ct, [])
        if ct == 'unix':
            # Only add flags which pass the `flag_filter`
            opts += flag_filter(
                self.compiler,
                "-Wno-misleading-indentation",
                "-fno-var-tracking-assignments"
                )

        for ext in self.extensions:
            ext.extra_compile_args = list(set(opts) & set(ext.extra_compile_args))

        build_ext.build_extensions(self)


Options.fast_fail = True

PYTHON_REQUIRES = ">=3.6"
TRACE_CYTHON = bool(int(os.getenv("TRACE_CYTHON", 0)))

cython_macros: List[Tuple[str, Optional[str]]] = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
]

if TRACE_CYTHON:
    cython_macros.append(("CYTHON_TRACE_NOGIL", None))

cflags_list = sysconfig.get_config_var('CFLAGS')
if cflags_list is None:
    cflags_list = []
else:
    cflags_list = cflags_list.split()

extra_compile_args = set(cflags_list)
extra_compile_args.discard('-Wstrict-prototypes')

extensions = [
    Extension(
        "*",
        ["src/commonnn/*.pyx"],
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

with open("README.md", "r") as readme:
    desc = readme.read()

sdesc = "A Python package for common-nearest-neighbours clustering"

requirements_map = {
    "mandatory": "",
    "optional": "-optional",
    "dev": "-dev",
    "docs": "-docs",
    "test": "-test",
}

requirements = {}
for category, fname in requirements_map.items():
    with open(f"requirements{fname}.txt") as fp:
        requirements[category] = fp.read().strip().split("\n")

setup(
    name="commonnn-clustering",
    version="0.0.3",
    keywords=["density-based clustering"],
    author="Jan-Oliver Joswig",
    author_email="jan.joswig@fu-berlin.de",
    description=sdesc,
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/bkellerlab/CommonNNClustering",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"cnnclustering": ["*.pxd"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={'build_ext': BuildExt},
    ext_modules=extensions,
    install_requires=requirements["mandatory"],
    extras_require={
        "optional": requirements["optional"],
        "dev": requirements["dev"],
        "docs": requirements["docs"],
        "test": requirements["test"],
    },
    zip_safe=False,
)
