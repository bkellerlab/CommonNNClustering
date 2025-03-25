import sysconfig
import tempfile
from distutils.errors import CompileError
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from typing import List, Optional, Tuple

from Cython.Build import cythonize
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


cython_macros: List[Tuple[str, Optional[str]]] = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
]

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
setup(cmdclass={'build_ext': BuildExt}, ext_modules=extensions, package_dir={"": "test/unit"})
