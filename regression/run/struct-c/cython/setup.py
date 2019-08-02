from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
    name = 'Test structures',
#    ext_modules = cythonize(
#        "cstruct.pyx",
#        include_path=[".."],
#    ),
    ext_modules=cythonize(
        Extension(
            "cstruct",
            sources=["cstruct.pyx", "../struct.c"],
            include_dirs=[".."],
        ),
        annotate=True,
#        language='c++',
    ),
)

