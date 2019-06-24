from distutils.core import setup, Extension

extension_mod = Extension(
    "_cstruct",
    ["swigstruct_module.c", "../struct.c"],
    include_dirs=['..'],
)

setup(name = "cstruct", ext_modules=[extension_mod])
