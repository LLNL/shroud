
## Regression tests for Shroud

Each YAML file is a regression test.
When run with do-test.py, the results are compared against the 
files in the `reference` directory.

Tests which can be compiled are in the `run` directory.

### arrayclass

Test a C++ array class.  Test pointers/references and const/non-const.
Header only.

### debugfalse

Same as the tutorial test but with option debug set to False.
All other tests have debug set to True to aid development.

### clibrary

Uses assumed type in a test.  Does not work with PGI.

### cxxlibrary

Test C++ specific features.
Pass struct by reference since struct.yaml on tests C/C++ compatible features.

### names

Test name generation and splicer.
Does not have a directory under ``run``.
