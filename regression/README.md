
## Regression tests for Shroud

Each YAML file is a regression test.
When run with do-test.py, the results are compared against the 
files in the `reference` directory.

Tests which can be compiled are in the `run` directory.

### references

Test C++ references.  Test pointers with classes.



### clibrary

Uses assumed type in a test.  Does not work with PGI.
