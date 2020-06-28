#!/bin/sh
# Check variations of files.

tkdiff ../struct-class-c/python/test.py ../struct-class-cxx/python/test.py

cd ../../reference
tkdiff struct-class-c/pyCstruct_listtype.c struct-class-c/pyCstruct_numpytype.c

tkdiff struct-class-c/pyCstruct_listtype.c struct-class-cxx/pyCstruct_listtype.cpp

