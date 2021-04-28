#!/bin/sh
#
# Run from home directory
# scripts/test-lc.sh

date

# Compile Fortran tests with multiple compilers.
make -f scripts/lc.mk clean

srun make -f scripts/lc.mk target=test-fortran all -j

# Compile Python tests with gcc
srun make -f scripts/lc.mk target=test-python python -j

scripts/summary.py build/regression/run

date

# Create summary
