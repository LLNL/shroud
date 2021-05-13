#!/bin/sh
#
# Run from root directory
# scripts/test-lc.sh

date

# Compile Fortran tests with multiple compilers.
make -f scripts/lc.mk clean

srun make -f scripts/lc.mk target=test-fortran all -j

#srun make -f scripts/lc.mk target=test-cfi intel -j
#srun make -f scripts/lc.mk target=test-cfi gcc-10.2.1 -j

# rzwhamo
#srun make -f scripts/lc.mk target=test-fortran cray -j
#srun make -f scripts/lc.mk target=test-cfi cray -j

# rzansel
#srun make -f scripts/lc.mk target=test-fortran ibm -j
#srun make -f scripts/lc.mk target=test-cfi ibm -j

# Compile Python tests with gcc
srun make -f scripts/lc.mk target=test-python python -j

# Create summary
scripts/summary.py build/regression/run

date

# rzansel
#srun make -f scripts/lc.mk target=test-fortran ibm -j
