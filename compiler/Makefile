# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

compiler = gcc
cmake = cmake ../config

all:
	rm -rf test && mkdir -p test && $(MAKE) -C test -f ../config/Makefile compiler=$(compiler)

all-cmake:
	rm -rf $^ && mkdir -p $^ && cd $^ && $(cmake) \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_Fortran_COMPILER=gfortran

