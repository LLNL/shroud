# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include defaults.mk

VPATH = ../config

all : assumed-type.txt

assumed-type-test : assumed-type-f.o assumed-type-c.o
	$(FC) $^ -o $@

assumed-type.txt : assumed-type
	./$^ > $@




cfi1-test: demo-cfi-f.o demo-cfi-c.o
	$(FC) $^ -o $@


clean :
	rm -rf fi1-test
.PHONY: clean

