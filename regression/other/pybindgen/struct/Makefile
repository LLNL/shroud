# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../defaults.mk

all: struct.c struct.h struct-binding.c
	python setup.py build

struct-binding.c : gen.py
	$(python.exe) -B $^ > $@

struct.c: $(rundir)/struct/struct.c
	cp $^ $@
struct.h: $(rundir)/struct/struct.h
	cp $^ $@

clean:
	rm -rf build
	rm -f struct.c struct.h

.PHONY: all clean test
