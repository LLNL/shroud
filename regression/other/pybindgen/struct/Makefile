# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../../defaults.mk

all: struct.c struct-binding.c
	python setup.py build_ext --inplace

struct-binding.c : gen.py
	$(python.exe) -B $< > $@

struct.c: $(rundir)/struct/struct.c struct.h
	cp $< $@
struct.h: $(rundir)/struct/struct.h
	cp $< $@

test:
	PYTHONPATH=$(cwd) python test.py

clean:
	rm -rf build
	rm -f cstruct.so *~
	rm -f struct.c struct.h

.PHONY: all clean test
