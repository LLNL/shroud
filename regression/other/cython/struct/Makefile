# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../../defaults.mk

build: struct.c
	python setup.py build_ext --inplace

cythonize:
	cythonize -a -i cstruct.pyx

struct.c: $(rundir)/struct/struct.c struct.h
	cp $< $@
struct.h: $(rundir)/struct/struct.h
	cp $< $@

clean:
	rm -rf build
	rm -f cstruct.so cstruct.html *~
	rm -f struct.c struct.h

.PHONY: build cythonize clean

