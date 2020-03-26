# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

#SWIG_OPTS = -c++

build: struct.c struct.h
	swig -python -o swigstruct_module.c struct.i
	python setup.py build_ext --inplace

struct.c: ../../../run/struct/struct.c
	cp $^ $@
struct.h: ../../../run/struct/struct.h
	cp $^ $@

clean:
	rm -rf build
	rm -f cstruct.py* swigstruct_module.c *.so *.o *~
	rm -f struct.c struct.h

.PHONY : build clean
