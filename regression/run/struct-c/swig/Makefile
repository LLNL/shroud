# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

SWIG_OPTS = -c++

build:
	swig -python -o swigstruct_module.c struct.i
	python setup.py build_ext --inplace

clean:
	rm -f cstruct.py* swigstruct_module.c *.o *~
	rm -rf build

.PHONY : build clean
