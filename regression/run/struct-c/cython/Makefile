# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

build:
	python setup.py build_ext --inplace

cythonize:
	cythonize -a -i cstruct.pyx

clean:
	rm -f cstruct.so cstruct.c cstruct.html *~
	rm -rf build

.PHONY: build cythonize clean

