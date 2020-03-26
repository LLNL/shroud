# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../defaults.mk

all: ownership.cpp ownership.hpp ownership-binding.cpp
	python setup.py build_ext --inplace

ownership-binding.cpp : gen.py
	$(python.exe) -B $^ > $@

ownership.cpp: $(rundir)/ownership/ownership.cpp
	cp $^ $@
ownership.hpp: $(rundir)/ownership/ownership.hpp
	cp $^ $@

test:
	PYTHONPATH=$(cwd) python test.py

clean:
	rm -rf build
	rm -f ownership.cpp ownership.hpp

.PHONY: all clean test
