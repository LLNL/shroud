# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../defaults.mk

all: classes.cpp classes.hpp classes-binding.cpp
	python setup.py build

classes-binding.cpp : gen.py
	$(python.exe) -B $^ > $@

classes.cpp: $(rundir)/classes/classes.cpp
	cp $^ $@
classes.hpp: $(rundir)/classes/classes.hpp
	cp $^ $@

clean:
	rm -rf build
	rm -f classes.cpp classes.hpp

.PHONY: all clean test
