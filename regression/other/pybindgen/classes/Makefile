# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../../defaults.mk

all: classes.cpp classes-binding.cpp
	python setup.py build_ext --inplace

classes-binding.cpp : gen.py
	$(python.exe) -B $< > $@

classes.cpp: $(rundir)/classes/classes.cpp classes.hpp
	cp $< $@
classes.hpp: $(rundir)/classes/classes.hpp
	cp $< $@

test:
	PYTHONPATH=$(cwd) python test.py

clean:
	rm -rf build
	rm -f classes.so *~
	rm -f classes.cpp classes.hpp

.PHONY: all clean test
