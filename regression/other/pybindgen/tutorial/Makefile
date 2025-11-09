# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../../defaults.mk

all: tutorial.cpp tutorial-binding.cpp
	python setup.py build_ext --inplace

tutorial-binding.cpp : gen.py
	$(python.exe) -B $< > $@

tutorial.cpp: $(rundir)/tutorial/tutorial.cpp tutorial.hpp
	cp $< $@
tutorial.hpp: $(rundir)/tutorial/tutorial.hpp
	cp $< $@

test:
	PYTHONPATH=$(cwd) python test.py

clean:
	rm -rf build
	rm -f tutorial.so *~
	rm -f tutorial.cpp tutorial.hpp

.PHONY: all clean test
