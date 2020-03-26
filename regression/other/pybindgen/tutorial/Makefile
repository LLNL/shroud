# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../defaults.mk

all: tutorial.cpp tutorial.hpp tutorial-binding.cpp
	python setup.py build

tutorial-binding.cpp : gen.py
	$(python.exe) -B $^ > $@

tutorial.cpp: ../../../run/tutorial/tutorial.cpp
	cp $^ $@
tutorial.hpp: ../../../run/tutorial/tutorial.hpp
	cp $^ $@

clean:
	rm -rf build
	rm -f tutorial.cpp tutorial.hpp tutorial-binding.cpp

.PHONY: all clean test
