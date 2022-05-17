# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

include ../../defaults.mk

all: strings.cpp strings-binding.cpp
	python setup.py build_ext --inplace

rebuild:
	touch gen.py

strings-binding.cpp: gen.py
	$(python.exe) -B $< > $@

strings.cpp: $(rundir)/strings/strings.cpp strings.hpp
	cp $< $@
strings.hpp: $(rundir)/strings/strings.hpp
	cp $< $@

test:
	PYTHONPATH=$(cwd) python test.py

clean:
	rm -rf build
	rm -f strings.so
	rm -f strings.cpp strings.hpp

.PHONY: all clean test
