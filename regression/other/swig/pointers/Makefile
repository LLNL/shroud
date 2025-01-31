# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.

include ../../defaults.mk

build: pointers.c pointers.i
	python setup.py build_ext --inplace

pointers.c:  $(rundir)/pointers/pointers.cpp pointers.hpp
	cp $< $@
pointers.hpp:  $(rundir)/pointers/pointers.hpp
	cp $< $@

clean:
	rm -rf build
	rm -f pointers.py pointers_wrap.c _pointers.so numpy.i *~
	rm -f pointers.c pointers.hpp

test:
	PYTHONPATH=$(shell pwd) python test.py

.PHONY: build clean test
