# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.

build: pointers.c pointers.hpp
	python setup.py build_ext --inplace

pointers.c:  ../../../run/pointers/pointers.cpp
	cp $^ $@
pointers.hpp:  ../../../run/pointers/pointers.hpp
	cp $^ $@

clean:
	rm -rf build
	rm -f pointers.py pointers_wrap.c _pointers.so numpy.i *~
	rm -f pointers.c pointers.hpp

test:
	PYTHONPATH=$(shell pwd) python test.py

.PHONY: build clean test
