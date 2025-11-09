# Copyright Shroud Project Developers. See LICENSE file for details.

#	swig -python -c++ -o tutorial_module.cpp tutorial.i

include ../../defaults.mk

build: tutorial.cpp tutorial.i
	python setup.py build_ext --inplace

tutorial.cpp: $(rundir)/tutorial/tutorial.cpp tutorial.hpp
	cp $< $@
tutorial.hpp: $(rundir)/tutorial/tutorial.hpp
	cp $< $@

clean:
	rm -rf build
	rm -f tutorial.py* tutorial_wrap.cpp _tutorial.so numpy.i *~
	rm -f tutorial.cpp tutorial.hpp

test:
	PYTHONPATH=$(shell pwd) python test.py

.PHONY: build clean test
