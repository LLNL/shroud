# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.

build:
	swig -python -c++ -o tutorial_module.cpp tutorial.i
	python setup.py build_ext --inplace

# setup.py can call swig directly but not for C++?

.PHONY: build
