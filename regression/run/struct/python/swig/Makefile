
SWIG_OPTS = -c++

all:
	swig -python -o swigstruct_module.c struct.i
	python setup.py build_ext --inplace

