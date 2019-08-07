
SWIG_OPTS = -c++

build:
	swig -python -o swigstruct_module.c struct.i
	python setup.py build_ext --inplace

clean:
	rm -f cstruct.py* swigstruct_module.c *.o *~
	rm -rf build

.PHONY : build clean
