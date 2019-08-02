
build:
	python setup.py build_ext --inplace

cythonize:
	cythonize -a -i cstruct.pyx

clean:
	rm -f cstruct.so cstruct.c cstruct.html *~
	rm -rf build

.PHONY: build cythonize clean

