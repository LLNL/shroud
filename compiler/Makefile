
cmake = cmake ../config

all:
	rm -rf test && mkdir -p test && cd test && $(MAKE) \
	-f ../config/Makefile \
        CC=/home/taylor16/apps/gcc-4.9.4/bin/gcc \
        FC=/home/taylor16/apps/gcc-4.9.4/bin/gfortran

all-cmake:
	rm -rf $^ && mkdir -p $^ && cd $^ && $(cmake) \
        -DCMAKE_C_COMPILER=/home/taylor16/apps/gcc-4.9.4/bin/gcc \
        -DCMAKE_Fortran_COMPILER=/home/taylor16/apps/gcc-4.9.4/bin/gfortran

