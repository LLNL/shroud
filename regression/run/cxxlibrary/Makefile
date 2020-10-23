# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# run/struct-cxx
#
ifndef top
top = ../../..
endif


include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/cxxlibrary \
    -I$(top)/regression/run/cxxlibrary

VPATH = \
    $(top)/regression/reference/cxxlibrary \
    $(top)/regression/run/cxxlibrary \
    $(top)/regression/run/fruit

OBJS = \
    cxxlibrary.o \
    wrapfcxxlibrary.o \
    wrapfcxxlibrary_structns.o \
    wrapcxxlibrary.o \
    wrapcxxlibrary_structns.o \
    fruit.o \
    main.o

cxxlibrary : $(OBJS)
	$(FC) $(FFLAGS) $(OBJS) -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod cxxlibrary

cxxlibrary.o : cxxlibrary.hpp
wrapcxxlibrary.o : wrapcxxlibrary.h cxxlibrary.hpp
main.o : wrapfcxxlibrary.o
