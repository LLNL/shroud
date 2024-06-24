# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# run/wrap
#  header only library
#
ifndef top
top = ../../..
endif

testdir = wrap-cxx


include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run/wrap

VPATH = \
    $(top)/regression/reference/$(testdir) \
    $(top)/regression/run/wrap \
    $(top)/regression/run/fruit

OBJS = \
    wrapfwrap.o \
    wrapClass1.o \
    utilwrap.o \
    fruit.o \
    main.o

$(testdir) : $(OBJS)
	$(FC) $(FFLAGS) $(OBJS) -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod $(testdir)

#wrapcxxlibrary.o : wrapcxxlibrary.h cxxlibrary.hpp
wrapClass1.o : wrapClass1.cpp wrapClass1.h
wrapClass1.h : wrap.hpp
main.o : fruit.o wrapfwrap.o
