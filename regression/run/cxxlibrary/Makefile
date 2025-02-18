# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# run/cxxlibrary
#
ifndef top
top = ../../..
endif

testdir = cxxlibrary

INCLUDE = \
    -I$(top)/regression/reference/cxxlibrary \
    -I$(top)/regression/run/cxxlibrary

VPATH = \
    $(top)/regression/reference/cxxlibrary \
    $(top)/regression/run/cxxlibrary \
    $(top)/regression/run/fruit

C_OBJS = \
    cxxlibrary.o \
    wrapcxxlibrary.o \
    wrapcxxlibrary_structns.o \
    wrapClass1.o \
    utilcxxlibrary.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfcxxlibrary.o \
    wrapfcxxlibrary_structns.o

include $(top)/regression/run/defaults.mk

cxxlibrary.o : cxxlibrary.hpp
wrapcxxlibrary.o : wrapcxxlibrary.h cxxlibrary.hpp
wrapClass1.o : wrapClass1.cpp cxxlibrary.hpp
wrapClass1.h : cxxlibrary.hpp
main.o : fruit.o wrapfcxxlibrary.o wrapfcxxlibrary_structns.o
