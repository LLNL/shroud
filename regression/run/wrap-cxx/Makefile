# Copyright Shroud Project Developers. See LICENSE file for details.
#
# run/wrap
#  header only library
#
ifndef top
top = ../../..
endif

testdir = wrap-cxx

INCLUDE = \
    -I$(top)/regression/reference/$(testdir) \
    -I$(top)/regression/run/wrap

VPATH = \
    $(top)/regression/reference/$(testdir) \
    $(top)/regression/run/wrap \
    $(top)/regression/run/fruit

C_OBJS = \
    wrapClass1.o \
    utilwrap.o
F_OBJS = \
    wrapfwrap.o \
    main.o \
    fruit.o

include $(top)/regression/run/defaults.mk

#wrapcxxlibrary.o : wrapcxxlibrary.h cxxlibrary.hpp
wrapClass1.o : wrapClass1.cpp wrapClass1.h
wrapClass1.h : wrap.hpp
main.o : fruit.o wrapfwrap.o
