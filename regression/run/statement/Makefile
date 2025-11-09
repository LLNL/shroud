# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/statement
#
ifeq ($(top),)
top := ../../..
endif 

testdir = statement

INCLUDE = \
    -I$(top)/regression/reference/statement \
    -I$(top)/regression/run/statement

VPATH = \
    $(top)/regression/reference/statement \
    $(top)/regression/run/statement \
    $(top)/regression/run/fruit

C_OBJS = \
    statement.o \
    wrapstatement.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfstatement.o

include $(top)/regression/run/defaults.mk

statement.o : statement.hpp
wrapstatement.o : wrapstatement.h statement.hpp
main.o : fruit.o wrapfstatement.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapStatement.o wrapClass1.o wrapSingleton.o statement.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapstatement.h
