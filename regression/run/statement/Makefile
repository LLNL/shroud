# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/statement
#
ifeq ($(top),)
top := ../../..
endif 

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/statement \
    -I$(top)/regression/run/statement \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/statement \
    $(top)/regression/run/statement \
    $(top)/regression/run/fruit

C_OBJS = \
    statement.o \
    wrapstatement.o
F_OBJS = \
    wrapfstatement.o \
    fruit.o \
    main.o

statement : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod statement

statement.o : statement.hpp
wrapstatement.o : wrapstatement.h statement.hpp
main.o : fruit.o wrapfstatement.o

# useful to isolate load error with just C++ code
maincpp : maincpp.o wrapStatement.o wrapClass1.o wrapSingleton.o statement.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapstatement.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
