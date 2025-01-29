# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/sgroup
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/sgroup \
    -I$(top)/regression/run/sgroup

VPATH = \
    $(top)/regression/reference/sgroup \
    $(top)/regression/run/sgroup \
    $(top)/regression/run/fruit

C_OBJS = \
    sgroup.o \
    wrapsgroup.o
F_OBJS = \
    wrapfsgroup.o \
    fruit.o \
    main.o


sgroup : $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod sgroup

#sgroup.o : sgroup.cpp sgroup.hpp
sgroup.o : sgroup.hpp
main.o : main.f fruit.o wrapfsgroup.o
maincpp.o : maincpp.cpp sgroup.hpp

# useful to isolate load error with just C++ code
maincpp : maincpp.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapstd_vector_int.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
