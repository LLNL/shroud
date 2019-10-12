# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/templates
#
include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/reference/templates \
    -I$(top)/regression/run/templates \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/reference/templates \
    $(top)/regression/run/templates \
    $(top)/regression/run/fruit

C_OBJS = \
    wraptemplates.o \
    wrapvectorforint.o \
    wrapuser_int.o \
    wrapvector_double.o
F_OBJS = \
    wrapftemplates.o \
    wrapftemplates_internal.o \
    wrapftemplates_std.o \
    fruit.o \
    main.o


templates : $(C_OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

clean :
	rm -f $(OBJS) *.mod templates

#templates.o : templates.cpp templates.hpp
wrapvectorforint.o : wrapvectorforint.cpp wrapvector_int.h
wrapuser_int.o : wrapuser_int.cpp wrapuser_int.h typestemplates.h templates.hpp
wraptemplates.o : wraptemplates.cpp wraptemplates.h typestemplates.h templates.hpp
main.o : main.f wrapftemplates.o
maincpp.o : maincpp.cpp templates.hpp

# useful to isolate load error with just C++ code
maincpp : maincpp.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapvector_int.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
