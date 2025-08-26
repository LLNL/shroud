# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
# #######################################################################
#
# run/templates
#
testdir = templates

INCLUDE = \
    -I$(top)/regression/reference/templates \
    -I$(top)/regression/run/templates

VPATH = \
    $(top)/regression/reference/templates \
    $(top)/regression/run/templates \
    $(top)/regression/run/fruit

C_OBJS = \
    templates.o \
    wraptemplates.o \
    wrapvectorforint.o \
    wrapuser_int.o \
    wrapstd_vector_double.o \
    wrapstructAsClass_int.o \
    wrapstructAsClass_double.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapftemplates.o \
    wrapftemplates_internal.o \
    wrapftemplates_std.o

include $(top)/regression/run/defaults.mk

#templates.o : templates.cpp templates.hpp
templates.o : templates.hpp
wrapvectorforint.o : wrapvectorforint.cpp wrapstd_vector_int.h
wrapuser_int.o : wrapuser_int.cpp wrapuser_int.h typestemplates.h templates.hpp
wraptemplates.o : wraptemplates.cpp wraptemplates.h typestemplates.h templates.hpp
wrapstructAsClass_int.o : wrapstructAsClass_int.h
wrapstructAsClass_double.o : wrapstructAsClass_double.h
main.o : main.f fruit.o wrapftemplates.o wrapftemplates_internal.o wrapftemplates_std.o
maincpp.o : maincpp.cpp templates.hpp

# useful to isolate load error with just C++ code
maincpp : maincpp.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CXXLIBS)

testc.o : testc.c wrapstd_vector_int.h
