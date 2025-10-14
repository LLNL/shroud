# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# #######################################################################
#
# run/shared
#
ifeq ($(top),)
top := ../../..
endif 

testdir = shared

INCLUDE = \
    -I$(top)/regression/reference/shared \
    -I$(top)/regression/run/shared

VPATH = \
    $(top)/regression/reference/shared \
    $(top)/regression/run/shared \
    $(top)/regression/run/fruit

C_OBJS = \
    utilShared.o \
    wrapObject.o \
    wrapObject_shared.o \
    wrapObject_weak.o
F_OBJS = \
    main.o \
    fruit.o \
    wrapfshared.o \
    test_object.o \
    test_shared.o \
    test_weak.o

include $(top)/regression/run/defaults.mk

#shared.o : shared.cpp shared.hpp
utilShared.o : utilShared.cpp shared.hpp typesShared.h
wrapObject.o : wrapObject.cpp shared.hpp wrapObject.h
wrapObject_shared.o : wrapObject_shared.cpp shared.hpp wrapObject_shared.h
wrapObject_weak.o : wrapObject_weak.cpp shared.hpp wrapObject_weak.h
test_object.o : fruit.o wrapfshared.o test_object.f
test_shared.o : fruit.o wrapfshared.o test_shared.f
test_weak.o : fruit.o wrapfshared.o test_weak.f
main.o : fruit.o wrapfshared.o test_object.o test_shared.o test_weak.o

# useful to isolate load error with just C++ code
maincpp : shared.hpp maincpp.o
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(CXXLIBS)

# Check overloading assignment(=) overload with subclasses
testmod : testmod.f
	$(FC) $(FFLAGS) $^ -o $@

testc.o : testc.c wrapShared.h shared.hpp


clean_local :
	rm -f maincpp maincpp.o testmod test_mod.mod
.PHONY : clean_local
