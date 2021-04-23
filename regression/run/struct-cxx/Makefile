# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
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
    -I$(top)/regression/reference/struct-cxx \
    -I$(top)/regression/run/struct

VPATH = \
    $(top)/regression/reference/struct-cxx \
    $(top)/regression/run/struct-cxx \
    $(top)/regression/run/fruit

C_OBJS = \
    struct.o \
    wrapstruct.o \
    wrapCstruct_as_class.o \
    wrapCstruct_as_subclass.o
F_OBJS = \
    wrapfstruct.o \
    fruit.o \
    main.o

struct-cxx : main.f struct.h struct.cpp $(C_OBJS) $(F_OBJS)
	$(FC) $(FFLAGS) $(C_OBJS) $(F_OBJS) -o $@ $(FLIBS)

# Copy all source files from run/struct, need to avoid VPATH.
# "convert" to c++
struct.cpp : $(top)/regression/run/struct/struct.c
	cp $< $@
struct.h : $(top)/regression/run/struct/struct.h
	cp $< $@
main.f : $(top)/regression/run/struct/main.f
	cp $< $@

clean :
	rm -f $(OBJS) *.mod struct-cxx

struct.o : struct.h
wrapstruct.o : wrapstruct.h struct.h
main.o : fruit.o wrapfstruct.o

testc.o : testc.c wrapstruct.h struct.h
testc : testc.o $(C_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(CLIBS)
