# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

testdir = pointers
tester : $(testdir)
.PHONY : tester

INCLUDE = \
    -I$(rundir)/pointers \
    -I$(rundir)/fruit

VPATH = \
    $(rundir)/pointers \
    $(rundir)/fruit

F_OBJS = \
    pointers_mod.o \
    fruit.o \
    main.o

C_OBJS = \
    pointers.o \
    pointers_wrap.o

include ../../defaults.mk
include $(rundir)/defaults.mk

swig:
#	swig -fortran -o swigpointers_module.c pointers.i
	$(swig-fortran)  -fortran pointers.i

test:
	./pointers

.PHONY : swig test
