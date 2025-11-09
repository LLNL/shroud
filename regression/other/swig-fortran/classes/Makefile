# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

testdir = classes
tester : $(testdir)
.PHONY : tester

INCLUDE = \
    -I$(rundir)/classes \
    -I$(rundir)/fruit

VPATH = \
    $(rundir)/classes \
    $(rundir)/fruit

F_OBJS = \
    classes_mod.o \
    fruit.o \
    main.o

C_OBJS = \
    classes.o \
    classes_wrap.o

include ../../defaults.mk
include $(rundir)/defaults.mk

swig:
#	swig -fortran -o swigclasses_module.c classes.i
	$(swig-fortran)  -fortran -c++ classes.i

main.o : main.f classes_mod.o

test:
	./classes

.PHONY : swig test tester
