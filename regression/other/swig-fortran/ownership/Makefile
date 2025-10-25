# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

testdir = ownership
tester : $(testdir)
.PHONY : tester

INCLUDE = \
    -I$(rundir)/ownership \
    -I$(rundir)/fruit

VPATH = \
    $(rundir)/ownership \
    $(rundir)/fruit

F_OBJS = \
    ownership_mod.o \
    fruit.o \
    main.o

C_OBJS = \
    ownership.o \
    ownership_wrap.o

include ../../defaults.mk
include $(rundir)/defaults.mk

swig:
#	swig -fortran -o swigownership_module.c ownership.i
	$(swig-fortran)  -fortran -c++ ownership.i

test:
	./ownership

.PHONY : swig test tester
