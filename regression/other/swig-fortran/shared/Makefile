# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

testdir = shared
tester : $(testdir)
.PHONY : tester

INCLUDE = \
    -I$(rundir)/shared \
    -I$(rundir)/fruit

VPATH = \
    $(rundir)/shared \
    $(rundir)/fruit

F_OBJS = \
    shared_mod.o \
    fruit.o \
    main.o

C_OBJS = \
    shared_wrap.o

include ../../defaults.mk
include $(rundir)/defaults.mk

swig:
#	swig -fortran -o swigshared_module.c shared.i
	$(swig-fortran) -I$(rundir)/$(testdir) -fortran -c++ $(testdir).i

test:
	./shared

.PHONY : swig test tester
