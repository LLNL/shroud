# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

testdir = strings
tester : $(testdir)
.PHONY : tester

INCLUDE = \
    -I$(rundir) \
    -I$(rundir)/strings \
    -I$(rundir)/fruit

VPATH = \
    $(rundir)/strings \
    $(rundir)/fruit

F_OBJS = \
    strings_mod.o \
    fruit.o \
    main.o

C_OBJS = \
    strings.o \
    strings_wrap.o

include ../../defaults.mk
include $(rundir)/defaults.mk

swig:
#	swig -fortran -o swigstrings_module.c strings.i
	$(swig-fortran)  -fortran -c++ strings.i

test:
	./strings

.PHONY : swig test
