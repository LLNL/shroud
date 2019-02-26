# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
#
# Produced at the Lawrence Livermore National Laboratory 
#
# LLNL-CODE-738041.
#
# All rights reserved. 
#
# This file is part of Shroud.
#
# For details about use and distribution, please read LICENSE.
# 
########################################################################
#
# shroud/regression/run/defaults.mk

#compiler = gcc
#compiler = intel
ifeq ($(compiler),)
compiler = gcc
endif

ifeq ($(compiler),gcc)
# -fno-strict-aliasing
#python2.7/object.h:769:22: warning:
# dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
#     ((PyObject*)(op))->ob_refcnt++)

CC = gcc
# -Wextra
# -O3 generates additional warnings
CXXWARNINGS = -O3
CFLAGS = -g -Wall -Wstrict-prototypes -fno-strict-aliasing
CLIBS = -lstdc++
CXX = g++
CXXFLAGS = -g $(CXXWARNINGS) -Wall -std=c++11 -fno-strict-aliasing
FC = gfortran
FFLAGS = -g -Wall -ffree-form -fbounds-check
#FFLAGS += -std=f2003
FLIBS = -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

ifeq ($(compiler),intel)
CC = icc
CFLAGS = -g
CLIBS = -lstdc++
CXX = icpc
CXXFLAGS = -g -std=c++11
FC = ifort
FFLAGS = -g -free -check all
FLIBS = -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

ifeq ($(compiler),pgi)
CC = pgcc
CFLAGS = -g
CLIBS = -lstdc++
CXX = pgc++
CXXFLAGS = -g -std=c++11
FC = pgf90
FFLAGS = -g -Mfree -Mstandard
FLIBS = -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

ifeq ($(compiler),ibm)
CC = xlc
CFLAGS = -g
CXX = xlc
CXXFLAGS = -g -std=c++0x 
FC = xlf2003
FFLAGS = -g -qfree=f90
#LIBS = -lstdc++ -L/opt/ibmcmp/lib64/bg -libmc++
LIBS = -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

# BG/Q with clang and xlf
ifeq ($(compiler),bgq)
CC = /collab/usr/gapps/opnsrc/gnu/dev/lnx-2.12-ppc/bgclang/r284961-stable/llnl/bin/mpiclang
CFLAGS = -g
CXX = /collab/usr/gapps/opnsrc/gnu/dev/lnx-2.12-ppc/bgclang/r284961-stable/llnl/bin/mpiclang++
CXXFLAGS = -g -std=c++0x 
FC = /opt/ibmcmp/xlf/bg/14.1/bin/bgxlf2003
FFLAGS = -g -qfree=f90
FLIBS = \
  -L/usr/apps/gnu/bgclang/r284961-stable/libc++/lib \
  -L/collab/usr/gapps/opnsrc/gnu/dev/lnx-2.12-ppc/bgclang/toolchain-4.7.2-fixup/lib \
  -L/usr/local/tools/toolchain-4.7.2/V1R2M2_4.7.2-efix014/gnu-linux-4.7.2-efix014/powerpc64-bgq-linux/lib \
  -lc++ -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

# Simple string functions, to reduce the clutter below.
sf_01 = "from sysconfig import get_config_var; print(get_config_var('$1'))"
sf_02 = "import sys; print(getattr(sys,'$1',''))"

python.exe = $(PYTHONEXE)
python.libdir   = $(eval python.libdir := $$(call shell,$(python.exe) \
  -c $(call sf_01,LIBDIR) 2>&1))$(python.libdir)
#python.configversion  = $(eval python.configversion := $$(call shell,$(python.exe) \
#  -c $(call sf_01,VERSION) 2>&1))$(python.configversion)
#python.abiflags = $(eval python.abiflags := $$(call shell,$(python.exe) \
#  -c $(call sf_02,abiflags) 2>&1))$(python.abiflags)
python.incdir   = $(eval python.incdir := $$(call shell,$(python.exe) \
  -c $(call sf_01,INCLUDEPY) 2>&1))$(python.incdir)

$(info  exe = $(python.exe))
$(info  libdir = $(python.libdir))
$(info  incdir = $(python.incdir))
#$(error done)

# 2.7
ifdef PYTHON
PYTHON_VER := $(shell $(PYTHON) -c "import sys;sys.stdout.write('{v[0]}.{v[1]}'.format(v=sys.version_info))")
PLATFORM := $(shell $(PYTHON) -c "import sys, sysconfig;sys.stdout.write(sysconfig.get_platform())")
PYTHON_PREFIX := $(shell $(PYTHON) -c "import sys;sys.stdout.write(sys.exec_prefix)")
PYTHON_NUMPY := $(shell $(PYTHON) -c "import sys, numpy;sys.stdout.write(numpy.get_include())")
PYTHON_BIN := $(PYTHON)
ifeq ($(PYTHONEXE),python2)
PYTHON_INC := -I$(PYTHON_PREFIX)/include/python$(PYTHON_VER) -I$(PYTHON_NUMPY)
PYTHON_LIB := -L$(PYTHON_PREFIX)/lib/python$(PYTHON_VER)/config -lpython$(PYTHON_VER) -ldl -lutil
else
PYTHON_INC := -I$(python.incdir) -I$(PYTHON_NUMPY)
PYTHON_LIB := -L$(python.libdir) -lpython$(PYTHON_VER)m -ldl -lutil
endif
endif

ifdef LUA
LUA_PREFIX = $(abspath $(dir $(LUA))/..)
LUA_BIN = $(LUA)
LUA_INC = -I$(LUA_PREFIX)/include
LUA_LIB = -L$(LUA_PREFIX)/lib -llua -ldl
endif

%.o : %.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

%.o %.mod  : %.f
	$(FC) $(FFLAGS) $(INCLUDE) -c -o $*.o $<

%.o %.mod  : %.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c -o $*.o $<
