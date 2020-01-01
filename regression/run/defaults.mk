# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
########################################################################
#
# shroud/regression/run/defaults.mk

# The fortran flags turn on preprocessing.

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
CFLAGS = -g -Wall -Wstrict-prototypes -fno-strict-aliasing -std=c99
# silence warning in enum.yaml test
CFLAGS += -Wno-enum-compare
CLIBS = -lstdc++
CXX = g++
CXXFLAGS = -g $(CXXWARNINGS) -Wall -std=c++11 -fno-strict-aliasing
FC = gfortran
FFLAGS = -g -cpp -Wall -ffree-form -fbounds-check
#FFLAGS += -std=f2003
FLIBS = -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

ifeq ($(compiler),intel)
CC = icc
CFLAGS = -g -std=c99
CLIBS = -lstdc++
CXX = icpc
CXXFLAGS = -g -std=c++11
FC = ifort
FFLAGS = -g -fpp -free -check all
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
# rzansel
TCE = /usr/tce/packages/xl/xl-2019.08.20/
CC = xlc
CFLAGS = -g
CXX = xlc
CXXFLAGS = -g -std=c++0x 
FC = xlf2003
FFLAGS = -g -qfree=f90 -qsuffix=cpp=f
# -qlanglvl=2003std
FLIBS = -lstdc++ -L$(TCE)/alllibs -libmc++ -lstdc++
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

ifdef PYTHON
# Simple string functions, to reduce the clutter below.
sf_01 = "from sysconfig import get_config_var; print(get_config_var('$1'))"
sf_02 = "import sys; print(getattr(sys,'$1',''))"

python.exe = $(PYTHON)
python.libpl   = $(eval python.libpl := $$(call shell,$(python.exe) \
  -c $(call sf_01,LIBPL) 2>&1))$(python.libpl)
python.libs    = $(eval python.libs := $$(call shell,$(python.exe) \
  -c $(call sf_01,LIBS) 2>&1))$(python.libs)
python.bldlibrary  = $(eval python.bldlibrary := $$(call shell,$(python.exe) \
  -c $(call sf_01,BLDLIBRARY) 2>&1))$(python.bldlibrary)
python.incdir   = $(eval python.incdir := $$(call shell,$(python.exe) \
  -c $(call sf_01,INCLUDEPY) 2>&1))$(python.incdir)

# python 2.7
# libpl      - .../lib/python2.7/config
# libs       - -lpthreads -ldl -lutil
# bldlibrary - -L. -lpython2.7
#
# python 3.6
# libpl      -  .../lib/python3.6/config-3.6m-x86_64-linux-gnu
# libs       -  -lpthreads -ldl -lutil
# bldlibrary - -L. -lpython3.6m

PYTHON_VER := $(shell $(PYTHON) -c "import sys;sys.stdout.write('{v[0]}.{v[1]}'.format(v=sys.version_info))")
PLATFORM := $(shell $(PYTHON) -c "import sys, sysconfig;sys.stdout.write(sysconfig.get_platform())")
PYTHON_PREFIX := $(shell $(PYTHON) -c "import sys;sys.stdout.write(sys.exec_prefix)")
PYTHON_NUMPY := $(shell $(PYTHON) -c "import sys, numpy;sys.stdout.write(numpy.get_include())")

PYTHON_INC := -I$(python.incdir) -I$(PYTHON_NUMPY)
PYTHON_LIB := -L$(python.libpl) $(python.bldlibrary) $(python.libs)
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
