# shroud/regression/run/defaults.mk

# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
# Produced at the Lawrence Livermore National Laboratory 
#
# LLNL-CODE-738041.
# All rights reserved. 
#
# This file is part of Shroud.  For details, see
# https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the disclaimer (as noted below)
#   in the documentation and/or other materials provided with the
#   distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
# LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
########################################################################


#compiler = gcc
#compiler = intel
ifeq ($(compiler),)
compiler = gcc
endif

ifeq ($(compiler),gcc)
CC = gcc
# -Wextra
CFLAGS = -g -Wall
CLIBS = -lstdc++
CXX = g++
CXXFLAGS = -g -Wall -std=c++11
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
PYTHON_INC := -I$(PYTHON_PREFIX)/include/python$(PYTHON_VER)m -I$(PYTHON_NUMPY)
PYTHON_LIB := -L$(PYTHON_PREFIX)/lib -lpython$(PYTHON_VER)m -ldl -lutil
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
