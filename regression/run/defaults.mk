# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
########################################################################
#
# shroud/regression/run/defaults.mk

# Defined in regression/run/*/Makefile:
# TEST_CFLAGS   - Test specific C flags.
# TEST_CXXFLAGS - Test specific C++ flags.
# TEST_FFLAGS   - Test specific Fortran flags.
# TEST_LDFLAGS  - Test specific load flags.

# The fortran flags turn on preprocessing.

#cwrapper := valgrind

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
ASAN = -fsanitize=address
# -Wextra
# -O3 generates additional warnings, but makes it harder to debug.
#CXXWARNINGS = -O3
LOCAL_CFLAGS = -g -Wall -Wstrict-prototypes -fno-strict-aliasing -std=c99
# silence warning in enum.yaml test
LOCAL_CFLAGS += -Wno-enum-compare
CLIBS = -lstdc++
CXX = g++
LOCAL_CXXFLAGS = -g $(CXXWARNINGS) -Wall -std=c++11 -fno-strict-aliasing
FC = gfortran
LOCAL_FFLAGS = -g -cpp -Wall -ffree-form -fbounds-check
#FFLAGS += -std=f2003
FLIBS = -lstdc++
SHARED = -fPIC
#LOCAL_LDFLAGS := -local
LD_SHARED = -shared
endif

ifeq ($(compiler),intel)
CC = icc
LOCAL_CFLAGS = -g -std=c99
CLIBS = -lstdc++
CXX = icpc
LOCAL_CXXFLAGS = -g -std=c++11
#LOCAL_CXXFLAGS += 
FC = ifort
LOCAL_FFLAGS = -g -fpp -free
# test-fortran-pointers-cfi
# forrtl: severe (194): Run-Time Check Failure.
# The variable 'test_out_ptrs$ISCALAR$_276' is being used in 'main.f(177,10)' without being defined
# This runtime check seems wrong since iscalar is passed as intent(OUT), pointer
# which will nullify the pointer in the subroutine.
LOCAL_FFLAGS += -check all,nopointers
FLIBS = -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

ifeq ($(compiler),oneapi)
CC = icx
LOCAL_CFLAGS = -g -O0 -std=c99
CLIBS = -lstdc++
#CLIBS += -Wl,--verbose
CXX = icpx
LOCAL_CXXFLAGS = -g -O0 -std=c++11
#LOCAL_CXXFLAGS += 
FC = ifx
LOCAL_FFLAGS = -g -O0 -fpp -free
# test-fortran-pointers-cfi
# forrtl: severe (194): Run-Time Check Failure.
# The variable 'test_out_ptrs$ISCALAR$_276' is being used in 'main.f(177,10)' without being defined
# This runtime check seems wrong since iscalar is passed as intent(OUT), pointer
# which will nullify the pointer in the subroutine.
# XXX - Using check is triggering a msan error in fruit.
oneapi_check = -check all,nopointers
oneapi_check =
LOCAL_FFLAGS += $(oneapi_check)
FLIBS = -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
LD_FC = $(oneapi_check)
endif

ifeq ($(compiler),pgi)
CC = pgcc
LOCAL_CFLAGS = -g
CLIBS = -lstdc++
CXX = pgc++
LOCAL_CXXFLAGS = -g -std=c++11
FC = pgf90
LOCAL_FFLAGS = -g -Mfree -Mstandard -cpp
FLIBS = -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

ifeq ($(compiler),ibm)
# rzansel
#TCE = /usr/tce/packages/xl/xl-2019.08.20
#TCE = /usr/tce/packages/xl/xl-2020.11.12
#TCE = /usr/tce/packages/xl/xl-2021.03.11
#TCE = /usr/tce/packages/xl/xl-2021.12.22
#TCE = /usr/tce/packages/xl/xl-2022.08.19
#TCE = /usr/tce/packages/xl/xl-2023.03.13
#CFI_INCLUDE = -I$(TCE)/xlf/16.1.1/include
CC = xlc
LOCAL_CFLAGS = -g
LOCAL_CFLAGS += $(CFI_INCLUDE)
CLIBS = -lstdc++
CXX = xlC
LOCAL_CXXFLAGS = -g -std=c++0x 
LOCAL_CXXFLAGS += $(CFI_INCLUDE)
FC = xlf2003
FC = xlf
LOCAL_FFLAGS = -g -qfree=f90
LOCAL_FFLAGS += -qlanglvl=ts
#LOCAL_FFLAGS += -qlanglvl=2003std
LOCAL_FFLAGS += -qxlf2003=polymorphic
LOCAL_FFLAGS += -qcheck=all
LOCAL_FFLAGS += -qflag=w:w -qsource
# The #line directive is not permitted by the Fortran TS29113 standard.
# -P  Inhibit generation of linemarkers 
LOCAL_FFLAGS += -qpreprocess -WF,-P
# keep preprocessor output
#LOCAL_FFLAGS += -d
# -qsuffix=cpp=f
CLIBS = -lstdc++ -Lalllibs -libmc++ -lstdc++
FLIBS = -lstdc++ -Lalllibs -libmc++ -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

# BG/Q with clang and xlf
ifeq ($(compiler),bgq)
CC = /collab/usr/gapps/opnsrc/gnu/dev/lnx-2.12-ppc/bgclang/r284961-stable/llnl/bin/mpiclang
LOCAL_CFLAGS = -g
CXX = /collab/usr/gapps/opnsrc/gnu/dev/lnx-2.12-ppc/bgclang/r284961-stable/llnl/bin/mpiclang++
LOCAL_CXXFLAGS = -g -std=c++0x 
FC = /opt/ibmcmp/xlf/bg/14.1/bin/bgxlf2003
LOCAL_FFLAGS = -g -qfree=f90
FLIBS = \
  -L/usr/apps/gnu/bgclang/r284961-stable/libc++/lib \
  -L/collab/usr/gapps/opnsrc/gnu/dev/lnx-2.12-ppc/bgclang/toolchain-4.7.2-fixup/lib \
  -L/usr/local/tools/toolchain-4.7.2/V1R2M2_4.7.2-efix014/gnu-linux-4.7.2-efix014/powerpc64-bgq-linux/lib \
  -lc++ -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

# Cray Compiler Environment
ifeq ($(compiler),cray)
CC = cc
LOCAL_CFLAGS = -g -std=c99
CLIBS = -lstdc++
CXX = CC
LOCAL_CXXFLAGS = -g -std=c++11
#LOCAL_CXXFLAGS += 
FC = ftn
# -e F preprocessor
LOCAL_FFLAGS = -g -e F -f free
# ftn-1077 ftn: This compilation contains OpenMP directives.
#   -h omp is not active so the directives are ignored.
LOCAL_FFLAGS += -M 1077
# Enables support for automatic memory allocation for allocatable
# variables and arrays that are on the left handside of intrinsic
# assignment statements.
#LOCAL_FFLAGS += -ew
# test-fortran-pointers-cfi
# forrtl: severe (194): Run-Time Check Failure.
# The variable 'test_out_ptrs$ISCALAR$_276' is being used in 'main.f(177,10)' without being defined
# This runtime check seems wrong since iscalar is passed as intent(OUT), pointer
# which will nullify the pointer in the subroutine.
#LOCAL_FFLAGS += -check all,nopointers
FLIBS = -lstdc++
SHARED = -fPIC
LD_SHARED = -shared
endif

ifeq ($(AS_SHARED),TRUE)
LOCAL_CFLAGS += $(SHARED)
LOCAL_CXXFLAGS += $(SHARED)
LOCAL_FFLAGS += $(SHARED)
LOCAL_LDFLAGS += $(LD_SHARED)
endif

# Only use ASAN if requested
ifndef use_asan
ASAN :=
endif

# Prefix required local flags to user flags from the command line.
override CFLAGS   := $(LOCAL_CFLAGS) $(ASAN) $(TEST_CFLAGS) $(CFLAGS)
override CXXFLAGS := $(LOCAL_CXXFLAGS) $(ASAN) $(TEST_CXXFLAGS) $(CXXFLAGS)
override FFLAGS   := $(LOCAL_FFLAGS) $(ASAN) $(TEST_FFLAGS) $(FFLAGS)
override LDFLAGS  := $(LOCAL_LDFLAGS) $(ASAN) $(LDFLAGS)

ifdef PYTHON
# Simple string functions, to reduce the clutter below.
sf_01 = "from sysconfig import get_config_var; print(get_config_var('$1'))"
sf_02 = "import sys; print(getattr(sys,'$1',''))"

python.exe = $(PYTHON)
python.libpl   = $(eval python.libpl := $$(call shell,$(python.exe) \
  -c $(call sf_01,LIBPL) 2>&1))$(python.libpl)
python.libs    = $(eval python.libs := $$(call shell,$(python.exe) \
  -c $(call sf_01,LIBS) 2>&1))$(python.libs)
python.ldflags = $(eval python.ldflags := $$(call shell,$(python.exe) \
  -c $(call sf_01,LDFLAGS) 2>&1))$(python.ldflags)
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
PYTHON_LIB := -L$(python.libpl) $(python.ldflags) $(python.bldlibrary) $(python.libs)
endif

python-print-debug:
	@echo PYTHON=$(PYTHON)
	@echo PYTHON_PREFIX=$(PYTHON_PREFIX)
	@echo PYTHON_VER=$(PYTHON_VER)
.PHONY : print-debug

ifdef LUA
LUA_PREFIX = $(abspath $(dir $(LUA))/..)
LUA_BIN = $(LUA)
LUA_INC = -I$(LUA_PREFIX)/include
LUA_LIB = -L$(LUA_PREFIX)/lib -llua -ldl
endif

lua-print-debug:
	@echo LUA=$(LUA)
	@echo LUA_PREFIX=$(LUA_PREFIX)
	@echo LUA_INC=$(LUA_INC)
	@echo LUA_LIB=$(LUA_LIB)

%.o : %.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

%.o : %.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

%.o %.mod  : %.f
	$(FC) $(FFLAGS) $(INCLUDE) -c -o $*.o $<

%.o %.mod  : %.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c -o $*.o $<

# Fortran test
$(testdir) : $(F_OBJS) $(C_OBJS)
	$(FC) $(LD_FC) $(LDFLAGS) $^ -o $@ $(CLIBS)

# C test
testc : testc.o $(C_OBJS)
	$(CC) $(LDFLAGS) $^ -o $@ $(CLIBS) -lm

# Python module
#$(testdir).so : $(PY_OBJS)
#	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

clean :
	rm -f $(F_OBJS) $(C_OBJS)  *.mod *.so $(testdir) testc
.PHONY : clean

