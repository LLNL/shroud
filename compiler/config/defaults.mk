# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
########################################################################
#
# compiler/config/defaults.mk

# The fortran flags turn on preprocessing.

#compiler = gcc
compiler = intel
ifeq ($(compiler),)
compiler = gcc
endif

ifeq ($(compiler),gcc)
CC = gcc
##-# -Wextra
##-# -O3 generates additional warnings
##-CXXWARNINGS = -O3
##-CFLAGS = -g -Wall -Wstrict-prototypes -fno-strict-aliasing -std=c99
##-# silence warning in enum.yaml test
##-CFLAGS += -Wno-enum-compare
##-CLIBS = -lstdc++
CXX = g++
##-CXXFLAGS = -g $(CXXWARNINGS) -Wall -std=c++11 -fno-strict-aliasing
FC = gfortran
##-FFLAGS = -g -cpp -Wall -ffree-form -fbounds-check
##-#FFLAGS += -std=f2003
##-FLIBS = -lstdc++
##-SHARED = -fPIC
##-LD_SHARED = -shared
VERSION = --version
endif

ifeq ($(compiler),intel)
CC = icc
CFLAGS = -g -std=c99
##-CLIBS = -lstdc++
CXX = icpc
##-CXXFLAGS = -g -std=c++11
FC = ifort
##-FFLAGS = -g -fpp -free -check all
FFLAGS = -g -free -check all
##-FLIBS = -lstdc++
##-SHARED = -fPIC
##-LD_SHARED = -shared
VERSION = --version
endif

ifeq ($(compiler),pgi)
CC = pgcc
##-CFLAGS = -g
##-CLIBS = -lstdc++
CXX = pgc++
##-CXXFLAGS = -g -std=c++11
FC = pgf90
##-FFLAGS = -g -Mfree -Mstandard
##-FLIBS = -lstdc++
##-SHARED = -fPIC
##-LD_SHARED = -shared
endif

ifeq ($(compiler),ibm)
# rzansel
TCE = /usr/tce/packages/xl/xl-2019.08.20/
CC = xlc
##-CFLAGS = -g
CXX = xlc
##-CXXFLAGS = -g -std=c++0x 
FC = xlf2003
##-FFLAGS = -g -qfree=f90 -qsuffix=cpp=f
##-# -qlanglvl=2003std
##-FLIBS = -lstdc++ -L$(TCE)/alllibs -libmc++ -lstdc++
##-SHARED = -fPIC
##-LD_SHARED = -shared
endif

# BG/Q with clang and xlf
ifeq ($(compiler),bgq)
CC = /collab/usr/gapps/opnsrc/gnu/dev/lnx-2.12-ppc/bgclang/r284961-stable/llnl/bin/mpiclang
##-CFLAGS = -g
CXX = /collab/usr/gapps/opnsrc/gnu/dev/lnx-2.12-ppc/bgclang/r284961-stable/llnl/bin/mpiclang++
##-CXXFLAGS = -g -std=c++0x 
FC = /opt/ibmcmp/xlf/bg/14.1/bin/bgxlf2003
##-FFLAGS = -g -qfree=f90
##-FLIBS = \
##-  -L/usr/apps/gnu/bgclang/r284961-stable/libc++/lib \
##-  -L/collab/usr/gapps/opnsrc/gnu/dev/lnx-2.12-ppc/bgclang/toolchain-4.7.2-fixup/lib \
##-  -L/usr/local/tools/toolchain-4.7.2/V1R2M2_4.7.2-efix014/gnu-linux-4.7.2-efix014/powerpc64-bgq-linux/lib \
##-  -lc++ -lstdc++
##-SHARED = -fPIC
##-LD_SHARED = -shared
endif
