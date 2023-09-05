# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

rundir = ../../../run
python.exe = ../../../../build/temp.linux-x86_64-3.7/venv/bin/python

swig-fortran = $(HOME)/local/swig-fortran/bin/swig

#-cwd := $(shell pwd)
#-
#-CC = gcc
#-CFLAGS = -g -Wall -Wstrict-prototypes -std=c99
#-
#-CXX = g++
#-CXXFLAGS = -g -Wall
#-CXXLIBS = -lstdc++
#-
#-FC = gfortran
#-FFLAGS = -g -cpp -Wall -ffree-form -fbounds-check
#-FLIBS = -lstdc++
#-
#-%.o : %.c
#-	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<
#-
#-%.o : %.cpp
#-	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<
#-
#-%.o : %.cxx
#-	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<
#-
#-%.o %.mod  : %.f
#-	$(FC) $(FFLAGS) $(INCLUDE) -c -o $*.o $<
#-
#-%.o %.mod  : %.f90
#-	$(FC) $(FFLAGS) $(INCLUDE) -c -o $*.o $<

