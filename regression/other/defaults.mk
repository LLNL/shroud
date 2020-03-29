# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

rundir = ../../../run
python.exe = ../../../../build/temp.linux-x86_64-2.7/venv/bin/python

swig-fortran = $(HOME)/local/swig-fortran/bin/swig

cwd := $(shell pwd)

CC = gcc
CFLAGS = -g -Wall -Wstrict-prototypes -std=c99

FC = gfortran
FFLAGS = -g -cpp -Wall -ffree-form -fbounds-check

%.o : %.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $*.o $<

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $*.o $<

%.o %.mod  : %.f
	$(FC) $(FFLAGS) $(INCLUDE) -c -o $*.o $<

%.o %.mod  : %.f90
	$(FC) $(FFLAGS) $(INCLUDE) -c -o $*.o $<

