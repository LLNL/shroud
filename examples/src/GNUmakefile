# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

srcdir = ../src

CC = gcc
CFLAGS = -g -I$(srcdir)
CXX = g++
CXXFLAGS = -g -I$(srcdir)
FC = gfortran
FFLAGS = -g -ffree-form

#shroud.exe := $(shell which shroud)
shroud.exe := ../../build/temp.linux-x86_64-2.7/venv/bin/shroud

python.exe := $(shell which python)

sf_01 = "from sysconfig import get_config_var; print(get_config_var('$1'))"
sf_02 = "import sys; print(getattr(sys,'$1',''))"
sf_03 = "import sys;import numpy;sys.stdout.write(numpy.get_include())"

python.home = $(abspath $(dir $(python.exe))/..)

# See http://make.mad-scientist.net/deferred-simple-variable-expansion/
# for discussion of the form "foo = $(eval foo := <something>)$(foo)".

python.version = $(eval python.version := $$(call shell,$(python.exe) \
  -c $(call sf_01,py_version) 2>&1))$(python.version)

python.root     = $(eval python.root := $$(call shell,$(python.exe) \
  -c $(call sf_01,prefix) 2>&1))$(python.root)
python.libpl    = $(eval python.libpl := $$(call shell,$(python.exe) \
  -c $(call sf_01,LIBPL) 2>&1))$(python.libpl)
python.library  = $(eval python.library := $$(call shell,$(python.exe) \
  -c $(call sf_01,LIBRARY) 2>&1))$(python.library)
python.shlibs   = $(eval python.shlibs := $$(call shell,$(python.exe) \
  -c $(call sf_01,SHLIBS) 2>&1))$(python.shlibs)

python.incdir      = $(eval python.incdir := $$(call shell,$(python.exe) \
  -c $(call sf_01,INCLUDEPY) 2>&1))$(python.incdir)
python.numpy       = $(eval python.numpy := $$(call shell,$(python.exe) \
  -c $(sf_03)))$(python.numpy)

python.Iflags = -I$(python.incdir) -I$(python.numpy)
python.lflags = $(python.libpl)/$(python.library) $(python.shlibs)

.PHONY: all
all: generate

generate : sample.yaml
	$(shroud.exe) $^ && touch $@

OBJS = \
    sample.o \
    wrapfsample.o \
    wrapsample.o

sample.o : sample.hpp

libsample.a : $(OBJS)
	$(AR) -r $@ $^


do-sample : do-sample.o libsample.a
	$(FC) $^ -o $@

#CFLAGS = -DPYTHONHOME=$(python.home) $(python.Iflags)
#LDFLAGS = $(python.lflags) -lm
#LDFLAGS += -Wl,--export-dynamic

#tester : testpython.o
#	$(CC) $^ $(LDFLAGS) -o $@


.PHONY: print-%
# Print the value of a variable named "foo".
# Usage: make print-foo
print-%  : ; @echo $* = $($*)

.PHONY: debug
debug:
	$(info  shroud   = $(shroud.exe))
	$(info  home     = $(python.home))
	$(info  version  = $(python.version))
	$(info  root     = $(python.root))
	$(info  shlibs   = $(python.shlibs))
	$(info  incdir   = $(python.incdir))
	$(info  numpy    = $(python.numpy))
	$(info  Iflags   = $(python.Iflags))
	$(info  lflags   = $(python.lflags))

vpath %.f $(srcdir)
vpath %.cpp $(srcdir)
vpath %.hpp $(srcdir)
vpath %.yaml $(srcdir)

.PHONY: clean
clean :
	rm -f generate
	rm -f sample.json sample.log typessample.h wrapfsample.f wrapsample.cpp wrapsample.h
	rm -f *.o
