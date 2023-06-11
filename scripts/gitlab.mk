#
# Test with various compilers at Livermore Computing
#
# LC installs compilers in a consistent pattern which allows gmake
# pattern rules to be used.
#
# Usage:  srun make -f scripts/lc.mk target=test-all -j
#
#  make gcc-target version=10.3.1


pythondir = /usr/tce/packages/python

tempdir = build/regression

target = test-fortran-strings
#target = test-all
# Flags for all uses of $(MAKE)
makeargs = LOGOUTPUT=1
# Location of build directory.
makeargs += tempdir=$(tempdir)
# Keep going if a test fails.
makeargs += --keep-going
# Run each compiler serially to avoid too many tasks
# and to keep output in the same order.
makeargs += -j 1
makeargs += $(target)

CMAKE = /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake
#CMAKE = /usr/tce/packages/cmake/cmake-3.18.0/bin/cmake

.PHONY : clean
clean :
	rm -rf $(tempdir)

######################################################################
# gcc

gccdir = /usr/tce/packages/gcc/gcc-$(version)/bin

cc-gcc  = $(gccdir)/gcc
cxx-gcc = $(gccdir)/g++
fc-gcc  = $(gccdir)/gfortran

.PHONY : gcc-target
gcc-target :
	$(MAKE) $(makeargs) testdir=gcc-$(version) compiler=gcc \
	CC=$(cc-gcc) \
	CXX=$(cxx-gcc) \
	FC=$(fc-gcc)

######################################################################
# Intel

inteldir = /usr/tce/packages/intel/intel-$(version)/compiler/$(version)/linux/bin/intel64

cc-intel  = $(inteldir)/icc
cxx-intel = $(inteldir)/icpc
fc-intel  = $(inteldir)/ifort

.PHONY : intel-target
intel-target :
	$(MAKE) $(makeargs) testdir=intel-$(version) compiler=intel \
	CC=$(cc-intel) \
	CXX=$(cxx-intel) \
	FC=$(fc-intel)

#CFLAGS="$(cflags-$@)" \
#	CXXFLAGS="$(cxxflags-$@)" \
#	FFLAGS="$(fflags-$@)"

######################################################################
# ibm
# -qversion
# xl-2021.03.31 - 16.01

ibmdir = /usr/tce/packages/xl/xl-$(version)/bin

cc-ibm  = $(ibmdir)/xlc
cxx-ibm = $(ibmdir)/xlC
fc-ibm  = $(ibmdir)/xlf

ibm-target :
	$(MAKE) $(makeargs) testdir=xl-$(version) compiler=ibm \
	CC=$(cc-ibm) \
	CXX=$(cxx-ibm) \
	FC=$(fc-ibm)

######################################################################
# cray

#craydir = /opt/cray/pe/craype
ccedir = /usr/tce/packages/cce/cce-$(version)-magic/bin

cc-cce  = $(ccedir)/craycc
cxx-cce = $(ccedir)/crayCC
fc-cce  = $(ccedir)/crayftn

.PHONY : cce-target
cce-target :
	$(MAKE) $(makeargs) testdir=cce-$(version) compiler=cray \
	CC=$(cc-cce) \
	CXX=$(cxx-cce) \
	FC=$(fc-cce)

######################################################################
# Python

python-list = \
  python-2.7.16 \
  python-3.5.1 \
  python-3.6.4 \
  python-3.7.2 \
  python-3.8.2

python-compiler = \
  compiler=gcc \
  CC=$(gccdir)/gcc-8.3.1/bin/gcc \
  CXX=$(gccdir)/gcc-8.3.1/bin/g++

python-exe-2.7.16 = $(pythondir)/python-2.7.16/bin/python2
python-exe-3.5.1  = $(pythondir)/python-3.5.1/bin/python3
python-exe-3.6.4  = $(pythondir)/python-3.6.4/bin/python3
python-exe-3.7.2  = $(pythondir)/python-3.7.2/bin/python3
python-exe-3.8.2  = $(pythondir)/python-3.8.2/bin/python3

.PHONY : python
python : $(python-list)

.PHONY : $(python-list)
$(python-list) : python-% :
	$(MAKE) $(makeargs) testdir=$@ \
	PYTHON=$(python-exe-$*) $(python-compiler)

######################################################################
# CMake

cmake-list = $(foreach v,$(gcc-list) $(intel-list),cmake-$v)

.PHONY : cmake
cmake : $(cmake-list)

.PHONY : $(cmake-list)
$(cmake-list) : cmake-% :
	mkdir -p build/cmake/$* && cd build/cmake/$* && \
	$(CMAKE) ../../../compiler/cmake \
	-DCMAKE_C_COMPILER=$(cc-$*) \
	-DCMAKE_Fortran_COMPILER=$(fc-$*) \
	-DCMAKE_C_FLAGS="$(cflags-$*)"  \
	-DCMAKE_Fortran_FLAGS="$(fflags-$*)"

#	-DCMAKE_CXX_COMPILER=$(cxx-$*)
#       -DCMAKE_CXX_FLAGS="$(cxxflags-$*)"


########################################################################

# ANSI color codes
none    := \033[0m
red     := \033[0;31m
green   := \033[0;32m
yellow  := \033[0;33m
blue    := \033[0;34m
magenta := \033[0;35m
cyan    := \033[0;36m
all_colors := none red green yellow blue magenta cyan
export $(all_colors) all_colors

# Shell command to unset the exported colors, when not on terminal.
setcolors = { [ -t 1 ] || unset $${all_colors}; }

# Macro cprint to be used in rules.
# Example: $(call cprint,"$${red}warning: %s$${none}\n" "a is not defined")
cprint = $(setcolors) && printf $(1)

# Macro cprint2 to be used in rules generated with $(eval ), i.e. expanded twice
# $(1) - Text to print, $(2) - color-name (optional)
cprint2 = $(setcolors) && \
  printf "$(if $(2),$$$${$(2)},$$$${green})%s$$$${none}\n" '$(1)'

.PHONY: printvars print-%
# Print the value of a variable named "foo".
# Usage: make print-foo
print-%:
	@$(call cprint,"%s is $${green}%s$${none} ($${cyan}%s$${none})\
	  (from $${magenta}%s$${none})\n" '$*' '$($*)' '$(value $*)'\
	  '$(origin $*)')

# Print the value of (nearly) all the variables.
# Usage: make printvars
printvars:
	@:;$(foreach V,$(sort $(.VARIABLES)),\
	$(if $(filter-out environ% default automatic,$(origin $V)),\
	$(info $(V)=$($V) ($(value $(V))))))
	@:

