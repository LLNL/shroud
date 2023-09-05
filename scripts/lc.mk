#
# Test with various compilers at Livermore Computing
#
# LC installs compilers in a consistent pattern which allows gmake
# pattern rules to be used.
#
# Usage:  srun make -f scripts/lc.mk target=test-all -j

gccdir = /usr/tce/packages/gcc
inteldir = /usr/tce/packages/intel
pgidir = /usr/tce/packages/pgi
ibmdir = /usr/tce/packages/xl
craydir = /opt/cray/pe/craype
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

.PHONY : all
all : gcc intel pgi

.PHONY : clean
clean :
	rm -rf $(tempdir)

######################################################################
# gcc

gcc-list = \
  gcc-4.9.3 \
  gcc-6.1.0 \
  gcc-7.3.0 \
  gcc-8.3.1 \
  gcc-9.3.1 \
  gcc-10.2.1

$(foreach v,$(gcc-list),$(eval cc-$v=$(gccdir)/$v/bin/gcc))
$(foreach v,$(gcc-list),$(eval cxx-$v=$(gccdir)/$v/bin/g++))
$(foreach v,$(gcc-list),$(eval fc-$v=$(gccdir)/$v/bin/gfortran))

.PHONY : gcc
gcc : $(gcc-list)

.PHONY : $(gcc-list)
$(gcc-list) : gcc-% :
	$(MAKE) $(makeargs) testdir=$@ compiler=gcc \
	CC=$(cc-$@) \
	CXX=$(cxx-$@) \
	FC=$(fc-$@)

######################################################################
# Intel

#  intel-14.0.3
intel-list = \
  intel-15.0.6 \
  intel-16.0.4 \
  intel-17.0.2 \
  intel-18.0.2 \
  intel-19.1.2 \
  intel-2021.2

# Match up gcc stdlib with intel compiler.
gccbin-intel-14.0.3 = $(gccdir)/gcc-4.9.3/bin
gccbin-intel-15.0.6 = $(gccdir)/gcc-4.9.3/bin
gccbin-intel-16.0.4 = $(gccdir)/gcc-4.9.3/bin
gccbin-intel-17.0.2 = $(gccdir)/gcc-4.9.3/bin
gccbin-intel-18.0.2 = $(gccdir)/gcc-8.3.1/bin
gccbin-intel-19.1.2 = $(gccdir)/gcc-8.3.1/bin
gccbin-intel-2021.2 = $(gccdir)/gcc-8.3.1/bin

$(foreach v,$(intel-list),$(eval cc-$v=$(inteldir)/$v/bin/icc))
$(foreach v,$(intel-list),$(eval cxx-$v=$(inteldir)/$v/bin/icpc))
$(foreach v,$(intel-list),$(eval fc-$v=$(inteldir)/$v/bin/ifort))

$(foreach v,$(intel-list),$(eval cflags-$v=-gcc-name=$(gccbin-$v)/gcc))
$(foreach v,$(intel-list),$(eval cxxflags-$v=-gxx-name=$(gccbin-$v)/g++))
$(foreach v,$(intel-list),$(eval fflags-$v=-gcc-name=$(gccbin-$v)/gcc))

#intel-14.0.3-cxxflags = -std=gnu++98 -Dnullptr=NULL

# Add F2003 feature.
fflags-intel-15.0.6 += -assume realloc_lhs

.PHONY : intel
intel : $(intel-list)

.PHONY : $(intel-list)
$(intel-list) : intel-% :
	$(MAKE) $(makeargs) testdir=$@ compiler=intel \
	CC=$(cc-$@) \
	CXX=$(cxx-$@) \
	FC=$(fc-$@)
	CFLAGS="$(cflags-$@)" \
	CXXFLAGS="$(cxxflags-$@)" \
	FFLAGS="$(fflags-$@)"

######################################################################
# pgi

#  pgi-16.9  missing -cpp flag
pgi-list = \
 pgi-17.10 \
 pgi-18.5 \
 pgi-19.7 \
 pgi-20.1 \
 pgi-21.1 \

$(foreach v,$(pgi-list),$(eval cc-$v=$(pgidir)/$v/bin/pgcc))
$(foreach v,$(pgi-list),$(eval cxx-$v=$(pgidir)/$v/bin/pgc++))
$(foreach v,$(pgi-list),$(eval fc-$v=$(pgidir)/$v/bin/pgf90))

.PHONY : pgi
pgi : $(pgi-list)

.PHONY : $(pgi-list)
$(pgi-list) : pgi-% :
	$(MAKE) $(makeargs) testdir=$@ compiler=pgi \
	CC=$(cc-$@) \
	CXX=$(cxx-$@) \
	FC=$(fc-$@)

######################################################################
# ibm
# -qversion
# xl-2021.03.31 - 16.01

ibm-list = \
 xl-2021.03.31

$(foreach v,$(ibm-list),$(eval cc-$v=$(ibmdir)/$v/bin/xlc))
$(foreach v,$(ibm-list),$(eval cxx-$v=$(ibmdir)/$v/bin/xlC))
$(foreach v,$(ibm-list),$(eval fc-$v=$(ibmdir)/$v/bin/xlf2003))

.PHONY : ibm
ibm : $(ibm-list)

.PHONY : $(ibm-list)
$(ibm-list) : xl-% :
	$(MAKE) $(makeargs) testdir=$@ compiler=ibm \
	CC=$(cc-$@) \
	CXX=$(cxx-$@) \
	FC=$(fc-$@)

######################################################################
# cray

cray-list = \
 cray-2.7.1 \
 cray-2.7.6

cray-ver = $(patsubst cray-%,%,$(cray-list))

$(foreach v,$(cray-ver),$(eval cc-cray-$v=$(craydir)/$v/bin/cc))
$(foreach v,$(cray-ver),$(eval cxx-cray-$v=$(craydir)/$v/bin/CC))
$(foreach v,$(cray-ver),$(eval fc-cray-$v=$(craydir)/$v/bin/ftn))

.PHONY : cray
cray : $(cray-list)

.PHONY : $(cray-list)
$(cray-list) : cray-% :
	$(MAKE) $(makeargs) testdir=$@ compiler=cray \
	CC=$(cc-$@) \
	CXX=$(cxx-$@) \
	FC=$(fc-$@)

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

cmake-list = $(foreach v,$(gcc-list) $(intel-list) $(pgi-list),cmake-$v)

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

