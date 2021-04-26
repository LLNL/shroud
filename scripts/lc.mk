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

target = test-fortran-strings
#target = test-all
# Flags for all uses of $(MAKE)
makeargs = LOGOUTPUT=1 $(target)
# Keep going if a test fails.
makeargs += --ignore-errors

all : gcc intel pgi
.PHONY : all

clean :
	$(MAKE) test-clean
.PHONY : clean

######################################################################
gcc-list = \
  gcc-4.9.3 \
  gcc-6.1.0 \
  gcc-7.3.0 \
  gcc-8.3.1 \
  gcc-9.3.1 \
  gcc-10.2.1

.PHONY : gcc
gcc : $(gcc-list)

.PHONY : $(gcc-list)
$(gcc-list) : gcc-% :
	$(MAKE) $(makeargs) testdir=$@ compiler=gcc \
	CC=$(gccdir)/$@/bin/gcc \
	CXX=$(gccdir)/$@/bin/g++ \
	FC=$(gccdir)/$@/bin/gfortran

######################################################################
#  intel-14.0.3

intel-list = \
  intel-15.0.6 \
  intel-16.0.4 \
  intel-17.0.2 \
  intel-18.0.2 \
  intel-19.1.2 \
  intel-2021.2


# Match up gcc stdlib with intel compiler.
gccbin-14.0.3 = $(gccdir)/gcc-4.9.3/bin
gccbin-15.0.6 = $(gccdir)/gcc-4.9.3/bin
gccbin-16.0.4 = $(gccdir)/gcc-4.9.3/bin
gccbin-17.0.2 = $(gccdir)/gcc-4.9.3/bin
gccbin-18.0.2 = $(gccdir)/gcc-8.3.1/bin
gccbin-19.1.2 = $(gccdir)/gcc-8.3.1/bin
gccbin-2021.2 = $(gccdir)/gcc-8.3.1/bin

#intel-14.0.3-cxxflags = -std=gnu++98 -Dnullptr=NULL

# Add F2003 feature.
intel-15.0.6-fflags = -assume realloc_lhs

.PHONY : intel
intel : $(intel-list)

.PHONY : $(intel-list)
$(intel-list) : intel-% :
	$(MAKE) $(makeargs) testdir=$@ compiler=intel \
	CC=$(inteldir)/$@/bin/icc \
	CXX=$(inteldir)/$@/bin/icpc \
	FC=$(inteldir)/$@/bin/ifort \
	CFLAGS=-gcc-name=$(gccbin-$*)/gcc \
	CXXFLAGS=-gxx-name=$(gccbin-$*)/g++ \
	FFLAGS="-gcc-name=$(gccbin-$*)/gcc $($@-fflags)"

######################################################################
#  pgi-16.9  missing -cpp flag
pgi-list = \
 pgi-17.10 \
 pgi-18.5 \
 pgi-19.7 \
 pgi-20.1 \
 pgi-21.1 \

.PHONY : pgi
pgi : $(pgi-list)

.PHONY : $(pgi-list)
$(pgi-list) : pgi-% :
	$(MAKE) $(makeargs) testdir=$@ compiler=pgi \
	CC=$(pgidir)/$@/bin/pgcc \
	CXX=$(pgidir)/$@/bin/pgc++ \
	FC=$(pgidir)/$@/bin/pgf90

