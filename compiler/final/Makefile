
compiler = gcc

# rzgenie
ifeq ($(compiler),gcc)
FC = gfortran -g -ffree-form -fcray-pointer
FC += -cpp
FC += -Wimplicit-interface
FC += -fimplicit-none
FC += -Warray-temporaries
FC += -fcheck=array-temps
ifdef OMP_NUM_THREADS
FC += -fopenmp
endif
#FC += -std=f2008
#FC += -std=f2018
FC += -Wrealloc-lhs
FC += -fno-realloc-lhs 
CC = gcc -g
endif

ifeq ($(compiler),intel)
FC = ifort -g -free
#FC +=  -stand f08
FC +=  -safe-cray-ptr
FC +=  -fpp
FC +=  -check arg_temp_created
FC +=  -check all
ifdef OMP_NUM_THREADS
FC += -qopenmp
endif
CC = icc -g -DTS29113
endif

ifeq ($(compiler),oneapi)
FC = ifx -g -free
#FC +=  -stand f08
FC +=  -safe-cray-ptr
FC +=  -fpp
FC +=  -check arg_temp_created
FC +=  -check all
ifdef OMP_NUM_THREADS
FC += -qopenmp
endif
CC = icx -g -DTS29113
endif

ifeq ($(compiler),pgi)
FC = ${PGIPATH}pgf90 -g -Mfree
#FC += -Mstandard
FC += -Mcray=pointer
CC = ${PGIPATH}pgcc -g
endif

#XLFPATH = /opt/ibmcmp/xlf/bg/14.1
ifeq ($(compiler),ibm)
# /opt/ibmcmp/vacpp/bg/12.1/bin
FC = xlf90 -g -qfree=f90 -qsuffix=cpp=f:f=f
#CC = xlc -g -DMANGLE2
endif

#FC = sunf90 -g -free
#CC = suncc -g -DMANGLE1
ifeq ($(compiler),sun)
endif

# rzansel
# module avail xl
ifeq ($(compiler),bluexl)
# -qlanglvl=2008std
FC = xlf2008 -g -qfree=f90 -qsuffix=cpp=f:f=f -qzerosize
FC += -u # implicit none
FC += -qalias=intptr
FC += -qassert=contiguous
FC += -qnosave
ifdef OMP_NUM_THREADS
FC += -WF,-qfpp
FC += -qsmp=omp
endif
#CC = xlc -g
endif


# rzvernal
ifeq ($(compiler),cce)
FC = crayftn -g -f free
FC += -e I # implicit none
FC += -ef  # write to modulename.mod rather than MODULENAME.mod
ifdef OMP_NUM_THREADS
FC += -eT
FC += -fopenmp
endif

#FC += -qalias=intptr
#FC += -qassert=contiguous
CC = $(CCEPATH)/craycc -g
endif


##################################################

tester : testmod.o
	$(FC) $^ -o $@

clean :
	rm -f *.o *.mod
.PHONY : clean
