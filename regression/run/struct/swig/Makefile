ifndef top
top = ../../../..
endif

include $(top)/regression/run/defaults.mk

INCLUDE = \
    -I$(top)/regression/run/struct \
    -I$(top)/regression/run/fruit

VPATH = \
    $(top)/regression/run/struct \
    $(top)/regression/run/fruit

OBJS = \
    struct.o \
    cstruct.o \
    struct_wrap.o \
    fruit.o \
    main.o

struct : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@ $(FLIBS)

swig:
#	swig -fortran -o swigstruct_module.c struct.i
	/home/taylor16/local-swig/bin/swig -fortran struct.i

clean :
	rm -f $(OBJS) *.mod struct

.PHONY : swig clean
