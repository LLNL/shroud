
VPATH = ../config

all : assumed-type.txt

assumed-type : assumed-type-f.o assumed-type-c.o
	$(FC) $^ -o $@

assumed-type.txt : assumed-type
	./$^ > $@
