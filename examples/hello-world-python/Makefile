# Specify extensions of files to delete when cleaning
CLEANEXTS   = o a 

# Specify the target file and the install directory
OUTPUTFILE  = bin/helloworld
SRCFILES = helloworld.cpp
CFLAGS = -g -Wall 

# Default target
.PHONY: all
all: $(OUTPUTFILE) $(OUTPUTFILE).o $(OUTPUTFILE).so

shroud:
	cp setup.py _setup.py | true
	shroud helloworld.yaml --outdir-python helloworld
	cp _setup.py setup.py | true

# These examples are not for shroud, if we wanted to build the binaries separately
$(OUTPUTFILE): $(SRCFILES)
	g++ -o $(OUTPUTFILE) -Iinclude $(SRCFILES) $(CFLAGS)

$(OUTPUTFILE).o: $(SRCFILES)
	g++ -c -o $(OUTPUTFILE).o -fPIC -Iinclude $(SRCFILES) $(CFLAGS)

$(OUTPUTFILE).so: $(OUTPUTFILE).o
	g++ -shared -o $(OUTPUTFILE).so $(OUTPUTFILE).o

.PHONY: clean 
clean:
	rm $(OUTPUTFILE) $(OUTPUTFILE).o $(OUTPUTFILE.so)
