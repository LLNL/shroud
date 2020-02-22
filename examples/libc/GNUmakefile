

shroud.exe = $(HOME)/shroud/shroud-quest/build/temp.linux-x86_64-2.7/venv/bin/shroud
python.exe = python3
debug.setup = --debug 

all : generate
	$(python.exe) setup.py build_ext --inplace $(debug.setup)

generate: src/.. libc.log

.PHONY: all generate test

libc.log : libc.yaml
	$(shroud.exe) --outdir-python src libc.yaml

# Pattern rule to make directories.
%/.. : ; $(at)test -d $(dir $@) || mkdir -p $(dir $@)

test:
	$(python.exe) setup.py test

clean: clean-shroud clean-module
	rm -f *~

clean-shroud:
	rm -f libc.log libc.json setup.py
	rm -rf src 

clean-module:
	rm -rf build libc.egg-info *.so

.PHONY: clean clean-shroud clean-module
