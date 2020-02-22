

shroud.exe = $(HOME)/shroud/shroud-quest/build/temp.linux-x86_64-2.7/venv/bin/shroud
python.exe = python3

all : generate
	$(python.exe) setup.py build_ext --inplace

generate: src/.. libc.log

.PHONY: all generate

libc.log : libc.yaml
	$(shroud.exe) --outdir-python src libc.yaml

# Pattern rule to make directories.
%/.. : ; $(at)test -d $(dir $@) || mkdir -p $(dir $@)

clean:
	rm -f libc.log libc.json setup.py *~ *.so
	rm -rf src build
