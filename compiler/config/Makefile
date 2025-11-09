# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include $(ROOT_DIR)/defaults.mk

VPATH = ../config

all : version assumed-type.txt

assumed-type-test : assumed-type-f.o assumed-type-c.o
	$(FC) $^ -o $@

assumed-type.txt : assumed-type-test
	./$^ > $@




cfi1-test: demo-cfi-f.o demo-cfi-c.o
	$(FC) $^ -o $@

version:
	$(FC) $(VERSION)

clean :
	rm -rf fi1-test
.PHONY: clean version

