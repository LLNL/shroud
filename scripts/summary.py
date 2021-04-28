#!/bin/env python3

"""
Create a summary of regression test coverage created by lc.mk.

$tempdir
  gcc-4.9.3
    tutorial
      build.out
      build.PASS | build.FAIL
      test.out
      test.PASS | test.FAIL
  python-3.8.3
    tutorial
      python
      build.out
      build.PASS | build.FAIL
      test.out
      test.PASS | test.FAIL
"""

import os
import sys

RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"

FAIL = RED + "FAIL" + RESET
PASS = GREEN + "PASS" + RESET
 

def print_pass_fail(rootDir):
    os.chdir(rootDir)
    for dirName, subdirList, fileList in os.walk('.'):
        dirName = dirName[2:]  # Remove leading ./
        if 'build.FAIL' in fileList:
            print(f'build {FAIL} {dirName}')
        elif 'build.PASS' in fileList:
            if 'test.FAIL' in fileList:
                print(f'test {FAIL} {dirName}')
            elif 'test.PASS' in fileList:
                print(f'test {PASS} {dirName}')
            else:
                print(f'missing PASS/FAIL {dirName}')


if __name__ == "__main__":
    print("HERE", sys.argv)
    if len(sys.argv) < 2:
        print("usage: {} directory".format(sys.argv[0]))
        raise SystemExit
    print_pass_fail(sys.argv[1])

    
