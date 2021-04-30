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

FAILBUILD = CYAN + "FAIL" + RESET
FAIL = RED + "FAIL" + RESET
PASS = GREEN + "PASS" + RESET
 

def collect_pass_fail(rootDir):
    """Traverse directory tree and look for pass/fail.

    summary
      fortran
        compiler-version
      python
        python-version

    """
    os.chdir(rootDir)
    summary = dict(
        fortran={},
        python={},
    )
    
    for dirName, subdirList, fileList in os.walk('.'):
        if dirName == '.':
            continue
        dirName = dirName[2:]  # Remove leading ./
        dirs = dirName.split("/")
        if len(dirs) == 1:
            continue
        # intel-16.0.4/ownership
        # python-2.7.16/enum-c/python
        if dirName.startswith("python"):
            work = summary["python"]
        else:
            work = summary["fortran"]
        work = work.setdefault(dirs[0], {})
        work = work.setdefault(dirs[1], {})
        if 'build.FAIL' in fileList:
            work['status'] = FAILBUILD
        elif 'build.PASS' in fileList:
            if 'test.FAIL' in fileList:
                work['status'] = FAIL
            elif 'test.PASS' in fileList:
                work['status'] = PASS
            else:
                work['status'] = 'missing PASS/FAIL'
    return summary

def print_summary(dct, indent=0):
    for key in sorted(dct.keys()):
        value = dct[key]
        if isinstance(value, dict):
            print("  " * indent, key)
            print_summary(dct[key], indent + 2)
        else:
            print("  " * indent, value)

def print_legend():
    print(f"{FAILBUILD}  build failed")
    print(f"{FAIL}  test failed")
    print(f"{PASS}  test passed")

def print_table(dct):
    """
    dct["gcc-"]["test"]["status"]
    """

    all_compilers = sorted(dct.keys())

    # Gather all tests (all compilers may not have all tests)
    work = {}
    for compiler in all_compilers:
        work.update(dct[compiler])
    del work['cxx']
    all_tests = sorted(work.keys())

    # Find width of test names
    test_width = -1
    for test in all_tests:
        if len(test) > test_width:
            test_width = len(test)

    for family in ['gcc', 'intel', 'pgi', 'xl', 'python']:
        subset_compilers = [x for x in all_compilers if x.startswith(family)]
        if not subset_compilers:
            continue
        print()
        print("Compiler ", family)
    
        line = "| ".join(str(x[len(family)+1:]).ljust(8) for x in subset_compilers)
        print(" ".ljust(test_width), "|", line)

        # Transpose table
        for test in all_tests:
            cline = []
            for compiler in subset_compilers:
                cline.append(dct[compiler][test].get("status", "---"))
            print(test.ljust(test_width), "|",
                  "    | ".join(str(x) for x in cline))
        


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: {} directory".format(sys.argv[0]))
        raise SystemExit
    summary = collect_pass_fail(sys.argv[1])
#    print_summary(summary)

    print_legend()
    print_table(summary["fortran"])
    print_table(summary["python"])

    
