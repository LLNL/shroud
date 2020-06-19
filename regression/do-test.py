#!/usr/bin/env python
# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#
########################################################################

#
# Run tests for shroud
# run input file, compare results to reference files
#
# Directory structure
#  src-dir/input/name.yaml
#  src-dir/reference/name/      reference results
#
#  bin-dir/test/name/     generated files

# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical error message')

from __future__ import print_function

import argparse
import errno
import filecmp
import logging
import os
import subprocess
import sys

# from io import StringIO
from io import BytesIO as StringIO

import shroud.main

# subprocess.call("ls -l", shell=True)

# proc = subprocess.Popen(['tail', '-500', 'mylogfile.log'], stdout=subprocess.PIPE)
# for line in proc.stdout.readlines():
#    print line.rstrip()


class Tester:
    def __init__(self):
        self.test_input_dir = ""
        self.test_output_dir = ""

        self.code_path = ""

        self.testyaml = ""  # input file
        self.ref_dir = ""  # reference directory
        self.result_dir = ""

    def open_log(self, logname):
        filename = os.path.join(self.test_output_dir, logname)
        print("Log file: {}".format(filename))
        logging.basicConfig(
            filename=filename, filemode="w", level=logging.DEBUG
        )

    def close_log(self):
        logging.shutdown()

    def setup_environment(self, input, output, executable=None):
        """Set environment for all tests.
        """
        self.test_input_dir = input
        self.test_output_dir = output

        status = True
        if not os.path.isdir(input):
            status = False
            print("Missing source directory:", input)
        if executable:
            if not os.path.exists(executable):
                status = False
                print("Missing executable:", executable)
            self.code_path = executable
        makedirs(output)
        return status

    def setup_test(self, desc, replace_ref=False):
        """Setup for a single test.

        Args:
           desc - TestDesc instance.
        """
        name = desc.name
        self.testDesc = desc
        self.testname = name
        logging.info("--------------------------------------------------")
        logging.info("Testing " + name)

        self.testyaml = os.path.join(self.test_input_dir, "input", desc.yaml)
        logging.info("Input file: " + self.testyaml)
        if not os.path.isfile(self.testyaml):
            logging.error("Input file does not exist")
            return False

        self.ref_dir = os.path.join(self.test_input_dir, "reference", name)
        logging.info("Reference directory: " + self.ref_dir)

        if replace_ref:
            # replacing reference, just create directly in ref directory
            self.result_dir = self.ref_dir
        else:
            self.result_dir = os.path.join(self.test_output_dir, name)
            logging.info("Result directory: " + self.result_dir)
            makedirs(self.result_dir)
            clear_files(self.result_dir)

        return True

    def push_stdout(self):
        # redirect stdout and stderr
        self.stdout = StringIO()
        self.saved_stdout = sys.stdout
        sys.stdout = self.stdout

        self.stderr = StringIO()
        self.saved_stderr = sys.stderr
        sys.stderr = self.stderr

    def pop_stdout(self):
        self.stdout_lines = self.stdout.getvalue()
        self.stdout.close()
        sys.stdout = self.saved_stdout

        self.stderr_lines = self.stderr.getvalue()
        self.stderr.close()
        sys.stderr = self.saved_stderr

    def XXX_do_module(self):
        """Run Shroud via a method."""
        args = argparse.Namespace(
            outdir=self.result_dir,
            outdir_c_fortran="",
            outdir_python="",
            outdir_lua="",
            logdir=self.result_dir,
            cfiles="",
            ffiles="",
            path=[self.test_input_dir],
            filename=[self.testyaml],
        )
        logging.info("Arguments: " + str(args))

        status = True
        self.push_stdout()
        try:
            shroud.main.main_with_args(args)
        except:
            logging.error("Shroud failed")
            status = False
        self.pop_stdout()

        # write output to a file
        output_file = os.path.join(self.result_dir, "output")
        fp = open(output_file, "w")
        fp.write(self.stdout_lines)
        fp.close()

        if status:
            status = self.do_compare()

        return status

    def do_test(self):
        """ Run test, return True/False for pass/fail.
        Files must compare, with no extra or missing files.
        """
        logging.info("Code to test: " + self.code_path)

        cmd = [
            self.code_path,
            "--path",
            os.path.join(self.test_input_dir, "input"),
            "--logdir",
            self.result_dir,
            "--outdir",
            self.result_dir,
            # Avoid printing things which may vary (path, date, time).
            "--option",
            "debug_testsuite=true",
        ]

        # test specific flags
        cmd += self.testDesc.cmdline

        cmd.append(self.testyaml)
        logging.debug(" ".join(cmd))

        try:
            output = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, universal_newlines=True
            )
        except subprocess.CalledProcessError as exc:
            logging.error("Exit status: %d" % exc.returncode)
            logging.error(exc.output)
            return False

        output_file = os.path.join(self.result_dir, "output")
        fp = open(output_file, "w")
        fp.write(output)
        fp.close()

        return True

    def do_compare(self):
        status = True  # assume it passes

        cmp = filecmp.dircmp(
            self.ref_dir,
            self.result_dir,
            # ignore directories with code for other wrappers
            ignore=["pybindgen", "cython", "swig"],
        )
        if not os.path.exists(self.ref_dir):
            logging.info("Reference directory does not exist: " + self.ref_dir)
            return False

        match, mismatch, errors = filecmp.cmpfiles(
            self.ref_dir, self.result_dir, cmp.common
        )
        for file in cmp.common:
            logging.info("Compare: " + file)
        if mismatch:
            status = False
            for file in mismatch:
                logging.warning("Does not compare: " + file)
        if errors:
            status = False
            for file in errors:
                logging.warning("Unable to compare: " + file)

        if cmp.left_only:
            status = False
            for file in cmp.left_only:
                logging.warning("Only in reference: " + file)
        if cmp.right_only:
            status = False
            for file in cmp.right_only:
                logging.warning("Only in result: " + file)

        if status:
            logging.info("Test {} pass".format(self.testname))
        else:
            logging.info("Test {} fail".format(self.testname))
        return status


def makedirs(path):
    """ Make sure directory exists.
    """
    try:
        # Python >=3.2
        os.makedirs(path, exist_ok=True)
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST or not os.path.isdir(path):
                raise


def clear_files(path):
    """Remove all files in a directory.
    """
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        try:
            if os.path.isfile(full_path):
                os.unlink(full_path)
        except Exception as e:
            logging.warning("Unable to remove file: " + full_path)
            logging.warning(e)


class TestDesc(object):
    """Information to describe a test.
    name = name of test.
    yaml = basename of yaml file, defaults to name.
    cmdline = list of command line arguments to append.
    """
    def __init__(self, name, yaml=None, cmdline=None):
        self.name = name
        self.yaml = (yaml or name) + ".yaml"
        self.cmdline = cmdline or []

if __name__ == "__main__":
    # XXX raise KeyError(key)

    parser = argparse.ArgumentParser(prog="do-test")
    parser.add_argument("-r", action="store_true", help="Replace test results")
    parser.add_argument("testname", nargs="*", help="test to run")
    args = parser.parse_args()

    replace_ref = args.r

    # XXX - get directories from environment or command line options

    tester = Tester()

    status = tester.setup_environment(
        os.environ["TEST_INPUT_DIR"],
        os.environ["TEST_OUTPUT_DIR"],
        os.environ["EXECUTABLE_DIR"],
    )
    if not status:
        raise SystemExit("Error in environment")
    tester.open_log("test.log")

    availTests = [
        TestDesc("none",
                 cmdline=[
                     "--write-helpers", "helpers",
                     "--yaml-types", "def_types.yaml",
                 ]),
        TestDesc("tutorial"),
        TestDesc("debugfalse", yaml="tutorial",
                 cmdline=[
                     "--option", "debug=False",
                 ]),
        TestDesc("types"),
        TestDesc("classes"),

        # enum
        TestDesc("enum-c", yaml="enum",
                 cmdline=[
                     "--language", "c",
                 ]),
        TestDesc("enum-cxx", yaml="enum",
                 cmdline=[
                     "--language", "c++",
                 ]),

        # pointers
        TestDesc("pointers-c", yaml="pointers",
                 cmdline=[
                     "--language", "c",
                     "--option", "wrap_python=false",
                 ]),
        TestDesc("pointers-cxx", yaml="pointers",
                 cmdline=[
                     "--option", "wrap_python=false",
                     # Create literal blocks for documentation
                     "--option", "literalinclude2=true",
                 ]),
        TestDesc("pointers-numpy-cxx", yaml="pointers",
                 cmdline=[
                     # Create literal blocks for documentation
                     "--option", "literalinclude2=true",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                 ]),
        TestDesc("pointers-list-cxx", yaml="pointers",
                 cmdline=[
                     "--option", "PY_array_arg=list",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                 ]),
        TestDesc("pointers-numpy-c", yaml="pointers",
                 cmdline=[
                     "--language", "c",
                     "--option", "PY_array_arg=numpy",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                 ]),
        TestDesc("pointers-list-c", yaml="pointers",
                 cmdline=[
                     "--language", "c",
                     "--option", "PY_array_arg=list",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                 ]),

        TestDesc("arrayclass"),

        # struct
        TestDesc("struct-c", yaml="struct",
                 cmdline=[
                     "--language", "c",
                     "--option", "literalinclude2=true",
                     "--option", "wrap_fortran=true",
                     "--option", "wrap_c=true",
                     "--option", "wrap_python=false",
                 ]),
        TestDesc("struct-cxx", yaml="struct",
                 cmdline=[
                     "--language", "c++",
                     "--option", "wrap_fortran=true",
                     "--option", "wrap_c=true",
                     "--option", "wrap_python=false",
                 ]),
        TestDesc("struct-numpy-c", yaml="struct",
                 cmdline=[
                     "--language", "c",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                     "--option", "wrap_python=true",
                     "--option", "PY_struct_arg=numpy",
                 ]),
        TestDesc("struct-numpy-cxx", yaml="struct",
                 cmdline=[
                     "--language", "c++",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                     "--option", "wrap_python=true",
                     "--option", "PY_struct_arg=numpy",
                 ]),
        TestDesc("struct-class-c", yaml="struct",
                 cmdline=[
                     "--language", "c",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                     "--option", "wrap_python=true",
                     "--option", "PY_struct_arg=class",
                 ]),
        TestDesc("struct-class-cxx", yaml="struct",
                 cmdline=[
                     "--language", "c++",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                     "--option", "wrap_python=true",
                     "--option", "PY_struct_arg=class",
                 ]),
        TestDesc("struct-list-cxx", yaml="struct",
                 cmdline=[
                     "--language", "c++",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                     "--option", "wrap_python=true",
                     "--option", "PY_struct_arg=list",
                 ]),

        TestDesc("structlist"),

        TestDesc("struct-py-c", yaml="struct-py",
                 cmdline=[
                     "--language", "c",
                 ]),
        TestDesc("struct-py-cxx", yaml="struct-py",
                 cmdline=[
                     "--language", "c++",
                 ]),

        # vectors
        TestDesc("vectors", yaml="vectors"),
        TestDesc("vectors-numpy", yaml="vectors",
                 cmdline=[
                     "--option", "PY_array_arg=numpy",
                     "--option", "wrap_python=true",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                 ]),
        TestDesc("vectors-list", yaml="vectors",
                 cmdline=[
                     "--option", "PY_array_arg=list",
                     "--option", "wrap_python=true",
                     "--option", "wrap_fortran=false",
                     "--option", "wrap_c=false",
                 ]),
        
        TestDesc("cdesc"),
        TestDesc("forward"),
        TestDesc("example"),
        TestDesc("include"),
        TestDesc("preprocess"),
        TestDesc("scope"),
        TestDesc("names"),
        TestDesc("names2"),
        TestDesc("namespace"),
        TestDesc("namespacedoc"),
        TestDesc("strings"),
        TestDesc("clibrary"),
        TestDesc("cxxlibrary"),
        TestDesc("interface"),
        TestDesc("statement"),
        TestDesc("templates"),
        TestDesc("ownership"),
        TestDesc("generic"),
        TestDesc("memdoc"),
    ]

    if args.testname:
        # Run a test if the name or yaml file matches
        runTests = []
        predefined = { desc.name:desc for desc in availTests }
        for testname in args.testname:
            found = False
            for predefined in availTests:
                if testname == predefined.name:
                    runTests.append(predefined)
                    found = True
                elif testname + ".yaml" == predefined.yaml:
                    runTests.append(predefined)
                    found = True
            if not found:
                # If not predefined, assume testname.yaml
                runTests.append(TestDesc(testname))
    else:
        runTests = availTests

#    logging.info("Tests to run: {}".format(" ".join(test_names)))

    pass_names = []
    fail_names = []
    for test in runTests:
        status = tester.setup_test(test, replace_ref)

        if status:
            status = tester.do_test()

            if status and not replace_ref:
                status = tester.do_compare()

        name = test.name
        if status:
            pass_names.append(name)
            print("{} pass".format(name))
        else:
            fail_names.append(name)
            print("{} ***FAILED".format(name))

    # summarize results
    if fail_names:
        exit_status = 1
        msg = "Not all tests passed"
    else:
        exit_status = 0
        msg = "All tests passed"
    print(msg)
    logging.info(msg)

    tester.close_log()
    sys.exit(exit_status)
