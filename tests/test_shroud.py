"""
Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
Produced at the Lawrence Livermore National Laboratory 

LLNL-CODE-738041.
All rights reserved. 

This file is part of Shroud.  For details, see
https://github.com/LLNL/shroud. Please also read shroud/LICENSE.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the disclaimer below.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the disclaimer (as noted below)
  in the documentation and/or other materials provided with the
  distribution.

* Neither the name of the LLNS/LLNL nor the names of its contributors
  may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

########################################################################
"""
from __future__ import print_function
from __future__ import absolute_import

# from __future__ import unicode_literals

from shroud import main

import os
import sys
import sysconfig
import unittest

from . import do_test


def distutils_dir_name(dname):
    """Returns the name of a distutils build directory"""
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(
        dirname=dname,
        platform=sysconfig.get_platform(),
        version=sys.version_info,
    )


class MainCase(unittest.TestCase):
    def setUp(self):

        # python -m unittest tests
        #   __file__ is relative
        # python setup.py test
        #   __file__ is absolute
        self.cwd = os.path.abspath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )

        self.testdir = os.path.abspath(
            os.path.join(self.cwd, "..", "build", distutils_dir_name("temp"))
        )
        self.tester = do_test.Tester()
        self.assertTrue(self.tester.set_environment(self.cwd, self.testdir))

    def run_shroud(self, input):
        tester = self.tester
        tester.open_log(input + ".log")
        tester.set_test(input)
        status = tester.do_module()
        tester.close_log()
        self.assertTrue(status)

    def test_example(self):
        self.run_shroud("example")

    def test_include(self):
        self.run_shroud("include")

    def test_names(self):
        self.run_shroud("names")

    def test_strings(self):
        self.run_shroud("strings")

    def test_tutorial(self):
        self.run_shroud("tutorial")


if __name__ == "__main__":
    unittest.main()
