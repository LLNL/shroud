# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Wrap a library using the Python API.
"""

from __future__ import print_function

import shroud

library = shroud.LibraryNode("testapi")
library.add_function(decl="void foo()")

if __name__ == "__main__":
    #    print(library._to_dict())
    import sys

    shroud.print_as_json(library, sys.stdout)
