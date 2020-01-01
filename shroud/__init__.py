# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################
"""
Shroud - generate language bindings
"""
from __future__ import absolute_import

from .main import create_wrapper

# from ast import LibraryNode, ClassNode, FunctionNode


def print_as_json(node, fp):
    """Use the _to_dict methods to convert to a dictonary."""
    import json

    json.dump(node, fp, cls=util.ExpandedEncoder, sort_keys=True, indent=4)


__version__ = "0.10.1"
version_info = (0, 10, 1, "beta", 0)
# 'alpha', 'beta', 'candidate', or 'final'.
