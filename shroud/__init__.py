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
from .metadata import (__version__, __version_info__)

# from ast import LibraryNode, ClassNode, FunctionNode


def print_as_json(node, fp):
    """Use the _to_dict methods to convert to a dictonary."""
    import json

    json.dump(node, fp, cls=util.ExpandedEncoder, sort_keys=True, indent=4)

