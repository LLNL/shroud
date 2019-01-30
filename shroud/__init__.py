# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-738041.
#
# All rights reserved.
#
# This file is part of Shroud.
#
# For details about use and distribution, please read LICENSE.
#
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
