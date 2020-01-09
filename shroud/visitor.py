# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################


# derived from
# http://peter-hoffmann.com/2010/extrinsic-visitor-pattern-python-inheritance.html
# with caching from http://peak.telecommunity.com/DevCenter/VisitorRevisited


class Visitor(object):
    """
    Subclasses of Visitor need to have methods like:
    def visit_Class(self, node)
       # work on instances of Class
    """

    def __init__(self):
        self._cache = {}

    def visit(self, node, *args, **kwargs):
        meth = None
        klass = node.__class__
        meth = self._cache.get(klass, None)
        if meth is None:
            for cls in node.__class__.__mro__:
                meth_name = "visit_" + cls.__name__
                meth = getattr(self, meth_name, None)
                if meth:
                    break

            if not meth:
                meth = self.generic_visit
            self._cache[klass] = meth
        return meth(node, *args, **kwargs)

    def generic_visit(self, node, *args, **kwargs):
        raise NotImplementedError(
            "Visitor.generic_visit for {}: {}".format(type(node), str(node))
        )
