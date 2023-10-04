# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)


"""
Keep track of the current state of the application so that
errors can be reported with a context.
"""

cursor = None

def get_cursor():
    global cursor
    if cursor is None:
        cursor = Cursor()
    return cursor

class NodeCursor(object):
    """Record current position in AST
    """
    def __init__(self, node):
        self.node = node    # ast.AstNode
        self.arg  = None    # declast.Declaration
        self.stmt = None    # str

    def print_context(self):
        if self.node:
#            print("Node:", self.node.declgen)
            print("Node:", self.node.name)
        if self.stmt:
            print("Statement:", self.stmt.name)
        

class Cursor(object):
    def __init__(self):
        self.phase_list = []
        self.push_phase("initial")

        self.current = NodeCursor(None)
        self.node_list = [ self.current ]

        self.last_phase = None
        self.nwarning = 0

    def push_phase(self, name):
        self.phase_list.append(name)
        self.phase = name

    def pop_phase(self, name):
        if name != self.phase:
            raise RuntimeError("pop_phase: does not match: %s" % name)
        self.phase_list.pop()
        self.phase = self.phase_list[-1]
        
    def push_node(self, node):
#        if node is None:
#            import pdb;pdb.set_trace()
#        print("XXXX push_node:", node.name if node else "None")
        self.current = NodeCursor(node)
        self.node_list.append(self.current)
#        import pdb;pdb.set_trace()
        return self.current

    def pop_node(self, node):
#        print("XXXX pop_node:", node.name if node else "None")
        if node != self.current.node:
            raise RuntimeError("pop_node: does not match")
        self.node_list.pop()
        self.current = self.node_list[-1]

    def context(self):
        if self.last_phase != self.phase:
            print()
            print("----------------------------------------")
            print("Phase:", self.phase)
            print("----------------------------------------")
        else:
            print("--------------------")
        self.current.print_context()
        self.last_phase = self.phase

    def decl_line(self, node):
        print("line {}".format(node.linenumber))
        print(node.ast.gen_decl())
        
    def warning(self, message):
        self.nwarning += 1
        self.context()
        print(message)

    def generate(self, message):
        """
        If there is a generate error, do not wrap.
        """
        node = self.node_list[-1].node
        self.nwarning += 1
        self.context()
        self.decl_line(node)
        print(message)
        node.wrap.clear()

    def ast(self, linenumber, decl, err=None):
        """Error from decl field in YAML file."""
        self.nwarning += 1
        self.context()
        print("line {}".format(linenumber))
        if err:
            print("Error in decl field")
            print("".join(err.message))
        else:
            print(decl)
        

class ShroudError(Exception):
    def __init__(self, message):
        self.message = message

class ShroudParseError(ShroudError):
    pass

