# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################
"""
Test parser.
Useful when changing the grammar to make sure the AST is good.

Use Python from env to get the shroud package.
../build/temp.linux-x86_64-3.7/venv/bin/python3 check_decl.py
"""

from shroud import ast
from shroud import declast
from shroud import todict
from shroud import typemap

import yaml
import pprint
import sys

def test_enum(namespace):
    out = []

    decl = "enum Color { RED, WHITE, BLUE };"
    ast = declast.check_decl(decl, namespace, trace=True)
    asdict = todict.to_dict(ast)
    asdict["_ast"] = ast.__class__.__name__
    out.append(asdict)
    namespace.add_enum(decl, ast)

    decl = "enum Color global;"
    ast = declast.check_decl(decl, namespace, trace=True)
    asdict = todict.to_dict(ast)
    asdict["_ast"] = ast.__class__.__name__
    out.append(asdict)

    decl = "Color var = RED;"
    ast = declast.check_decl(decl, namespace, trace=True)
    asdict = todict.to_dict(ast)
    asdict["_ast"] = ast.__class__.__name__
    out.append(asdict)

    yaml.safe_dump(out, sys.stdout)

def test_struct(library):
    """
    struct Point { int x; int y;};
    struct Point end;
    Point start;
    """
    out = []

    decl = "struct likeclass"
    ast = declast.check_decl(decl, library, trace=True)
    asdict = todict.to_dict(ast)
    asdict["_ast"] = ast.__class__.__name__
    out.append(asdict)
    library.symtab.pop_scope()  # Normally done by closing curly brace

    decl = "struct Point { int x; int y;};"
    ast = declast.check_decl(decl, library, trace=True)
    asdict = todict.to_dict(ast)
    asdict["_ast"] = ast.__class__.__name__
    out.append(asdict)

    decl = "struct Point end;"
    ast = declast.check_decl(decl, library, trace=True)
    asdict = todict.to_dict(ast)
    asdict["_ast"] = ast.__class__.__name__
    out.append(asdict)

    decl = "Point start;"
    ast = declast.check_decl(decl, library, trace=True)
    asdict = todict.to_dict(ast)
    asdict["_ast"] = ast.__class__.__name__
    out.append(asdict)

    yaml.safe_dump(out, sys.stdout)


def test_code(library):

    decl = """
int i;
double d;
"""
    xdecl = """
namespace ns1 {
  int i;
  namespace ns2 {
    int j;
  }
}
"""
    decl = """
namespace ns {
  class name {
     int imem;
  };
}
"""
    decl = """
template<T> class user {
  template<U> void nested(T arg1, U arg2 );
};
user<int> returnUserType(void);
"""
    decl = """
struct list_s {
  struct list_s *next;
  list_s *prev;
};
"""
#  } listvar;
    xdecl = """
enum Color {RED, WHITE, BLUE};
Color flag;
"""

    trace = True
    out = []
    ast = declast.Parser(decl, library, trace).top_level()
    asdict = todict.to_dict(ast, labelast=True)
    out.append(asdict)
    print("XXXXXXXXXXXXXXXXXX AST")
    yaml.safe_dump(out, sys.stdout)
    print("XXXXXXXXXXXXXXXXXX SymbolTable")
    todict.print_scope(library.symtab.scope_stack[0])

    
if __name__ == "__main__":
#    decl = "extern int global;"

#    if not typemap.get_global_typemaps():
#        typemap.initialize()
    

    library = ast.LibraryNode()  # creates library.symtab
#    import pdb;pdb.set_trace()
    library.symtab.language = "c"
#    symtab = declast.SymbolTable()
#    print("XXXXXXXXXXXX0", symtab)


#    test_enum(library)
#    test_struct(library)
    test_code(library)

