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

top level:
make test-decl
make test-decl-diff
make test-decl-replace
"""

from shroud import ast
from shroud import declast
from shroud import todict
from shroud import typemap

import yaml
import pprint
import sys

lines = """
# variable declarations
int i;
double d;
--------------------
# nested namespace
namespace ns1 {
  int i;
  namespace ns2 {
    int j;
  }
}
--------------------
# class in namespace
namespace ns {
  class name {
     int imem;
  };
}
--------------------
# template
template<T> class user {
  template<U> void nested(T arg1, U arg2 );
};
user<int> returnUserType(void);
--------------------
# Structure for C++
struct Point { int x; int y;};
struct Point end;
Point start;
--------------------
# Recursive structure
struct list_s {
  struct list_s *next;
  list_s *prev;
};
#  } listvar;
--------------------
# enumerations C++
enum Color {RED, WHITE, BLUE};
enum Color global;
Color flag = RED;
--------------------
"""

Xlines = """
# Recursive structure
struct list_s {
  struct list_s *next;
  list_s *prev;
};
#  } listvar;
--------------------
"""


def test_block(comments, code, symtab):
    print("XXXXXXXXXXXXXXXXXXXX")
    for cmt in comments:
        print(f"{cmt}")
    trace = True
    trace = False
    decl = "\n".join(code)
    print("XXXX CODE")
    print(decl)
    symtab = declast.SymbolTable()
    ast = declast.Parser(decl, symtab, trace).top_level()
    asdict = todict.to_dict(ast, labelast=True)
    print("XXXX AST")
    yaml.safe_dump(asdict, sys.stdout)
    print("XXXX SymbolTable")
    todict.print_scope(symtab.scope_stack[0])

def test_file():
    code = []
    comments = []
    symtab = None
    for line in lines.split("\n"):
        if line.startswith("#"):
            comments.append(line)
        elif line.startswith("-----"):
            test_block(comments, code, symtab)
            comments = []
            code = []
        else:
            code.append(line)
                

        
if __name__ == "__main__":
    test_file()
