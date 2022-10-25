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
# language=c
struct list_s {
  struct list_s *next;
};
#  } listvar;
--------------------
# Recursive structure
# Error: C does not automatically declare a type for structs
# language=c
struct list_s {
  list_s *prev;
};
#  } listvar;
--------------------
# Recursive structure
struct list_s {
  struct list_s *next;
  list_s *prev;
};
#  } listvar;
--------------------
# enumerations
# language=c
enum Color {RED, WHITE, BLUE};
enum Color global;
--------------------
# enumerations
# Error: C does not automatically declare a type for enums
# language=c
enum Color {RED, WHITE, BLUE};
Color flag = RED;
--------------------
# enumerations C++
enum Color {RED, WHITE, BLUE};
enum Color global;
Color flag = RED;
--------------------
"""

# Run only one test by assigning here and
# rename Xlines to lines.
Xlines = """
struct Point { int x; int y;};
struct Point end;
Point start;
--------------------
"""


def test_block(comments, code, symtab):
    """Parse a single block of code.
    """
    print("")
    print("XXXXXXXXXXXXXXXXXXXX")
    language = "cxx"
    for cmt in comments:
        if cmt.find("language=c++") != -1:
            language = "cxx"
        elif cmt.find("language=c") != -1:
            language = "c"
        print(f"{cmt}")
    trace = True
    trace = False
    decl = "\n".join(code)
    print("XXXX CODE")
    print(decl)
    symtab = declast.SymbolTable(language=language)
    try:
        ast = declast.Parser(decl, symtab, trace).top_level()
        asdict = todict.to_dict(ast, labelast=True)
        print("XXXX AST")
        yaml.safe_dump(asdict, sys.stdout)
        print("XXXX SymbolTable")
        todict.print_scope(symtab.scope_stack[0])
    except RuntimeError as err:
        print(err)

def test_file():
    """Parse a group of lines
    which are delimited by dashes lines.
    """
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