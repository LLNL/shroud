# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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

from shroud import declast
#from shroud import declstr
from shroud import error
from shroud import todict

from shroud import declstr

decl_str = declstr.decl_str
decl_str_noparams = declstr.decl_str_noparams

gen_arg_as_c = declstr.gen_arg_as_c
gen_arg_as_cxx = declstr.gen_arg_as_cxx

# Turn off continuation for testing (avoids adding tabs into output)
declstr.gen_arg_instance.update(continuation=False)


import yaml
import pprint
import sys

lines = """
# create_std
--------------------
# variable declarations
int i, j;
const double d, *d2;
--------------------
# variable pointer declarations
int *i1;
int **i2;
int &i3;
--------------------
typedef int footype;
--------------------
# Class statement
class Class1;
--------------------
# Class constructor
class Class2 {
  Class2();
  ~Class2();
};
--------------------
# Structure for C++
struct Point { int x, x2; int y;};
struct Point end;
Point start;
void func1(struct Point arg1, Point arg2);
--------------------
# Typedef structure
# language=c
struct list_s {
  int i;
};
struct list_s var1;
typedef struct list_s list_typ;
list_typ var2;
--------------------
# Recursive structure
# language=c
struct list_s {
  struct list_s *next;
};
struct list_s var1;
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
typedef enum Color Color_typ;
Color_typ local;
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
void func1(enum Color arg1, Color arg2);
--------------------
# pointer typedef
# language=c
typedef void *address;
address var;
void caller(address arg1);
--------------------
# function pointer typedef
# language=c
typedef int (*fcn)(int);
void caller(fcn callback);
--------------------
# template
template<T> class user {
  template<U> void nested(T arg1, U arg2 );
};
user<int> returnUserType(void);
--------------------
# nested namespace
# XXX - fix printing
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
     int imem, jmem;
  };
}
--------------------
# declstr language=c
int fun1(int arg1, int *arg2, const int *arg3);
int callback1(int in, int (*incr)(int));
--------------------
# declstr language=c++ create_std
int fun1(std::vector<int> arg1, std::vector<int> *arg2, std::vector<int> &arg3);
int callback1(int in, int (*incr)(int));
--------------------
"""  # end line

# Run only one test by assigning here and
# rename Xlines to lines.
Xlines = """
# language=c
struct Point_s { int x; int y;};
struct Point_s foo;
#typedef struct Point_s Point;
#Point start;
--------------------
"""
Xlines = """
# language=c
typedef struct Point_s { int x, x2; int y;} Point;
--------------------
"""
Xlines = """
template<typename T> struct structAsClass
--------------------
"""

Xlines = """
# language=c
#struct tag_s { int i; };
#struct tag_s var1;
#typedef struct tag_s tagname;
#void caller(tagname *arg1);
typedef int (*fcn)(int);
void caller(fcn callback);
--------------------
"""

Xlines = """
# declstr  create_std language=c++
#int fun1(std::vector<int> arg1, std::vector<int> *arg2, std::vector<int> &arg3);
int callback1(int in, int (*incr)(int));
--------------------
"""

def test_decl_str(idx, declaration, indent):
    """Convert function declaration to C and C++.
    Along with its arguments.
    """
    indent = indent + "    "
    s = decl_str.gen_decl(declaration)
    print(indent, "decl_str:", idx, s)
    s = gen_arg_as_c(declaration, add_params=False)
    print(indent, "as_c    :", idx, s)
    s = gen_arg_as_cxx(declaration, add_params=False)
    print(indent, "as_cxx  :", idx, s)
    
    if declaration.declarator.params is not None:
        s = decl_str_noparams.gen_decl(declaration)
        print(indent, "no params:", s)
        indent = indent + "    "
        for i,  arg in enumerate(declaration.declarator.params):
            s = decl_str.gen_decl(arg)
            print(indent, "decl_str:", i, s)
            s = gen_arg_as_c(arg)
            print(indent, "as_c    :", i, s)
            s = gen_arg_as_cxx(arg)
            print(indent, "as_cxx  :", i, s)

def test_block(comments, code, symtab):
    """Parse a single block of code.
    """
    print("")
    print("XXXXXXXXXXXXXXXXXXXX")
    language = "cxx"
    create_std = False
    do_declstr = False
    for cmt in comments:
        if cmt.find("language=c++") != -1:
            language = "cxx"
        elif cmt.find("language=c") != -1:
            language = "c"
        if cmt.find("create_std") != -1:
            create_std = True
        if cmt.find("declstr") != -1:
            do_declstr = True
        print(f"{cmt}")
    trace = True
    trace = False
    decl = "\n".join(code)
    print("XXXX CODE")
    print(decl)
    symtab = declast.SymbolTable(language=language)
    if create_std:
        symtab.create_std_names()
        symtab.create_std_namespace()
    try:
        ast = declast.Parser(decl, symtab, trace).top_level()
        asdict = todict.to_dict(ast, labelast=True)

        print("XXXX PRINT")
        for i, stmt in enumerate(ast.stmts):
            if isinstance(stmt, declast.Declaration):
                print(i, stmt)
                for d2 in stmt.declarators:
                    print("  ", d2)

                if do_declstr:
                    print("XXXX DeclStr")
                    test_decl_str(i, stmt, "")

            elif isinstance(stmt, declast.Template):
                print(i, stmt)

        print("XXXX PRINT_NODE")
        s = todict.print_node(ast)
        print(s)

        print("XXXX AST")
        yaml.safe_dump(asdict, sys.stdout)

        print("XXXX SymbolTable")
        symbols = declast.symtab_to_dict(symtab.scope_stack[0])
        yaml.safe_dump(symbols, sys.stdout)
    except error.ShroudParseError as err:
        print("Parse Error line {}:".format(err.line))
        print(err.message)

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
