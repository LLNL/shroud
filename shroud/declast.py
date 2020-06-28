# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
A top-down, recursive descent parser for C/C++ expressions with
additions for shroud attributes.

"""

from __future__ import print_function
import collections
import copy
import re

from . import todict
from . import typemap

Token = collections.namedtuple("Token", ["typ", "value", "line", "column"])

# https://docs.python.org/3.2/library/re.html#writing-a-tokenizer
type_specifier = {
    "void",
    "bool",
    "char",
    "short",
    "int",
    "long",
    "float",
    "double",
    "signed",
    "unsigned",
}
type_qualifier = {"const", "volatile"}
storage_class = {"auto", "register", "static", "extern", "typedef"}

cxx_keywords = {"class", "enum", "namespace", "struct", "template", "typename"}

# Just to avoid passing it into each call to check_decl
global_namespace = None

token_specification = [
    ("REAL", r"((((\d+[.]\d*)|(\d*[.]\d+))([Ee][+-]?\d+)?)|(\d+[Ee][+-]?\d+))"),
    ("INTEGER", r"\d+"),
    ("DQUOTE", r'["][^"]*["]'),  # double quoted string
    ("SQUOTE", r"['][^']*[']"),  # single quoted string
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LCURLY", r"{"),
    ("RCURLY", r"}"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("STAR", r"\*"),
    ("EQUALS", r"="),
    ("REF", r"\&"),
    ("PLUS", r"\+"),
    ("MINUS", r"\-"),
    ("SLASH", r"/"),
    ("COMMA", r","),
    ("SEMICOLON", r";"),
    ("LT", r"<"),
    ("GT", r">"),
    ("TILDE", r"\~"),
    ("NAMESPACE", r"::"),
    ("VARARG", r"\.\.\."),
    ("ID", r"[A-Za-z_][A-Za-z0-9_]*"),  # Identifiers
    ("NEWLINE", r"[\n]"),  # Line endings
    ("SKIP", r"[ \t]"),  # Skip over spaces and tabs
    ("OTHER", r"."),
]
tok_regex = "|".join("(?P<%s>%s)" % pair for pair in token_specification)
get_token = re.compile(tok_regex).match

canonical_typemap = dict(
    # explict 'int'
    short_int="short",
    long_int="long",
    long_long_int="long_long",
    unsigned_short_int="unsigned_short",
    unsigned_long_int="unsigned_long",
    unsigned_long_long_int="unsigned_long_long",
    # implied 'int'
    unsigned="unsigned_int",
)

def tokenize(s):
    line = 1
    pos = line_start = 0
    mo = get_token(s)
    while mo is not None:
        typ = mo.lastgroup
        if typ == "NEWLINE":
            line_start = pos
            line += 1
        elif typ != "SKIP":
            val = mo.group(typ)
            if typ == "ID":
                if val in type_specifier:
                    typ = "TYPE_SPECIFIER"
                elif val in type_qualifier:
                    typ = "TYPE_QUALIFIER"
                elif val in storage_class:
                    typ = "STORAGE_CLASS"
                elif val in cxx_keywords:
                    typ = val.upper()
            yield Token(typ, val, line, mo.start() - line_start)
        pos = mo.end()
        mo = get_token(s, pos)
    if pos != len(s):
        raise RuntimeError(
            "Unexpected character %r on line %d" % (s[pos], line)
        )


# indent = 0
# def trace(name, indent=0):
#    def wrap(function):
#        def wrapper(*args):
#            global indent
#            print(' ' *indent, "enter", name)
#            indent += 4
#            function(*args)
#            indent -= 4
#            print(' '*indent, "exit", name)
#        return wrapper
#    return wrap


class RecursiveDescent(object):
    """Recursive Descent Parsing helper functions.
    """

    def peek(self, typ):
        return self.token.typ == typ

    def next(self):
        """Get next token."""
        try:
            self.token = next(self.tokenizer)
        except StopIteration:
            self.token = Token("EOF", None, 0, 0)
        self.info("next", self.token)

    def have(self, typ):
        """Peek at token, if found consume."""
        if self.token.typ == typ:
            self.next()
            return True
        else:
            return False

    def mustbe(self, typ):
        """Consume a token of type typ"""
        token = self.token
        if self.token.typ == typ:
            self.next()
            return token
        else:
            self.error_msg("Expected {}, found {}", typ, self.token.typ)

    def error_msg(self, format, *args):
        msg = format.format(*args)
        ptr = " " * self.token.column + "^"
        raise RuntimeError("\n".join(["Parse Error", self.decl, ptr, msg]))

    def enter(self, name, *args):
        """Print message when entering a function."""
        if self.trace:
            print(" " * self.indent, "enter", name, *args)
            self.indent += 4

    def exit(self, name, *args):
        """Print message when exiting a function."""
        if self.trace:
            self.indent -= 4
            print(" " * self.indent, "exit", name, *args)

    def info(self, *args):
        """Print debug message during parse."""
        if self.trace:
            print(" " * self.indent, *args)


######################################################################

# For each operator, a (precedence, associativity) pair.
OpInfo = collections.namedtuple("OpInfo", "prec assoc")

OPINFO_MAP = {
    "+": OpInfo(1, "LEFT"),
    "-": OpInfo(1, "LEFT"),
    "*": OpInfo(2, "LEFT"),
    "/": OpInfo(2, "LEFT"),
    #    '^':    OpInfo(3, 'RIGHT'),
}


class ExprParser(RecursiveDescent):
    """
    Parse implied attribute expressions.
    Expand functions into Fortran or Python.

    Examples:
      size(var)
    """

    def __init__(self, expr, trace=False):
        self.decl = expr
        self.expr = expr
        self.trace = trace
        self.indent = 0
        self.token = None
        self.tokenizer = tokenize(expr)
        self.next()  # load first token

    def expression(self, min_prec=0):
        """Parse expressions.
        Preserves precedence and associativity.

        https://eli.thegreenplace.net/2012/08/02/parsing-expressions-by-precedence-climbing
        """
        self.enter("expression")
        atom_lhs = self.primary()

        while True:
            op = self.token.value
            if op not in OPINFO_MAP or OPINFO_MAP[op].prec < min_prec:
                break

            # Inside this loop the current token is a binary operator

            # Get the operator's precedence and associativity, and compute a
            # minimal precedence for the recursive call
            prec, assoc = OPINFO_MAP[op]
            next_min_prec = prec + 1 if assoc == "LEFT" else prec

            # Consume the current token and prepare the next one for the
            # recursive call
            self.next()
            atom_rhs = self.expression(next_min_prec)

            # Update lhs with the new value
            atom_lhs = BinaryOp(atom_lhs, op, atom_rhs)

        self.exit("expression")
        return atom_lhs

    def primary(self):
        self.enter("primary")
        if self.peek("ID"):
            node = self.identifier()
        elif self.token.typ in ["REAL", "INTEGER"]:
            self.enter("constant")
            node = Constant(self.token.value)
            self.next()
        elif self.have("LPAREN"):
            node = ParenExpr(self.expression())
            self.mustbe("RPAREN")
        elif self.token.typ in ["PLUS", "MINUS"]:
            self.enter("unary")
            value = self.token.value
            self.next()
            node = UnaryOp(value, self.primary())
        else:
            self.error_msg("Unexpected token {} in primary", self.token.value)
        self.exit("primary")
        return node

    def identifier(self):
        """
        <expr> ::= name '(' arglist ')'
        """
        self.enter("identifier")
        name = self.mustbe("ID").value
        if self.peek("LPAREN"):
            args = self.argument_list()
            node = Identifier(name, args)
        else:
            node = Identifier(name)
        self.exit("identifier")
        return node

    def argument_list(self):
        """
        <argument-list> ::= '(' <expression>?  [ , <expression> ]* ')'

        """
        self.enter("argument_list")
        params = []
        self.next()  # consume LPAREN peeked at in caller
        while self.token.typ != "RPAREN":
            node = self.expression()
            params.append(node)
            if not self.have("COMMA"):
                break
        self.mustbe("RPAREN")
        self.exit("argument_list", str(params))
        return params

    def dimension_shape(self):
        """Parse dimension.
        A comma delimited list of expressions:
           expr [ , expr ]*
        Only the upper bound is set.
        Use with attribute +dimension().
        Return the shape as a list of expressions.
        """
        # similar to argument_list but without parens
        self.enter("argument_list")
        shape = []
        while True:
            node = self.expression()
            shape.append(node)
            if not self.have("COMMA"):
                break
        self.exit("argument_list", str(shape))
        return shape

def check_expr(expr, trace=False):
    a = ExprParser(expr, trace=trace).expression()
    return a

def check_dimension(dim, trace=False):
    """return a list of expressions"""
    a = ExprParser(dim, trace=trace).dimension_shape()
    return a

######################################################################


class Parser(ExprParser):
    """
    Parse a C/C++ declaration with Shroud annotations.

    An abstract-declarator is a declarator without an identifier,
    consisting of one or more pointer, array, or function modifiers.
    """

    def __init__(self, decl, namespace, trace=False):
        self.decl = decl  # declaration to parse
        self.namespace = namespace  # An ast.AstNode subclass.
        self.trace = trace
        self.indent = 0
        self.token = None
        self.tokenizer = tokenize(decl)
        self.next()  # load first token

    def update_namespace(self, node):
        """Push another level of the namespace.
        Accept a Template node and save parameters as symbols
        in a namespace.
        Used while parsing a template_statement to add TemplateParams to the
        symbol table.
        """
        node.fill_symbols(self.namespace)
        self.namespace = node

    def parameter_list(self):
        """Parse function parameters."""
        # look for ... var arg at end
        """
        <parameter-list> ::= '(' <declaration>?  [ , <declaration ]* ')'

        """
        self.enter("parameter_list")
        params = []
        self.next()  # consume LPAREN peeked at in caller
        while self.token.typ != "RPAREN":
            node = self.declaration()
            params.append(node)
            if self.have("COMMA"):
                if self.have("VARARG"):
                    raise NotImplementedError("varargs")
            else:
                break
        self.mustbe("RPAREN")
        self.exit("parameter_list", str(params))
        return params

    def nested_namespace(self, namespace):
        """Found start of namespace.

        <nested-namespace> ::= { namespace :: }* identifier
        """
        self.enter("nested_namespace")
        nested = [self.token.value]
        self.next()
        while self.have("NAMESPACE"):
            # make sure nested scope is a namespaceNode
            tok = self.mustbe("ID")
            name = tok.value
            ns = namespace.qualified_lookup(name)
            if not ns:
                self.error_msg(
                    "Symbol '{}' is not in namespace '{}'".format(
                        name, nested[-1]
                    )
                )
            nested.append(name)
            namespace = ns
        qualified_id = "::".join(nested)
        self.exit("nested_namespace", qualified_id)
        return namespace, qualified_id

    def declaration_specifier(self, node):
        """
        Set attributes on node corresponding to next token
        node - Declaration node.
        <declaration-specifier> ::= <storage-class-specifier>
                                  | <type-specifier>
                                  | <type-qualifier>
                                  | (ns_name :: )+ name
                                  | :: (ns_name :: )+ name    # XXX - todo
                                  | ~ ID
        """
        self.enter("declaration_specifier")
        found_type = False
        more = True

        # destructor
        if self.have("TILDE"):
            if not self.namespace.is_class:
                raise RuntimeError("Destructor is not in a class")
            tok = self.mustbe("ID")
            if tok.value != self.namespace.name:
                raise RuntimeError("Expected class-name after ~")
            node.specifier.append(tok.value)
            self.parse_template_arguments(node)
            #  class Class1 { ~Class1(); }
            self.info("destructor", self.namespace.typemap.name)
            node.attrs["_name"] = "dtor"
            node.attrs["_destructor"] = True
            node.typemap = self.namespace.typemap
            found_type = True
            more = False

        while more:
            # if self.token.type = 'ID' and  typedef-name
            if not found_type and self.token.typ == "ID":
                # Find typedef'd names, classes and namespaces
                ns = self.namespace.unqualified_lookup(self.token.value)
                if ns:
                    ns, ns_name = self.nested_namespace(ns)
                    node.specifier.append(ns_name)
                    self.parse_template_arguments(node)
                    if (
                        self.namespace.is_class
                        and self.namespace is ns
                        and self.token.typ == "LPAREN"
                    ):
                        # template<T> vector { vector<T>(); }
                        # class Class1 { Class1(); }
                        self.info("constructor")
                        node.attrs["_name"] = "ctor"
                        node.attrs["_constructor"] = True
                        more = False
                    # Save fully resolved typename
                    node.typemap = ns.typemap
                    found_type = True
                else:
                    more = False
            elif self.token.typ == "TYPE_SPECIFIER":
                node.specifier.append(self.token.value)
                self.info("type-specifier:", self.token.value)
                self.next()
            elif self.token.typ == "TYPE_QUALIFIER":
                # const volatile
                setattr(node, self.token.value, True)
                self.info("type-qualifier:", self.token.value)
                self.next()
            elif self.token.typ == "STORAGE_CLASS":
                node.storage.append(self.token.value)
                self.info("storage-class-specifier:", self.token.value)
                self.next()
            else:
                more = False
        if not node.specifier:
            self.error_msg(
                "Expected TYPE_SPECIFIER, found {} '{}'".format(
                    self.token.typ, self.token.value
                )
            )
        self.exit("declaration_specifier")
        return

    def parse_template_arguments(self, node):
        """Parse vector parameters.
        vector<T>
        map<Key,T>

        Used while parsing function arguments.
        similar to template_argument_list
        """
        lst = node.template_arguments
        if self.have("LT"):
            while self.token.typ != "GT":
                temp = Declaration()
                self.declaration_specifier(temp)
                self.get_canonical_typemap(temp)
                lst.append(temp)
                if not self.have("COMMA"):
                    break
                self.error_msg("Only single template argument accepted")
            self.mustbe("GT")

    def decl_statement(self):
        """Check for optional semicolon and stray stuff at the end of line.
        """
        if self.token.typ == "CLASS":
            node = self.class_statement()
        elif self.token.typ == "ENUM":
            node = self.enum_statement()
        elif self.token.typ == "STRUCT":
            node = self.struct_statement()
        elif self.token.typ == "NAMESPACE":
            node = self.namespace_statement()
        elif self.token.typ == "TEMPLATE":
            node = self.template_statement()
        else:
            node = self.declaration()
        self.have("SEMICOLON")
        self.mustbe("EOF")
        return node

    def declaration(self):
        """Parse a declaration statement.
        Use with decl_statement and function arguments

        <declaration> ::= {<declaration-specifier>}+ <declarator>?
                           ( '['  <constant-expression>?  ']'  |
                             '('  <parameter-list>            ')' [ const ] )
                           [ = <initializer> ]
        """
        self.enter("declaration")
        node = Declaration()
        self.declaration_specifier(node)
        self.get_canonical_typemap(node)

        if node.attrs["_destructor"]:
            pass
        elif node.attrs["_constructor"]:
            pass
        else:
            node.declarator = self.declarator()

        if self.token.typ == "LPAREN":  # peek
            node.params = self.parameter_list()

            # Look for (void), set to no parameters.
            if len(node.params) == 1:
                chk = node.params[0]
                if (chk.declarator is None and
                    chk.specifier == ["void"]):
                    node.params = []

            #  method const
            if self.token.typ == "TYPE_QUALIFIER":
                if self.token.value == "const":
                    self.next()
                    node.func_const = True
                else:
                    raise RuntimeError(
                        "'{}' unexpected after function declaration".format(
                            self.token.value
                        )
                    )
        while self.token.typ == "LBRACKET":
            self.next() # consume bracket
            node.array.append(self.expression())
            self.mustbe("RBRACKET")
        self.attribute(node.attrs)  # variable attributes

        # Attribute are parsed before default value since
        # default value may have a +.
        # (int value = 1+size)
        if self.have("EQUALS"):
            node.init = self.initializer()
        self.exit("declaration", str(node))
        return node

    def declarator(self):
        """
        <declarator> ::=  <pointer>* [ ID ]
                       |  '(' <declarator> ')'
        """
        self.enter("declarator")
        node = Declarator()
        node.pointer = self.pointer()

        if self.token.typ == "ID":  # variable identifier
            node.name = self.token.value
            self.info("declarator ID:", self.token.value)
            self.next()
        elif self.token.typ == "LPAREN":  # (*var)
            self.next()
            node.func = self.declarator()
            self.mustbe("RPAREN")
        else:
            if not node.pointer:
                node = None

        self.exit("declarator", str(node))
        return node

    def pointer(self):
        """
        <pointer> ::= * {<type-qualifier>}* {<pointer>}?

        Allow for multiple layers of indirection.
        This will also accept illegal input like '*&'
        """
        self.enter("pointer")
        ptrs = []
        while self.token.typ in ["STAR", "REF"]:
            node = Ptr(self.token.value)
            ptrs.append(node)
            self.info("pointer:", self.token.value)
            self.next()
            while self.token.typ == "TYPE_QUALIFIER":  # const, volatile
                setattr(node, self.token.value, True)
                self.info("type-qualifier:", self.token.value)
                self.next()
        self.exit("pointer", str(ptrs))
        return ptrs

    def get_canonical_typemap(self, decl):
        """Convert specifier to typemap.
        Map specifier as needed.
        specifier = ['long', 'int']

        long int -> long

        Args:
            decl: ast.Declaration
        """
        if decl.typemap is not None:
            return
        typename = "_".join(decl.specifier)
        typename = canonical_typemap.get(typename, typename)
        ntypemap = typemap.lookup_type(typename)
# XXX - incorporate pointer into typemap
#        nptr = decl.is_pointer()
#        if nptr == 0:
#            ntypemap = typemap.lookup_type(typename)
#        else:
#            for i in range(nptr, -1, -1):
#                ntypemap = typemap.lookup_type(typename + "***"[0:i])
#                if ntypemap is not None:
#                    break
        if ntypemap is None:
            self.error_msg(
                "Unknown typemap '{}".format("_".join(decl.specifier))
            )
        decl.typemap = ntypemap

    def initializer(self):
        """
        TODO: This should support expressions
        """
        self.enter("initializer")
        value = self.token.value
        if self.have("REAL"):
            value = float(value)
        elif self.have("INTEGER"):
            value = int(value)
        elif self.have("DQUOTE"):
            value = value
        elif self.have("SQUOTE"):
            value = value
        elif self.have("ID"):
            pass
        else:
            value = None
        self.exit("initializer")
        return value

    def attribute(self, attrs):
        """Collect attributes of the form:
           +name
           +name(expression)
           +name=scalar
        """
        self.enter("attribute")
        while self.have("PLUS"):
            name = self.mustbe("ID").value
            if self.have("LPAREN"):
                parens = 1
                parts = []
                # collect tokens until found balanced paren
                while True:
                    if self.token.typ == "LPAREN":
                        parens += 1
                    elif self.token.typ == "RPAREN":
                        parens -= 1
                    elif self.token.typ == "EOF":
                        raise RuntimeError(
                            "Unbalanced parens in attribute {}".format(name)
                        )
                    if parens == 0:
                        self.next()
                        break
                    parts.append(self.token.value)
                    self.next()
                attrs[name] = "".join(parts)
            elif self.have("EQUALS"):
                attrs[name] = self.initializer()
            else:
                attrs[name] = True
        self.exit("attribute", attrs)

    def class_statement(self):
        """  class ID
        """
        self.enter("class_statement")
        self.mustbe("CLASS")
        name = self.mustbe("ID")
        node = CXXClass(name.value)
        self.exit("class_statement")
        return node

    def namespace_statement(self):
        """  namespace ID
        """
        self.enter("namespace_statement")
        self.mustbe("NAMESPACE")
        name = self.mustbe("ID")
        node = Namespace(name.value)
        self.exit("namespace_statement")
        return node

    def template_statement(self):
        """  template < template-parameter-list > declaration
        template-parameter ::= [ class | typename] ID
        """
        self.enter("template_statement")
        self.mustbe("TEMPLATE")
        node = Template()
        name = self.mustbe("LT")
        while self.token.typ != "GT":
            if self.have("TYPENAME"):
                name = self.mustbe("ID").value
            elif self.have("CLASS"):
                name = self.mustbe("ID").value
            else:
                name = self.mustbe("ID").value
            node.parameters.append(TemplateParam(name))
            if not self.have("COMMA"):
                break
        self.mustbe("GT")

        if self.token.typ == "CLASS":
            node.decl = self.class_statement()
        else:
            self.update_namespace(node)
            node.decl = self.declaration()

        self.exit("template_statement")
        return node

    def template_argument_list(self):
        """Parse template argument list
        < template_argument [ , template_argument ]* >

        Must be abstract declarations.
        ex.  <int,double>
        not <int foo>

        Return a list of Declaration.

        Used while parsing YAML
        - instantiation: <int>
        """
        self.mustbe("LT")
        lst = []
        while self.token.typ != "GT":
            temp = Declaration()
            self.declaration_specifier(temp)
            self.get_canonical_typemap(temp)
            lst.append(temp)
            if not self.have("COMMA"):
                break
        self.mustbe("GT")
        return lst

    def enum_statement(self):
        self.enter("enum_statement")
        self.mustbe("ENUM")
        if self.have("STRUCT"):
            scope = "struct"
        elif self.have("CLASS"):
            scope = "class"
        else:
            scope = None
        name = self.mustbe("ID")
        self.mustbe("LCURLY")
        node = Enum(name.value, scope)
        members = node.members
        while self.token.typ != "RCURLY":
            name = self.mustbe("ID")
            if self.have("EQUALS"):
                value = self.expression()
            else:
                value = None
            members.append(EnumValue(name.value, value))
            if not self.have("COMMA"):
                break
        self.mustbe("RCURLY")
        self.exit("enum_statement", str(members))
        return node

    def struct_statement(self):
        self.enter("struct_statement")
        self.mustbe("STRUCT")
        name = self.mustbe("ID")
        self.mustbe("LCURLY")
        node = Struct(name.value)
        members = node.members
        while self.token.typ != "RCURLY":
            members.append(self.declaration())
            self.mustbe("SEMICOLON")
        self.mustbe("RCURLY")
        self.exit("struct_statement")
        return node


######################################################################


class Node(object):
    pass


class Identifier(Node):
    def __init__(self, name, args=None):
        self.name = name
        self.args = args


class BinaryOp(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right


class UnaryOp(Node):
    def __init__(self, op, node):
        self.op = op
        self.node = node


class ParenExpr(Node):
    def __init__(self, node):
        self.node = node


class Constant(Node):
    def __init__(self, value):
        self.value = value


class Ptr(Node):
    """ A pointer or reference. """

    def __init__(self, ptr=""):
        self.ptr = ptr  # * or &
        self.const = False
        self.volatile = False

    def gen_decl_work(self, decl, **kwargs):
        """Generate string by appending text to decl.
        """
        if self.ptr:
            decl.append(" ")
            if kwargs.get("as_c", False):
                # references become pointers with as_c
                decl.append("*")
            elif kwargs.get("as_ptr", False):
                # Change reference to pointer
                decl.append("*")
            else:
                decl.append(self.ptr)
        if self.const:
            decl.append(" const")
        if self.volatile:
            decl.append(" volatile")

    def __str__(self):
        if self.const:
            return self.ptr + " const"
        else:
            return self.ptr


class Declarator(Node):
    """
    If both name and func are are None, then this is an abstract
    declarator: ex. 'int *' in 'void foo(int *)'.
    """

    def __init__(self):
        self.pointer = []  # Multiple levels of indirection
        self.name = None  # *name
        self.func = None  # (*name)     declarator

    def gen_decl_work(self, decl, **kwargs):
        """Generate string by appending text to decl.

        Replace name with value from kwargs.
        name=None will skip appending any existing name.
        """
        if kwargs.get("force_ptr", False):
            # Force to be a pointer
            decl.append(" *")
        elif kwargs.get("as_scalar", False):
            pass  # Do not print pointer
        else:
            for ptr in self.pointer:
                ptr.gen_decl_work(decl, **kwargs)
        if self.func:
            decl.append(" (")
            self.func.gen_decl_work(decl, **kwargs)
            decl.append(")")
        elif "name" in kwargs:
            if kwargs["name"]:
                decl.append(" ")
                decl.append(kwargs["name"])
        elif self.name:
            decl.append(" ")
            decl.append(self.name)

    def __str__(self):
        out = ""
        for ptr in self.pointer:
            out += str(ptr)
            out += " "

        if self.name:
            out += self.name
        elif self.func:
            out += "(" + str(self.func) + ")"

        return out


class Declaration(Node):
    """
    specifier = const  int
    init =         a  *a   a=1

    attrs     - Attributes set by the user.
    metaattrs - Attributes set by Shroud.
        struct_member - map ctor argument to struct member.
    """

    fortran_ranks = [
        "",
        "(:)",
        "(:,:)",
        "(:,:,:)",
        "(:,:,:,:)",
        "(:,:,:,:,:)",
        "(:,:,:,:,:,:)",
        "(:,:,:,:,:,:,:)",
    ]

    def __init__(self):
        self.specifier = []  # int, long, ...
        self.storage = []  # static, tyedef, ...
        self.const = False
        self.volatile = False
        self.declarator = None
        self.params = None  # None=No parameters, []=empty parameters list
        self.array = []
        self.init = None  # initial value
        self.template_arguments = []
        self.attrs = collections.defaultdict(lambda: None)
        self.metaattrs = collections.defaultdict(lambda: None)

        self.func_const = False
        self.typemap = None

        self.stmts_suffix = ''  # Used to find statements in typemap (ex. buf)
        self.ftrim_char_in = False # Pass string as TRIM(arg)//C_NULL_CHAR

    def get_name(self, use_attr=True):
        """Get name from declarator
        use_attr - True, check attr for name
        ctor and dtor should have _name set
        """
        if use_attr:
            name = self.attrs["name"] or self.attrs["_name"]
            if name is not None:
                return name
        if self.declarator is None:
            # abstract declarator
            return None
        name = self.declarator.name
        if name is None:
            if self.declarator.func:
                name = self.declarator.func.name
        return name

    def set_name(self, name):
        """Set name in declarator"""
        if self.declarator.name:
            self.declarator.name = name
        else:
            self.declarator.func.name = name

    name = property(get_name, set_name, None, "Declaration name")

    def set_type(self, ntypemap):
        """Set type specifier from a typemap."""
        self.typemap = ntypemap
        # 'long long' into ['long', 'long']
        self.specifier = ntypemap.cxx_type.split()

    def get_full_type(self):
        return ' '.join(self.specifier)

    def is_ctor(self):
        """Return True if self is a constructor."""
        return self.attrs["_constructor"]

    def is_dtor(self):
        """Return True if self is a constructor."""
        return self.attrs["_destructor"]

    def is_pointer(self):
        """Return number of levels of pointers.
        """
        nlevels = 0
        if self.declarator is None:
            return nlevels
        for ptr in self.declarator.pointer:
            if ptr.ptr == "*":
                nlevels += 1
        return nlevels

    def is_reference(self):
        """Return number of levels of references.
        """
        nlevels = 0
        if self.declarator is None:
            return nlevels
        for ptr in self.declarator.pointer:
            if ptr.ptr == "&":
                nlevels += 1
        return nlevels

    def is_indirect(self):
        """Return number of indirections.
        pointer or reference.
        """
        nlevels = 0
        if self.declarator:
            for ptr in self.declarator.pointer:
                if ptr.ptr:
                    nlevels += 1
        return nlevels

    def is_array(self):
        """Return number of indirections.
        array, pointer or reference.
        """
        nlevels = 0
        if self.array:
            nlevels += 1
        if self.declarator is None:
            return nlevels
        for ptr in self.declarator.pointer:
            if ptr.ptr:
                nlevels += 1
        return nlevels
        
    def is_function_pointer(self):
        """Return number of levels of pointers.
        """
        if self.declarator is None:
            return False
        if self.declarator.func is None:
            return False
        if not self.declarator.func.pointer:
            return False
        return True

    def XXXget_indirect(self):
        """Return indirect operators.
        '*', '**', '&*', '[]'
        """
        out = ''
        if self.declarator is None:
            return out
        for ptr in self.declarator.pointer:
            out += ptr.ptr
        if self.array:
            out += "[]"   # XXX - multidimensional?
        return out

    def get_indirect_stmt(self):
        """Return statement field for pointers.
        'scalar', 'pointer', '**'
        """
        out = ''
        if self.declarator is None:
            return "scalar"
        for ptr in self.declarator.pointer:
            out += ptr.ptr
        if self.array:
            out += "[]"   # XXX - multidimensional?
        if out == "":
            return "scalar"
        return out

    def get_array_size(self):
        """Return size of array by multiplying dimensions."""
        array = self.array
        if not array:
            return None
        if len(array) == 1:
            return todict.print_node(array[0])
        out = []
        for dim in array:
            out.append("({})".format(todict.print_node(dim)))
        return '*'.join(out)

    def get_subprogram(self):
        """Return Fortran subprogram - subroutine or function.
        Return None for variable declarations.
        """
        if self.params is None:
            return None
        if self.typemap.name != "void":
            return "function"
        if self.is_pointer():
            return "function"
        return "subroutine"

    def find_arg_by_name(self, name):
        """Find argument in params with name."""
        return find_arg_by_name(self.params, name)

    def find_arg_index_by_name(self, name):
        """Return index of argument in params with name."""
        return find_arg_index_by_name(self.params, name)

    def _as_arg(self, name):
        """Create an argument to hold the function result.
        This is intended for pointer arguments, char, string or vector.
        Move template_arguments from function to argument.
        """
        new = Declaration()
        new.specifier = self.specifier[:]
        new.storage = self.storage[:]
        new.const = self.const
        new.volatile = self.volatile
        new.declarator = copy.deepcopy(self.declarator)
        new.declarator.name = name
        if not new.declarator.pointer:
            # make sure the return type is a pointer
            new.declarator.pointer = [Ptr("*")]
        # new.array = None
        new.attrs = copy.deepcopy(self.attrs)
        new.typemap = self.typemap
        new.template_arguments = self.template_arguments
        return new

    def _set_to_void(self):
        """Change function to void"""
        self.specifier = ["void"]
        self.typemap = typemap.lookup_type("void")
        self.const = False
        self.volatile = False
        self.declarator.pointer = []
        self.template_arguments = []

    def result_as_arg(self, name):
        """Pass the function result as an argument.
        Change function result to 'void'.
        """
        newarg = self._as_arg(name)
        self.params.append(newarg)
        self._set_to_void()
        return newarg

    def instantiate(self, node):
        """Instantiate a template argument.
        node - Declaration node of template argument.
        Return a new copy of self and fill in type from node.
        If node is 'int *', the pointer is in the declarator.
        """
        # XXX - what if T = 'int *' and arg is 'T *arg'?
        new = copy.copy(self)
        new.set_type(node.typemap)
        return new

    def __str__(self):
        out = []
        if self.const:
            out.append("const ")
        if self.volatile:
            out.append("volatile ")
        if self.attrs["_destructor"]:
            out.append("~")
        if self.storage:
            out.append(" ".join(self.storage))
            out.append(" ")
        if self.specifier:
            out.append(" ".join(self.specifier))
        else:
            out.append("int")
        if self.declarator:
            out.append(" ")
            out.append(str(self.declarator))
        if self.params is not None:
            out.append("(")
            if self.params:
                out.append(str(self.params[0]))
                for param in self.params[1:]:
                    out.append(",")
                    out.append(str(param))
            out.append(")")
            if self.func_const:
                out.append(" const")
        if self.array:
            for dim in self.array:
                out.append("[")
                out.append(todict.print_node(dim))
                out.append("]")
        if self.init:
            out.append("=")
            out.append(str(self.init))
        return "".join(out)

    def gen_decl(self, **kwargs):
        """Return a string of the unparsed declaration.

        Args:
            params - None do not print parameters.
        """
        decl = []
        self.gen_decl_work(decl, **kwargs)
        return "".join(decl)

    def gen_decl_work(self, decl, **kwargs):
        """Generate string by appending text to decl.

        Replace params with value from kwargs.
        Most useful to call with params=None to skip parameters
        and only get function result.

        attrs=False give compilable code.
        """
        use_attrs = kwargs.get("attrs", True)
        if self.const:
            decl.append("const ")

        if self.attrs["_destructor"]:
            decl.append("~")
        if self.storage:
            decl.append(" ".join(self.storage))
            decl.append(" ")
        decl.append(" ".join(self.specifier))
        if self.template_arguments:
            decl.append("<")
            for targ in self.template_arguments:
                decl.append(str(targ))
                decl.append(",")
            decl[-1] = ">"

        if self.declarator:
            self.declarator.gen_decl_work(decl, **kwargs)

        if self.init is not None:
            decl.append("=")
            decl.append(str(self.init))
        #        if use_attrs:
        #            self.gen_attrs(self.attrs, decl)

        params = kwargs.get("params", self.params)
        if params is not None:
            decl.append("(")
            comma = ""
            for arg in params:
                decl.append(comma)
                arg.gen_decl_work(decl)
                comma = ", "
            decl.append(")")
            if self.func_const:
                decl.append(" const")
        for dim in self.array:
            decl.append("[")
            decl.append(todict.print_node(dim))
            decl.append("]")
        if use_attrs:
            self.gen_attrs(self.attrs, decl)

    _skip_annotations = ["template"]

    def gen_attrs(self, attrs, decl):
        space = " "
        for attr in sorted(attrs):
            if attr[0] == "_":  # internal attribute
                continue
            if attr in self._skip_annotations:
                continue
            value = attrs[attr]
            if value is None:  # unset
                continue
            decl.append(space)
            decl.append("+")
            if value is True:
                decl.append(attr)
            else:
                decl.append("{}({})".format(attr, value))
            space = ""

    def gen_arg_as_cxx(self, **kwargs):
        """Generate C++ declaration of variable.
        No parameters or attributes.
        """
        decl = []
        self.gen_arg_as_lang(decl, lang="cxx_type", **kwargs)
        return "".join(decl)

    def gen_arg_as_c(self, **kwargs):
        """Return a string of the unparsed declaration.
        """
        decl = []
        self.gen_arg_as_lang(decl, lang="c_type", **kwargs)
        return "".join(decl)

    def gen_arg_as_lang(
        self,
        decl,
        lang,
        continuation=False,
        asgn_value=False,
        remove_const=False,
        with_template_args=False,
        **kwargs
    ):
        """Generate an argument for the C wrapper.
        C++ types are converted to C types using typemap.

        Args:
            lang = c_type or cxx_type
            continuation - True - insert tabs to aid continuations.
                           Defaults to False.
            asgn_value - If True, make sure the value can be assigned
                         by removing const. Defaults to False.
            remove_const - Defaults to False.
            as_ptr - Change reference to pointer
            force_ptr - Change a scalar into a pointer
            as_scalar - Do not print Ptr
            params - if None, do not print function parameters.
            with_template_args - if True, print template arguments

        If a templated type, assume std::vector.
        The C argument will be a pointer to the template type.
        'std::vector<int> &'  generates 'int *'
        The length info is lost but will be provided as another argument
        to the C wrapper.
        """
        const_index = None
        if self.const:
            const_index = len(decl)
            decl.append("const ")

        if with_template_args and self.template_arguments:
            # Use template arguments from declaration
            typ = getattr(self.typemap, lang)
            decl.append(self.typemap.name)
            decl.append("<")
            for targ in self.template_arguments:
                decl.append(str(targ))
                decl.append(",")
            decl[-1] = ">"
        else:
            # Convert template_argument.
            # ex vector<int> -> int
            if self.template_arguments:
                ntypemap = self.template_arguments[0].typemap
            else:
                ntypemap = self.typemap
            typ = getattr(ntypemap, lang)
            decl.append(typ)

        if self.declarator is None:
            # XXX - used with constructor but seems wrong for abstract arguments
            # The C wrapper wants a pointer to the type.
            declarator = Declarator()
            declarator.name = self.name
        else:
            declarator = self.declarator

        if asgn_value and const_index is not None and not self.is_indirect():
            # Remove 'const' so the variable can be assigned to.
            decl[const_index] = ""
        elif remove_const and const_index is not None:
            decl[const_index] = ""

        if lang == "c_type":
            declarator.gen_decl_work(decl, as_c=True, **kwargs)
        else:
            declarator.gen_decl_work(decl, **kwargs)

        params = kwargs.get("params", self.params)
        if params is not None:
            decl.append("(")
            if continuation:
                decl.append("\t")
            comma = ""
            for arg in params:
                decl.append(comma)
                arg.gen_decl_work(decl, attrs=None, continuation=continuation)
                if continuation:
                    comma = ",\t "
                else:
                    comma = ", "
            decl.append(")")
            if self.func_const:
                decl.append(" const")
        if self.array:
            for dim in self.array:
                decl.append("[")
                decl.append(todict.print_node(dim))
                decl.append("]")

    def as_cast(self, language="c"):
        """
        Ignore const, name.
        Array as pointer.

        (as_cast) arg
        """
        decl = []
        typ = getattr(self.typemap, language + '_type')
        decl.append(typ)
        ptrs = []
        if self.declarator:
            for ptr in self.declarator.pointer:
                ptrs.append("*")   # ptr.ptr)
        if self.array:
            ptrs.append("*")
        if ptrs:
            decl.append(" ")
            decl.extend(ptrs)
        return ''.join(decl)
        
    ##############

    def bind_c(self, intent=None, **kwargs):
        """Generate an argument used with the bind(C) interface from Fortran.

        Args:
            intent - Explicit intent 'in', 'inout', 'out'.
                     Defaults to None to use intent from attrs.

            name   - Set name explicitly, else self.name.
        """
        t = []
        attrs = self.attrs
        ntypemap = self.typemap
        basedef = ntypemap
        if self.template_arguments:
            # If a template, use its type
            ntypemap = self.template_arguments[0].typemap

        typ = ntypemap.f_c_type or ntypemap.f_type
        if typ is None:
            raise RuntimeError(
                "Type {} has no value for f_c_type".format(self.typename)
            )
        t.append(typ)
        if attrs["value"]:
            t.append("value")
        intent = intent or attrs["intent"]
        if intent:
            t.append("intent(%s)" % intent.upper())

        decl = []
        decl.append(", ".join(t))
        decl.append(" :: ")

        if kwargs.get("name", None):
            decl.append(kwargs["name"])
        else:
            decl.append(self.name)

        if basedef.base == "vector":
            decl.append("(*)")  # is array
        elif ntypemap.base == "string":
            decl.append("(*)")
        elif attrs["dimension"]:
            # Any dimension is changed to assumed length.
            decl.append("(*)")
        elif attrs["rank"] is not None and attrs["rank"] > 0:
            # Any dimension is changed to assumed length.
            decl.append("(*)")
        elif attrs["allocatable"]:
            # allocatable assumes dimension
            decl.append("(*)")
        return "".join(decl)

    def gen_arg_as_fortran(
        self,
        bindc=False,
        local=False,
        **kwargs
    ):
        """Geneate declaration for Fortran variable.

        bindc - Use C interoperable type. Used with hidden and implied arguments.
        If local==True, this is a local variable, skip attributes
          OPTIONAL, VALUE, and INTENT
        """
        t = []
        attrs = self.attrs
        ntypemap = self.typemap
        if self.template_arguments:
            # If a template, use its type (std::vector)
            ntypemap = self.template_arguments[0].typemap

        is_allocatable = False
        is_pointer = False
        deref = attrs["deref"]
        if deref == "allocatable":
            is_allocatable = True
        elif deref == "pointer":
            is_pointer = True

        if not is_allocatable:
            is_allocatable = attrs["allocatable"]

        if ntypemap.base == "string":
            if attrs["len"] and local:
                # Also used with function result declaration.
                t.append("character(len={})".format(attrs["len"]))
            elif is_allocatable:
                t.append("character(len=:)")
            elif self.array:
                t.append("character(kind=C_CHAR)")
            elif not local:
                t.append("character(len=*)")
            else:
                t.append("character")
        elif bindc:
            t.append(ntypemap.f_c_type or ntypemap.f_type)
        else:
            t.append(ntypemap.f_type)

        if not local:  # must be dummy argument
            if attrs["value"]:
                t.append("value")
            intent = attrs["intent"]
            if intent:
                t.append("intent(%s)" % intent.upper())

        if is_allocatable:
            t.append("allocatable")
        if is_pointer:
            t.append("pointer")

        decl = []
        decl.append(", ".join(t))
        decl.append(" :: ")

        if "name" in kwargs:
            decl.append(kwargs["name"])
        else:
            decl.append(self.name)

        dimension = attrs["dimension"]
        rank = attrs["rank"]
        if rank is not None:
            decl.append(self.fortran_ranks[rank])
        elif dimension:
            if is_allocatable:
                # Assume 1-d.
                decl.append("(:)")
            elif is_pointer:
                decl.append("(:)")  # XXX - 1d only
            else:
                decl.append("(" + dimension + ")")
        elif is_allocatable:
            # Assume 1-d.
            if ntypemap.base != "string":
                decl.append("(:)")
        elif self.array:
            decl.append("(")
            # Convert to column-major order.
            for dim in reversed(self.array):
                decl.append(todict.print_node(dim))
                decl.append(",")
            decl[-1] = ")"

        return "".join(decl)


class CXXClass(Node):
    """A C++ class statement.
    """

    def __init__(self, name):
        self.name = name


class Namespace(Node):
    """A C++ namespace statement.
    """

    def __init__(self, name):
        self.name = name


class Enum(Node):
    """An enumeration statement.
    enum Color { RED, BLUE, WHITE }
    enum class Color { RED, BLUE, WHITE }
    """

    def __init__(self, name, scope=None):
        self.name = name
        self.scope = scope
        self.members = []


class EnumValue(Node):
    """A single name in an enum statment with optional value"""

    def __init__(self, name, value=None):
        self.name = name
        self.value = value


class Struct(Node):
    """A struct statement.
    struct name { int i; double d; };
    """

    def __init__(self, name):
        self.name = name
        self.members = []


class Template(Node):
    """A template statement.

    parameters - list of TemplateParam instances.
    decl - Declaration or CXXClass Node.
    """

    def __init__(self):
        self.parameters = []
        self.decl = None

        self.parent = None
        self.symbols = {}
        self.is_class = False

    def fill_symbols(self, parent):
        """Add the TemplateParams into the symbol table.
        This allows them to be looked up via unqualified_lookup.
        """
        self.parent = parent
        for param in self.parameters:
            self.symbols[param.name] = param

    def unqualified_lookup(self, name):
        """Lookup template parameter."""
        if name in self.symbols:
            return self.symbols[name]
        return self.parent.unqualified_lookup(name)


class TemplateParam(Node):
    """A template parameter.
    template < TemplateParameter >

    Create a Typemap for the TemplateParam.
    XXX - class and typename are discarded while parsing.

    self.typemap = a typemap.Typemap with base='template'.
                   Used as a place holder for the Template argument.
                   The typemap is not registered.
    """

    def __init__(self, name):
        # Set cxx_type so flat_name will be set.
        # But use an illegal identifer name since it should never be used.
        self.name = name
        self.typemap = typemap.Typemap(
            name, base="template", cxx_type="-TemplateParam-")


def check_decl(decl, namespace=None, template_types=None, trace=False):
    """ parse expr as a declaration, return list/dict result.

    namespace - An ast.AstNode subclass.
    """
    if not namespace:
        # grab global namespace if not passed in.
        namespace = global_namespace
    if template_types:
        global type_specifier
        old_types = type_specifier
        type_specifier = set(old_types)
        type_specifier.update(template_types)
        a = Parser(decl, namespace, trace).decl_statement()
        type_specifier = old_types
    else:
        a = Parser(decl, namespace, trace).decl_statement()
    return a


def create_voidstar(ntypemap, name, const=False):
    """Create a Declaration for an argument as 'typ *name'.
    """
    arg = Declaration()
    arg.const = const
    arg.declarator = Declarator()
    arg.declarator.name = name
    arg.declarator.pointer = [Ptr("*")]
    arg.specifier = ntypemap.cxx_type.split()
    arg.typemap = ntypemap
    return arg


def create_struct_ctor(cls):
    """Create a ctor function for a struct (aka C++ class).
    Use with PY_struct_arg==class.
    Used as __init__ function.
    """
    name = cls.name + "_ctor"
    ast = Declaration()
    ast.attrs["_constructor"] = True
    ast.attrs["name"] = name
##        _name="ctor",
    ast.typemap = cls.typemap
    ast.specifier = [ cls.name ]
    ast.params = []
    return ast


def find_arg_by_name(decls, name):
    """Find argument in params with name.
    Return None if not found.

    Args:
        decls - list of Declaration
        name  - argument to find
    """
    if decls is None:
        return None
    for decl in decls:
        if decl.name == name:
            return decl
    return None

def find_arg_index_by_name(decls, name):
    """Return index of argument in decls with name.

    Args:
        decls - list of Declaration
        name  - argument to find
    """
    if decls is None:
        return -1
    for i, decl in enumerate(decls):
        if decl.name == name:
            return i
    return -1

