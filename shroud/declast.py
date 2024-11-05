# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
A top-down, recursive descent parser for C/C++ expressions with
additions for shroud attributes.

Typical usage:
  declast.check_decl(decl, self.symtab)
"""

from __future__ import print_function
import collections
import copy
import re

from . import error
from . import todict
from . import typemap

Token = collections.namedtuple("Token", ["typ", "value", "line", "column"])

# https://docs.python.org/3.10/library/re.html#writing-a-tokenizer
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
    "complex",    # C _Complex
}
type_qualifier = {"const", "volatile"}
storage_class = {"auto", "register", "static", "extern", "typedef"}

cxx_keywords = {
    "class", "enum", "namespace", "struct", "template", "typename",
    "public", "private", "protected",
}

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
    ("COLON", r":"),
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
    complex_double="double_complex",
    complex_float="float_complex",
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
        lines = self.decl.split("\n")
        lineno = self.token.line
#        msg = "line {}: ".format(lineno) + format.format(*args)
        msg = format.format(*args)
        ptr = " " * (self.token.column-1) + "^"
        raise error.ShroudParseError("\n".join([lines[lineno-1], ptr, msg]),
                                     lineno, self.token.column)

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

######################################################################


class Parser(ExprParser):
    """
    Parse a C/C++ declaration with Shroud annotations.

    An abstract-declarator is a declarator without an identifier,
    consisting of one or more pointer, array, or function modifiers.
    """

    def __init__(self, decl, symtab, trace=False):
        """
        Args:
            decl - str, declaration to parse.
            namespace - ast.NamespaceNode, ast.ClassNode
        """
        self.decl = decl
        self.symtab = symtab
        self.trace = trace
        self.indent = 0
        self.token = None
        self.tokenizer = tokenize(decl)
        self.next()  # load first token

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
            node.declarator.arg_name = "arg" + str(len(params))
            if self.have("COMMA"):
                if self.have("VARARG"):
                    raise NotImplementedError("varargs")
            else:
                break
        self.mustbe("RPAREN")
        self.exit("parameter_list", str(params))
        return params

    def nested_namespace(self, namespace):
        """Look for qualified name.
        Current token.typ is an ID.

        <nested-namespace> ::= { namespace :: }* identifier

        Return namespace which owns qualified name and
        the fully qualified name (aa:bb:cc)

        Args:
            namespace - Node
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
        Args:
            node - declast.Declaration
        """
        self.enter("declaration_specifier")
        found_type = False
        more = True
        parent = self.symtab.current

        # destructor
        if self.have("TILDE"):
            if not hasattr(parent, "has_ctor"):  # CXXClass or Struct
                raise RuntimeError("Destructor is not in a class")
            tok = self.mustbe("ID")
            if tok.value != parent.name:
                raise RuntimeError("Expected class-name after ~")
            node.specifier.append("void")
            self.parse_template_arguments(node)
            #  class Class1 { ~Class1(); }
            self.info("destructor", parent.typemap.name)
            node.is_dtor = tok.value
            #node.typemap = self.namespace.typemap # The class' typemap
            node.typemap = typemap.void_typemap
            found_type = True
            more = False

        while more:
            # if self.token.type = 'ID' and  typedef-name
            if not found_type and self.token.typ == "ID":
                # Find typedef'd names, classes and namespaces
                ns = self.symtab.current.unqualified_lookup(self.token.value)
                if ns:
                    ns, ns_name = self.nested_namespace(ns)
                    node.specifier.append(ns_name)
                    self.parse_template_arguments(node)
                    if (
                        hasattr(ns, "has_ctor")  # CXXClass or Struct
                        and parent is ns
                        and self.token.typ == "LPAREN"
                    ):
                        # template<T> vector { vector<T>(); }
                        # class Class1 { Class1(); }
                        self.info("constructor")
                        node.is_ctor = True
                        more = False
                    # Save fully resolved typename
                    node.typemap = ns.typemap
                    #self.info("Typemap {}".format(ns.typemap.name))
                    if node.typemap.sgroup == "smart_ptr":
                        create_smart_ptr_typemap(node, self.symtab)
                    elif node.typemap.base == "template":
                        node.template_argument = ns_name
                    found_type = True
                else:
                    more = False
            elif self.token.typ == "TYPE_SPECIFIER":
                found_type = True
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
            elif self.token.typ == "CLASS":
                self.class_decl(node)
                found_type = True
            elif self.token.typ == "ENUM":
                self.enum_decl(node)
                found_type = True
            elif self.token.typ == "STRUCT":
                self.struct_decl(node)
                found_type = True
            else:
                more = False
        if not found_type:
            # GGG ValueError: Single '}' encountered in format string
            if self.token.value == '}':
                value = '}}'
            else:
                value = self.token.value
            self.error_msg(
                "Expected TYPE_SPECIFIER, found {} '{}'".format(
                    self.token.typ, value
                )
            )
        self.exit("declaration_specifier")
        return

    def parse_template_arguments(self, node):
        """Parse template parameters.
        vector<T>
        map<Key,T>
        vector<const double *>

        Used while parsing function arguments.
        similar to template_argument_list
        """
        lst = node.template_arguments
        if self.have("LT"):
            while self.token.typ != "GT":
                temp = self.declaration()
                lst.append(temp)
                if not self.have("COMMA"):
                    break
            self.mustbe("GT")

    def top_level(self):
        """Parse top/file level scope."""
        self.enter("top_statement")
        node = Block()
        while self.token.typ != "EOF":
            self.group_statement(node.stmts)
        self.exit("top_statement")
        return node

    def group_statement(self, group):
        """Parse statements and any associated block.

        class name { };
        struct tag { };
        namespace name { }
        """
        self.enter("group_statement")
        node = self.line_statement()
        group.append(node)
        if isinstance(node, Namespace):
            self.block_statement(node.group)
        elif isinstance(node, CXXClass):
            self.block_statement(node.group)
            self.mustbe("SEMICOLON")
        elif isinstance(node, Template):
            ast = node.decl
            if isinstance(node.decl, CXXClass):
                self.block_statement(ast.group)
            self.mustbe("SEMICOLON")
        else:
            self.mustbe("SEMICOLON")
        self.exit("group_statement")
    
    def block_statement(self, group):
        """Parse curly block.
        Appends Nodes to group.
        Block following class, struct, namespace:
          '{' [ line_statement* ] '}'
        """
        self.enter("block_statement")
        self.mustbe("LCURLY")
        while self.token.typ != "RCURLY":
            self.group_statement(group)
        self.mustbe("RCURLY")
        self.exit("block_statement")
    
    def line_statement(self):
        """Check for optional semicolon and stray stuff at the end of line.
        """
        if self.token.typ == "NAMESPACE":
            node = self.namespace_statement()
        elif self.token.typ == "TEMPLATE":
            node = self.template_statement()
        else:
            node = self.declaration(stmt=True)
        return node

    def decl_statement(self):
        """Check for optional semicolon and stray stuff at the end of line.
        Used when parsing decl from YAML which may not have semicolon.
        """
        node = self.line_statement()
        self.have("SEMICOLON")
        self.mustbe("EOF")
        return node

    def declaration(self, stmt=False):
        """Parse a declaration statement.
        Use with decl_statement and function arguments

        <declaration> ::= {<declaration-specifier>}+ <declarator_item>*
        """
        self.enter("declaration")
        node = Declaration(self.symtab)
        self.declaration_specifier(node)
        self.get_canonical_typemap(node)

        node.declarator = self.declarator_item(node)
        node.declarators.append(node.declarator)
        if stmt:
            # A declaration statement may have multiple declarators
            while self.have("COMMA"):
                d2 = self.declarator_item(node)
                node.declarators.append(d2)

        # SSS Share fields between Declaration and Declarator for now
        for d2 in node.declarators:
            if d2.func:
                d2.func.typemap = node.typemap
                if "typedef" in node.storage:
                    d2.typemap = node.typemap
                else:
                    d2.typemap = add_funptr_typemap(self.symtab, node, d2)
                    d2.typemap = node.typemap  # work in progress
            else:
                d2.typemap = node.typemap
                
        if "typedef" in node.storage:
            self.symtab.create_typedef(node)
        self.exit("declaration", str(node))
        return node

    def declarator_item(self, node):
        """
        <declarator_item> ::= (
                             '['  <constant-expression>?  ']'  |
                             '('  <parameter-list>            ')' [ const ]
                            ) [ = <initializer> ]

        node - declast.Declaration
        """
        self.enter("declarator_item")
        if node.is_dtor:
            declarator = Declarator()
            declarator.is_dtor = True
            declarator.ctor_dtor_name = True
            declarator.default_name = "dtor"
        elif node.is_ctor:
            declarator = Declarator()
            declarator.is_ctor = True
            declarator.ctor_dtor_name = True
            declarator.default_name = "ctor"
        else:
            declarator = self.declarator()

        if self.token.typ == "LPAREN":  # peek
            # Function parameters.
            params = self.parameter_list()
            declarator.params = params

            # Look for (void), set to no parameters.
            if len(params) == 1:
                chk = params[0]
                if (chk.declarator.name is None and  # abstract declarator
                    len(chk.declarator.pointer) == 0 and
                    chk.specifier == ["void"] and
                    chk.declarator.func is None    # Function pointers
                ):
                    declarator.params = []

            #  method const
            if self.token.typ == "TYPE_QUALIFIER":
                if self.token.value == "const":
                    self.next()
                    declarator.func_const = True
                else:
                    raise RuntimeError(
                        "'{}' unexpected after function declaration".format(
                            self.token.value
                        )
                    )
        while self.token.typ == "LBRACKET":
            self.next() # consume bracket
            declarator.array.append(self.expression())
            self.mustbe("RBRACKET")
        self.attribute(declarator.attrs)  # variable attributes

        # Attribute are parsed before default value since
        # default value may have a +.
        # (int value = 1+size)
        if self.have("EQUALS"):
            declarator.init = self.initializer()

        if declarator.ctor_dtor_name:
            declarator.ctor_dtor_name = declarator.attrs.get("name", declarator.default_name)
            
        self.exit("declarator_item", str(node))
        return declarator

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
            # Promote name.
            node.name = node.func.name
            self.mustbe("RPAREN")

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
        ntypemap = self.symtab.lookup_typemap(typename)
# XXX - incorporate pointer into typemap
#        nptr = decl.is_pointer()
#        if nptr == 0:
#            ntypemap = typemap.lookup_typemap(typename)
#        else:
#            for i in range(nptr, -1, -1):
#                ntypemap = typemap.lookup_typemap(typename + "***"[0:i])
#                if ntypemap is not None:
#                    break
        if ntypemap is None:
            self.error_msg(
                "(get_canonical_typemap) Unknown typemap '{}' - '{}'".format("_".join(decl.specifier), typename)
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

    def class_decl(self, node):
        """Create a class.

        class ID [ : ( PUBLIC | PRIVATE | PROTECTED ) ID '{'  '}'
        class ID ;
          Forward declare
        class ID <EOF>
          Declare in YAML.

        node : declast.Declaration
        """
        self.enter("class_decl")
        self.mustbe("CLASS")
        name = self.mustbe("ID")
        clsnode = CXXClass(name.value, self.symtab)
        node.specifier.append("class " + name.value)
        node.class_specifier = clsnode
        node.typemap = clsnode.typemap
        if self.have("EOF"):
            # Body added by other lines in YAML.
            node.tag_body = True
        elif self.have("COLON"):
            node.tag_body = True
            if self.token.typ in ["PUBLIC", "PRIVATE", "PROTECTED"]:
                access_specifier = self.token.value
                self.next()
            else:
                access_specifier = 'private'
            if self.token.typ == "ID":
                ns = self.symtab.current.unqualified_lookup(self.token.value)
                if ns:
                    ns, ns_name = self.nested_namespace(ns)
                    # XXX - make sure ns is a ast.ClassNode (and not a namespace)
                    clsnode.baseclass.append((access_specifier, ns_name, ns))
                else:
                    self.error_msg("unknown class '{}'", self.token.value)
            else:
                self.mustbe("ID")

        if self.have("LCURLY"):
            node.tag_body = True
            members = clsnode.members
            while self.token.typ != "RCURLY":
#                members.append(self.declaration()) # GGG, accepts too much  - template
                members.append(self.line_statement())
                self.mustbe("SEMICOLON")
            self.mustbe("RCURLY")
            self.symtab.pop_scope()
                    
        self.exit("class_decl")
        return node

    def namespace_statement(self):
        """  namespace ID
        """
        self.enter("namespace_statement")
        self.mustbe("NAMESPACE")
        name = self.mustbe("ID")
        node = Namespace(name.value, self.symtab)
        self.exit("namespace_statement")
        return node

    def template_statement(self):
        """  template < template-parameter-list > declaration
        template-parameter ::= [ class | typename] ID
        """
        self.enter("template_statement")
        self.mustbe("TEMPLATE")
        node = Template(self.symtab)
        name = self.mustbe("LT")
        while self.token.typ != "GT":
            if self.have("TYPENAME"):
                name = self.mustbe("ID").value
            elif self.have("CLASS"):
                name = self.mustbe("ID").value
            else:
                name = self.mustbe("ID").value
            node.append_template_param(name)
            if not self.have("COMMA"):
                break
        self.mustbe("GT")

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
            temp = Declaration(self.symtab)
            self.declaration_specifier(temp)
            self.get_canonical_typemap(temp)
            lst.append(temp)
            if not self.have("COMMA"):
                break
        self.mustbe("GT")
        return lst

    def enum_decl(self, node):
        """Creating an enumeration.

        ENUM [ CLASS | STRUCT ] ID {  }
           Add to typemap table as "enum-{name}"
        ENUM ID
           Lookup enum in typemap table.

        node : declast.Declaration
        """
        self.enter("enum_decl")
        self.mustbe("ENUM")
        if self.have("STRUCT"):
            scope = "struct"
        elif self.have("CLASS"):
            scope = "class"
        else:
            scope = None
        name = self.mustbe("ID")  # GGG - optional
        ename = name.value
        if scope:
            node.specifier.append("enum {} {}".format(scope, ename))
        else:
            node.specifier.append("enum " + ename)
        if self.have("LCURLY"):
            #        self.mustbe("LCURLY")
            node.tag_body = True
            enumnode = Enum(ename, self.symtab, scope)
            members = enumnode.members
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
            self.symtab.pop_scope()
            node.enum_specifier = enumnode
            node.typemap = enumnode.typemap
        else:
            enumnode = self.symtab.current.lookup_tag("enum", ename)
            if enumnode is None:
                raise RuntimeError("Enum tag '%s' is not defined" % ename)
            ntypemap = enumnode.typemap
            node.typemap = ntypemap
        self.exit("enum_decl")#, str(members))
        return node

    def struct_decl(self, node):
        """Create a struct.

        STRUCT ID {  }
           Add to typemap table as "struct-{name}"
        STRUCT ID ;
           Forward declare
        STRUCT ID <EOF>
           Declare in YAML.
        STRUCT ID [ Declaration ]
           Lookup struct tag in typemap table.

        node : declast.Declaration
        """
        self.enter("struct_decl")
        self.mustbe("STRUCT")
        name = self.mustbe("ID")  # GGG name is optional
        sname = name.value
        node.specifier.append("struct " + sname)
        if self.have("LCURLY"):
            structnode = Struct(sname, self.symtab)
            members = structnode.members
            while self.token.typ != "RCURLY":
                members.append(self.declaration(stmt=True))
                self.mustbe("SEMICOLON")
            self.mustbe("RCURLY")
            self.symtab.pop_scope()
            node.class_specifier = structnode
            node.typemap = structnode.typemap
            node.tag_body = True
        elif self.have("EOF"):
            structnode = Struct(sname, self.symtab)
            node.class_specifier = structnode
            node.typemap = structnode.typemap
            # Body added by other lines in YAML.
            node.tag_body = True
            # GGG - Caller must call symtab.pop_scope when finished with members.
        else:
            structnode = self.symtab.current.lookup_tag("struct", sname)
            if structnode is None:
                raise RuntimeError("Struct tag '%s' is not defined" % sname)
            node.class_specifier = structnode
            ntypemap = structnode.typemap
            node.typemap = ntypemap
        self.exit("struct_decl")
        return node


######################################################################
# Attribute only parser

class AttrParser(Parser):
    def attrs(self, bind, bindfcn):
        """
        <arg> ::= name [ +attrs ]*
        <list> ::= [ ( ]  <arg> [ , <arg> ]* [ ) ] [ +attrs ]*

        Attributes are filled directly into bind[name].meta
        """
        more = True
        have_func = False
        if self.have("LPAREN"):
            have_func = True
        if self.peek("RPAREN"):
            more = False

        while more:
            tok = self.mustbe("ID")

            attrs = {}
            self.attribute(attrs)
            if attrs:
                bindarg = bind.setdefault(tok.value, bindfcn())
                if bindarg.meta is None:
                    bindarg.meta = collections.defaultdict(lambda: None)
                bindarg.meta.update(attrs)

            if not self.have("COMMA"):
                break

        if have_func and self.mustbe("RPAREN"):
            attrs = {}
            self.attribute(attrs)
            if attrs:
                bindarg = bind.setdefault("+result", bindfcn())
                if bindarg.meta is None:
                    bindarg.meta = collections.defaultdict(lambda: None)
                bindarg.meta.update(attrs)
            
        self.mustbe("EOF")
        
def check_attrs(decl, bind, bindfcn, trace=False):
    """ parse an attribute expression.
           (arg1 +attr, arg2+attr)  +attr

    namespace - An ast.AstNode subclass.
    """
    #trace = True   # GGG For debug
    a = AttrParser(decl, None, trace).attrs(bind, bindfcn)
    return a

######################################################################
# Abstract Syntax Tree Nodes

class Node(object):
    """Abstract Symtax Tree base object.

    Contains the symbol table for the scope.

    children - Symbol table nodes of types.
    group - Parse tree nodes.
    """
    def init_symtab(self, parent, prefix):
        """This node can contain nested symbols.
        Used for looking up scoped names.
        Used with Struct, CXXClass, Namespace, Global

        scope_prefix is the fully qualifed name (includes scope_name).
        """
        if parent is self:
            print("XXXXXXX parent is self")
        self.parent = parent
        self.children = []    # Like symbols but in creation order.
        self.scope_prefix = prefix
        self.symbols = {}
        self.using = []
#        if parent is not None: # GGG
#            parent.add_symbol(self.name, self)

    def add_child(self, name, node):
        """Add child node.

        Done as part of SymbolTable.add_child_to_current.
        Call explicilty for non-scope Nodes like Declaration.
        (Actually a function creates a scope, but not one
        we care about since wrapping is not involved with
        local variables.)

        node : Node
        """
        self.symbols[name] = node
        self.children.append(node)

    def add_tag(self, tag, node):
        """Add a Node to symbols.
        Mangle tag name before adding to symbols dictionary.
        Used with struct and enum tags.
        """
        self.symbols["{}-{}".format(tag, node.name)] = node

    def lookup_tag(self, tag, name):
        """
        Mangle tag name before looking up.
        """
        return self.unqualified_lookup("{}-{}".format(tag, name))
        
    def XXXcreate_template_typemaps(self, node, symtab):
        """
        Create typemaps for each template argument.
        This is done after we know if if is a class/struct/function template.
        node - Template
        """
        for param in node.parameters:
            self.symbols[param.name] = param
            type_name = symtab.scopename + param.name
            ntypemap = typemap.Typemap(
                type_name, base="template", cxx_type="-TemplateParam-")
            param.typemap = ntypemap
            symtab.register_typemap(type_name, ntypemap)

    def check_forward_declaration(self, symtab):
        """Return Node of any previously declared name.
        Used with CXXClass and Struct.
        If parent already declares name, assume it is a forward
        declaration (both same Python class)

        Keep both Nodes in children but only one in symbols.
        Both declartions share typemap.
        """
        forward = symtab.current.symbols.get(self.name)
        if forward:
            # GGG check types the same and forward has no children.
            self.typemap = forward.typemap  # GGG
            self.newtypemap = forward.newtypemap
        return forward

    def qualified_lookup(self, name):
        """Look for symbols within this Node.
        A qualified name is a name that appears on the
        right hand side of the scope resolution operator ::.
        """
        return self.symbols.get(name, None)

    def unqualified_lookup(self, name):
        """Look for symbols in this node or its parent.
        Also look in symbols from a USING NAMESPACE statement.
        [basic.lookup.unqual]
        """
        if name in self.symbols:
            return self.symbols[name]
        for ns in self.using:
            if name in ns.symbols:
                return ns.symbols[name]
        if self.parent is self:
            raise RuntimeError("recursion")
        if self.parent is None:
            return None
        else:
            return self.parent.unqualified_lookup(name)

    def using_directive(self, name):
        """Implement 'using namespace <name>'
        """
        ns = self.unqualified_lookup(name)
        if ns is None:
            raise RuntimeError("{} not found in namespace".format(name))
        if ns not in self.using:
            self.using.append(ns)


class Global(Node):
    """Represent the global namespace"""
    def __init__(self):
        self.name = "***global***"
        self.init_symtab(None, "")


class Block(Node):
    """Represent a group of statements"""
    def __init__(self):
        self.name = "***block***"
        self.stmts = []


class Identifier(Node):
    def __init__(self, name, args=None):
        self.name = name
        self.args = args


class AssumedRank(Node):
    """Assumed-rank dimension i.e. (..)"""
    pass


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

    def __str__(self):
        if self.const:
            return self.ptr + " const"
        else:
            return self.ptr


class Declarator(Node):
    """
    If both name and func are None, then this is an abstract
    declarator: ex. 'int *' in 'void foo(int *)'.
    """

    def __init__(self):
        self.pointer = []  # Multiple levels of indirection
        self.name = None  # *name
        self.func = None  # (*name)     Declarator
        
        self.ctor_dtor_name = False
        self.default_name = None
        self.arg_name = None         # default abstract-declarator name - arg%n

        self.params = None  # None=No parameters, []=empty parameters list
        self.array = []
        self.init = None  # initial value
        self.attrs = {}
        self.func_const = False
        self.typemap = None
        self.is_ctor = False
        self.is_dtor = False

    def get_user_name(self, use_attr=True):
        """Get name from declarator
        use_attr - True, check attr for name
        ctor and dtor should have default_name set
        """
        if use_attr:
            name = self.attrs.get("name", self.default_name)
            if name is not None:
                return name
        return self.name
    user_name = property(get_user_name, None, None, "Declaration user_name")

    def is_pointer(self):
        """Return number of levels of pointers.
        """
        nlevels = 0
        for ptr in self.pointer:
            if ptr.ptr == "*":
                nlevels += 1
        return nlevels

    def is_reference(self):
        """Return number of levels of references.
        """
        nlevels = 0
        for ptr in self.pointer:
            if ptr.ptr == "&":
                nlevels += 1
        return nlevels

    def is_indirect(self):
        """Return number of indirections.
        pointer or reference.
        """
        nlevels = 0
        for ptr in self.pointer:
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
        for ptr in self.pointer:
            if ptr.ptr:
                nlevels += 1
        return nlevels
        
    def is_function_pointer(self):
        """Return number of levels of pointers.
        """
        if self.func is None:
            return False
        if not self.func.pointer:
            return False
        return True

    def XXXget_indirect(self):
        """Return indirect operators.
        '*', '**', '&*', '[]'
        """
        out = ''
        for ptr in self.pointer:
            out += ptr.ptr
        if self.array:
            out += "[]"   # XXX - multidimensional?
        return out

    def get_indirect_stmt(self):
        """Return statement field for pointers.
        'scalar', '*', '**'
        """
        out = ''
        for ptr in self.pointer:
            out += ptr.ptr
        if self.array:
            out += "[]"   # XXX - multidimensional?
        if out == "":
            return "scalar"
        return out

    def get_abstract_declarator(self):
        """Return abstract declarator.
        """
        out = ""
        for ptr in self.pointer:
            out += ptr.ptr
        if self.array:
            out += "[]"   # XXX - multidimensional?
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

    def gen_attrs(self, attrs, parts):
        space = " "
        for attr in sorted(attrs):
            if attr[0] == "_":  # internal attribute, __line__
                continue
            value = attrs[attr]
            parts.append(space)
            parts.append("+")
            if value is True:
                parts.append(attr)
            else:
                parts.append("{}({})".format(attr, value))
            space = ""

    def to_string(self, abstract=False, name=None):
        out = []
        for ptr in self.pointer:
            out.append(str(ptr))

        if self.func:
            out.append("(" + self.func.to_string(abstract) + ")")
        elif abstract:
            pass
        elif name is not None:
            out.append(name)
        elif self.name:
            out.append(self.name)

        if self.params is not None:
            out.append("(")
            if self.params:
                comma = ""
                for param in self.params:
                    out.append(comma)
                    out.append(str(param))
                    s = param.declarator.to_string(abstract)
                    if abstract:
                        if s:
                            out.append(s)
                        comma = ","
                    else:
                        if s:
                            out.append(" ")
                            out.append(s)
                        comma = ", "
            else:
                out.append("void")
            out.append(")")
            if self.func_const:
                out.append(" const")
        if self.array:
            for dim in self.array:
                out.append("[")
                out.append(todict.print_node(dim))
                out.append("]")
        if self.init is not None:
            out.append("=")
            out.append(str(self.init))
        if not abstract:
            self.gen_attrs(self.attrs, out)

        return "".join(out)

    def abstract(self):
        return self.to_string(abstract=True)

    def __str__(self):
        return self.to_string()


class Declaration(Node):
    """
    specifier = const  int
    init =         a  *a   a=1

    attrs     - Attributes set by the user.
    """

    def __init__(self, symtab=None):
#        self.symtab = symtab  # GGG -lots of problems with copy
        self.specifier = []  # int, long, ...
        self.storage = []  # static, tyedef, ...
        self.enum_specifier = None   # Enum
        self.class_specifier = None  # CXXClass, Struct (union)
        self.tag_body = False        # if True, members are defined.
        self.const = False
        self.volatile = False
        self.declarator = None       # declarators[0]
        self.declarators = []
        self.template_arguments = []    # vector<int>, list of Declaration
        self.template_argument = None   # T arg, str
        self.is_ctor = False
        self.is_dtor = False

        self.typemap = None

    def set_type(self, ntypemap):
        """Set type specifier from a typemap."""
        self.typemap = ntypemap
        # 'long long' into ['long', 'long']
        self.specifier = ntypemap.cxx_type.split()

    def get_full_type(self):
        return ' '.join(self.specifier)

    def instantiate(self, node):
        """Instantiate a template argument.
        node - Declaration node of template argument.
        Return a new copy of self and fill in type from node.
        Also copy the declarators and replace typemap.
        If node is 'int *', the pointer is in the declarator.
        """
        # XXX - what if T = 'int *' and arg is 'T *arg'?
        new = copy.copy(self)
        new.set_type(node.typemap)
        declarators = []
        for declarator in self.declarators:
            newd = copy.copy(declarator)
            newd.typemap = node.typemap
            declarators.append(newd)
        new.declarators = declarators
        new.declarator = declarators[0]
        return new

    def __str__(self):
        out = []
        if self.const:
            out.append("const ")
        if self.volatile:
            out.append("volatile ")
        if self.is_dtor:
            out.append("~")
            out.append(self.is_dtor)
        else:
            if self.storage:
                out.append(" ".join(self.storage))
                out.append(" ")
            if self.specifier:
                out.append(" ".join(self.specifier))
            else:
                out.append("int")
        if self.template_arguments:
            out.append(self.gen_template_arguments())
        return "".join(out)

    def __repr__(self):
        return "<Declaration('{}')>".format(str(self))

    def to_string_declarator(self, abstract=False, name=None):
        """Return the declaration for the first declarator"""
        declarator = self.declarator.to_string(abstract, name)
        if declarator:
            decl = "{} {}".format(str(self), declarator)
        else:
            decl = str(self)
        return decl
        
    def get_first_abstract_declarator(self):
        """Return an abstract declarator for the first declarator.
        The wrapping will split "int i,j" into "int i;int j"
        """
        declarator = self.declarator.get_abstract_declarator()
        if declarator:
            decl = "{} {}".format(str(self), declarator)
        else:
            decl = str(self)
        return decl
            
    def gen_template_argument(self):
        """
        ex  "int, double *"
        """
        # template_arguments is a list of Declarations
        return ",".join([targ.get_first_abstract_declarator()
                         for targ in self.template_arguments])
        
    def gen_template_arguments(self):
        """Return string for template_arguments."""
        return "<" + self.gen_template_argument() + ">"
        
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
        for ptr in self.declarator.pointer:
            ptrs.append("*")   # ptr.ptr)
        if self.declarator.array:
            ptrs.append("*")
        if ptrs:
            decl.append(" ")
            decl.extend(ptrs)
        return ''.join(decl)
        

class CXXClass(Node):
    """A C++ class statement.

    members are populated by function class_decl.
    children is populated by ast.py
    """

    def __init__(self, name, symtab):
        """
        ntypemap from YAML file.
        """
        self.name = name
        self.baseclass = []
        self.members = []
        self.has_ctor = True
        self.group = []

        forward = self.check_forward_declaration(symtab)

        if not forward:
            type_name = symtab.scopename + name
            ntypemap = typemap.Typemap(
                type_name,
                base="shadow",
                sgroup="shadow",
            )
            symtab.register_typemap(type_name, ntypemap)
            self.newtypemap = ntypemap
            self.typemap = ntypemap
        symtab.add_child_to_current(self)
        symtab.push_scope(self)

def create_smart_ptr_typemap(node, symtab):
    """Create a Typemap entry for a shared pointer instantiation.

    ex: std::shared_ptr<Object> *return_ptr(void);

    base_typemap is referencing type Object.

    Parameters:
      node - declast.Declaration
      symtab - declast.SymbolTable
    """
    targs = node.gen_template_arguments()
    type_name = node.typemap.name + targs
    base_typemap = node.template_arguments[0].typemap
    node.typemap = fetch_smart_ptr_typemap(type_name, base_typemap, symtab)

def fetch_smart_ptr_typemap(type_name, base_typemap, symtab):
    """Return a Typemap entry for a shared pointer instantiation.

    Return existing entry if it already exists, otherwise create one.
    Create with sgroup=shadow. It servers a a place holder and 
    will be filled in later.

    Parameters:
      type_name - str
      symtab - declast.SymbolTable
    """
    ntypemap = symtab.lookup_typemap(type_name)
    if ntypemap is None:
        ntypemap = typemap.Typemap(
            type_name,
            base="smartptr",
            sgroup="smartptr",
            cxx_type=type_name,
            base_typemap=base_typemap,
            ntemplate_args=1,
        )
        symtab.register_typemap(type_name, ntypemap)
    return ntypemap


class Namespace(Node):
    """A C++ namespace statement.
    """

    def __init__(self, name, symtab):
        self.name = name
        symtab.add_child_to_current(self)
        symtab.push_scope(self)
        self.group = []


class Enum(Node):
    """An enumeration statement.
    enum Color { RED, BLUE, WHITE }
    enum class Color { RED, BLUE, WHITE }

    For C and C++, the enum tag is registered.
    For C++, it is registered as a type.
    """

    def __init__(self, name, symtab, scope=None):
        self.name = name
        self.scope = scope
        self.members = []

        type_name = symtab.scopename + name
        ntypemap = typemap.Typemap(
            type_name,
        )
        ntypemap.is_enum = True  # GGG kludge to identify enums
        symtab.add_tag_to_current("enum", self)
        if symtab.language == "cxx":
            symtab.add_child_to_current(self)
            symtab.register_typemap(type_name, ntypemap)
        self.typemap = ntypemap

        symtab.push_scope(self)


class EnumValue(Node):
    """A single name in an enum statment with optional value"""

    def __init__(self, name, value=None):
        self.name = name
        self.value = value


class Struct(Node):
    """A struct statement.
    struct name
    struct name { int i; double d; };

    Add a typemap to the symbol table.
    For C and C++, the struct tag is registered.
    For C++, it is registered as a type.

    members are populated by function struct_decl.
    children is populated by ast.py
    """

    def __init__(self, name, symtab):
        self.name = name
        self.members = []
        self.has_ctor = True
        forward = self.check_forward_declaration(symtab)

        if not forward:
            type_name = symtab.scopename + name
            ntypemap = typemap.Typemap(
                type_name,
                base="struct",
                sgroup="struct",
                ntemplate_args = symtab.find_ntemplate_args()
            )
            symtab.add_tag_to_current("struct", self)
            if symtab.language == "cxx":
                symtab.add_child_to_current(self)
                symtab.register_typemap(type_name, ntypemap)
            self.newtypemap = ntypemap
            self.typemap = ntypemap
        symtab.push_scope(self)

    def __str__(self):
        return "struct " + self.name


class Template(Node):
    """A template statement.

    parameters - list of TemplateParam instances.
    decl - Declaration or CXXClass Node.
    """

    def __init__(self, symtab):
        self.parameters = []
        self.decl = None    # GGG maybe rename to ast or decl_ast

        self.name = "template-"
        self.is_class = False
        self.paramtypemap = symtab.lookup_typemap("--template-parameter--")
        self.ntemplate_args = 0

        symtab.push_template_scope(self)

    def append_template_param(self, name):
        """append a TemplateParam to this Template.
        """
        node = TemplateParam(name)
        node.typemap = self.paramtypemap
        self.parameters.append(node)
        self.symbols[name] = node
        self.ntemplate_args += 1

    def add_child(self, name, node):
        """
        Add the templated function into the parent,
        not the Template scope.
          template<U> class name
        """
        self.parent.add_child(name, node)

    def __str__(self):
        s = []
        for param in self.parameters:
            s.append(str(param))
        return "<" + ",".join(s) + ">"

class TemplateParam(Node):
    """A template parameter.
    template < TemplateParameter >

    Create a Typemap for the TemplateParam.
    XXX - class and typename are discarded while parsing.

    Must call Node.create_template_typemaps after the templated 
    declaration in order to get scope correct for typemaps.
       template<T> class cname
    Typemap name will be cname::T

    self.typemap = a typemap.Typemap with base='template'.
                   Used as a place holder for the Template argument.
                   The typemap is not registered.
    """

    def __init__(self, name):
        # Set cxx_type so flat_name will be set.
        # But use an illegal identifer name since it should never be used.
        self.name = name

    def __str__(self):
        return self.name
        

class Typedef(Node):
    """
    Added to SymbolTable to record a typedef name.

    When used with 'int', 'std::string', ...
    ast will be None.
    """
    def __init__(self, name, ast, ntypemap):
        self.name = name
        self.ast = ast
        if ast:
            ntypemap.is_typedef = True  # GGG kludge to identify typedef
        self.typemap = ntypemap
        
        
class SymbolTable(object):
    """Keep track of current scope of names while parsing.

    These names become part of the grammar and must be known
    while parsing and not part of the AST.

    scope_stack - stack (list) of AstNodes.
    name_stack - stack (list) of names of scope_stack nodes.
       useful to create scoped names - ns::class::enum

    Maintain a dictionary of typemaps.
    """
    def __init__(self, language="cxx"):
        self.scope_stack = []
        self.scope_len   = []
        self.scopename = ''
        self.typemaps = typemap.default_typemap()
        self.language = language

        # Create the global scope.
        self.top = Global()
        self.scope_stack.append(self.top)
        self.scope_len.append(0)
        self.current = self.top

    def push_template_scope(self, node):
        """
        Template scopes do not add to scopename.

        node - Template
        """
        node.init_symtab(self.current, self.scopename)
        self.scope_stack.append(node)
        self.scope_len.append(self.scope_len[-1])
        self.current = node

    def push_scope(self, node):
        """
        node creates a new named scope.
        
        node = Node subclass: CXXClass, Struct
        """
        # node := Struct
        name = node.name
        self.scopename = self.scopename[:self.scope_len[-1]] + name + '::'
        self.scope_len.append(len(self.scopename))

        if not hasattr(node, "scope_prefix"):
            # Only initialize once.
            node.init_symtab(self.current, self.scopename)
        self.scope_stack.append(node)
        self.current = node

    def pop_scope(self):
        self.scope_stack.pop()
        self.scope_len.pop()
        self.current = self.scope_stack[-1]
        self.scopename = self.scopename[:self.scope_len[-1]]

    def add_child_to_current(self, node, name=None):
        """Add symbol to current scope."""
        self.current.add_child(name or node.name, node)

    def add_tag_to_current(self, tag, node):
        """Add tag name to symbols."""
        self.current.add_tag(tag, node)

    def using_directive(self, name):
        """Implement 'using namespace <name>' for current scope
        """
        ns = self.current.using_directive(name)

    def register_typemap(self, name, ntypemap):
        self.typemaps[name] = ntypemap
        
    def lookup_typemap(self, name):
        ntypemap = self.typemaps.get(name)
        return ntypemap

    def create_nested_namespaces(self, names):
        """
        Create a possibly nested namespace.
        self.current is set to the last namespace.

        Use add_typedef_by_name to add member to the namespace.
        Call restore_depth to pop this created namespace.
        """
        depth = self.save_depth()
        ns = self.current
        for name in names:
            if name in ns.symbols:
                ns = ns.symbols[name]
                # GGG make sure it is a Namespace node
                self.add_child_to_current(ns)
                self.push_scope(ns)
            else:
                node = Namespace(name, self)
                ns.symbols[name] = node
                ns = node
        return depth
    
    def add_namespaces(self, ntypemap, as_typedef=False):
        """Create nested namespaces from list of names.

        Args:
            name - list of names for namespaces.
        """
        names = ntypemap.name.split("::")
        cxx_name = names.pop()
        depth = self.create_nested_namespaces(names)
        sgroup = ntypemap.sgroup
        if as_typedef:
            node = Typedef(cxx_name, None, ntypemap)
            self.current.add_child(node.name, node)
        elif sgroup == "shadow":
            node = CXXClass(cxx_name, self, ntypemap)
        elif sgroup == "struct":
            node = Struct(cxx_name, self, ntypemap)
        else:
            raise RuntimeError("add_namespaces {}".format(sgroup))
        self.restore_depth(depth)

    def save_depth(self):
        """Save current depth of the stack
        Allow nested scopes to be pushed
        then restore pop stack to using restore_depth.
        """
        return len(self.scope_stack)

    def restore_depth(self, depth):
        """Restore stack to depth"""
        n = len(self.scope_stack)
        for i in range(n - depth):
            self.pop_scope()

    def stash_stack(self):
        """
        Save current state of stack then reset to empty.
        Use restore_stack to restore saved state.
        """
        self.old_scope_stack = self.scope_stack
        self.old_scope_len = self.scope_len
        self.old_scopename = self.scopename
        self.old_current = self.current

        self.current = self.scope_stack[0]
        self.scope_stack = [self.current]
        self.scope_len = [0]
        self.scopename = ""

    def restore_stack(self):
        """Reset to values from last stash_stack"""
        self.scope_stack = self.old_scope_stack
        self.scope_len = self.old_scope_len
        self.scopename = self.old_scopename
        self.current = self.old_current
    
    def create_typedef(self, ast):
        """Add a typedef to the symbol table.

        ast - ast.Declaration
        """
#        if ast.declarator.pointer:
#            # typedef int *foo;
#            name = ast.declarator.name
#            ntypemap = self.lookup_typemap("--typedef--")
#            node = Typedef(name, ntypemap)
#            self.add_child_to_current(node, name)
        if ast.declarator.func:
            # typedef int (*fcn)(int);
            name = ast.declarator.user_name
            type_name = self.scopename + name
            ntypemap = typemap.Typemap(
                type_name,
                base="procedure",
                sgroup="procedure",
#                ast=ast,  # XXX - maybe needed later
            )
            self.register_typemap(ntypemap.name, ntypemap)
            node = Typedef(name, ast, ntypemap)
#            ntypemap.compute_flat_name() GGG
            self.add_child_to_current(node, name)
        else:
            # typedef int TypeID;
            # GGG At this point, just creating an alias for type.
            # typedef void *address;
            name = ast.declarator.name
            type_name = self.scopename + name
            orig = ast.typemap
            ntypemap = orig.clone_as(type_name)
            ntypemap.typedef = orig
            ntypemap.cxx_type = ntypemap.name
            ntypemap.compute_flat_name()
            self.register_typemap(ntypemap.name, ntypemap)
            node = Typedef(name, ast, ntypemap)
            self.add_child_to_current(node, name)
        ast.typemap = ntypemap

    def add_typedef_by_name(self, name):
        """
        Add name into the current scope as a type.
        Used with predefined types like std::string.
        The typemap must already exist.
        """
        tname = self.scopename + name
        ntypemap = self.lookup_typemap(tname)
        if ntypemap is None:
            cursor = error.get_cursor().warning(
                "add_typedef_by_name: Unknown type {}".format(tname))
        else:
            node = Typedef(name, None, ntypemap)
            self.current.add_child(node.name, node)

    def add_typedef(self, name, ntypemap):
        """
        Add typedef from YAML typemap dictionary.
        typemap:
        - type: ns::name
        """
        self.add_namespaces(ntypemap, as_typedef=True)
        self.register_typemap(name, ntypemap)  ### GGG move into Typdef.__init__?

    def create_std_names(self):
        """Add standard types to the Library."""
        self.add_typedef_by_name("size_t")
        self.add_typedef_by_name("int8_t")
        self.add_typedef_by_name("int16_t")
        self.add_typedef_by_name("int32_t")
        self.add_typedef_by_name("int64_t")
        self.add_typedef_by_name("uint8_t")
        self.add_typedef_by_name("uint16_t")
        self.add_typedef_by_name("uint32_t")
        self.add_typedef_by_name("uint64_t")
        self.add_typedef_by_name("MPI_Comm")

    def create_std_namespace(self):
        """Create some types in std."""
        depth = self.create_nested_namespaces(["std"])
        # create_typedef_typemap  - GGG must be in typemap
        self.add_typedef_by_name("string")
        self.add_typedef_by_name("vector")
        self.add_typedef_by_name("shared_ptr")
        self.add_typedef_by_name("weak_ptr")
        self.restore_depth(depth)

    def find_ntemplate_args(self):
        """If currently defining a template, return number of arguments."""
        if isinstance(self.current, Template):
            return self.current.ntemplate_args
        return 0

def symtab_to_dict(node):
    """Return SymbolTable as a dictionary.
    Used for debugging/testing.
    """
    d = dict(cls=node.__class__.__name__)
    if hasattr(node, "symbols"):
        if node.symbols:
            symbols = {}
            for k, v in node.symbols.items():
                symbols[k] = symtab_to_dict(v)
            d['symbols'] = symbols
    if hasattr(node, "typemap"):
        # Global and Namespace do not have typemaps.
        d["typemap"] = node.typemap.name
    return d

def symtab_to_typemap(node):
    """Return SymbolTable as a dictionary.
    Used for debugging/testing.
    """
    if hasattr(node, "typemap"):
        # Global and Namespace do not have typemaps.
        if node.typemap.sgroup in ["shadow", "struct", "template", "enum"]:
            return node.typemap.name
        elif node.typemap.is_enum:
            return node.typemap.name
        else:
            return None
    symbols = {}
    if hasattr(node, "symbols"):
        for k, v in node.symbols.items():
            # If a tag exists, just add name of tag.
            if "enum-" + k in node.symbols:
                symbols[k] = "enum-" + k
            elif "struct-" + k in node.symbols:
                symbols[k] = "struct-" + k
            else:
                out = symtab_to_typemap(v)
                if out is not None:
                    symbols[k] = out
    if not symbols:
        return None
    else:
        return symbols

def check_decl(decl, symtab, trace=False):
    """ parse expr as a declaration, return Node.

    Args:
        decl   - str, string to parse
        symtab - declast.SymbolTable
        trace  - bool
    """
    #trace = True   # GGG For debug
    a = Parser(decl, symtab, trace).decl_statement()
    return a

def check_block(decl, namespace=None, trace=False):
    """ parse expr as a declaration, return list/dict result.

    namespace - An ast.AstNode subclass.
    """
#    trace = True   # GGG For debug
    if namespace is None:
        # grab global namespace if not passed in.
        namespace = global_namespace
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
    arg.declarator.typemap = ntypemap
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
    ast.is_ctor = True
    ast.typemap = cls.typemap
    ast.specifier = [ cls.name ]
    declarator = Declarator()
    ast.declarator = declarator
    declarator.params = []
    declarator.typemap = cls.typemap
    declarator.is_ctor = True
    declarator.attrs["name"] = name
    return ast


def find_arg_by_name(decls, name):
    """Find argument in params with name.
    Return None if not found.
    Does not check name attribute.

    Args:
        decls - list of Declaration
        name  - argument to find
    """
    if decls is None:
        return None
    for decl in decls:
        if decl.declarator.name == name:
            return decl
    return None

def find_arg_index_by_name(decls, name):
    """Return offset of argument in decls with name.
    Does not check name attribute.

    Args:
        decls - list of Declaration
        name  - argument to find
    """
    if decls is None:
        return -1
    for i, decl in enumerate(decls):
        if decl.declarator.name == name:
            return i
    return -1

def check_dimension(dim, trace=False):
    """Return AST of dim.

    Look for assumed-rank, "..", first.
    Else a comma delimited list of expressions.

    Parameters
    ----------
    dim : str
    trace : boolean
    """
    if dim == "..":
        return AssumedRank()
    else:
        return ExprParser(dim, trace=trace).dimension_shape()

def find_rank_of_dimension(dim):
    """Return the rank of a dimension string."""
    if dim is None:
        return None
    elif dim == "..":
        return None
    dim_ast = ExprParser(dim).dimension_shape()
    return len(dim_ast)

def add_funptr_typemap(symtab, declaration, declarator):
    """Create a typemap for a function pointer.
    """
    type_name = str(declaration) + declarator.abstract()
    ntypemap = symtab.lookup_typemap(type_name)
    if not ntypemap:
        ntypemap = typemap.Typemap(
            type_name,
            base="procedure",
            sgroup="procedure",

            f_type="f-function-pointer",
            #                ast=ast,  # XXX - maybe needed later
        )
        symtab.register_typemap(ntypemap.name, ntypemap)
    return ntypemap
