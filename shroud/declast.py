# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
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
        msg = "line {}: ".format(lineno) + format.format(*args)
        ptr = " " * (self.token.column-1) + "^"
        raise RuntimeError("\n".join(["Parse Error:", lines[lineno-1], ptr, msg]))

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
                    if node.typemap.base == "template":
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
                specifier = self.struct_decl(node)
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
        """Parse vector parameters.
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
                self.error_msg("Only single template argument accepted")
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
            node = self.declaration()
        return node

    def decl_statement(self):
        """Check for optional semicolon and stray stuff at the end of line.
        Used when parsing decl from YAML which may not have semicolon.
        """
        node = self.line_statement()
        self.have("SEMICOLON")
        self.mustbe("EOF")
        return node

    def declaration(self):
        """Parse a declaration statement.
        Use with decl_statement and function arguments

        <declaration> ::= {<declaration-specifier>}+ <declarator_item>*
        """
        self.enter("declaration")
        node = Declaration(self.symtab)
        self.declaration_specifier(node)
        self.get_canonical_typemap(node)

        self.declarator_item(node)

        # SSS Share fields between Declaration and Declarator for now
        declarator = node.declarator
        declarator.typemap = node.typemap
        if declarator.func:
            declarator.func.typemap = node.typemap
        
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
        """
        self.enter("declarator_item")
        if node.is_dtor:
            declarator = Declarator()
            declarator.ctor_dtor_name = True
            declarator.attrs["_name"] = "dtor"
            declarator.attrs["_destructor"] = node.is_dtor
        elif node.is_ctor:
            declarator = Declarator()
            declarator.ctor_dtor_name = True
            declarator.attrs["_name"] = "ctor"
            declarator.attrs["_constructor"] = True
        else:
            declarator = self.declarator()
        node.declarator = declarator

        if self.token.typ == "LPAREN":  # peek
            # Function parameters.
            params = self.parameter_list()
            declarator.params = params

            # Look for (void), set to no parameters.
            if len(params) == 1:
                chk = params[0]
                if (chk.declarator.name is None and
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
            declarator.ctor_dtor_name = declarator.attrs["name"] or declarator.attrs["_name"]
            
        self.exit("declarator_item", str(node))
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
#                members.append(self.declaration()) # GGG, accepts too much
                members.append(self.line_statement())
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
# Abstract Syntax Tree Nodes

class Node(object):
    """
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
        
    def create_template_typemaps(self, node, symtab):
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
    If both name and func are None, then this is an abstract
    declarator: ex. 'int *' in 'void foo(int *)'.
    """

    def __init__(self):
        self.pointer = []  # Multiple levels of indirection
        self.name = None  # *name
        self.func = None  # (*name)     Declarator
        
        self.ctor_dtor_name = False

        self.params = None  # None=No parameters, []=empty parameters list
        self.array = []
        self.init = None  # initial value
        self.attrs = collections.defaultdict(lambda: None)
        self.metaattrs = collections.defaultdict(lambda: None)
        self.func_const = False
        self.typemap = None

    def get_user_name(self, use_attr=True):
        """Get name from declarator
        use_attr - True, check attr for name
        ctor and dtor should have _name set
        """
        if use_attr:
            name = self.attrs["name"] or self.attrs["_name"]
            if name is not None:
                return name
        return self.name
    user_name = property(get_user_name, None, None, "Declaration user_name")

    def is_ctor(self):
        """Return True if self is a constructor."""
        return self.attrs["_constructor"]

    def is_dtor(self):
        """Return destructor attribute.
        Will be False for non-destructors, else class name.
        """
        return self.attrs["_destructor"]

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

    def gen_decl_work(self, decl, force_ptr=False, ctor_dtor=False,
                      append_init=True, continuation=False,
                      attrs=True, **kwargs):
        """Generate string by appending text to decl.

        Replace name with value from kwargs.
        name=None will skip appending any existing name.

        attrs=False give compilable code.
        """
        if force_ptr:
            # Force to be a pointer
            decl.append(" *")
        elif kwargs.get("as_scalar", False):
            pass  # Do not print pointer
        else:
            for ptr in self.pointer:
                ptr.gen_decl_work(decl, **kwargs)
        if self.func:
            decl.append(" (")
            self.func.gen_decl_work(decl, attrs=attrs, **kwargs)
            decl.append(")")
        elif "name" in kwargs:
            if kwargs["name"]:
                decl.append(" ")
                decl.append(kwargs["name"])
        elif self.name:
            decl.append(" ")
            decl.append(self.name)
        elif ctor_dtor and self.ctor_dtor_name:
            decl.append(" ")
            decl.append(self.ctor_dtor_name)

        if append_init and self.init is not None:
            decl.append("=")
            decl.append(str(self.init))
        #        if use_attrs:
        #            self.gen_attrs(self.attrs, decl)

        params = kwargs.get("params", self.params)
        if params is not None:
            decl.append("(")
            if continuation:
                decl.append("\t")
            if params:
                comma = ""
                for arg in params:
                    decl.append(comma)
                    arg.gen_decl_work(decl, attrs=attrs, continuation=continuation)
                    if continuation:
                        comma = ",\t "
                    else:
                        comma = ", "
            else:
                decl.append("void")
            decl.append(")")
            if self.func_const:
                decl.append(" const")
        for dim in self.array:
            decl.append("[")
            decl.append(todict.print_node(dim))
            decl.append("]")
        if attrs:
            self.gen_attrs(self.attrs, decl)

    _skip_annotations = ["template"]

    def gen_attrs(self, attrs, decl, skip={}):
        space = " "
        for attr in sorted(attrs):
            if attr[0] == "_":  # internal attribute
                continue
            if attr in self._skip_annotations:
                continue
            if attr in skip:
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

    def __str__(self):
        out = []
        for ptr in self.pointer:
            out.append(str(ptr))
            out.append(" ")

        if self.func:
            out.append("(" + str(self.func) + ")")
        elif self.name:
            out.append(self.name)

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

    def __init__(self, symtab=None):
#        self.symtab = symtab  # GGG -lots of problems with copy
        self.specifier = []  # int, long, ...
        self.storage = []  # static, tyedef, ...
        self.enum_specifier = None   # Enum
        self.class_specifier = None  # CXXClass, Struct (union)
        self.tag_body = False        # if True, members are defined.
        self.const = False
        self.volatile = False
        self.declarator = None
        self.template_arguments = []    # vector<int>, list of Declaration
        self.template_argument = None   # T arg, str
        self.is_ctor = False
        self.is_dtor = False

        self.typemap = None

        self.ftrim_char_in = False # Pass string as TRIM(arg)//C_NULL_CHAR
        self.blanknull = False     # Convert blank CHARACTER to NULL pointer.

    def set_type(self, ntypemap):
        """Set type specifier from a typemap."""
        self.typemap = ntypemap
        # 'long long' into ['long', 'long']
        self.specifier = ntypemap.cxx_type.split()

    def get_full_type(self):
        return ' '.join(self.specifier)

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
        new.typemap = self.typemap
        new.template_arguments = self.template_arguments

        new.declarator = copy.deepcopy(self.declarator)
        new.declarator.name = name
        if not new.declarator.pointer:
            # make sure the return type is a pointer
            new.declarator.pointer = [Ptr("*")]
        # new.array = None
        new.declarator.attrs = copy.deepcopy(self.declarator.attrs) # XXX no need for deepcopy in future
        new.declarator.metaattrs = copy.deepcopy(self.declarator.metaattrs)
        new.declarator.metaattrs["intent"] = "out"
        new.declarator.params= None
        new.declarator.typemap = new.declarator.typemap
        return new

    def set_return_to_void(self):
        """Change function to void"""
        self.specifier = ["void"]
        self.typemap = typemap.void_typemap
        self.const = False
        self.volatile = False
        self.declarator.pointer = []
        self.declarator.typemap = typemap.void_typemap
        self.template_arguments = []

    def result_as_arg(self, name):
        """Pass the function result as an argument.
        Change function result to 'void'.
        """
        newarg = self._as_arg(name)
        self.declarator.params.append(newarg)
        self.set_return_to_void()
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

        var = str(self.declarator)
        if var:
            out.append(" ")
            out.append(var)
        return "".join(out)

    def gen_decl(self, **kwargs):
        """Return a string of the unparsed declaration.

        Args:
            params - None do not print parameters.
        """
        decl = []
        self.gen_decl_work(decl, **kwargs)
        return "".join(decl)

    def gen_decl_work(self, decl, attrs=True, **kwargs):
        """Generate string by appending text to decl.

        Replace params with value from kwargs.
        Most useful to call with params=None to skip parameters
        and only get function result.
        """
        if self.const:
            decl.append("const ")

        if self.is_dtor:
            decl.append("~")
            decl.append(self.is_dtor)
        else:
            if self.storage:
                decl.append(" ".join(self.storage))
                decl.append(" ")
            decl.append(" ".join(self.specifier))
        if self.template_arguments:
            decl.append(self.gen_template_arguments())

        self.declarator.gen_decl_work(decl, attrs=attrs, **kwargs)

    def gen_template_arguments(self):
        """Return string for template_arguments."""
        decl = ["<"]
        for targ in self.template_arguments:
            decl.append(str(targ))
            decl.append(",")
        decl[-1] = ">"
        return ''.join(decl)

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

    def gen_arg_as_language(self, lang, **kwargs):
        """Generate C++ declaration of variable.
        No parameters or attributes.

        Parameters
        ----------
        lang : str
            "c_type" or "cxx_type"
        """
        decl = []
        self.gen_arg_as_lang(decl, lang=lang, **kwargs)
        return "".join(decl)

    def gen_arg_as_lang(
        self,
        decl,
        lang,
        continuation=False,
        asgn_value=False,
        remove_const=False,
        with_template_args=False,
        force_ptr=False,
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
            if self.typemap.sgroup == "vector":
                # Vector types are not explicitly instantiated in the YAML file.
                decl.append(self.typemap.name)
                decl.append(self.gen_template_arguments())
            else:
                # cxx_type includes template  ex. user<int>
                decl.append(self.typemap.cxx_type)
        else:
            # Convert template_argument.
            # ex vector<int> -> int
            if self.template_arguments:
                ntypemap = self.template_arguments[0].typemap
            else:
                ntypemap = self.typemap
            typ = getattr(ntypemap, lang) or "--NOTYPE--"
            decl.append(typ)

        declarator = self.declarator
        if self.is_ctor and lang == "c_type":
            # The C wrapper wants a pointer to the type.
            force_ptr = True

        if asgn_value and const_index is not None and not self.declarator.is_indirect():
            # Remove 'const' so the variable can be assigned to.
            decl[const_index] = ""
        elif remove_const and const_index is not None:
            decl[const_index] = ""

        if lang == "c_type":
            declarator.gen_decl_work(decl, as_c=True, force_ptr=force_ptr,
                                     append_init=False, ctor_dtor=True,
                                     attrs=False, continuation=continuation, **kwargs)
        else:
            declarator.gen_decl_work(decl, force_ptr=force_ptr,
                                     append_init=False, ctor_dtor=True,
                                     attrs=False, continuation=continuation, **kwargs)

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
        
    ##############

    def bind_c(self, intent=None, **kwargs):
        """Generate an argument used with the bind(C) interface from Fortran.

        Args:
            intent - Explicit intent 'in', 'inout', 'out'.
                     Defaults to None to use intent from attrs.

            name   - Set name explicitly, else self.name.
        """
        t = []
        attrs = self.declarator.attrs
        meta = self.declarator.metaattrs
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
        intent = intent or meta["intent"]
        if intent in ["in", "out", "inout"]:
            t.append("intent(%s)" % intent.upper())
        elif intent == "setter":
            # Argument to setter function.
            t.append("intent(IN)")

        decl = []
        decl.append(", ".join(t))
        decl.append(" :: ")

        if kwargs.get("name", None):
            decl.append(kwargs["name"])
        else:
            decl.append(self.declarator.user_name)

        if basedef.base == "vector":
            decl.append("(*)")  # is array
        elif ntypemap.base == "string":
            decl.append("(*)")
        elif attrs["dimension"]:
            # Any dimension is changed to assumed-size.
            decl.append("(*)")
        elif attrs["rank"] is not None and attrs["rank"] > 0:
            # Any dimension is changed to assumed-size.
            decl.append("(*)")
        elif attrs["allocatable"]:
            # allocatable assumes dimension
            decl.append("(*)")
        return "".join(decl)

    def gen_arg_as_fortran(
        self,
        bindc=False,
        local=False,
        pass_obj=False,
        optional=False,
        **kwargs
    ):
        """Geneate declaration for Fortran variable.

        bindc - Use C interoperable type. Used with hidden and implied arguments.
        If local==True, this is a local variable, skip attributes
          OPTIONAL, VALUE, and INTENT
        """
        t = []
        declarator = self.declarator
        attrs = declarator.attrs
        meta = declarator.metaattrs
        ntypemap = self.typemap
        if ntypemap.sgroup == "vector":
            # If std::vector, use its type (<int>)
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
            elif declarator.array:
                t.append("character(kind=C_CHAR)")
            elif not local:
                t.append("character(len=*)")
            else:
                t.append("character")
        elif pass_obj:
            # Used with wrap_struct_as=class for passed-object dummy argument.
            t.append(ntypemap.f_class)
        elif bindc:
            t.append(ntypemap.f_c_type or ntypemap.f_type)
        else:
            t.append(ntypemap.f_type)

        if not local:  # must be dummy argument
            if attrs["value"]:
                t.append("value")
            intent = meta["intent"]
            if intent in ["in", "out", "inout"]:
                t.append("intent(%s)" % intent.upper())
            elif intent == "setter":
                # Argument to setter function.
                t.append("intent(IN)")

        if is_allocatable:
            t.append("allocatable")
        if is_pointer:
            t.append("pointer")
        if optional:
            t.append("optional")

        decl = []
        decl.append(", ".join(t))
        decl.append(" :: ")

        if "name" in kwargs:
            decl.append(kwargs["name"])
        else:
            decl.append(self.declarator.user_name)

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
        elif declarator.array:
            decl.append("(")
            # Convert to column-major order.
            for dim in reversed(declarator.array):
                decl.append(todict.print_node(dim))
                decl.append(",")
            decl[-1] = ")"

        return "".join(decl)


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
    """

    def __init__(self, name, symtab, scope=None):
        self.name = name
        self.scope = scope
        self.members = []

        type_name = symtab.scopename + name
        inttypemap = symtab.lookup_typemap("int")  # XXX - all enums are not ints
        ntypemap = inttypemap.clone_as(type_name)
#        ntypemap = typemap.Typemap( # GGG - do not assume enum is int
#            type_name,
#            base="enum",
#            sgroup="enum",
#        )
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
            )
            symtab.add_tag_to_current("struct", self)
            if symtab.language == "cxx":
                symtab.add_child_to_current(self)
                symtab.register_typemap(type_name, ntypemap)
            self.newtypemap = ntypemap
            self.typemap = ntypemap
        symtab.push_scope(self)


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

        symtab.push_template_scope(self)

    def append_template_param(self, name):
        """append a TemplateParam to this Template.
        """
        node = TemplateParam(name)
        node.typemap = self.paramtypemap
        self.parameters.append(node)
        self.symbols[name] = node

    def add_child(self, name, node):
        """
        Add the templated function into the parent,
        not the Template scope.
          template<U> class name
        """
        self.parent.add_child(name, node)
            

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
    """
    def __init__(self, language="cxx"):
        self.scope_stack = []
        self.scope_len   = []
        self.scopename = ''
        self.typemaps = typemap.default_typemap()
        self.language = language

        # Create the global scope.
        glb = Global()
        self.scope_stack.append(glb)
        self.scope_len.append(0)
        self.current = glb

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
                base="fcnptr",
                sgroup="fcnptr",
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
            ntypemap.typedef = orig.name
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
            raise RuntimeError("Unknown type {}".format(tname))
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
        self.restore_depth(depth)

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
        elif hasattr(node.typemap, "is_enum"):
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
    """ parse expr as a declaration, return list/dict result.

    namespace - An ast.AstNode subclass.
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
    declarator.attrs["_constructor"] = True
    declarator.attrs["name"] = name
##        _name="ctor",
    declarator.metaattrs["intent"] = "ctor"
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
    """Return index of argument in decls with name.
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
