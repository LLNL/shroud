# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-738041.
# All rights reserved.
#  
# This file is part of Shroud.  For details, see
# https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
#  
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the disclaimer (as noted below)
#   in the documentation and/or other materials provided with the
#   distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
# LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
########################################################################
"""
A top-down, recursive descent parser for C/C++ expressions with
additions for shroud attributes.

"""

from __future__ import print_function
import collections
import copy
import re

from . import typemap

Token = collections.namedtuple('Token', ['typ', 'value', 'line', 'column'])

# https://docs.python.org/3.2/library/re.html#writing-a-tokenizer
type_specifier = { 'void', 'bool', 'char', 'short', 'int', 'long', 'float', 'double',
                   'signed', 'unsigned',
                   'MPI_Comm',
                   'string', 'vector'}
type_qualifier = { 'const', 'volatile' }
storage_class = { 'auto', 'register', 'static', 'extern', 'typedef' }
namespace = { 'std' }

token_specification = [
    ('REAL',      r'((((\d+[.]\d*)|(\d*[.]\d+))([Ee][+-]?\d+)?)|(\d+[Ee][+-]?\d+))'),
    ('INTEGER',   r'\d+'),
    ('DQUOTE',    r'["][^"]*["]'),  # double quoted string
    ('SQUOTE',    r"['][^']*[']"),  # single quoted string
    ('LPAREN',    r'\('),
    ('RPAREN',    r'\)'),
    ('STAR',      r'\*'),
    ('EQUALS',    r'='),
    ('REF',       r'\&'),
    ('PLUS',      r'\+'),
    ('MINUS',     r'\-'),
    ('SLASH',     r'/'),
    ('COMMA',     r','),
    ('SEMICOLON', r';'),
    ('LT',        r'<'),
    ('GT',        r'>'),
    ('TILDE',     r'\~'),
    ('NAMESPACE', r'::'),
    ('VARARG',    r'\.\.\.'),
    ('ID',        r'[A-Za-z_][A-Za-z0-9_]*'),   # Identifiers
    ('NEWLINE',   r'[\n]'),        # Line endings
    ('SKIP',      r'[ \t]'),       # Skip over spaces and tabs
    ('OTHER',     r'.'),
    ]
tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
get_token = re.compile(tok_regex).match

def tokenize(s):
    line = 1
    pos = line_start = 0
    mo = get_token(s)
    while mo is not None:
        typ = mo.lastgroup
        if typ == 'NEWLINE':
            line_start = pos
            line += 1
        elif typ != 'SKIP':
            val = mo.group(typ)
            if typ == 'ID':
                if val in type_specifier:
                    typ = 'TYPE_SPECIFIER'
                elif val in type_qualifier:
                    typ = 'TYPE_QUALIFIER'
                elif val in storage_class:
                    typ = 'STORAGE_CLASS'
            yield Token(typ, val, line, mo.start()-line_start)
        pos = mo.end()
        mo = get_token(s, pos)
    if pos != len(s):
        raise RuntimeError('Unexpected character %r on line %d' %(s[pos], line))


def add_type(name):
    """Add a user type (typedef, class) to the parser.
    """
    type_specifier.add(name)

def add_typemap():
    """Add all types from the typemap to the parser.
    """
    for name in typemap.Typedef._typedict.keys():
        type_specifier.add(name)

#indent = 0
#def trace(name, indent=0):
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
            self.token = Token('EOF', None, 0, 0)
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
        ptr = ' '*self.token.column + '^'
        raise RuntimeError('\n'.join(["Parse Error", self.decl, ptr, msg]))

    def enter(self, name, *args):
        """Print message when entering a function."""
        if self.trace:
            print(' ' * self.indent, 'enter', name, *args)
            self.indent += 4

    def exit(self, name, *args):
        """Print message when exiting a function."""
        if self.trace:
            self.indent -= 4
            print(' ' * self.indent, 'exit', name, *args)

    def info(self, *args):
        """Print debug message during parse."""
        if self.trace:
            print(' ' * self.indent, *args)


class Parser(RecursiveDescent):
    """
    Parse a C/C++ declaration with Shroud annotations.

    An abstract-declarator is a declarator without an identifier,
    consisting of one or more pointer, array, or function modifiers.
    """
    def __init__(self, decl, current_class=None, trace=False):
        self.decl = decl          # declaration to parse
        self.current_class = current_class
        self.trace = trace
        self.indent = 0
        self.token = None
        self.tokenizer = tokenize(decl)
        self.next()  # load first token

    def parameter_list(self):
        # look for ... var arg at end
        """
        <parameter-list> ::= '(' <declaration>?  [ , <declaration ]* ')'

        """
        self.enter('parameter_list')
        params = []
        self.next()  # consume LPAREN peeked at in caller
        while self.token.typ != 'RPAREN':
            node = self.declaration()
            params.append(node)
            if self.have('COMMA'):
                if self.have('VARARG'):
                    raise NotImplementedError("varargs")
            else:
                break
        self.mustbe('RPAREN')
        self.exit('parameter_list', str(params))
        return params

    def nested_namespace(self):
        """Found start of namespace.

        <nested-namespace> ::= { namespace :: }* identifier
        """
        self.enter('nested_namespace')
        nested = [ self.token.value ]
        self.next()
        while self.have('NAMESPACE'):
            if self.token.typ == 'ID':
                nested.append(self.token.value)
                self.next()
            elif self.token.typ == 'TYPE_SPECIFIER':
                # This is a bit of a kludge to support both
                # std::string and string
                # As if "using std::string"
                nested.append(self.token.value)
                self.next()
            else:
                raise self.error_msg("Error in namespace")
        qualified_id = '::'.join(nested)
        self.exit('nested_namespace', qualified_id)
        return qualified_id

    def declaration_specifier(self, node):
        """
        Set attributes on node corresponding to next token
        node - Declaration node.
        <declaration-specifier> ::= <storage-class-specifier>
                                  | <type-specifier>
                                  | <type-qualifier>
                                  | { nested-namespace } [ < { nested-namespace } > }?
                                  | <current_class>       # constructor
                                  | ~ <current_class>

        Returns True if need declarator after, else False (i.e. ctor or dtor)
        """
        self.enter('declaration_specifier')
        need = True
        more = True
        if self.have('TILDE'):
            specifier = self.mustbe('TYPE_SPECIFIER').value
            if specifier == self.current_class:
                self.info('destructor')
                node.specifier.append(specifier)
                node.fattrs['_name'] = 'dtor'
                node.fattrs['_destructor'] = True
                more = False
                need = False
            else:
                raise RuntimeError("Expected destructor")
        elif self.token.typ == 'TYPE_SPECIFIER' and self.token.value == self.current_class:
            # class constructor
            self.info('type-specifier0:', self.token.value)
            node.specifier.append(self.token.value)
            self.next()
            if self.token.typ == 'LPAREN':
                self.info('constructor')
                node.fattrs['_name'] = 'ctor'
                node.fattrs['_constructor'] = True
                more = False
                need = False

        while more:
            # if self.token.type = 'ID' and  typedef-name
            if self.token.typ == 'ID' and self.token.value in namespace:
                node.specifier.append(self.nested_namespace())
                if self.have('LT'):
                    node.attrs['template'] = self.nested_namespace()
                    self.mustbe('GT')
            elif self.token.typ == 'TYPE_SPECIFIER':
                node.specifier.append(self.token.value)
                self.info('type-specifier:', self.token.value)
                self.next()
            elif self.token.typ == 'TYPE_QUALIFIER':
                # const volatile
                setattr(node, self.token.value, True)
                self.info('type-qualifier:', self.token.value)
                self.next()
            elif self.token.typ == 'STORAGE_CLASS':
                node.storage.append(self.token.value)
                self.info('storage-class-specifier:', self.token.value)
                self.next()
            else:
                more= False
        self.exit('declaration_specifier', need)
        return need

    def decl_statement(self):
        """Check for optional semicolon and stray stuff at the end of line.
        """
        node = self.declaration()
        self.have('SEMICOLON')
        self.mustbe('EOF')
        return node

    def declaration(self):
        """Parse a declaration statement.
        Use with decl_statement and function arguments

        <declaration> ::= {<declaration-specifier>}+ <declarator>?
                           ( '['  <constant-expression>?  ']'  |
                             '('  <parameter-list>            ')' [ const ] ) 
                           [ = <initializer> ]
        """
        self.enter('declaration')
        node = Declaration()
        if self.declaration_specifier(node):
            node.declarator = self.declarator()
        self.attribute(node.attrs)  # this is ambiguous   'void foo+attr(int arg1)'

        if self.token.typ == 'LPAREN':     # peek
            node.params = self.parameter_list()

            #  method const
            if self.token.typ == 'TYPE_QUALIFIER':
                if self.token.value == 'const':
                    self.next()
                    node.func_const = True
                else:
                    raise RuntimeError(
                        "'{}' unexpected after function declaration"
                        .format(self.token.value))
#        elif self.token.typ == 'LBRACKET':
#            node.array  = self.constant_expression()
        self.attribute(node.fattrs)

        if self.have('EQUALS'):
            node.init = self.initializer()
        self.exit('declaration', str(node))
        return node

    def declarator(self):
        """
        <declarator> ::=  <pointer>* [ ID ]
                       |  '(' <declarator> ')'
        """
        self.enter('declarator')
        node = Declarator()
        node.pointer = self.pointer()

        if self.token.typ == 'ID':         # variable identifier
            node.name = self.token.value
            self.info("declarator ID:", self.token.value)
            self.next()
        elif self.token.typ == 'LPAREN':   # (*var)
            self.next()
            node.func = self.declarator()
            self.mustbe('RPAREN')

        self.exit('declarator', str(node))
        return node

    def pointer(self):
        """
        <pointer> ::= * {<type-qualifier>}* {<pointer>}?

        Allow for multiple layers of indirection.
        This will also accept illegal input like '*&'
        """
        self.enter('pointer')
        ptrs = []
        while self.token.typ in ['STAR', 'REF']:
            node = Ptr(self.token.value)
            ptrs.append(node)
            self.info("pointer:", self.token.value)
            self.next()
            while self.token.typ == 'TYPE_QUALIFIER':  # const, volatile
                setattr(node, self.token.value, True)
                self.info("type-qualifier:", self.token.value)
                self.next()
        self.exit('pointer', str(ptrs))
        return ptrs

    def initializer(self):
        """
        TODO: This should support expressions
        """
        self.enter('initializer')
        value = self.token.value
        if self.have('REAL'):
            value = float(value)
        elif self.have('INTEGER'):
            value = int(value)
        elif self.have('DQUOTE'):
            value = value
        elif self.have('SQUOTE'):
            value = value
        elif self.have('ID'):
            pass
        else:
            value = None
        self.exit('initializer')
        return value

    def attribute(self, attrs):
        """Collect attributes of the form:
           +name
           +name(expression)
           +name=scalar
        """
        self.enter('attribute')
        while self.have('PLUS'):
            name = self.mustbe('ID').value
            if self.have('LPAREN'):
                parens = 1
                parts = []
                # collect tokens until found balanced paren
                while True:
                    if self.token.typ == 'LPAREN':
                        parens += 1
                    elif self.token.typ == 'RPAREN':
                        parens -= 1
                    elif self.token.typ == 'EOF':
                        raise RuntimeError(
                            "Unbalanced parens in attribute {}".format(name))
                    if parens == 0:
                        self.next()
                        break
                    parts.append(self.token.value)
                    self.next()
                attrs[name] = ''.join(parts)
            elif self.have('EQUALS'):
                attrs[name] = self.initializer()
            else:
                attrs[name] = True
        self.exit('attribute', attrs)


class Node(object):
    pass


class Ptr(Node):
    """ A pointer or reference. """
    def __init__(self, ptr=''):
        self.ptr   = ptr     # * or &
        self.const = False
        self.volatile = False

    def gen_decl_work(self, decl, **kwargs):
        """Generate string by appending text to decl.
        """
        if self.ptr:
            decl.append(' ')
            if kwargs.get('as_c', False):
                # references become pointers with as_c
                decl.append('*')
            else:
                decl.append(self.ptr)
        if self.const:
            decl.append(' const')
        if self.volatile:
            decl.append(' volatile')

    def __str__(self):
        if self.const:
            return self.ptr + ' const'
        else:
            return self.ptr


class Declarator(Node):
    """
    If both name and func are are None, then this is an abstract
    declarator: ex. 'int *' in 'void foo(int *)'.
    """
    def __init__(self):
        self.pointer = []     # Multiple levels of indirection
        self.name    = None   #  *name
        self.func    = None   # (*name)     declarator

    def gen_decl_work(self, decl, **kwargs):
        """Generate string by appending text to decl.

        Replace name with value from kwargs.
        name=None will skip appending any existing name.
        """
        for ptr in self.pointer:
            ptr.gen_decl_work(decl, **kwargs)
        if self.func:
            decl.append(' (')
            self.func.gen_decl_work(decl, **kwargs)
            decl.append(')')
        elif 'name' in kwargs:
            if kwargs['name']:
                decl.append(' ')
                decl.append(kwargs['name'])
        elif self.name:
            decl.append(' ')
            decl.append(self.name)

    def __str__(self):
        out = ''
        for ptr in self.pointer:
            out += str(ptr)
            out += ' '

        if self.name:
            out += self.name
        elif self.func:
            out += '(' + str(self.func) + ')'

        return out


class Declaration(Node):
    """
    specifier = const  int
    init =         a  *a   a=1
    """
    def __init__(self):
        self.specifier  = []    # int, long, ...
        self.storage    = []    # static, ...
        self.const      = False
        self.volatile   = False
        self.declarator = None
        self.params     = None   # None=No parameters, []=empty parameters list
        self.array      = None
        self.init       = None   # initial value
        self.attrs      = {}     # declarator attributes

        self.func_const = False
        self.fattrs     = {}     # function attributes

    def get_name(self):
        """Get name from declarator
        ctor and dtor should have _name set
        """
        name = self.fattrs.get('name', None) or self.fattrs.get('_name', None)
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

    def get_type(self):
        """Return type.
        Mulitple specifies are joined by an underscore. i.e. long_long
        """
        if self.specifier:
            typ = '_'.join(self.specifier)
        else:
            typ = 'int'
        return typ

    def set_type(self, typ):
        self.specifier = typ.split()

    typename = property(get_type, set_type, None, "Declaration type")

    def is_pointer(self):
        """Return number of levels of pointers.
        """
        nlevels = 0
        if self.declarator is None:
            return nlevels
        for ptr in self.declarator.pointer:
            if ptr.ptr == '*':
                nlevels += 1
        return nlevels

    def is_reference(self):
        """Return number of levels of references.
        """
        nlevels = 0
        if self.declarator is None:
            return nlevels
        for ptr in self.declarator.pointer:
            if ptr.ptr == '&':
                nlevels += 1
        return nlevels

    def is_indirect(self):
        """Return number of indirections, pointer or reference.
        """
        nlevels = 0
        if self.declarator is None:
            return nlevels
        for ptr in self.declarator.pointer:
            if ptr.ptr:
                nlevels += 1
        return nlevels

    def is_function_pointer(self):
        """Return number of levels of pointers.
        """
        nlevels = 0
        if self.declarator is None:
            return False
        if self.declarator.func is None:
            return False
        if not self.declarator.func.pointer:
            return False
        return True

    def get_subprogram(self):
        """Return Fortran subprogram - subroutine or function"""
        if self.typename != 'void':
            return 'function'
        if self.is_pointer():
            return 'function'
        return 'subroutine'

    def _as_arg(self, name):
        """Create an argument to hold the function result.
        This is intended for pointer arguments, char or string.
        """
        new = Declaration()
        new.specifier  = self.specifier[:]
        new.storage    = self.storage[:]
        new.const      = False
        new.volatile   = False
        new.declarator = copy.deepcopy(self.declarator)
        new.declarator.name = name
        if not new.declarator.pointer:
            # make sure the return type is a pointer
            new.declarator.pointer = [ Ptr('*') ]
#        new.array      = None
        new.attrs      = copy.deepcopy(self.attrs)
        return new

    def _set_to_void(self):
        """Change function to void"""
        self.specifier = ['void']
        self.const = False
        self.declarator.pointer = []

    def result_as_arg(self, name):
        """Pass the function result as an argument.
        """
        newarg = self._as_arg(name)
        self.params.append(newarg)
        self._set_to_void()
        return newarg

    def __str__(self):
        out = []
        if self.const:
            out.append('const ')
        if self.volatile:
            out.append('volatile ')
        if '_destructor' in self.fattrs:
            out.append('~')
        if self.specifier:
            out.append(' '.join(self.specifier))
        else:
            out.append('int')
        out.append(' ')
        if self.declarator:
            out.append(str(self.declarator))
        if self.params is not None:
            out.append('(')
            if self.params:
                out.append(str(self.params[0]))
                for param in self.params[1:]:
                    out.append(',')
                    out.append(str(param))
            out.append(')')
            if self.func_const:
                out.append(' const')
        elif self.array:
            out.append('[AAAA]')
        if self.init:
            out.append('=')
            out.append(str(self.init))
        return ''.join(out)

    def gen_decl(self, **kwargs):
        """Return a string of the unparsed declaration.
        """
        decl = []
        self.gen_decl_work(decl, **kwargs)
        return ''.join(decl)

    def gen_decl_work(self, decl, **kwargs):
        """Generate string by appending text to decl.

        Replace params with value from kwargs.
        Most useful to call with params=None to skip parameters
        and only get function result.

        attrs=False give compilable code.
        """
        use_attrs = kwargs.get('attrs', True)
        if self.const:
            decl.append('const ')

        if '_destructor' in self.fattrs:
            decl.append('~')
        decl.append(' '.join(self.specifier))
        if 'template' in self.attrs:
            decl.append('<')
            decl.append(self.attrs['template'])
            decl.append('>')

        if self.declarator:
            self.declarator.gen_decl_work(decl, **kwargs)

        if self.init is not None:
            decl.append('=')
            decl.append(str(self.init))
        if use_attrs:
            self.gen_attrs(self.attrs, decl)

        params = kwargs.get('params', self.params)
        if params is not None:
            decl.append('(')
            comma = ''
            for arg in params:
                decl.append(comma)
                arg.gen_decl_work(decl)
                comma = ', ' 
            decl.append(')')
            if self.func_const:
                decl.append(' const')
            if use_attrs:
                self.gen_attrs(self.fattrs, decl)

    _skip_annotations = ['template']

    def gen_attrs(self, attrs, decl):
        space = ' '
        for attr in sorted(attrs):
            if attr[0] == '_':  # internal attribute
                continue
            if attr in self._skip_annotations:
                continue
            value = attrs[attr]
            if value is False:
                continue
            decl.append(space)
            decl.append('+')
            if value is True:
                decl.append(attr)
            else:
                decl.append('{}({})'.format(attr, value))
            space = ''

    def gen_arg_as_cxx(self, **kwargs):
        """Generate C++ declaration of variable.
        No parameters or attributes.
        """
        decl = []
        self.gen_arg_as_lang(decl, lang='cxx_type', **kwargs)
        return ''.join(decl)

    def gen_arg_as_c(self, **kwargs):
        """Return a string of the unparsed declaration.
        """
        decl = []
        self.gen_arg_as_lang(decl, lang='c_type', **kwargs)
        return ''.join(decl)

    def gen_arg_as_lang(self, decl, lang,
                        continuation=False,
                        asgn_value=False,
                        **kwargs):
        """Generate an argument for the C wrapper.
        C++ types are converted to C types using typemap.

        lang = c_type or cxx_type
        continuation = True - insert tabs to aid continuations
        asgn_value = If True, make sure the value can be assigned
                     by removing const.

        If a templated type, assume std::vector.
        The C argument will be a pointer to the template type.
        'std::vector<int> &'  generates 'int *'
        The length info is lost but will be provided as another argument
        to the C wrapper.
        """
        const_index = None
        if self.const:
            const_index = len(decl)
            decl.append('const ')

        if 'template' in self.attrs:
            typedef = typemap.Typedef.lookup(self.attrs['template'])
        else:
            typename = self.typename
            typedef = typemap.Typedef.lookup(typename)
        if typedef is None:
            raise RuntimeError("No such type: {}".format(typename))

        typ = getattr(typedef, lang)
        decl.append(typ)

        if self.declarator is None:
            # The C wrapper wants a pointer to the type.
            declarator = Declarator()
            declarator.pointer = [ Ptr('*') ]
            declarator.name = self.name
        else:
            declarator = self.declarator

        if asgn_value and const_index is not None and not self.is_indirect():
            # Remove 'const' do the variable can be assigned to.
            decl[const_index] = ''

        if lang == 'c_type':
            declarator.gen_decl_work(decl, as_c=True, **kwargs)
        else:
            declarator.gen_decl_work(decl, **kwargs)

        params = kwargs.get('params', self.params)
        if params is not None:
            decl.append('(')
            if continuation:
                decl.append('\t')
            comma = ''
            for arg in params:
                decl.append(comma)
                arg.gen_decl_work(decl, attrs=None,
                                  continuation=continuation)
                if continuation:
                    comma = ',\t '
                else:
                    comma = ', '
            decl.append(')')
            if self.func_const:
                decl.append(' const')

##############

    def bind_c(self, **kwargs):
        """Generate an argument used with the bind(C) interface from Fortran.
        """
        t = []
        typedef = typemap.Typedef.lookup(self.typename)
        basedef = typedef
        attrs = self.attrs
        if 'template' in attrs:
            # If a template, use its type
            typedef = typemap.Typedef.lookup(attrs['template'])

        typ = typedef.f_c_type or typedef.f_type
        if typ is None:
            raise RuntimeError("Type {} has no value for f_c_type".format(self.typename))
        t.append(typ)
        if attrs.get('value', False):
            t.append('value')
        intent = attrs.get('intent', None)
        if intent:
            t.append('intent(%s)' % intent.upper())

        decl = []
        decl.append(', '.join(t))
        decl.append(' :: ')

        if 'name' in kwargs:
            decl.append(kwargs['name'])
        else:
            decl.append(self.name)

        if basedef.base == 'vector':
            decl.append('(*)') # is array
        elif typedef.base == 'string':
            decl.append('(*)')
        else:
            dimension = attrs.get('dimension', '')
            if dimension:
                # Any dimension is changed to assumed length.
                decl.append('(*)')
        return ''.join(decl)

    def gen_arg_as_fortran(self, local=False, **kwargs):
        """Geneate declaration for Fortran variable.

        If local==True, this is a local variable, skip attributes
          OPTIONAL, VALUE, and INTENT
        """
        t = []
        typedef = typemap.Typedef.lookup(self.typename)
        attrs = self.attrs
        if 'template' in attrs:
            # If a template, use its type
            typedef = typemap.Typedef.lookup(attrs['template'])

        typ = typedef.f_type
        t.append(typ)
        if not local:  # must be dummy argument
            if attrs.get('value', False):
                t.append('value')
            intent = attrs.get('intent', None)
            if intent:
                t.append('intent(%s)' % intent.upper())

        decl = []
        decl.append(', '.join(t))
        decl.append(' :: ')

        if 'name' in kwargs:
            decl.append(kwargs['name'])
        else:
            decl.append(self.name)

        dimension = attrs.get('dimension', '')
        if dimension:
            decl.append('(' + dimension + ')')

        return ''.join(decl)


def check_decl(decl, current_class=None, template_types=[],trace=False):
    """ parse expr as a declaration, return list/dict result.
    """
    if template_types or current_class:
        global type_specifier
        old_types = type_specifier
        type_specifier = set(old_types)
        type_specifier.update(template_types)
        if current_class:
            type_specifier.add(current_class)
        a = Parser(decl,current_class=current_class,trace=trace).decl_statement()
        type_specifier = old_types
    else:
        a = Parser(decl,current_class=current_class,trace=trace).decl_statement()
    return a


def create_this_arg(name, typ, const=True):
    """Create a Declaration for an argument for the 'this' argument.
    """
    arg = Declaration()
    arg.const = const
    arg.declarator = Declarator()
    arg.declarator.name = name
    arg.declarator.pointer = [ Ptr('*') ]
    arg.specifier = [ typ ]
    return arg


######################################################################

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


# For each operator, a (precedence, associativity) pair.
OpInfo = collections.namedtuple('OpInfo', 'prec assoc')

OPINFO_MAP = {
    '+':    OpInfo(1, 'LEFT'),
    '-':    OpInfo(1, 'LEFT'),
    '*':    OpInfo(2, 'LEFT'),
    '/':    OpInfo(2, 'LEFT'),
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
        self.enter('expression')
        atom_lhs = self.primary()

        while True:
            op = self.token.value
            if (op not in OPINFO_MAP
                or OPINFO_MAP[op].prec < min_prec):
                break

            # Inside this loop the current token is a binary operator

            # Get the operator's precedence and associativity, and compute a
            # minimal precedence for the recursive call
            prec, assoc = OPINFO_MAP[op]
            next_min_prec = prec + 1 if assoc == 'LEFT' else prec

            # Consume the current token and prepare the next one for the
            # recursive call
            self.next()
            atom_rhs = self.expression(next_min_prec)

            # Update lhs with the new value
            atom_lhs = BinaryOp(atom_lhs, op, atom_rhs)

        self.exit('expression')
        return atom_lhs

    def primary(self):
        self.enter('primary')
        if self.peek('ID'):
            node = self.identifier()
        elif self.token.typ in ['REAL', 'INTEGER']:
            self.enter('constant')
            node = Constant(self.token.value)
            self.next()
        elif self.have('LPAREN'):
            node = ParenExpr(self.expression())
            self.mustbe('RPAREN')
        elif self.token.typ in ['PLUS', 'MINUS']:
            self.enter('unary')
            value = self.token.value
            self.next()
            node = UnaryOp(value, self.primary())
        else:
            self.error_msg("Unexpected token {}", self.token.value)
        self.exit('primary')
        return node

    def identifier(self):
        """
        <expr> ::= name '(' arglist ')'
        """
        self.enter('identifier')
        name = self.mustbe('ID').value
        if self.peek('LPAREN'):
            args = self.argument_list()
            node = Identifier(name, args)
        else:
            node = Identifier(name)
        self.exit('identifier')
        return node

    def argument_list(self):
        """
        <argument-list> ::= '(' <name>?  [ , <name ]* ')'

        """
        self.enter('argument_list')
        params = []
        self.next()  # consume LPAREN peeked at in caller
        while self.token.typ != 'RPAREN':
            node = self.identifier()
            params.append(node)
            if not self.have('COMMA'):
                break
        self.mustbe('RPAREN')
        self.exit('argument_list', str(params))
        return params


def check_expr(expr, trace=False):
    a = ExprParser(expr, trace=trace).expression()
    return a
