#!/bin/env python3
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
    ('REAL',      r'[+-]?((((\d+[.]\d*)|(\d*[.]\d+))([Ee][+-]?\d+)?)|(\d+[Ee][+-]?\d+))'),
    ('INTEGER',   r'[+-]?\d+'),
    ('DQUOTE',    r'["][^"]*["]'),  # double quoted string
    ('SQUOTE',    r"['][^']*[']"),  # single quoted string
    ('LPAREN',    r'\('),
    ('RPAREN',    r'\)'),
    ('STAR',      r'\*'),
    ('EQUALS',    r'='),
    ('REF',       r'\&'),
    ('PLUS',      r'\+'),
    ('COMMA',     r','),
    ('SEMICOLON', r';'),
    ('LT',        r'<'),
    ('GT',        r'>'),
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
    """Add a use type (typedef, class) to the parser.
    """
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

class Parser(object):
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
        self.next()

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

        Returns a list of specifiers
        """
        self.enter('declaration_specifier')
        while True:
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
                break
        self.exit('declaration_specifier')

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
        self.declaration_specifier(node)
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
        elif self.token.typ == 'TYPE_SPECIFIER' and \
             self.token.value == self.current_class:
            # class constructor
            node.name = self.token.value
            self.info("declarator ID(class):", self.token.value)
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


class Node(object):
    pass


class Ptr(Node):
    """ A pointer or reference. """
    def __init__(self, ptr=''):
        self.ptr   = ptr     # * or &
        self.const = False
        self.volatile = False

    def to_dict(self, attrs):
        if self.ptr == '&':
            attrs['reference'] = True
        elif self.ptr == '*':
            attrs['ptr'] = True
# old implementation did not support const pointers
#        if self.const:
#            attrs['const'] = True

    def _to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        d = dict(
            ptr = self.ptr,
            const = self.const,
#            volatile = self.volatile,
        )
        return d

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

    def to_dict(self, d):
        if self.name:
            d['name'] = self.name
        if self.pointer:
            self.pointer[0].to_dict(d['attrs'])

    def _to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        d = dict(
            pointer = [p._to_dict() for p in self.pointer],
        )
        if self.name:
            d['name'] = self.name
        elif self.func:
            d['func'] = self.func._to_dict()
        return d

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
        self.func_const = False
        self.array      = None
        self.init       = None   # initial value
        self.attrs      = {}     # declarator attributes
        self.fattrs     = {}     # function attributes

    def get_name(self):
        """Extract name from declarator."""
        name = self.declarator.name
        if name is None:
            if self.declarator.func:
                name = self.declarator.func.name
        return name

    def get_type(self):
        """Extract type.
        TODO: deal with 'long long', 'unsigned int'
        """
        if self.specifier:
            if len(self.specifier) > 1:
                raise RuntimeError("too many type specifiers '{}'"
                                   .format(' '.join(self.specifier)))
            typ = self.specifier[0]
        else:
            typ = 'int'
        return typ

    def set_type(self, typ):
        self.specifier[0] = typ

    def is_pointer(self):
        """Return number of levels of pointers.
        """
        nlevels = 0
        for ptr in self.declarator.pointer:
            if ptr.ptr == '*':
                nlevels += 1
        return nlevels

    def is_reference(self):
        """Return number of levels of references.
        """
        nlevels = 0
        for ptr in self.declarator.pointer:
            if ptr.ptr == '&':
                nlevels += 1
        return nlevels

    def is_indirect(self):
        """Return number of indirections, pointer or reference.
        """
        nlevels = 0
        for ptr in self.declarator.pointer:
            if ptr.ptr:
                nlevels += 1
        return nlevels

    def _as_arg(self, name):
        """Create an argument to hold the result.
        """
        new = Declaration()
        new.specifier  = self.specifier[:]
        new.storage    = self.storage[:]
        new.const      = False
        new.volatile   = False
        new.declarator = copy.deepcopy(self.declarator)
        new.declarator.name = name
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

    def to_dict(self, d=None):
        """
        Create a dictionary to match the one created by parse_decl.py
        """
        attrs = {}
        if d is None:
            d = {}
            top = dict(
                result = d,
                args = [],
                func_const = self.func_const,
                attrs = {}
            )
            top['attrs'].update(self.fattrs)
        else:
            top = None
        if self.specifier:
            d['type'] = self.specifier[0]
        else:
            d['type'] = 'int'
        d['attrs'] = attrs
        attrs.update(self.attrs)
        d['const'] = self.const
        if self.init is not None:
            attrs['default'] = self.init
        self.declarator.to_dict(d)
        if self.params is not None:
            for param in self.params:
                arg = {}
                top['args'].append(arg)
                param.to_dict(arg)
        return top

    def _to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        d = dict(
            specifier = self.specifier,
            storage = self.storage,
            const = self.const,
#            volatile = self.volatile,
            declarator = self.declarator._to_dict(),
            func_const = self.func_const,
#            self.array,
            attrs = self.attrs,
        )
        if self.params is not None:
            d['args'] = [ x._to_dict() for x in self.params]
            d['fattrs'] = self.fattrs
        else:
            if self.fattrs:
                raise RuntimeError("fattrs is not empty for non-function")
        if self.init is not None:
            d['init'] = self.init
        return d

    def __str__(self):
        out = ' '.join(self.specifier)
        if self.const:
            out += 'const '
        if self.volatile:
            out += 'volatile'
        if self.specifier:
            out += ' '.join(self.specifier)
        else:
            out += 'int'
        out += ' '
        if self.declarator:
            out += str(self.declarator)
        if self.params is not None:
            out += '('
            if self.params:
                out += str(self.params[0])
                for param in self.params[1:]:
                    out += ','
                    out += str(param)
            out += ')'
            if self.func_const:
                out += ' const'
        elif self.array:
            out += '[AAAA]'
        if self.init:
            out += '=' + str(self.init)
        return out

    def gen_arg_decl(self, decl):
        """ Generate declaration for a single Declaration node.
        decl - array of strings
        """
        if self.const:
            decl.append('const ')
        decl.append(self.get_type())
        if 'template' in self.attrs:
            decl.append('<{}>'.format(self.attrs['template']))
        decl.append(' ')
        for ptr in self.declarator.pointer:
            if ptr.ptr:
                decl.append(ptr.ptr)
        # XXX - deal with function pointers
        decl.append(self.get_name())
        self.gen_attrs(self.attrs, decl)
        if self.init is not None:
            decl.append('=')
            decl.append(str(self.init))

    def gen_decl(self):
        """Generate declaration.
        """
        decl = []
        self.gen_arg_decl(decl)

        if self.params is not None:
            decl.append('(')
            comma = ''
            for arg in self.params:
                decl.append(comma)
                arg.gen_arg_decl(decl)
                comma = ', ' 
            decl.append(')')

        if self.func_const:
            decl.append(' const')
        self.gen_attrs(self.fattrs, decl)

        return ''.join(decl)

    def gen_attrs(self, attrs, decl):
        for attr in sorted(attrs):
            if attr == 'template':
                continue
            value = attrs[attr]
            if value is True:
                decl.append('+{}'.format(attr))
            else:
                decl.append('+{}({})'.format(attr, value))


def check_decl(decl, current_class=None, template_types=[]):
    """ parse expr as a declaration, return list/dict result.
    """
    if template_types or current_class:
        global type_specifier
        old_types = type_specifier
        type_specifier = set(old_types)
        type_specifier.update(template_types)
        if current_class:
            type_specifier.add(current_class)
        a = Parser(decl,current_class=current_class,trace=False).decl_statement()
        type_specifier = old_types
    else:
        a = Parser(decl,current_class=current_class,trace=False).decl_statement()
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
    
##################################################
def is_pointer(decl):
    """Old dictionary based."""
    return decl['attrs'].get('ptr', False)

def is_reference(decl):
    """Old dictionary based."""
    return decl['attrs'].get('reference', False)

##################################################

def str_declarator(decl):
    """ Convert declaration dict to string.
    Used with output from check_decl.
    Helpful in error messages.
      a = check_decl()
      str_declarator( a['result'] )
      str_declarator( a['args'][0] )
    """
    attrs = decl['attrs']
    out = ''
    if 'const' in attrs:
        out += 'const '
    out += decl['type']
    if 'template' in attrs:
        out += '<' + attrs['template'] + '>'
    out += ' '
    if 'reference' in attrs:
        out += '&'
    if 'ptr' in attrs:
        out += '*'
#    out += decl['name']
    out += decl.get('name','XXXNAME')
    return out


statements = '''
 10.
 11.0
   .12
-13.
+14.0
-15e100
-16e-100
 17e+100
 18
+19
'''

statements = """
unsigned int a
void funptr1(double (*get)())
int *
int *()
int (*) ( void )
char ** cc
void aaa0(int)
void aaa1(int a)
void aaa2(int *a)
int bbb(int a, int * b, const int * c, int const * d, int **e)
"""

later = """
int *[3];
int (*) [5];
int (*const []) ( unsigned int, ... );
long long foo()
const void (*someFunc)()
void (*const timer_func)()
"""

if True:
    # tests from test_declast.py
    # used to generate baselines
    statements = """
void foo
void foo +alias(junk)
void foo()
void foo() const
void foo(int arg1)
void foo(int arg1, double arg2)
const std::string& getName() const
const void foo+attr1(30)+len=30(int arg1+in, double arg2+out)+attr2(True)
Class1 *Class1()  +constructor
void name(int arg1 = 0, double arg2 = 0.0, std::string arg3 = \"name\",bool arg4 = true)
void decl11(ArgType arg)
void decl12(std::vector<std::string> arg1, string arg2)
"""
    current_class='Class1'
    add_type('Class1')
    add_type('ArgType')


#void foo()
#void funptr1(double (*get)())
#const void foo(int arg1+in, double arg2+out = 0.0)
xstatements = """
static long int **foo
"""
#add_type('Class1')
#current_class = ''

if __name__ == '__main__':
    import json
    for line in statements.split('\n'):
        if line:
            a = Parser(line,current_class=current_class,trace=False).decl_statement()
            print(line)
            print(a)

            dd = a.to_dict()
            print(str_declarator(dd['result']))
            for arg in dd['args']:
                print('arg:', str_declarator(arg))
            print(json.dumps(dd, indent=4, sort_keys=True))

            dd = a._to_dict()
            print(json.dumps(dd, indent=4, sort_keys=True))

            if line.replace(' ','') == str(a).replace(' ',''):
                print('PASS')
            else:
                print('**', line.replace(' ',''))
                print('**', str(a).replace(' ',''))
                print('***** FAIL *****')
#            print('123456789 123456789 123456789')
