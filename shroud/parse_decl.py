"""
Copyright (c) 2017, Lawrence Livermore National Security, LLC. 
Produced at the Lawrence Livermore National Laboratory 

LLNL-CODE-738041.
All rights reserved. 
 
This file is part of Shroud.  For details, see
https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
 
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the disclaimer below.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the disclaimer (as noted below)
  in the documentation and/or other materials provided with the
  distribution.

* Neither the name of the LLNS/LLNL nor the names of its contributors
  may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

########################################################################
Parse a declaration and return a named-tuple tree.

decl = ( type, name, attrs, [ args ] )

arg = ( type, name, attrs )

attrs = { key : value }


This module just parses syntax.  Semantics, like valid type names
and duplicate argument names are check later.
"""
from __future__ import print_function
from __future__ import absolute_import

import parsley


def add_to_dict(d, key, value):
    d[key] = value
    return d


x = parsley.makeGrammar("""
name = < (letter | '_') (letter | digit | '_' | ':')* >

#type = name:t ?( t in types ) ^(C-type) -> t
type = name:t

digits = <digit*>
floatPart :sign :ds = <('.' digits exponent?) | exponent>:tail
                     -> float(sign + ds + tail)
exponent = ('e' | 'E') ('+' | '-')? digits

number = spaces ('-' | -> ''):sign (digits:ds (floatPart(sign ds)
                                               | -> int(sign + ds)))

string = (('"' | "'"):q <(~exactly(q) anything)*>:xs exactly(q))
                     -> xs

# remove parens
parens = '('  (~')' anything)*:v ')'
        -> ''.join(v)

value = name | string | number

attr = '+' ws name:n ( ('=' value) | parens | -> True ):v
        -> (n,v)

qualifier = 'const' -> [('const', True)]
            |       -> []

pointer = '*' -> [('ptr', True)]
        | '&' -> [('reference', True)]
        |     -> []

default = ws '=' ws value:default -> [('default', default)]
                 |                -> []

declarator = qualifier:qu ws type:t ws pointer:pp ws name:n  ( ws attr )*:at default:df
        -> dict(type=t, name=n, attrs=dict(qu+pp+at+df))

parameter_list = declarator:first ( ws ',' ws declarator)*:rest -> [first] + rest
                 | -> []

argument_list = ( '(' ws parameter_list:l ws ')' ) -> l
                | -> []

decl = declarator:dd ws argument_list:args ws qualifier:qual (ws attr)*:at
        -> dict( result=dd, args=args, attrs=dict(qual + at))
""", {})


def check_decl(expr, parser=x):
    """ parse expr as a declaration, return list/dict result.
    """
    return parser(expr).decl()
