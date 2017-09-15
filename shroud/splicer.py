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
Read a file and extract the splicer blocks.
"""
from __future__ import print_function
from __future__ import absolute_import

import os


def get_splicers(fname, out):
    """
    fname - input file name
    out - dictionary to update

    tags of the form
    begin  value ...
    """
    str_begin = 'splicer begin'
    str_end = 'splicer end'
    str_push = 'splicer push'
    str_pop = 'splicer pop'

    state_look = 1
    state_collect = 2
    state = state_look

    top = out
    stack = [top]

    begin_tag = ''

    with open(fname, 'r') as fp:
        for line in fp.readlines():
            if state == state_look:
                i = line.find(str_begin)
                if i > 0:
                    fields = line[i+len(str_begin):].split()
                    tag = fields[0]
                    begin_tag = tag
                    subtags = tag.split('.')
                    begin_subtag = subtags[-1]
                    for subtag in subtags[:-1]:
                        top = top.setdefault(subtag, {})
                        stack.append(top)
#                    print("BEGIN", begin_tag)
                    save = []
                    state = state_collect
                    continue

            elif state == state_collect:
                i = line.find(str_end)
                if i > 0:
                    fields = line[i+len(str_end):].split()
                    end_tag = fields[0]
#                    print("END", end_tag)
                    if begin_tag != end_tag:
                        raise RuntimeError(
                            "Mismatched tags  '%s' '%s'", (begin_tag, end_tag))
                    if end_tag in top:
                        raise RuntimeError(
                            "Tag already exists - '%s'" % begin_tag)
                    top[begin_subtag] = save
                    top = out
                    stack = [top]
                    state = state_look
                else:
                    save.append(line.rstrip())


def get_splicer_based_on_suffix(name, out):
        fileName, fileExtension = os.path.splitext(name)
        if fileExtension in ['.f', '.f90']:
            d = out.setdefault('f', {})
            get_splicers(name, d)
        elif fileExtension in [
                '.c', '.h', '.cpp', '.hpp', '.cxx', '.hxx', '.cc', '.C']:
            d = out.setdefault('c', {})
            get_splicers(name, d)
        elif fileExtension in ['.py']:
            d = out.setdefault('py', {})
            get_splicers(config, name, d)
        elif fileExtension in ['.lua']:
            d = out.setdefault('lua', {})
            get_splicers(config, name, d)


# def print_tree(out):
#     import json
#     print(json.dumps(out, indent=4, sort_keys=True))

if __name__ == '__main__':
    import glob
    import json

    out = {}
    for name in glob.glob('../tests/example/*'):
        get_splicer_based_on_suffix(name, out)
    print(json.dumps(out, indent=4, sort_keys=True))

    for name in ['fsplicer.f', 'csplicer.c', 'pysplicer.c']:
        out = {}
        get_splicers(os.path.join('..', 'tests', name), out)
        print(json.dumps(out, indent=4, sort_keys=True))
