"""
Wrap a library using the Python API.
"""

from __future__ import print_function

import shroud

library = shroud.LibraryNode('testapi')
library.add_function(decl='void foo()')

if __name__ == '__main__':
#    print(library._to_dict())
    import sys
    shroud.print_as_json(library, sys.stdout)
