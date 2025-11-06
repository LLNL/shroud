# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Memory management for Fortran and C wrappers.
"""

from collections import OrderedDict

from . import util
from .util import append_format, wformat

class CapsuleFmt(object):
    """
    Methods to compute the destructors for a capsule (shadow) type.
    """
    def __init__(self, newlibrary):
        # Include files required by wrapper implementations.
        self.newlibrary = newlibrary
        self.language = newlibrary.language
        self.capsule_typedef_nodes = OrderedDict()  # [typemap.name] = typemap
        self.capsule_code = {}
        self.capsule_order = []
        self.capsule_include = {}  # includes needed by C_memory_dtor_function
        self.destructors = newlibrary.destructors
        # reserved the 0 slot of capsule_order
        self.add_capsule_code("--none--", None, ["// Nothing to delete"])

    def compute_idtor(self, node):
        """Create a capsule destructor for type.

        Only call add_capsule_code if the destructor is wrapped.
        Otherwise, there is no way to delete the object.
        i.e. the class has a private destructor.

        Args:
            node - ast.ClassNode.
        """
        has_dtor = False
        for method in node.functions:
            if method.ast.is_dtor:
                has_dtor = True
                break

        ntypemap = node.typemap
        if has_dtor:
            cxx_type = ntypemap.cxx_type
            del_lines = [
                "{cxx_type} *cxx_ptr = \treinterpret_cast<{cxx_type} *>(ptr);".format(
                    cxx_type=cxx_type
                ),
                "delete cxx_ptr;",
            ]
            ntypemap.idtor = self.add_capsule_code(
                cxx_type, ntypemap, del_lines
            )
            self.capsule_typedef_nodes[ntypemap.name] = ntypemap
        else:
            ntypemap.idtor = "0"

    def set_capsule_headers(self, headers):
        """Headers used by C_memory_dtor_function.
        """
        fmt = self.newlibrary.fmtdict
        headers.add_shroud_file(fmt.C_header_utility)
        headers.add_shroud_dict(self.capsule_include)
        if self.language == "c":
            # Add header for NULL. C++ uses nullptr.
            headers.add_shroud_file("<stdlib.h>")
        for ntypedefs in self.capsule_typedef_nodes.values():
            headers.add_typemap_list(ntypedefs.impl_header)
            
    def write_capsule_delete_code(self, output):
        """Write a function used to delete C/C++ memory.

        Parameters
        ----------
        output : list of str
            Accumulation of line to file being created.
        """
        library = self.newlibrary
        options = library.options

        fmt = library.fmtdict

        output.append("")
        if options.literalinclude2:
            output.append("// start release allocated memory")
        append_format(
            output,
            "// Release library allocated memory.\n"
            "void {C_memory_dtor_function}\t({c_capsule_data_type} *cap)\n"
            "{{+",
            fmt,
        )

        if len(self.capsule_order) > 1:
            # If more than slot 0 is added, create switch statement
            append_format(
                output, "void *ptr = cap->addr;\n" "switch (cap->idtor) {{", fmt
            )

            for i, name in enumerate(self.capsule_order):
                output.append("case {}:   // {}\n{{+".format(i, name))
                output.extend(self.capsule_code[name][1])
                output.append("break;\n-}")

            output.append(
                "default:\n{+\n"
                "// Unexpected case in destructor\n"
                "break;\n"
                "-}\n"
                "}"
            )

        append_format(output,
                      "cap->addr = {nullptr};\n"
                      "cap->idtor = 0;  // avoid deleting again\n"
                      "cap->cmemflags = cap->cmemflags & ~SWIG_MEM_OWN;\n"
                      "-}}",
                      fmt
        )
        if options.literalinclude2:
            output.append("// end release allocated memory")

    def add_capsule_code(self, name, var_typemap, lines):
        """Add unique names to capsule_code.
        Return index of name.

        Args:
            name - ex.  std::vector<int>
            var_typemap - typemap.Typemap.
            lines -
        """
        if name not in self.capsule_code:
            self.capsule_code[name] = (str(len(self.capsule_code)), lines)
            self.capsule_order.append(name)

            # include files required by the type
            if var_typemap:
                for include in var_typemap.cxx_header:
                    self.capsule_include[include] = True

        return self.capsule_code[name][0]

    def add_destructor(self, fmt, name, cmd_list, arg_typemap):
        """Add a capsule destructor with name and commands.

        Args:
            fmt -
            name -
            cmd_list -
            arg_typemap - typemap.Typemap.
        """
        if name not in self.capsule_code:
            del_lines = []
            for cmd in cmd_list:
                del_lines.append(wformat(cmd, fmt))
            idtor = self.add_capsule_code(name, arg_typemap, del_lines)
        else:
            idtor = self.capsule_code[name][0]
        return idtor

    def find_idtor(self, ntypemap, bind):
        """Find the destructor name.

        Set fmt.idtor as index of destructor.

        Check stmts.destructor_name

        XXX - no longer true...
        Only arguments have idtor's.
        For example,
            int * foo() +owner(caller)
        will convert to
            void foo(context+owner(caller) )

        Args:
            ntypemap - typemap.Typemap
            bind - statements.BindArg
        """
        intent_blk = bind.stmt
        meta = bind.meta
        fmt = bind.fmtdict

        destructor_name = intent_blk.destructor_name
        if destructor_name:
            # Custom destructor from statements.
            # Use destructor in typemap to remove intermediate objects
            # e.g. std::vector
            self.capsule_typedef_nodes[ntypemap.name] = ntypemap
            destructor_name = wformat(destructor_name, fmt)
            if destructor_name not in self.capsule_code:
                del_lines = []
                util.append_format_cmds(
                    del_lines, intent_blk, "destructor", fmt
                )
                fmt.idtor = self.add_capsule_code(
                    destructor_name, ntypemap, del_lines
                )
                for header in intent_blk.destructor_header:
                    self.capsule_include[header] = True
            else:
                fmt.idtor = self.capsule_code[destructor_name][0]
            return

        owner = intent_blk.owner or meta["owner"]

        destructor_name = meta["destructor_name"]
        if owner != "caller":
            # library, shared, weak. Do not let user release.
            pass
        elif destructor_name is not None:
            # destructor_name attribute.
            fmt.idtor = self.add_destructor(
                fmt, destructor_name, [self.destructors[destructor_name]], None
            )
        elif ntypemap.idtor != "0":
            # Return cached value.
            fmt.idtor = ntypemap.idtor
        elif self.language == "c":
            fmt.idtor = self.add_destructor(
                fmt,
                ntypemap.cxx_type,
                [
                    "{cxx_type} *cxx_ptr =\t ({cxx_type} *) ptr;",
                    "free(cxx_ptr);",
                ],
                ntypemap,
            )
            ntypemap.idtor = fmt.idtor
            self.capsule_typedef_nodes[ntypemap.name] = ntypemap
        elif ntypemap.cxx_to_c:
            # Class instance.
            # A C++ native type (std::string, std::vector)
            # XXX - vector does not assign cxx_to_c
            fmt.idtor = self.add_destructor(
                fmt,
                ntypemap.cxx_type,
                [
                    "{cxx_type} *cxx_ptr =\t reinterpret_cast<{cxx_type} *>(ptr);",
                    "delete cxx_ptr;",
                ],
                ntypemap,
            )
            ntypemap.idtor = fmt.idtor
            self.capsule_typedef_nodes[ntypemap.name] = ntypemap
        else:
            # A POD type
            fmt.idtor = self.add_destructor(
                fmt,
                ntypemap.cxx_type,
                [
                    "{cxx_type} *cxx_ptr =\t reinterpret_cast<{cxx_type} *>(ptr);",
                    "free(cxx_ptr);",
                ],
                ntypemap,
            )
            ntypemap.idtor = fmt.idtor
