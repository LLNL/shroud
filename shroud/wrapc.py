# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Generate C bindings for C++ classes

"""
from __future__ import print_function
from __future__ import absolute_import

import os
from collections import OrderedDict

from . import declast
from . import todict
from . import statements
from . import typemap
from . import whelpers
from . import util
from .util import append_format, wformat

default_owner = "library"

lang_map = {"c": "C", "cxx": "C++"}


class Wrapc(util.WrapperMixin):
    """Generate C bindings and Fortran helpers for C++ library.

    """
    capsule_code = {}
    capsule_order = []
    capsule_include = {}  # includes needed by C_memory_dtor_function

    def __init__(self, newlibrary, config, splicers):
        """
        Args:
            newlibrary - ast.LibraryNode
            config -
            splicers -
        """
        self.newlibrary = newlibrary
        self.patterns = newlibrary.patterns
        self.language = newlibrary.language
        self.config = config
        self.log = config.log
        self._init_splicer(splicers)
        self.comment = "//"
        self.cont = ""
        self.linelen = newlibrary.options.C_line_length
        self.doxygen_begin = "/**"
        self.doxygen_cont = " *"
        self.doxygen_end = " */"
        self.shared_helper = config.fc_shared_helpers  # Shared between Fortran and C.
        self.shared_proto_c = []
        # Include files required by wrapper implementations.
        self.capsule_typedef_nodes = OrderedDict()  # [typemap.name] = typemap

    def _begin_output_file(self):
        """Start a new class for output"""
        #        # forward declarations of C++ class as opaque C struct.
        #        self.header_forward = {}
        # Include files required by wrapper prototypes
        self.header_typedef_nodes = OrderedDict()  # [typemap.name] = typemap
        # Include files required by wrapper implementations.
        self.impl_typedef_nodes = OrderedDict()  # [typemap.name] = typemap
        # Headers needed by implementation, i.e. helper functions.
        self.header_impl = util.Header(self.newlibrary)
        # Headers needed by interface.
        self.header_iface = util.Header(self.newlibrary)
        # Prototype for wrapped functions.
        self.header_proto_c = []
        self.impl = []
        self.enum_impl = []
        self.struct_impl = []
        self.c_helper = {}
        self.c_helper_include = {}  # include files in generated C header

    def wrap_library(self):
        newlibrary = self.newlibrary
        fmt_library = newlibrary.fmtdict
        # reserved the 0 slot of capsule_order
        self.add_capsule_code("--none--", None, ["// Nothing to delete"])
        self.wrap_namespace(newlibrary.wrap_namespace, True)

        self.gather_helper_code(self.shared_helper)

    def write_post_fortran(self):
        """Write utility files.

        Fortran wrappers may produce C helper functions.
        i.e. implemented in C but call from Fortran via BIND(C).
        Write C utility file after creating Fortran wrappers.
        """
        self.write_impl_utility()
        self.write_header_utility()

    def wrap_namespace(self, node, top=False):
        """Wrap a library or namespace.

        Args:
            node - ast.LibraryNode, ast.NamespaceNode
            top - True = top level library/namespace, else nested.

        Wrap depth first to accumulate destructor information
        which is written at the library level.
        """
        if top:
            # have one namespace level, then replace name each time
            self._push_splicer("namespace")
            self._push_splicer("XXX") # placer holder
        for ns in node.namespaces:
            if ns.wrap.c:
                self.wrap_namespace(ns)
        if top:
            self._pop_splicer("XXX")  # This name will not match since it is replaced.
            self._pop_splicer("namespace")
        else:
            # Skip file component in scope_file for splicer name.
            self._update_splicer_top("::".join(node.scope_file[1:]))

        self._push_splicer("class")
        structs = []
        for cls in node.classes:
            if not node.wrap.c:
                continue
            if cls.wrap_as == "struct":
                structs.append(cls)
            else:
                self._push_splicer(cls.name)
                self.write_file(node, cls, None, False)
                self._pop_splicer(cls.name)
        self._pop_splicer("class")

        self.write_file(node, None, structs, top)

    def write_file(self, ns, cls, structs, top):
        """Write a file for the library, namespace or class.

        Args:
            ns - ast.LibraryNode or ast.NamespaceNode
            cls - ast.ClassNode
            structs -
        """
        node = cls or ns
        fmt = node.fmtdict
        self._begin_output_file()

        if structs:
            for struct in structs:
                self.wrap_struct(struct)

        if cls:
            if cls.wrap_as == "class":
                self.wrap_class(cls)
        else:
            self.wrap_enums(ns)
            self.wrap_functions(ns)

        c_header = fmt.C_header_filename
        c_impl = fmt.C_impl_filename

        self.gather_helper_code(self.c_helper)
        # always include utility header
        self.c_helper_include[ns.fmtdict.C_header_utility] = True
        self.shared_helper.update(self.c_helper)  # accumulate all helpers

        if not self.write_header(ns, cls, c_header):
            # The header will not be written if it is empty.
            c_header = None
        self.write_impl(ns, cls, c_header, c_impl)

    def wrap_enums(self, node):
        """Wrap all enums in a splicer block

        Args:
            node - ast.ClassNode.
        """
        self._push_splicer("enum")
        for node in node.enums:
            self.wrap_enum(None, node)
        self._pop_splicer("enum")

    def wrap_functions(self, library):
        """
        Args:
            library - ast.LibraryNode
        """
        # worker function for write_file
        self._push_splicer("function")
        for node in library.functions:
            self.wrap_function(None, node)
        self._pop_splicer("function")

    def add_c_helper(self, helpers, fmt):
        """Add a list of C helpers."""
        c_helper = wformat(helpers, fmt)
        for helper in c_helper.split():
            self.c_helper[helper] = True
        
    def _gather_helper_code(self, name, done):
        """Add code from helpers.

        First recursively process dependent_helpers
        to add code in order.

        Args:
            name - Name of helper.
            done - Dictionary of previously processed helpers.
        """
        if name in done:
            return  # avoid recursion
        done[name] = True

        helper_info = whelpers.CHelpers[name]
        if "dependent_helpers" in helper_info:
            for dep in helper_info["dependent_helpers"]:
                # check for recursion
                self._gather_helper_code(dep, done)

        if self.language == "c":
            lang_include = "c_include"
            lang_source = "c_source"
        else:
            lang_include = "cxx_include"
            lang_source = "cxx_source"

        scope = helper_info.get("scope", "file")
        if lang_include in helper_info:
            for include in helper_info[lang_include]:
                self.helper_include[scope][include] = True
        elif "include" in helper_info:
            for include in helper_info["include"]:
                self.helper_include[scope][include] = True

        if lang_source in helper_info:
            self.helper_source[scope].append(helper_info[lang_source])
        elif "source" in helper_info:
            self.helper_source[scope].append(helper_info["source"])

    def gather_helper_code(self, helpers):
        """Gather up all helpers requested.

        Sort into helper_source and helper_include.

        Parameters
        ----------
        helpers : dict
           Indexed by name of helper.
           Should be self.c_helper or self.shared_helper.
        """
        # per class
        self.helper_source = dict(file=[], cwrap_include=[], cwrap_impl=[])
        self.helper_include = dict(file={}, cwrap_include={}, cwrap_impl={})

        done = {}  # Avoid duplicates by keeping track of what's been written.
        for name in sorted(helpers.keys()):
            self._gather_helper_code(name, done)

    def write_impl_utility(self):
        """Write a utility source file with global helpers.

        Helpers which are implemented in C and called from Fortran.
        Named from fmt.C_impl_utility.
        """
        fmt = self.newlibrary.fmtdict
        fname = fmt.C_impl_utility
        write_file = False

        output = []
        headers = util.Header(self.newlibrary)
        
        capsule_code = []
        self.write_capsule_code(capsule_code)
        if capsule_code:
            append_format(
                self.shared_proto_c,
                "\nvoid {C_memory_dtor_function}\t({C_capsule_data_type} *cap);",
                fmt,
            )
            self.set_capsule_headers(headers)
            self.shared_helper["capsule_data_helper"] = True

        self.gather_helper_code(self.shared_helper)
        
        headers.add_shroud_file(fmt.C_header_utility)
        headers.add_shroud_dict(self.helper_include["cwrap_impl"])
        headers.write_headers(output)

        if self.language == "cxx":
            output.append("")
            #            if self._create_splicer('CXX_declarations', output):
            #                write_file = True
            output.extend(["", "#ifdef __cplusplus", 'extern "C" {', "#endif"])

        if self.helper_source["cwrap_impl"]:
            write_file = True
            output.extend(self.helper_source["cwrap_impl"])

        if capsule_code:
            write_file = True
            output.extend(capsule_code)

        if self.language == "cxx":
            output.extend(["", "#ifdef __cplusplus", "}", "#endif"])

        if write_file:
            self.config.cfiles.append(
                os.path.join(self.config.c_fortran_dir, fname)
            )
            self.write_output_file(fname, self.config.c_fortran_dir, output)

    def write_header_utility(self):
        """Write a utility header file with type definitions.

        One utility header is written for the library.
        Named from fmt.C_header_utility.
        Contains typedefs for each shadow class.

        Note that this file is written after Fortran is processed
        since Fortran wrappers may contribute code. i.e. C helpers
        which are called by Fortran wrappers.

        write_impl_utility is called after this function.
        """
        fmt = self.newlibrary.fmtdict
        fname = fmt.C_header_utility
        output = []

        guard = fname.replace(".", "_").upper()

        output.extend(
            [
                "// For C users and %s implementation" % lang_map[self.language],
                "",
                "#ifndef %s" % guard,
                "#define %s" % guard,
            ]
        )

        headers = util.Header(self.newlibrary)
        headers.add_shroud_dict(self.helper_include["cwrap_include"])
        headers.write_headers(output)

        if self.language == "cxx":
            output.append("")
            #            if self._create_splicer('CXX_declarations', output):
            #                write_file = True
            output.extend(["", "#ifdef __cplusplus", 'extern "C" {', "#endif"])

        output.extend(self.helper_source["cwrap_include"])

        if self.shared_proto_c:
            output.extend(self.shared_proto_c)

        if self.language == "cxx":
            output.extend(["", "#ifdef __cplusplus", "}", "#endif"])

        output.extend(["", "#endif  // " + guard])

        self.config.cfiles.append(
            os.path.join(self.config.c_fortran_dir, fname)
        )
        self.write_output_file(fname, self.config.c_fortran_dir, output)

    def write_header(self, library, cls, fname):
        """ Write header file for a library or class node.
        The header file can be used by C or C++.

        Args:
            library - ast.LibraryNode, ast.NamespaceNode
            cls - ast.ClassNode.
            fname -
        """
        guard = fname.replace(".", "_").upper()
        node = cls or library
        options = node.options

        # If no C wrappers are required, do not write the file
        write_file = False
        output = []

        if options.doxygen:
            self.write_doxygen_file(output, fname, node)

        output.extend(
            [
                "// For C users and %s implementation" % lang_map[self.language],
                "",
                "#ifndef %s" % guard,
                "#define %s" % guard,
            ]
        )
        if cls and cls.cpp_if:
            output.append("#" + node.cpp_if)

        # headers required by typedefs and helpers
        headers = util.Header(self.newlibrary)
        headers.add_typemaps_xxx(self.header_typedef_nodes)
        headers.add_shroud_dict(self.c_helper_include)
        headers.add_file_code_header(fname, library)
        headers.write_headers(output)
        
        if self.language == "cxx":
            output.append("")
            if self._create_splicer("CXX_declarations", output):
                write_file = True
            output.extend(["", "#ifdef __cplusplus", 'extern "C" {', "#endif"])

        # ISO_Fortran_binding.h needs to be in extern "C" block.
        self.header_iface.write_headers(output)

        if self.enum_impl:
            write_file = True
            output.extend(self.enum_impl)

        if self.struct_impl:
            write_file = True
            output.extend(self.struct_impl)

        #        if self.header_forward:
        #            output.extend([
        #                '',
        #                '// declaration of shadow types'
        #            ])
        #            for name in sorted(self.header_forward.keys()):
        #                write_file = True
        #                output.append(
        #                    'struct s_{C_type_name} {{+\n'
        #                    'void *addr;   /* address of C++ memory */\n'
        #                    'int idtor;    /* index of destructor */\n'
        #                    'int refcount; /* reference count */\n'
        #                    '-}};\n'
        #                    'typedef struct s_{C_type_name} {C_type_name};'.
        #                    format(C_type_name=name))
        output.append("")
        if self._create_splicer("C_declarations", output):
            write_file = True
        if self.header_proto_c:
            write_file = True
            output.extend(self.header_proto_c)
        if self.language == "cxx":
            output.extend(["", "#ifdef __cplusplus", "}", "#endif"])
        if cls and cls.cpp_if:
            output.append("#endif  // " + node.cpp_if)
        output.extend(["", "#endif  // " + guard])

        if write_file:
            self.config.cfiles.append(
                os.path.join(self.config.c_fortran_dir, fname)
            )
            self.write_output_file(fname, self.config.c_fortran_dir, output)
        return write_file

    def write_impl(self, ns, cls, hname, fname):
        """Write implementation.
        Write struct, function, enum for a
        namespace or class.

        Args:
            ns - ast.LibraryNode or ast.NamespaceNode
            cls - ast.ClassNode
            hname -
            fname -
        """
        node = cls or ns

        # If no C wrappers are required, do not write the file
        write_file = False
        output = []
        if cls and cls.cpp_if:
            output.append("#" + node.cpp_if)

        # Use headers from implementation
        self.header_impl.add_cxx_header(node)
        self.header_impl.add_shroud_dict(self.helper_include["file"])
        self.header_impl.add_typemaps_xxx(self.impl_typedef_nodes, "impl_header")
        if hname:
            self.header_impl.add_shroud_file(hname)
        self.header_impl.write_headers(output)

        if self.language == "cxx":
            output.append("")
            if self._create_splicer("CXX_definitions", output):
                write_file = True
            output.append('\nextern "C" {')
        output.append("")

        if self.helper_source["file"]:
            write_file = True
            output.extend(self.helper_source["file"])

        if self._create_splicer("C_definitions", output):
            write_file = True
        if self.impl:
            write_file = True
            output.extend(self.impl)

        if self.language == "cxx":
            output.append("")
            output.append('}  // extern "C"')

        if cls and cls.cpp_if:
            output.append("#endif  // " + node.cpp_if)

        if write_file:
            self.config.cfiles.append(
                os.path.join(self.config.c_fortran_dir, fname)
            )
            self.write_output_file(fname, self.config.c_fortran_dir, output)

    def wrap_struct(self, node):
        """Create a C copy of struct.
        All members must be POD types.
        XXX - Only need to wrap if in a C++ namespace.

        Args:
            node - ast.ClasNode.
        """
        if self.language == "c":
            # No need for wrapper with C.
            # Use struct definition in user's header from cxx_header.
            return
        if node.wrap.c is False:
            return
        self.log.write("struct {1.name}\n".format(self, node))
        cname = node.typemap.c_type

        output = self.struct_impl
        output.append("")
        output.extend(
            ["", "struct s_{C_type_name} {{".format(C_type_name=cname), 1]
        )
        for var in node.variables:
            ast = var.ast
            output.append(ast.gen_arg_as_c() + ";")
        output.extend(
            [
                -1,
                "};",
                "typedef struct s_{C_type_name} {C_type_name};".format(
                    C_type_name=cname
                ),
            ]
        )

        # Add a sanity check on sizes of structs
        if False:
            # XXX - add this to compiled code somewhere
            stypemap = node.typemap
            output.extend(
                [
                    "",
                    "0#if sizeof {} != sizeof {}".format(
                        stypemap.name, stypemap.c_type
                    ),
                    "0#error Sizeof {} and {} do not match".format(
                        stypemap.name, stypemap.c_type
                    ),
                    "0#endif",
                ]
            )

    def wrap_class(self, node):
        """
        Args:
            node - ast.ClassNode.
        """
        self.log.write("class {1.name}\n".format(self, node))

        fmt_class = node.fmtdict
        # call method syntax
        fmt_class.CXX_this_call = fmt_class.CXX_this + "->"

        # create a forward declaration for this type
        hname = whelpers.add_shadow_helper(node)
        self.shared_helper[hname] = True
        #        self.header_forward[cname] = True
        self.compute_idtor(node)

        self.wrap_enums(node)

        self._push_splicer("method")
        for method in node.functions:
            self.wrap_function(node, method)
        self._pop_splicer("method")

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
            if method.ast.attrs["_destructor"]:
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

    def wrap_enum(self, cls, node):
        """Wrap an enumeration.
        This largely echoes the C++ code.
        For classes, it adds prefixes.

        Args:
            cls - ast.ClassNode.
            node - ast.EnumNode.
        """
        options = node.options
        ast = node.ast
        output = self.enum_impl

        node.eval_template("C_enum")
        fmt_enum = node.fmtdict
        fmtmembers = node._fmtmembers

        output.append("")
        append_format(output, "//  {namespace_scope}{enum_name}", fmt_enum)
        append_format(output, "enum {C_enum} {{+", fmt_enum)
        for member in ast.members:
            fmt_id = fmtmembers[member.name]
            if member.value is not None:
                append_format(output, "{C_enum_member} = {C_value},", fmt_id)
            else:
                append_format(output, "{C_enum_member},", fmt_id)
        output[-1] = output[-1][:-1]  # Avoid trailing comma for older compilers
        append_format(output, "-}};", fmt_enum)

    def build_proto_list(self, fmt, ast, stmts_blk, proto_list):
        """Find prototype based on c_arg_decl in fc_statements.

        Parameters
        ----------
        fmt - util.Scope
            Format dictionary (fmt_arg or fmt_result).
        ast - declast.Declaration
            Abstract Syntax Tree from parser.
        stmts_blk  - typemap.CStmts or util.Scope.
        proto_list - list
            Prototypes are appended to list.
        """
        if stmts_blk.c_arg_decl is not None:
            for arg in stmts_blk.c_arg_decl:
                append_format(proto_list, arg, fmt)
        elif stmts_blk.intent == "function":
            # Functions do not pass arguments by default.
            pass
        else:
            # vector<int> -> int *
            proto_list.append(ast.gen_arg_as_c(continuation=True))

    def add_code_from_statements(
        self, fmt, intent_blk, pre_call, post_call, need_wrapper
    ):
        """Add pre_call and post_call code blocks.
        Also record the helper functions they need.

        Args:
            fmt -
            intent_blk -
            pre_call -
            post_call -
            need_wrapper -

        return need_wrapper
        A wrapper is needed if code is added.
        """
        self.header_impl.add_statements_headers("impl_header", intent_blk)
        self.header_iface.add_statements_headers("iface_header", intent_blk)

        if intent_blk.pre_call:
            need_wrapper = True
            # pre_call.append('// intent=%s' % intent)
            for line in intent_blk.pre_call:
                append_format(pre_call, line, fmt)

        if intent_blk.post_call:
            need_wrapper = True
            for line in intent_blk.post_call:
                append_format(post_call, line, fmt)

        if intent_blk.c_helper:
            self.add_c_helper(intent_blk.c_helper, fmt)
        return need_wrapper

    def set_fmt_fields(self, cls, fcn, ast, ntypemap, fmt, is_func):
        """
        Set format fields for ast.
        Used with arguments and results.

        Args:
            cls      - ast.ClassNode or None of enclosing class.
            fcn      - ast.FunctionNode of calling function.
            ast      - declast.Declaration
            ntypemap - typemap.Typemap
            fmt      - scope.Util
            is_func  - True if function.
        """

        if is_func:
            rootname = fmt.C_result
        else:
            rootname = ast.name
            if ast.const:
                fmt.c_const = "const "
            else:
                fmt.c_const = ""
            compute_c_deref(ast, None, fmt)
            fmt.c_type = ntypemap.c_type
            fmt.cxx_type = ntypemap.cxx_type
            fmt.sh_type = ntypemap.sh_type
            fmt.cfi_type = ntypemap.cfi_type
            fmt.idtor = "0"
            if ast.blanknull:
                # Argument to helper ShroudStrAlloc via attr[blanknull].
                fmt.c_blanknull = "1"
        
        attrs = ast.attrs
        statements.assign_buf_variable_names(attrs, ast.metaattrs, fcn.options, fmt, rootname)
        
        dim = attrs["dimension"]
        if dim:
            if cls is not None:
                cls.create_node_map()
                class_context = wformat("{CXX_this}->", fmt)
            else:
                class_context = ""
            visitor = ToDimension(cls, fcn, fmt, class_context)
            visitor.visit(ast.metaattrs["dimension"])
            fmt.rank = str(visitor.rank)
            if fmt.rank != "assumed":
                if hasattr(fmt, "c_var_cdesc"):
                    # array_type is assumed to be c_var_cdesc.
                    # Assign each rank of dimension.
                    fmtdim = []
                    fmtsize = []
                    for i, dim in enumerate(visitor.shape):
                        fmtdim.append("{}->shape[{}] = {};".format(
                            fmt.c_var_cdesc, i, dim))
                        fmtsize.append("{}->shape[{}]".format(
                            fmt.c_var_cdesc, i, dim))
                    fmt.c_array_shape = "\n" + "\n".join(fmtdim)
                    if fmtsize:
                        # Multiply extents together to get size.
                        fmt.c_array_size = "*\t".join(fmtsize)
                if hasattr(fmt, "c_var_extents"):
                    # Used with CFI_establish
                    fmtdim = []
                    for i, dim in enumerate(visitor.shape):
                        fmtdim.append("{}[{}] = {};\n".format(
                            fmt.c_var_extents, i, dim))
                    fmt.c_temp_extents_decl = (
                        "CFI_index_t {0}[{1}];\n{2}".
                        format(fmt.c_var_extents, fmt.rank,
                               "".join(fmtdim)))
                    # Used with CFI_setpointer to set lower bound to 1.
                    fmt.c_temp_lower_decl = (
                        "CFI_index_t {0}[{1}] = {{{2}}};\n".
                        format(fmt.c_var_lower, fmt.rank,
                               ",".join(["1" for x in range(visitor.rank)])))
                    fmt.c_temp_extents_use = fmt.c_var_extents
                    fmt.c_temp_lower_use = fmt.c_var_lower

    def set_cxx_nonconst_ptr(self, ast, fmt):
        """Set fmt.cxx_nonconst_ptr.
        A non-const pointer to cxx_var (which may be same as c_var).
        cxx_addr is used with references.
        """
        if self.language == "c":
            if ast.const:
                fmt.cxx_nonconst_ptr = wformat(
                    "({cxx_type} *) {cxx_addr}{cxx_var}", fmt)
            else:
                fmt.cxx_nonconst_ptr = wformat(
                    "{cxx_addr}{cxx_var}", fmt)
        elif ast.const:
            # cast away constness
            fmt.cxx_nonconst_ptr = wformat(
                "const_cast<{cxx_type} *>\t({cxx_addr}{cxx_var})",
                fmt
            )
        else:
            fmt.cxx_nonconst_ptr = wformat("{cxx_addr}{cxx_var}", fmt)
        
    def wrap_function(self, cls, node):
        """Wrap a C++ function with C.

        Args:
            cls  - ast.ClassNode or None for functions.
            node - ast.FunctionNode.
        """
        options = node.options
        if not node.wrap.c:
            return

        self.log.write("C {0.declgen} {1}\n".format(
            node, self.get_metaattrs(node.ast)))

        fmt_func = node.fmtdict
        fmtargs = node._fmtargs

        if node.C_force_wrapper:
            need_wrapper = True
        elif options.C_force_wrapper:
            # User may force a wrapper.  For example, function is really a
            # macro or function pointer.
            need_wrapper = True
        elif self.language == "c" or options.get("C_extern_C", False):
            # Fortran can call C directly and only needs wrappers when code is
            # inserted. For example, pre_call or post_call.
            need_wrapper = False
        else:
            # C++ will need C wrappers to deal with name mangling.
            need_wrapper = True

        # Look for C++ routine to call.
        # Usually the same node unless it is generated (i.e. bufferified).
        CXX_node = node
        generated = []
        if CXX_node._generated:
            generated.append(CXX_node._generated)
        while CXX_node._PTR_C_CXX_index is not None:
            CXX_node = self.newlibrary.function_index[CXX_node._PTR_C_CXX_index]
            if CXX_node._generated:
                generated.append(CXX_node._generated)
        CXX_ast = CXX_node.ast
        CXX_subprogram = CXX_ast.get_subprogram()

        # C return type
        ast = node.ast
        C_subprogram = ast.get_subprogram()
        result_typemap = ast.typemap
        result_api = ast.metaattrs["api"]

        result_is_const = ast.const
        is_ctor = CXX_ast.is_ctor()
        is_dtor = CXX_ast.is_dtor()
        is_static = False
        is_pointer = CXX_ast.is_pointer()
        is_const = ast.func_const

        # self.impl_typedef_nodes.update(node.gen_headers_typedef) Python 3.6
        self.impl_typedef_nodes.update(node.gen_headers_typedef.items())
        header_typedef_nodes = OrderedDict()
        #        if result_typemap.forward:
        #            # create forward references for other types being wrapped
        #            # i.e. This method returns a wrapped type
        #            self.header_forward[result_typemap.c_type] = True

        sintent = ast.metaattrs["intent"]
        if CXX_subprogram == "subroutine":
            fmt_result = fmt_func
            fmt_pattern = fmt_func
            # intent will be "subroutine" or "dtor".
            stmts = ["c", sintent]
            result_blk = statements.lookup_fc_stmts(stmts)
        else:
            fmt_result0 = node._fmtresult
            fmt_result = fmt_result0.setdefault("fmtc", util.Scope(fmt_func))
            #            fmt_result.cxx_type = result_typemap.cxx_type  # XXX

            spointer = ast.get_indirect_stmt()
            # intent will be "function" or "ctor".
            stmts = ["c", sintent, result_typemap.sgroup, spointer,
                     result_api, ast.metaattrs["deref"]]
            result_blk = statements.lookup_fc_stmts(stmts)

            fmt_result.idtor = "0"  # no destructor
            fmt_result.c_var = fmt_result.C_local + fmt_result.C_result
            fmt_result.c_type = result_typemap.c_type
            fmt_result.cxx_type = result_typemap.cxx_type
            fmt_result.sh_type = result_typemap.sh_type
            fmt_result.cfi_type = result_typemap.cfi_type
            if ast.template_arguments:
                template_typemap = ast.template_arguments[0].typemap
                fmt_result.cxx_T = template_typemap.name
                header_typedef_nodes[template_typemap.name] = template_typemap
            else:
                header_typedef_nodes[result_typemap.name] = result_typemap
            c_local_var = ""
            if result_blk.cxx_local_var == "result":
                # C result is passed in as an argument. Create local C++ name.
                fmt_result.cxx_var = fmt_result.CXX_local + fmt_result.C_result
            elif self.language == "c":
                fmt_result.cxx_var = fmt_result.c_var
            elif result_blk.c_local_var:
                c_local_var = result_blk.c_local_var
                fmt_result.c_var = fmt_result.C_local + fmt_result.C_result
                fmt_result.cxx_var = fmt_result.CXX_local + fmt_result.C_result
            elif result_typemap.cxx_to_c is None:
                # C and C++ are compatible
                fmt_result.cxx_var = fmt_result.c_var
            else:
                fmt_result.cxx_var = fmt_result.CXX_local + fmt_result.C_result

            if result_is_const:
                fmt_result.c_const = "const "
            else:
                fmt_result.c_const = ""
            self.name_temp_vars(fmt_result.C_result, result_blk, fmt_result)

            fmt_func.cxx_rv_decl = CXX_ast.gen_arg_as_cxx(
                name=fmt_result.cxx_var, params=None, continuation=True
            )

            compute_cxx_deref(
                CXX_ast, result_blk.cxx_local_var, fmt_result)
            fmt_pattern = fmt_result
        result_blk = statements.lookup_local_stmts(
            ["c", result_api], result_blk, node)

        proto_list = []  # arguments for wrapper prototype
        proto_tail = []  # extra arguments at end of call
        call_list = []  # arguments to call function
        final_code = []
        return_code = []
        stmts_comments = []

        # Useful for debugging.  Requested and found path.
        fmt_result.stmt0 = statements.compute_name(stmts)
        fmt_result.stmt1 = result_blk.name
        if options.debug:
            stmts_comments.append(
                "// ----------------------------------------")
            if options.debug_index:
                stmts_comments.append("// Index:     {}".format(node._function_index))
            c_decl = ast.gen_decl(params=None)
            stmts_comments.append("// Function:  " + c_decl)
            self.document_stmts(
                stmts_comments, ast, fmt_result.stmt0, fmt_result.stmt1)
        
        # Indicate which argument contains function result, usually none.
        # Can be changed when a result is converted into an argument (string/vector).
        result_arg = None
        setup_this = []
        pre_call = []  # list of temporary variable declarations
        post_call = []

        if cls:
            # Add 'this' argument
            need_wrapper = True
            is_static = "static" in ast.storage
            if is_ctor:
                pass
            else:
                if is_const:
                    fmt_func.c_const = "const "
                else:
                    fmt_func.c_const = ""
                fmt_func.c_deref = "*"
                fmt_func.c_member = "->"
                fmt_func.c_var = fmt_func.C_this
                if is_static:
                    fmt_func.CXX_this_call = (
                        fmt_func.namespace_scope + fmt_func.class_scope
                    )
                else:
                    # 'this' argument, always a pointer to a shadow type.
                    proto_list.append( "{}{} * {}".format(
                        fmt_func.c_const,
                        cls.typemap.c_type, fmt_func.C_this))

                    # LHS is class' cxx_to_c
                    cls_typemap = cls.typemap
                    if cls_typemap.base != "shadow":
                        raise RuntimeError(
                            "Wapped class is not a shadow type"
                        )
                    append_format(
                        setup_this,
                        "{c_const}{namespace_scope}{cxx_type} *{CXX_this} =\t "
                        "{cast_static}{c_const}{namespace_scope}{cxx_type} *{cast1}{c_var}->addr{cast2};",
                        fmt_func,
                    )

        self.find_idtor(node.ast, result_typemap, fmt_result, result_blk)

        self.set_fmt_fields(cls, node, ast, result_typemap, fmt_result, True)

        #    c_var      - argument to C function  (wrapper function)
        #    c_var_trim - variable with trimmed length of c_var
        #    c_var_len  - variable with length of c_var
        #    cxx_var    - argument to C++ function  (wrapped function).
        #                 Usually same as c_var but may be a new local variable
        #                 or the function result variable.

        # --- Loop over function parameters
        for arg in ast.params:
            arg_name = arg.name
            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg = fmt_arg0.setdefault("fmtc", util.Scope(fmt_func))
            c_attrs = arg.attrs
            c_meta = arg.metaattrs

            arg_typemap = arg.typemap  # XXX - look up vector
            sgroup = arg_typemap.sgroup

            if arg_typemap.base == "vector":
                fmt_arg.cxx_T = arg.template_arguments[0].typemap.name

            self.header_impl.add_typemap_list(arg_typemap.impl_header)
                    
            arg_typemap, specialize = statements.lookup_c_statements(arg)
            header_typedef_nodes[arg_typemap.name] = arg_typemap
            cxx_local_var = ""
            sapi = c_meta["api"]

            if c_meta["is_result"]:
                # This argument is the C function result
                arg_call = False

                fmt_pattern = fmt_arg
                result_arg = arg
                return_deref_attr = c_meta["deref"]
                spointer = CXX_ast.get_indirect_stmt()
                stmts = [
                    "c", "function", sgroup, spointer,
                    sapi, return_deref_attr,
                ]
                intent_blk = statements.lookup_fc_stmts(stmts)
                fmt_arg.c_var = arg.name
                self.name_temp_vars(arg_name, intent_blk, fmt_arg)
                self.set_fmt_fields(cls, node, arg, arg_typemap, fmt_arg, False)
                need_wrapper = True
                cxx_local_var = intent_blk.cxx_local_var

                # Note that result_type is void, so use arg_typemap.
                if cxx_local_var == "result":
                    fmt_arg.cxx_var = fmt_func.C_local + fmt_func.C_result
                elif self.language == "c":
                    fmt_arg.cxx_var = fmt_arg.c_var
                elif arg_typemap.cxx_to_c is None:
                    fmt_arg.cxx_var = fmt_func.C_local + fmt_func.C_result
                else:
                    fmt_arg.cxx_var = fmt_func.CXX_local + fmt_func.C_result
                # Set cxx_var for statement.final in fmt_result context
                fmt_result.cxx_var = fmt_arg.cxx_var

                fmt_func.cxx_rv_decl = CXX_ast.gen_arg_as_cxx(
                    name=fmt_arg.cxx_var, params=None, continuation=True
                )
                if cxx_local_var in ["pointer", "scalar"]:
                    fmt_func.cxx_rv_decl = "*" + fmt_arg.cxx_var
                compute_cxx_deref(CXX_ast, cxx_local_var, fmt_arg)
            else:
                # regular argument (not function result)
                arg_call = arg
                spointer = arg.get_indirect_stmt()
                if c_attrs["hidden"] and node._generated:
                    sapi = "hidden"
                stmts = ["c", c_meta["intent"], sgroup, spointer,
                         sapi, c_meta["deref"]] + specialize
                intent_blk = statements.lookup_fc_stmts(stmts)
                fmt_arg.c_var = arg.name
                # XXX - order issue - c_var must be set before name_temp_vars,
                #       but set by set_fmt_fields
                self.name_temp_vars(arg_name, intent_blk, fmt_arg)
                self.set_fmt_fields(cls, node, arg, arg_typemap, fmt_arg, False)

                if intent_blk.cxx_local_var:
                    # Explicit conversion must be in pre_call.
                    cxx_local_var = intent_blk.cxx_local_var
                    fmt_arg.cxx_var = fmt_arg.CXX_local + fmt_arg.c_var
                elif self.language == "c":
                    fmt_arg.cxx_var = fmt_arg.c_var
                elif arg_typemap.c_to_cxx is None:
                    # Compatible
                    fmt_arg.cxx_var = fmt_arg.c_var
                else:
                    # convert C argument to C++
                    fmt_arg.cxx_var = fmt_arg.CXX_local + fmt_arg.c_var
                    fmt_arg.cxx_val = wformat(arg_typemap.c_to_cxx, fmt_arg)
                    fmt_arg.cxx_decl = arg.gen_arg_as_cxx(
                        name=fmt_arg.cxx_var,
                        params=None,
                        as_ptr=True,
                        continuation=True,
                    )
                    append_format(
                        pre_call, "{cxx_decl} =\t {cxx_val};", fmt_arg
                    )
                compute_cxx_deref(arg, cxx_local_var, fmt_arg)
            
            # Useful for debugging.  Requested and found path.
            fmt_arg.stmt0 = statements.compute_name(stmts)
            fmt_arg.stmt1 = intent_blk.name
            if options.debug:
                stmts_comments.append(
                    "// ----------------------------------------")
                c_decl = arg.gen_decl()
                stmts_comments.append("// Argument:  " + c_decl)
                self.document_stmts(
                    stmts_comments, arg, fmt_arg.stmt0, fmt_arg.stmt1)

            if sapi != "hidden":
                self.build_proto_list(
                    fmt_arg,
                    arg,
                    intent_blk,
                    proto_list,
                )

            self.set_cxx_nonconst_ptr(arg, fmt_arg)
            self.find_idtor(arg, arg_typemap, fmt_arg, intent_blk)

            need_wrapper = self.add_code_from_statements(
                fmt_arg, intent_blk, pre_call, post_call, need_wrapper
            )

            if arg_call:
                # Collect arguments to pass to wrapped function.
                # Skips result_as_arg argument.
                if intent_blk.arg_call:
                    for arg_call in intent_blk.arg_call:
                        append_format(call_list, arg_call, fmt_arg)
                elif cxx_local_var == "scalar":
                    if arg.is_pointer():
                        call_list.append("&" + fmt_arg.cxx_var)
                    else:
                        call_list.append(fmt_arg.cxx_var)
                elif cxx_local_var == "pointer":
                    if arg.is_pointer():
                        call_list.append(fmt_arg.cxx_var)
                    else:
                        call_list.append("*" + fmt_arg.cxx_var)
                elif arg.is_reference():
                    # reference to scalar  i.e. double &max
                    # void tutorial::getMinMax(int &min);
                    # wrapper(int *min) {
                    #   tutorial::getMinMax(*min);
                    #}
                    call_list.append("*" + fmt_arg.cxx_var)
                else:
                    call_list.append(fmt_arg.cxx_var)

        #            if arg_typemap.forward:
        #                # create forward references for other types being wrapped
        #                # i.e. This argument is another wrapped type
        #                self.header_forward[arg_typemap.c_type] = True
        # --- End loop over function parameters

        self.build_proto_list(
            fmt_result,
            ast,
            result_blk,
            proto_list,
        )
        
        if call_list:
            fmt_func.C_call_list = ",\t ".join(call_list)

        if len(proto_list) + len(proto_tail) == 0:
            proto_list.append("void")
        fmt_func.C_prototype = options.get(
            "C_prototype", ",\t ".join(proto_list + proto_tail)
        )

        return_deref_attr = ast.metaattrs["deref"]
        if result_blk.return_type:
            # Override return type.
            fmt_func.C_return_type = wformat(
                result_blk.return_type, fmt_result)
        elif return_deref_attr == "scalar":
            # Need a wrapper since it will dereference the return pointer.
            need_wrapper = True
            fmt_func.C_return_type = ast.gen_arg_as_c(
                name=None, as_scalar=True, params=None, continuation=True
            )
        elif result_typemap.sgroup == "shadow":
            # The const does not apply to the capsule.
            fmt_func.C_return_type = ast.gen_arg_as_c(
                name=None, params=None, continuation=True,
                remove_const=True
            )
        else:
            fmt_func.C_return_type = ast.gen_arg_as_c(
                name=None, params=None, continuation=True
            )

        # generate the C body
        post_call_pattern = []
        if node.C_error_pattern is not None:
            C_error_pattern = statements.compute_name(
                [node.C_error_pattern, node.splicer_group])
            if C_error_pattern in self.patterns:
                need_wrapper = True
                post_call_pattern.append("// C_error_pattern")
                append_format(
                    post_call_pattern,
                    self.patterns[C_error_pattern],
                    fmt_pattern,
                )

        if result_blk.call:
            raw_call_code = result_blk.call
            need_wrapper = True
        elif CXX_subprogram == "subroutine":
            raw_call_code = [
                "{CXX_this_call}{function_name}"
                "{CXX_template}({C_call_list});",
            ]
        else:
            if result_blk.cxx_local_var is None:
                pass
            elif result_blk.cxx_local_var == "result":
                pass
            else:
                # A C++ var is created by pre_call.
                # Assign to it directly. ex c_function_shadow_scalar
                fmt_result.cxx_addr = ""
                fmt_func.cxx_rv_decl = "*" + fmt_result.cxx_var
            
            raw_call_code = [
                "{cxx_rv_decl} =\t {CXX_this_call}{function_name}"
                "{CXX_template}(\t{C_call_list});",
            ]
            if result_arg is None:
                # Return result from function
                # (It was not passed back in an argument)
                if self.language == "c":
                    pass
                elif result_blk.return_type == "void":
                    # Do not return C++ result in C wrapper.
                    # Probably assigned to an argument.
                    pass
                elif result_blk.c_local_var:
                    # c_var is created by the post_call clause or
                    # it may be passed in as an argument.
                    # For example, with struct and shadow.
                    pass
                elif result_typemap.cxx_to_c is not None:
                    # Make intermediate c_var value if a conversion
                    # is required i.e. not the same as cxx_var.
                    fmt_result.c_rv_decl = CXX_ast.gen_arg_as_c(
                        name=fmt_result.c_var, params=None, continuation=True
                    )
                    fmt_result.c_val = wformat(
                        result_typemap.cxx_to_c, fmt_result
                    )
                    append_format(
                        return_code, "{c_rv_decl} =\t {c_val};", fmt_result
                    )
                self.set_cxx_nonconst_ptr(ast, fmt_result)
                    
                self.header_impl.add_typemap_list(result_typemap.impl_header)

        need_wrapper = self.add_code_from_statements(
            fmt_result, result_blk, pre_call, post_call, need_wrapper
        )

        call_code = []
        for line in raw_call_code:
            append_format(call_code, line, fmt_result)

        if result_blk.final:
            need_wrapper = True
            final_code.append("{+")
            final_code.append("// final")
            for line in result_blk.final:
                append_format(final_code, line, fmt_result)
            final_code.append("-}")

        if result_blk.return_type == "void":
            raw_return_code = []
        elif result_blk.ret:
            raw_return_code = result_blk.ret
        elif return_deref_attr == "copy":
            raw_return_code = ["return {cxx_var};"]
        elif return_deref_attr == "scalar":
            # dereference pointer to return scalar
            raw_return_code = ["return *{cxx_var};"]
        elif result_arg is None and C_subprogram == "function":
            # Note: A C function may be converted into a Fortran subroutine
            # subprogram when the result is returned in an argument.
            fmt_result.c_get_value = statements.compute_return_prefix(ast, c_local_var)
            raw_return_code = ["return {c_get_value}{c_var};"]
        else:
            # XXX - No return for void statements.
            # XXX - Put on an option?
#            raw_return_code = ["return;"]
            raw_return_code = []
        for line in raw_return_code:
            append_format(return_code, line, fmt_result)

        splicer_name = statements.compute_name(["c", node.splicer_group])
        if splicer_name in node.splicer:
            need_wrapper = True
            C_force = node.splicer[splicer_name]
            C_code = None
        else:
            # copy-out values, clean up
            C_force = None
            C_code = pre_call + call_code + post_call_pattern + \
                     post_call + final_code + return_code

        if need_wrapper:
            self.header_typedef_nodes.update(header_typedef_nodes.items()) # Python 3.6
            self.header_proto_c.append("")
            if node.cpp_if:
                self.header_proto_c.append("#" + node.cpp_if)
            append_format(
                self.header_proto_c,
                "{C_return_type} {C_name}(\t{C_prototype});",
                fmt_func,
            )
            if node.cpp_if:
                self.header_proto_c.append("#endif")

            impl = self.impl
            impl.append("")
            if options.doxygen and node.doxygen:
                self.write_doxygen(impl, node.doxygen)
            if node.cpp_if:
                impl.append("#" + node.cpp_if)
            impl.extend(stmts_comments)
            if options.literalinclude:
                append_format(impl, "// start {C_name}", fmt_func)
            append_format(
                impl, "{C_return_type} {C_name}(\t{C_prototype})", fmt_func
            )
            impl.append("{+")
            impl.extend(setup_this)
            #sname = fmt_func.function_name # XXX ?
            sname = wformat("{underscore_name}{function_suffix}{template_suffix}",
                            fmt_func)
            self._create_splicer(sname, impl, C_code, C_force)
            impl.append("-}")
            if options.literalinclude:
                append_format(impl, "// end {C_name}", fmt_func)
            if node.cpp_if:
                impl.append("#endif  // " + node.cpp_if)
        else:
            # There is no C wrapper, have Fortran call the function directly.
            fmt_func.C_name = node.ast.name

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
            
    def write_capsule_code(self, output):
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
            "void {C_memory_dtor_function}\t({C_capsule_data_type} *cap)\n"
            "{{+",
            fmt,
        )

        if options.F_auto_reference_count:
            # check refererence before deleting
            append_format(
                output,
                "@--cap->refcount;\n"
                "if (cap->refcount > 0) {{+\n"
                "return;\n"
                "-}}",
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

    def find_idtor(self, ast, ntypemap, fmt, intent_blk):
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
            ast -
            ntypemap - typemap.Typemap
            fmt -
            intent_blk -
        """

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
            else:
                fmt.idtor = self.capsule_code[destructor_name][0]
            return

        from_stmt = False
        if ast.attrs["owner"]:
            owner = ast.attrs["owner"]
        elif intent_blk.owner:
            owner = intent_blk.owner
            from_stmt = True
        else:
            owner = default_owner

        free_pattern = ast.attrs["free_pattern"]
        if owner == "library":
            # Library owns memory, do not let user release.
            pass
        elif not ast.is_pointer() and not from_stmt:
            # Non-pointers do not return dynamic memory.
            # Unless it is a function which returns memory
            # by value. (like a class instance.)
            pass
        elif free_pattern is not None:
            # free_pattern attribute.
            fmt.idtor = self.add_destructor(
                fmt, free_pattern, [self.patterns[free_pattern]], None
            )
        elif ntypemap.idtor != "0":
            # Return cached value.
            fmt.idtor = ntypemap.idtor
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

######################################################################

class ToDimension(todict.PrintNode):
    """Convert dimension expression to C wrapper code.

    expression has already been checked for errors by generate.check_implied.
    Convert functions:
      size  -  PyArray_SIZE
    """

    def __init__(self, cls, fcn, fmt, context):
        """
        Args:
            cls  - ast.ClassNode or None
            fcn  - ast.FunctionNode of calling function.
            fmt  - util.Scope
            context - how to access Identifiers in cls.
                      Different for function arguments and
                      class/struct members.
        """
        super(ToDimension, self).__init__()
        self.cls = cls
        self.fcn = fcn
        self.fmt = fmt
        self.context = context

        self.rank = 0
        self.shape = []

    def visit_list(self, node):
        # list of dimension expressions
        self.rank = len(node)
        for dim in node:
            sh = self.visit(dim)
            self.shape.append(sh)

    def visit_Identifier(self, node):
        argname = node.name
        # Look for members of class/struct.
        if self.cls is not None and argname in self.cls.map_name_to_node:
            # This name is in the same class as the dimension.
            # Make name relative to the class.
            member = self.cls.map_name_to_node[argname]
            if member.may_have_args():
                if node.args is None:
                    print("{} must have arguments".format(argname))
                else:
                    return "{}{}({})".format(
                        self.context, argname, self.comma_list(node.args))
            else:
                if node.args is not None:
                    print("{} must not have arguments".format(argname))
                else:
                    return "{}{}".format(self.context, argname)
        else:
            deref = ''
            arg = self.fcn.ast.find_arg_by_name(argname)
            if arg:
                if arg.attrs["hidden"]:
                    # (int *arg +intent(out)+hidden)
                    # c_out_native_*_hidden creates a local scalar.
                    deref = ''
                elif arg.is_indirect():
                    # If argument is a pointer, then dereference it.
                    # i.e.  (int *arg +intent(out))
                    deref = '*'
            if node.args is None:
                return deref + argname  # variable
            else:
                return deref + self.param_list(node) # function
        return "--??--"

    def visit_AssumedRank(self, node):
        self.rank = "assumed"
        return "--assumed-rank--"
        raise RuntimeError("wrapc.py: Detected assumed-rank dimension")

######################################################################

def compute_c_deref(arg, local_var, fmt):
    """Compute format fields to dereference C argument."""
    if local_var == "scalar":
        fmt.c_deref = ""
        fmt.c_member = "."
        fmt.c_addr = "&"
    elif local_var == "pointer":
        fmt.c_deref = "*"
        fmt.c_member = "->"
        fmt.c_addr = ""
    elif arg.is_indirect(): #pointer():
        fmt.c_deref = "*"
        fmt.c_member = "->"
        fmt.c_addr = ""
    else:
        fmt.c_deref = ""
        fmt.c_member = "."
        fmt.c_addr = "&"

def compute_cxx_deref(arg, local_var, fmt):
    """Compute format fields to dereference C++ variable."""
    if local_var == "scalar":
#        fmt.cxx_deref = ""
        fmt.cxx_member = "."
        fmt.cxx_addr = "&"
    elif local_var == "pointer":
#        fmt.cxx_deref = "*"
        fmt.cxx_member = "->"
        fmt.cxx_addr = ""
    elif arg.is_pointer():
#        fmt.cxx_deref = "*"
        fmt.cxx_member = "->"
        fmt.cxx_addr = ""
    else:
#        fmt.cxx_deref = ""
        fmt.cxx_member = "."
        fmt.cxx_addr = "&"
