# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
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

from . import declast
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
        self.shared_helper = {}  # All accumulated helpers
        self.shared_proto_c = []

    _default_buf_args = ["arg"]

    def _begin_output_file(self):
        """Start a new class for output"""
        #        # forward declarations of C++ class as opaque C struct.
        #        self.header_forward = {}
        # include files required by typedefs
        self.header_typedef_nodes = {}  # [arg_typedef.name] = arg_typedef
        # headers needed by implementation, i.e. helper functions
        self.header_impl_include = {}  # header files in implementation file
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
        whelpers.set_literalinclude(newlibrary.options.literalinclude2)
        whelpers.add_copy_array_helper_c(fmt_library)
        self.wrap_namespace(newlibrary.wrap_namespace, True)
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
            if ns.options.wrap_c:
                self.wrap_namespace(ns)
        # Skip file component in scope_file for splicer name.
        if top:
            self._pop_splicer("XXX")  # This name will not match since it is replaced.
            self._pop_splicer("namespace")
        else:
            self._update_splicer_top("::".join(node.scope_file[1:]))

        self._push_splicer("class")
        structs = []
        for cls in node.classes:
            if not node.options.wrap_c:
                continue
            if cls.as_struct:
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
            if not cls.as_struct:
                self.wrap_class(cls)
        else:
            self.wrap_enums(ns)
            self.wrap_functions(ns)
            if top:
                self.write_capsule_code()

        c_header = fmt.C_header_filename
        c_impl = fmt.C_impl_filename

        self.gather_helper_code(self.c_helper)
        # always include utility header
        self.c_helper_include[ns.fmtdict.C_header_utility] = True
        self.shared_helper.update(self.c_helper)  # accumulate all helpers

        if not self.write_header(ns, cls, c_header):
            # The header will not be written if it is empty
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

    def _gather_helper_code(self, name, done):
        """Add code from helpers.

        First recursively process dependent_helpers
        to add code in order.

        Args:
            name -
            done -
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
            lang_header = "c_header"
            lang_source = "c_source"
        else:
            lang_header = "cxx_header"
            lang_source = "cxx_source"
        scope = helper_info.get("scope", "file")

        if lang_header in helper_info:
            for include in helper_info[lang_header].split():
                self.helper_header[scope][include] = True
        elif "header" in helper_info:
            for include in helper_info["header"].split():
                self.helper_header[scope][include] = True

        if lang_source in helper_info:
            self.helper_source[scope].append(helper_info[lang_source])
        elif "source" in helper_info:
            self.helper_source[scope].append(helper_info["source"])

    def gather_helper_code(self, helpers):
        """Gather up all helpers requested and insert code into output.

        helpers should be self.c_helper or self.shared_helper

        Args:
            helpers -
        """
        # per class
        self.helper_source = dict(file=[], utility=[])
        self.helper_header = dict(file={}, utility={})

        done = {}  # avoid duplicates and recursion
        for name in sorted(helpers.keys()):
            self._gather_helper_code(name, done)

    def write_header_utility(self):
        """Write a utility header file with type definitions.
        """
        self.gather_helper_code(self.shared_helper)

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

        # headers required helpers
        self.write_headers_nodes(
            "c_header", {}, self.helper_header["utility"].keys(), output
        )

        if self.language == "cxx":
            output.append("")
            #            if self._create_splicer('CXX_declarations', output):
            #                write_file = True
            output.extend(["", "#ifdef __cplusplus", 'extern "C" {', "#endif"])

        output.extend(self.helper_source["utility"])

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
        """ Write header file for a library node or a class node.

        Args:
            library - ast.LibraryNode.
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
        self.write_headers_nodes(
            "c_header",
            self.header_typedef_nodes,
            self.c_helper_include.keys(),
            output,
        )

        if self.language == "cxx":
            output.append("")
            if self._create_splicer("CXX_declarations", output):
                write_file = True
            output.extend(["", "#ifdef __cplusplus", 'extern "C" {', "#endif"])

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
        Writ struct, function, enum for a
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

        if hname:
            output.append('#include "%s"' % hname)

        # Use headers from class if they exist or else library
        self.header_impl_include.update(self.helper_header["file"])
        if cls and cls.cxx_header:
            for include in cls.cxx_header.split():
                self.header_impl_include[include] = True
        else:
            for include in self.newlibrary.cxx_header.split():
                self.header_impl_include[include] = True

        # headers required by implementation
        if self.header_impl_include:
            headers = self.header_impl_include.keys()
            self.write_headers(headers, output)

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
        A C++ struct must all POD.
        XXX - Only need to wrap if in a C++ namespace.

        Args:
            node - ast.ClasNode.
        """
        if self.language == "c":
            # No need for wrapper with C.
            # Use struct definition in user's header from cxx_header.
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
            if "_destructor" in method.ast.attrs:
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
            fmt_id.C_enum_member = wformat(
                options.C_enum_member_template, fmt_id
            )
            if member.value is not None:
                append_format(output, "{C_enum_member} = {cxx_value},", fmt_id)
            else:
                append_format(output, "{C_enum_member},", fmt_id)
        output[-1] = output[-1][:-1]  # Avoid trailing comma for older compilers
        append_format(output, "-}};", fmt_enum)

    def build_proto_list(self, fmt, ast, buf_args, proto_list, need_wrapper):
        """Find prototype based on buf_args in c_statements.

        Args:
            fmt - Format dictionary (fmt_arg or fmt_result).
            ast - Abstract Syntax Tree from parser.
            buf_args - List of arguments/metadata to add.
            proto_list - Prototypes are appended to list.
            need_wrapper -

        return need_wrapper
        A wrapper will be needed if there is meta data.
        i.e. No wrapper if the C function can be called directly.
        """
        attrs = ast.attrs

        for buf_arg in buf_args:
            if buf_arg == "arg":
                # vector<int> -> int *
                proto_list.append(ast.gen_arg_as_c(continuation=True))
                continue
            elif buf_arg == "shadow":
                proto_list.append(ast.gen_arg_as_c(continuation=True))
                continue

            need_wrapper = True
            if buf_arg == "size":
                fmt.c_var_size = attrs["size"]
                append_format(proto_list, "long {c_var_size}", fmt)
            elif buf_arg == "capsule":
                fmt.c_var_capsule = attrs["capsule"]
                append_format(
                    proto_list, "{C_capsule_data_type} *{c_var_capsule}", fmt
                )
            elif buf_arg == "context":
                fmt.c_var_context = attrs["context"]
                append_format(
                    proto_list, "{C_array_type} *{c_var_context}", fmt
                )
                if "dimension" in attrs:
                    # XXX - assumes dimension is a single variable.
                    fmt.c_var_dimension = attrs["dimension"]
            elif buf_arg == "len_trim":
                fmt.c_var_trim = attrs["len_trim"]
                append_format(proto_list, "int {c_var_trim}", fmt)
            elif buf_arg == "len":
                fmt.c_var_len = attrs["len"]
                append_format(proto_list, "int {c_var_len}", fmt)
            else:
                raise RuntimeError(
                    "wrap_function: unhandled case {}".format(buf_arg)
                )
        return need_wrapper

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
        if "pre_call" in intent_blk:
            need_wrapper = True
            # pre_call.append('// intent=%s' % intent)
            for line in intent_blk["pre_call"]:
                append_format(pre_call, line, fmt)

        if "post_call" in intent_blk:
            need_wrapper = True
            for line in intent_blk["post_call"]:
                append_format(post_call, line, fmt)

        if "c_helper" in intent_blk:
            c_helper = wformat(intent_blk["c_helper"], fmt)
            for helper in c_helper.split():
                self.c_helper[helper] = True
        return need_wrapper

    def wrap_function(self, cls, node):
        """Wrap a C++ function with C.

        Args:
            cls  - ast.ClassNode or None for functions.
            node - ast.FunctionNode.
        """
        options = node.options
        if not options.wrap_c:
            return

        if cls:
            cls_function = "method"
        else:
            cls_function = "function"
        self.log.write("C {0} {1.declgen}\n".format(cls_function, node))

        fmt_func = node.fmtdict
        fmtargs = node._fmtargs

        if self.language == "c" or options.get("C_extern_C", False):
            # Fortran can call C directly and only needs wrappers when code is
            # inserted. For example, precall or postcall.
            need_wrapper = False
        else:
            # C++ will need C wrappers to deal with name mangling.
            need_wrapper = True

        # Look for C++ routine to wrap
        # Usually the same node unless it is generated (i.e. bufferified)
        CXX_node = node
        generated = []
        if CXX_node._generated:
            generated.append(CXX_node._generated)
        while CXX_node._PTR_C_CXX_index is not None:
            CXX_node = self.newlibrary.function_index[CXX_node._PTR_C_CXX_index]
            if CXX_node._generated:
                generated.append(CXX_node._generated)
        CXX_ast = CXX_node.ast
        CXX_subprogram = CXX_node.CXX_subprogram

        # C return type
        ast = node.ast
        C_subprogram = node.C_subprogram
        result_typemap = node.CXX_result_typemap
        generated_suffix = node.generated_suffix

        result_is_const = ast.const
        is_ctor = CXX_ast.is_ctor()
        is_dtor = CXX_ast.is_dtor()
        is_static = False
        is_pointer = CXX_ast.is_pointer()
        is_const = ast.func_const
        is_shadow_scalar = False
        is_union_scalar = False
        shadow_arg_decl = None

        if result_typemap.c_header:
            # include any dependent header in generated header
            self.header_typedef_nodes[result_typemap.name] = result_typemap
        if result_typemap.cxx_header:
            # include any dependent header in generated source
            self.header_impl_include[result_typemap.cxx_header] = True
        #        if result_typemap.forward:
        #            # create forward references for other types being wrapped
        #            # i.e. This method returns a wrapped type
        #            self.header_forward[result_typemap.c_type] = True

        if result_is_const:
            fmt_func.c_const = "const "
        else:
            fmt_func.c_const = ""

        if CXX_subprogram == "subroutine":
            fmt_result = fmt_func
            fmt_pattern = fmt_func
        else:
            fmt_result0 = node._fmtresult
            fmt_result = fmt_result0.setdefault("fmtc", util.Scope(fmt_func))
            #            fmt_result.cxx_type = result_typemap.cxx_type  # XXX
            fmt_result.idtor = "0"  # no destructor
            fmt_result.c_var = fmt_result.C_local + fmt_result.C_result
            fmt_result.cxx_type = result_typemap.cxx_type
            if result_typemap.c_union and not is_pointer:
                # 'convert' via fields of a union
                # used with structs where casting will not work
                # XXX - maybe change to convert to pointer to C++ struct.
                is_union_scalar = True
                fmt_result.cxx_var = fmt_result.c_var
            elif result_typemap.cxx_to_c is None:
                # C and C++ are compatible
                fmt_result.cxx_var = fmt_result.c_var
            else:
                fmt_result.cxx_var = fmt_result.CXX_local + fmt_result.C_result

            if (
                result_typemap.base == "shadow"
                and not CXX_ast.is_indirect()
                and not is_ctor
            ):
                # decl: Class1 getClassNew()
                is_shadow_scalar = True
                fmt_func.cxx_rv_decl = CXX_ast.gen_arg_as_cxx(
                    name=fmt_result.cxx_var,
                    params=None,
                    continuation=True,
                    force_ptr=True,
                )
            elif is_union_scalar:
                fmt_func.cxx_rv_decl = (
                    result_typemap.c_union + " " + fmt_result.cxx_var
                )
            else:
                fmt_func.cxx_rv_decl = CXX_ast.gen_arg_as_cxx(
                    name=fmt_result.cxx_var, params=None, continuation=True
                )

            if result_typemap.base == "shadow":
                # Add an extra argument if function returns a shadow class
                shadow_arg_decl = ast.gen_arg_as_c(
                    name=fmt_result.c_var,
                    continuation=True,
                    params=None,
                    force_ptr=True,
                    remove_const=True,
                )

            if is_ctor or is_pointer:
                # The C wrapper always creates a pointer to the new instance in the ctor.
                fmt_result.cxx_member = "->"
                fmt_result.cxx_addr = ""
            else:
                fmt_result.cxx_member = "."
                fmt_result.cxx_addr = "&"
            fmt_pattern = fmt_result

        proto_list = []  # arguments for wrapper prototype
        call_list = []  # arguments to call function

        # indicate which argument contains function result, usually none
        result_arg = None
        pre_call = []  # list of temporary variable declarations
        call_code = []
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
                    # 'this' argument
                    rvast = declast.create_this_arg(
                        fmt_func.C_this, cls.typemap, is_const
                    )
                    arg = rvast.gen_arg_as_c(continuation=True)
                    proto_list.append(arg)

                    # LHS is class' cxx_to_c
                    cls_typemap = cls.typemap
                    if cls_typemap.c_to_cxx is None:
                        # This should be set in typemap.fill_shadow_typemap_defaults
                        raise RuntimeError(
                            "Wappped class does not have c_to_cxx set"
                        )
                    append_format(
                        pre_call,
                        "{c_const}{namespace_scope}{cxx_type} *{CXX_this} =\t "
                        + cls_typemap.c_to_cxx
                        + ";",
                        fmt_func,
                    )

        self.find_idtor(node.ast, result_typemap, fmt_result, None)

        if hasattr(node, "statements"):
            # Statements added to node in setup_allocatable_result.
            if "c" in node.statements:
                iblk = node.statements["c"]["result_buf"]
                need_wrapper = self.build_proto_list(
                    fmt_result,
                    ast,
                    iblk.get("buf_args", []),
                    proto_list,
                    need_wrapper,
                )
                need_wrapper = self.add_code_from_statements(
                    fmt_result, iblk, pre_call, post_call, need_wrapper
                )

        if is_shadow_scalar:
            # Allocate a new instance, then assign pointer to dereferenced cxx_var.
            append_format(
                pre_call,
                "{cxx_rv_decl} = new %s;" % result_typemap.cxx_type,
                fmt_func,
            )
            fmt_result.cxx_addr = ""
            fmt_result.idtor = result_typemap.idtor
            fmt_func.cxx_rv_decl = "*" + fmt_result.cxx_var

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

            arg_typemap = arg.typemap  # XXX - look up vector
            fmt_arg.update(arg_typemap.format)

            if arg_typemap.base == "vector":
                fmt_arg.cxx_T = arg.template_arguments[0].typemap.name

            arg_typemap, c_statements = typemap.lookup_c_statements(arg)

            fmt_arg.c_var = arg_name

            if arg.const:
                fmt_arg.c_const = "const "
            else:
                fmt_arg.c_const = ""
            arg_is_union_scalar = False
            if arg.is_indirect():  # is_pointer?
                fmt_arg.c_deref = "*"
                fmt_arg.c_member = "->"
                fmt_arg.cxx_member = "->"
                fmt_arg.cxx_addr = ""
            else:
                fmt_arg.c_deref = ""
                fmt_arg.c_member = "."
                fmt_arg.cxx_member = "."
                fmt_arg.cxx_addr = "&"
                if arg_typemap.c_union:
                    arg_is_union_scalar = True
            fmt_arg.cxx_type = arg_typemap.cxx_type
            fmt_arg.idtor = "0"
            cxx_local_var = ""

            have_idtor = False
            if c_attrs.get("_is_result", False):
                # This argument is the C function result
                arg_call = False

                # Note that result_type is void, so use arg_typemap.
                if arg_typemap.cxx_to_c is None:
                    fmt_arg.cxx_var = fmt_func.C_local + fmt_func.C_result
                else:
                    fmt_arg.cxx_var = fmt_func.CXX_local + fmt_func.C_result
                # Set cxx_var for C_finalize which evaluates in fmt_result context
                fmt_result.cxx_var = fmt_arg.cxx_var
                fmt_func.cxx_rv_decl = CXX_ast.gen_arg_as_cxx(
                    name=fmt_arg.cxx_var, params=None, continuation=True
                )

                fmt_pattern = fmt_arg
                result_arg = arg
                stmts = "result" + generated_suffix
                need_wrapper = True
                if is_pointer:
                    fmt_arg.cxx_member = "->"
                    fmt_arg.cxx_addr = ""
                else:
                    fmt_arg.cxx_member = "."
                    fmt_arg.cxx_addr = "&"

                result_return_pointer_as = c_attrs.get("deref", "")
                if result_return_pointer_as in ["pointer", "allocatable"]:
                    if not CXX_ast.is_indirect():
                        # As std::string is returned.
                        # Must allocate the std::string then assign to it via cxx_rv_decl.
                        # This allows the std::string to outlast the function return.
                        fmt_arg.cxx_addr = ""
                        fmt_arg.cxx_member = "->"
                        append_format(
                            pre_call,  # no const
                            "std::string * {cxx_var} = new std::string;",
                            fmt_arg,
                        )
                        fmt_func.cxx_rv_decl = wformat("*{cxx_var}", fmt_arg)
                        # XXX - delete string after copying its contents idtor=
                        fmt_arg.idtor = self.add_destructor(
                            fmt_arg,
                            "new_string",
                            [
                                "std::string *cxx_ptr = \treinterpret_cast<std::string *>(ptr);",
                                "delete cxx_ptr;",
                            ],
                            arg_typemap,
                        )
                        have_idtor = True

            else:
                # regular argument (not function result)
                arg_call = arg
                if arg_is_union_scalar and arg_typemap.c_to_cxx is not None:
                    # Argument is passed from Fortran to C by value.
                    # Take address of argument for cxx_var.
                    # It is dereferenced when passed to C++ to pass the value.
                    # This avoids copying the struct since only the pointer is cast.
                    #  tutorial::struct1 * SHCXX_arg =
                    #    static_cast<tutorial::struct1 *>
                    #      (static_cast<void *>(&arg));
                    # Preserves call-by-value semantics to allow C++ routine
                    # to change the value.
                    tmp = fmt_arg.c_var
                    fmt_arg.cxx_var = fmt_arg.CXX_local + fmt_arg.c_var
                    fmt_arg.c_var = "&" + tmp
                    fmt_arg.cxx_val = wformat(arg_typemap.c_to_cxx, fmt_arg)
                    fmt_arg.c_var = tmp
                    fmt_arg.cxx_decl = arg.gen_arg_as_cxx(
                        name=fmt_arg.cxx_var,
                        params=None,
                        as_ptr=True,
                        force_ptr=True,
                        continuation=True,
                    )
                    append_format(
                        pre_call, "{cxx_decl} =\t {cxx_val};", fmt_arg
                    )
                elif arg_typemap.c_to_cxx is None:
                    fmt_arg.cxx_var = fmt_arg.c_var  # compatible
                else:
                    # convert C argument to C++
                    if arg_typemap.base == 'shadow':
                        # When a shadow class is passed by value, the shadow
                        # class is passed by value and it contains a pointer
                        # to the actual class. Set force_ptr to get a pointer
                        # in the declaration.
                        # In addition, set cxx_local_var = "pointer" below
                        # in order to pass the value of the class, and not the
                        # pointer.
                        # See tutorial passClassByValue.
                        force_ptr = True
                    else:
                        force_ptr = False
                    fmt_arg.cxx_var = fmt_arg.CXX_local + fmt_arg.c_var
                    fmt_arg.cxx_val = wformat(arg_typemap.c_to_cxx, fmt_arg)
                    fmt_arg.cxx_decl = arg.gen_arg_as_cxx(
                        name=fmt_arg.cxx_var,
                        params=None,
                        as_ptr=True,
                        force_ptr=force_ptr,
                        continuation=True,
                    )
                    append_format(
                        pre_call, "{cxx_decl} =\t {cxx_val};", fmt_arg
                    )

                    if arg.is_indirect():
                        # Only pointers can be passed in and must cast to another pointer.
                        # By setting cxx_local_var=pointer, it will be dereferenced
                        # correctly when passed to C++.
                        # base==string will have a pre_call block which sets cxx_local_var
                        cxx_local_var = "pointer"
                    elif arg_typemap.base == 'shadow':
                        cxx_local_var = "pointer"

                stmts = "intent_" + c_attrs["intent"] + arg.stmts_suffix

            intent_blk = c_statements.get(stmts, {})

            need_wrapper = self.build_proto_list(
                fmt_arg,
                arg,
                intent_blk.get("buf_args", self._default_buf_args),
                proto_list,
                need_wrapper,
            )

            # Add any code needed for intent(IN).
            # Usually to convert types.
            # For example, convert char * to std::string
            # Skip input arguments generated by F_string_result_as_arg
            if "cxx_local_var" in intent_blk:
                cxx_local_var = intent_blk["cxx_local_var"]
                fmt_arg.cxx_var = fmt_arg.C_argument + fmt_arg.c_var
            #                    fmt_arg.cxx_var = fmt_arg.CXX_local + fmt_arg.c_var
            # This uses C_local or CXX_local for arguments.
            #                if 'cxx_T' in fmt_arg:
            #                    fmt_arg.cxx_var = fmt_func.CXX_local + fmt_arg.c_var
            #                elif arg_typemap.cxx_to_c is None:
            #                    fmt_arg.cxx_var = fmt_func.C_local + fmt_arg.c_var
            #                else:
            #                    fmt_arg.cxx_var = fmt_func.CXX_local + fmt_arg.c_var
            if cxx_local_var == "scalar":
                fmt_arg.cxx_member = "."
            elif cxx_local_var == "pointer":
                fmt_arg.cxx_member = "->"

            if self.language == "c":
                fmt_arg.cxx_cast_to_void_ptr = wformat(
                    "{cxx_addr}{cxx_var}", fmt_arg
                )
            elif arg.const:
                # cast away constness
                fmt_arg.cxx_type = arg_typemap.cxx_type
                fmt_arg.cxx_cast_to_void_ptr = wformat(
                    "static_cast<void *>\t(const_cast<"
                    "{cxx_type} *>\t({cxx_addr}{cxx_var}))",
                    fmt_arg,
                )
            else:
                fmt_arg.cxx_cast_to_void_ptr = wformat(
                    "static_cast<void *>({cxx_addr}{cxx_var})", fmt_arg
                )

            if not have_idtor:
                self.find_idtor(arg, arg_typemap, fmt_arg, intent_blk)

            need_wrapper = self.add_code_from_statements(
                fmt_arg, intent_blk, pre_call, post_call, need_wrapper
            )
            self.add_statements_headers(intent_blk)

            if arg_call:
                # Collect arguments to pass to wrapped function.
                # Skips result_as_arg argument.
                if arg_is_union_scalar:
                    # Pass by value
                    call_list.append("*" + fmt_arg.cxx_var)
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
                    call_list.append("*" + fmt_arg.cxx_var)
                else:
                    call_list.append(fmt_arg.cxx_var)

            if arg_typemap.c_header:
                # include any dependent header in generated header
                self.header_typedef_nodes[arg_typemap.name] = arg_typemap
            if arg_typemap.cxx_header:
                # include any dependent header in generated source
                self.header_impl_include[arg_typemap.cxx_header] = True
        #            if arg_typemap.forward:
        #                # create forward references for other types being wrapped
        #                # i.e. This argument is another wrapped type
        #                self.header_forward[arg_typemap.c_type] = True

        if shadow_arg_decl:
            # Add argument for shadow result.
            proto_list.append(shadow_arg_decl)

        fmt_func.C_call_list = ",\t ".join(call_list)

        fmt_func.C_prototype = options.get(
            "C_prototype", ",\t ".join(proto_list)
        )

        if node.return_this:
            fmt_func.C_return_type = "void"
        elif is_dtor:
            fmt_func.C_return_type = "void"
        elif result_typemap.base == "shadow":
            # Return pointer to capsule_data. It contains pointer to results.
            fmt_func.C_return_type = result_typemap.c_type + " *"
        elif fmt_func.C_custom_return_type:
            pass  # fmt_func.C_return_type = fmt_func.C_return_type
        elif ast.return_pointer_as == "scalar":
            fmt_func.C_return_type = ast.gen_arg_as_c(
                name=None, as_scalar=True, params=None, continuation=True
            )
        else:
            fmt_func.C_return_type = ast.gen_arg_as_c(
                name=None, params=None, continuation=True
            )

        post_call_pattern = []
        if node.C_error_pattern is not None:
            C_error_pattern = node.C_error_pattern + generated_suffix
            if C_error_pattern in self.patterns:
                post_call_pattern.append("// C_error_pattern")
                append_format(
                    post_call_pattern,
                    self.patterns[C_error_pattern],
                    fmt_pattern,
                )
        if post_call_pattern:
            need_wrapper = True
            fmt_func.C_post_call_pattern = "\n".join(post_call_pattern)

        # generate the C body
        C_return_code = "return;"
        if is_ctor:
            # Always create a pointer to the instance.
            fmt_func.cxx_rv_decl = (
                result_typemap.cxx_type + " *" + fmt_result.cxx_var
            )
            append_format(
                call_code,
                "{cxx_rv_decl} =\t new {namespace_scope}"
                "{cxx_type}({C_call_list});",
                fmt_func,
            )
            if result_typemap.cxx_to_c is not None:
                fmt_func.c_rv_decl = (
                    result_typemap.c_type + " *" + fmt_result.c_var
                )
                fmt_result.c_val = wformat(result_typemap.cxx_to_c, fmt_result)
            fmt_result.c_type = result_typemap.c_type
            fmt_result.idtor = "0"
            self.header_impl_include["<stdlib.h>"] = True  # for malloc
            # XXX - similar to c_statements.result
            append_format(
                post_call,
                "{c_var}->addr = {c_val};\n" "{c_var}->idtor = {idtor};",
                fmt_result,
            )
            C_return_code = wformat("return {c_var};", fmt_result)
        elif is_dtor:
            append_format(
                call_code,
                "delete {CXX_this};\n" "{C_this}->addr = NULL;",
                fmt_func,
            )
        elif CXX_subprogram == "subroutine":
            append_format(
                call_code,
                "{CXX_this_call}{function_name}"
                "{CXX_template}(\t{C_call_list});",
                fmt_func,
            )
        else:
            added_call_code = False

            if result_arg is None:
                # Return result from function
                # (It was not passed back in an argument)
                if self.language == "c":
                    pass
                elif result_typemap.base == "shadow":
                    # c_statements.post_call creates return value
                    if result_is_const:
                        # cast away constness
                        fmt_result.cxx_type = result_typemap.cxx_type
                        fmt_result.cxx_cast_to_void_ptr = wformat(
                            "static_cast<void *>\t(const_cast<"
                            "{cxx_type} *>\t({cxx_addr}{cxx_var}))",
                            fmt_result,
                        )
                    else:
                        fmt_result.cxx_cast_to_void_ptr = wformat(
                            "static_cast<void *>({cxx_addr}{cxx_var})",
                            fmt_result,
                        )
                elif is_union_scalar:
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
                        post_call, "{c_rv_decl} =\t {c_val};", fmt_result
                    )

                c_statements = result_typemap.c_statements
                intent_blk = c_statements.get("result" + ast.stmts_suffix, {})
                self.add_statements_headers(intent_blk)

                need_wrapper = self.add_code_from_statements(
                    fmt_result, intent_blk, pre_call, post_call, need_wrapper
                )
                if util.append_format_cmds(
                    call_code, intent_blk, "call", fmt_result
                ):
                    need_wrapper = True
                    added_call_code = True
                # XXX release rv if necessary

            if not added_call_code:
                if is_union_scalar:
                    # Call function within {}'s to assign to first field of union.
                    append_format(
                        call_code,
                        "{cxx_rv_decl} =\t {{{CXX_this_call}{function_name}"
                        "{CXX_template}(\t{C_call_list})}};",
                        fmt_func,
                    )
                else:
                    append_format(
                        call_code,
                        "{cxx_rv_decl} =\t {CXX_this_call}{function_name}"
                        "{CXX_template}(\t{C_call_list});",
                        fmt_func,
                    )

            if C_subprogram == "function":
                # Note: A C function may be converted into a Fortran subroutine
                # subprogram when the result is returned in an argument.
                if node.ast.is_reference():
                    if result_typemap.base in ["shadow", "string"]:
                        C_return_code = wformat("return {c_var};", fmt_result)
                    else:
                        # Return address of reference i.e. a pointer.
                        C_return_code = wformat("return &{c_var};", fmt_result)
                else:
                    C_return_code = wformat("return {c_var};", fmt_result)

        if fmt_func.inlocal("C_finalize" + generated_suffix):
            # maybe check C_finalize up chain for accumulative code
            # i.e. per class, per library.
            finalize_line = fmt_func.get("C_finalize" + generated_suffix)
            need_wrapper = True
            post_call.append("{")
            post_call.append("    // C_finalize")
            util.append_format_indent(post_call, finalize_line, fmt_result)
            post_call.append("}")

        if fmt_func.inlocal("C_return_code"):
            need_wrapper = True
            C_return_code = wformat(fmt_func.C_return_code, fmt_func)
        elif is_union_scalar:
            fmt_func.C_return_code = wformat("return {cxx_var}.c;", fmt_result)
        elif ast.return_pointer_as == "scalar":
            # dereference pointer to return scalar
            fmt_func.C_return_code = wformat("return *{cxx_var};", fmt_result)
        else:
            fmt_func.C_return_code = C_return_code

        if pre_call:
            fmt_func.C_pre_call = "\n".join(pre_call)
        fmt_func.C_call_code = "\n".join(call_code)
        if post_call:
            fmt_func.C_post_call = "\n".join(post_call)

        splicer_code = self.splicer_stack[-1].get(fmt_func.function_name, None)
        if fmt_func.inlocal("C_code"):
            need_wrapper = True
            C_code = [1, wformat(fmt_func.C_code, fmt_func), -1]
        elif splicer_code:
            need_wrapper = True
            C_code = splicer_code
        else:
            # copy-out values, clean up
            C_code = [1]
            C_code.extend(pre_call)
            C_code.extend(call_code)
            C_code.extend(post_call_pattern)
            C_code.extend(post_call)
            C_code.append(fmt_func.C_return_code)
            C_code.append(-1)

        if need_wrapper:
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
            if options.debug:
                impl.append("// " + node.declgen)
                if options.debug_index:
                    impl.append("// function_index=%d" % node._function_index)
            if options.doxygen and node.doxygen:
                self.write_doxygen(impl, node.doxygen)
            if node.cpp_if:
                impl.append("#" + node.cpp_if)
            if options.literalinclude:
                append_format(impl, "// start {C_name}", fmt_func)
            append_format(
                impl, "{C_return_type} {C_name}(\t{C_prototype})", fmt_func
            )
            impl.append("{")
            self._create_splicer(
                fmt_func.underscore_name +
                fmt_func.function_suffix +
                fmt_func.template_suffix,
                impl,
                C_code,
            )
            impl.append("}")
            if options.literalinclude:
                append_format(impl, "// end {C_name}", fmt_func)
            if node.cpp_if:
                impl.append("#endif  // " + node.cpp_if)
        else:
            # There is no C wrapper, have Fortran call the function directly.
            fmt_func.C_name = node.ast.name

    def write_capsule_code(self):
        """Write a function used to delete memory when C/C++
        memory is deleted.
        """
        library = self.newlibrary
        options = library.options

        self.c_helper["capsule_data_helper"] = True
        fmt = library.fmtdict

        self.header_impl_include.update(self.capsule_include)
        self.header_impl_include[fmt.C_header_utility] = True

        append_format(
            self.shared_proto_c,
            "\nvoid {C_memory_dtor_function}\t({C_capsule_data_type} *cap);",
            fmt,
        )

        output = self.impl
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
            self.header_impl_include["<stdlib.h>"] = True  # for free
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

        # Add header for NULL.
        self.header_impl_include["<stdlib.h>"] = True
        output.append(
            "cap->addr = NULL;\n"
            "cap->idtor = 0;  // avoid deleting again\n"
            "-}"
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
            if var_typemap and var_typemap.cxx_header:
                for include in var_typemap.cxx_header.split():
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

    def find_idtor(self, ast, atypemap, fmt, intent_blk):
        """Find the destructor based on the typemap.
        idtor = index of destructor.

        Only arguments have idtor's.
        For example,
            int * foo() +owner(caller)
        will convert to
            void foo(context+owner(caller) )

        Args:
            ast -
            atypemap - typemap.Typemap
            fmt -
            intent_blk -
        """

        if intent_blk:
            destructor_name = intent_blk.get("destructor_name", None)
            if destructor_name:
                # Use destructor in typemap to remove intermediate objects
                # e.g. std::vector
                destructor_name = wformat(destructor_name, fmt)
                if destructor_name not in self.capsule_code:
                    del_lines = []
                    util.append_format_cmds(
                        del_lines, intent_blk, "destructor", fmt
                    )
                    fmt.idtor = self.add_capsule_code(
                        destructor_name, atypemap, del_lines
                    )
                else:
                    fmt.idtor = self.capsule_code[destructor_name][0]
                return

        owner = ast.attrs.get("owner", default_owner)
        free_pattern = ast.attrs.get("free_pattern", None)
        if owner == "library":
            # Library owns memory, do not let user release.
            pass
        elif not ast.is_pointer():
            # Non-pointers do not return dynamic memory.
            pass
        elif free_pattern is not None:
            # free_pattern attribute.
            fmt.idtor = self.add_destructor(
                fmt, free_pattern, [self.patterns[free_pattern]], None
            )
        elif atypemap.idtor != "0":
            # Return cached value.
            fmt.idtor = atypemap.idtor
        elif atypemap.cxx_to_c:
            # A C++ native type (std::string, std::vector)
            # XXX - vector does not assign cxx_to_c
            fmt.idtor = self.add_destructor(
                fmt,
                atypemap.cxx_type,
                [
                    "{cxx_type} *cxx_ptr =\t reinterpret_cast<{cxx_type} *>(ptr);",
                    "delete cxx_ptr;",
                ],
                atypemap,
            )
            atypemap.idtor = fmt.idtor
        else:
            # A POD type
            fmt.idtor = self.add_destructor(
                fmt,
                atypemap.cxx_type,
                [
                    "{cxx_type} *cxx_ptr =\t reinterpret_cast<{cxx_type} *>(ptr);",
                    "free(cxx_ptr);",
                ],
                atypemap,
            )
            atypemap.idtor = fmt.idtor
