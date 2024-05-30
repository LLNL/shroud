# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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
from collections import OrderedDict, namedtuple

from . import error
from . import declast
from .declstr import gen_decl, gen_decl_noparams, gen_arg_as_c, DeclStr
from . import fcfmt
from . import todict
from . import statements
from . import typemap
from . import whelpers
from . import util
from .statements import get_func_bind, get_arg_bind
from .util import append_format, wformat

default_owner = "library"

lang_map = {"c": "C", "cxx": "C++"}

CPlusPlus = namedtuple("CPlusPlus", "start_cxx else_cxx end_cxx start_extern_c end_extern_c")
cplusplus = CPlusPlus(
    ["#ifdef __cplusplus"],      # start_cxx
    ["#else  // __cplusplus"],   # else_cxx
    ["#endif  // __cplusplus"],  # end_cxx
    ["", "#ifdef __cplusplus", 'extern "C" {', "#endif"], # start_extern_c
    ["", "#ifdef __cplusplus", "}", "#endif"],            # end_extern_c
)

class Wrapc(util.WrapperMixin, fcfmt.FillFormat):
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
        self.capsule_impl_cxx = []
        self.capsule_impl_c = []
        self.header_types = util.Header(self.newlibrary) # shared type header
        self.helper_summary = None
        # Include files required by wrapper implementations.
        self.capsule_typedef_nodes = OrderedDict()  # [typemap.name] = typemap
        self.cursor = error.get_cursor()

        global cplusplus
        if self.language == "c":
            cplusplus = CPlusPlus([], [], [], [], [])

    def _begin_output_file(self):
        """Start a new class for output"""
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
        self.typedef_impl = []
        self.enum_impl = []
        self.struct_impl_cxx = []
        self.struct_impl_c = []
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
            if not (node.wrap.c or node.wrap.fortran):
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
            if node.wrap.c:
                self.wrap_enums(ns)
                self.wrap_typedefs(ns)
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

    def wrap_typedefs(self, node):
        """Wrap all typedefs in a splicer block

        Args:
            node - ast.ClassNode, ast.LibraryNode
        """
#        if self.language == "c":
#            # No need for wrapper with C.
#            # Use typedef definition in user's header from cxx_header.
#            return
        self._push_splicer("typedef")
        for typ in node.typedefs:
            self.wrap_typedef(typ)
        self._pop_splicer("typedef")

    def wrap_enums(self, node):
        """Wrap all enums in a splicer block

        Args:
            node - ast.ClassNode, ast.LibraryNode
        """
        self._push_splicer("enum")
        for enum in node.enums:
            self.wrap_enum(enum)
        self._pop_splicer("enum")

    def wrap_functions(self, library):
        """
        Args:
            library - ast.LibraryNode
        """
        # worker function for write_file
        self.cursor.push_phase("Wrapc.wrap_function")
        self._push_splicer("function")
        for node in library.functions:
            if node.wrap.c:
                self.wrap_function("c", None, node)
            if node.wrap.fortran and node.options.F_create_bufferify_function:
                self.wrap_function("f", None, node)
        self._pop_splicer("function")
        self.cursor.pop_phase("Wrapc.wrap_function")

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

#        api = helper_info.get("api", self.language)
# XXX - For historical reasons, default to c
        api = helper_info.get("api", "c")
        scope = helper_info.get("scope", "file")
        if lang_include in helper_info:
            for include in helper_info[lang_include]:
                self.helper_include[scope][include] = True
        elif "include" in helper_info:
            for include in helper_info["include"]:
                self.helper_include[scope][include] = True

        if lang_source in helper_info:
            self.helper_summary[api][scope].append(helper_info[lang_source])
        elif "source" in helper_info:
            self.helper_summary[api][scope].append(helper_info["source"])

        proto = helper_info.get("proto")
        if proto:
            self.helper_summary[api]["proto"].append(proto)

        proto_include = helper_info.get("proto_include")
        if proto_include:
            for include in proto_include:
                self.helper_summary[api]["proto_include"][include] = True

    def gather_helper_code(self, helpers):
        """Gather up all helpers requested.

        Sort into helper_summary and helper_include.

        Parameters
        ----------
        helpers : dict
           Indexed by name of helper.
           Should be self.c_helper or self.shared_helper.
        """
        # per class
        self.helper_include = dict(file={}, cwrap_include={}, cwrap_impl={})

        self.helper_summary = dict(
            c=dict(
                proto=[],
                proto_include=OrderedDict(),
                file=[],
                cwrap_include=[],
                cwrap_impl=[],
            ),
            cxx=dict(
                proto=[],
                proto_include=OrderedDict(),
                file=[],
                cwrap_include=[],
                cwrap_impl=[],
            ),
        )
        
        done = {}  # Avoid duplicates by keeping track of what's been gathered.
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
            self.set_capsule_headers(headers)
            self.shared_helper["capsule_dtor"] = True

        self.gather_helper_code(self.shared_helper)
        
        headers.add_shroud_file(fmt.C_header_utility)
        headers.add_shroud_dict(self.helper_include["cwrap_impl"])
        headers.write_headers(output)

        if self.language == "cxx":
            output.append("")
            #            if self._create_splicer('CXX_declarations', output):
            #                write_file = True
        output.extend(cplusplus.start_extern_c)

        source = self.helper_summary["c"]["cwrap_impl"]
        if source:
            write_file = True
            output.extend(source)

        if capsule_code:
            write_file = True
            output.extend(capsule_code)

        output.extend(cplusplus.end_extern_c)

        source = self.helper_summary["cxx"]["cwrap_impl"]
        if source:
            write_file = True
            output.extend(source)

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

        headers = self.header_types
        headers.add_shroud_dict(self.helper_include["cwrap_include"])
        headers.write_headers(output, is_header=True)

        self._push_splicer("types")
        self._create_splicer('CXX_declarations', output, blank=True)
        
        if self.language == "cxx":
            output.extend(cplusplus.start_extern_c)
            self._create_splicer('C_declarations', output, blank=True)

        output.extend(self.helper_summary["c"]["cwrap_include"])
        self.write_class_capsule_structs(output)

        proto = self.helper_summary["c"]["proto"]
        if proto:
            output.append("")
            output.extend(proto)

        if self.language == "cxx":
            output.extend(["", "#ifdef __cplusplus", "}"])
            for header in self.helper_summary["cxx"]["proto_include"].keys():
                output.append("#include " + header)
            proto = self.helper_summary["cxx"]["proto"]
            if proto:
                output.append("")
                output.append("// C++ implementation prototypes")
                output.extend(proto)
            output.append("#endif")

        output.extend(["", "#endif  // " + guard])
        self._pop_splicer("util")

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
        headers.write_headers(output, is_header=True)
        
        if self.language == "cxx":
            if self._create_splicer("CXX_declarations", output, blank=True):
                write_file = True
        output.extend(cplusplus.start_extern_c)

        # ISO_Fortran_binding.h needs to be in extern "C" block.
        self.header_iface.write_headers(output)

        if self._create_splicer("C_declarations", output, blank=True):
            write_file = True

        if self.enum_impl:
            write_file = True
            output.extend(self.enum_impl)

        if self.typedef_impl:
            write_file = True
            output.extend(self.typedef_impl)

        if self.struct_impl_c:
            write_file = True
            output.extend(cplusplus.end_extern_c)
            output.extend(cplusplus.start_cxx)
            output.extend(self.struct_impl_cxx)
            output.extend(cplusplus.else_cxx)
            output.extend(self.struct_impl_c)
            output.extend(cplusplus.end_cxx)
            output.extend(cplusplus.start_extern_c)

        if self.header_proto_c:
            write_file = True
            output.extend(self.header_proto_c)
        output.extend(cplusplus.end_extern_c)
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
            if self._create_splicer("CXX_definitions", output, blank=True):
                write_file = True
            source = self.helper_summary["cxx"]["file"]
            if source:
                write_file = True
                output.extend(source)
            output.append('\nextern "C" {')

        source = self.helper_summary["c"]["file"]
        if source:
            write_file = True
            output.append("")
            output.extend(source)

        if self._create_splicer("C_definitions", output, blank=True):
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
        cxxname = node.typemap.cxx_type

        output = self.struct_impl_cxx
        output.append("using {} = {};".format(cname, cxxname))
    
        output = self.struct_impl_c
        output.extend(
            [
                "",
                "typedef struct s_{C_type_name} {C_type_name};".format(
                    C_type_name=cname
                ),
                "struct s_{C_type_name} {{".format(C_type_name=cname),
                1
            ]
        )
        for var in node.variables:
            ast = var.ast
            output.append(gen_arg_as_c(ast) + ";")
        output.extend(
            [
                -1,
                "};",
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

    def add_class_capsule_worker(self, output, fmt, literalinclude):
        output.append("")
        if literalinclude:
            append_format(output, "// start {lang} capsule {cname}", fmt)
        append_format(
            output,
            """// {lang} capsule {cname}
{cpp_if}struct s_{C_type_name} {{+
{capsule_type} *addr;     /* address of C++ memory */
int idtor;      /* index of destructor */
-}};
typedef struct s_{C_type_name} {C_type_name};{cpp_endif}""",
            fmt)
        if literalinclude:
            append_format(output, "// end {lang} capsule {cname}", fmt)
           
    def add_class_capsule_structs(self, node):
        """Create the capsule structs for a class.
        A C++ struct with the actual type,
        and a C struct with void pointer.
        """
        fmt_class = node.fmtdict
        literalinclude = node.options.literalinclude

        fmt = util.Scope(node.fmtdict)
        fmt.cname = node.typemap.c_type
        if node.cpp_if:
            fmt.cpp_if = "#" + node.cpp_if + "\n"
            fmt.cpp_endif = "\n#endif  // " + node.cpp_if
        else:
            fmt.cpp_if = ""
            fmt.cpp_endif = ""

        fmt.lang = "C"
        fmt.capsule_type = "void"
        self.add_class_capsule_worker(self.capsule_impl_c, fmt, literalinclude)

        fmt.lang = "C++"
        fmt.capsule_type = node.typemap.cxx_type
        self.add_class_capsule_worker(self.capsule_impl_cxx, fmt, literalinclude)

#        self.header_types.add_cxx_header(node)

    def write_class_capsule_structs(self, output):
        if self.capsule_impl_cxx:
            output.append("#if 0")
            #        output.extend(cplusplus.start_cxx)
            output.extend(self.capsule_impl_cxx)
            output.append("#endif")
#        output.extend(cplusplus.end_extern_c)

#        output.extend(cplusplus.start_cxx)
#        output.extend(self.capsule_impl_cxx)
#        output.extend(cplusplus.else_cxx)
        output.extend(self.capsule_impl_c)
#        if self.capsule_impl_c:
#            output.append("#ifndef __cplusplus")
#            output.append("// C version of class capsules")
#            output.extend(self.capsule_impl_c)
#            output.append("#endif")
#        output.extend(cplusplus.end_cxx)

#        output.extend(cplusplus.start_extern_c)
        
    def wrap_class(self, node):
        """
        Args:
            node - ast.ClassNode.
        """
        cursor = self.cursor
        cursor.push_node(node)

        self.log.write("class {}\n".format(node.name_instantiation or node.name))

        fmt_class = node.fmtdict
        # call method syntax
        fmt_class.CXX_this_call = fmt_class.CXX_this + "->"

        self.compute_idtor(node)
        self.add_class_capsule_structs(node)
        if node.wrap.c:
            self.wrap_enums(node)
            self.wrap_typedefs(node)

        self._push_splicer("method")
        for method in node.functions:
            if method.wrap.c:
                self.wrap_function("c", node, method)
            if method.wrap.fortran:
                self.wrap_function("f", node, method)
        self._pop_splicer("method")
        cursor.pop_node(node)

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

    def wrap_typedef(self, node):
        """Wrap a typedef declaration.

        Args:
            node - ast.TypedefNode.
        """
        options = node.options
        fmtdict = node.fmtdict
        ast = node.ast
        output = self.typedef_impl

        if "c" in node.splicer:
            C_code = None
            C_force = node.splicer["c"]
        else:
            # XXX - Should gen_arg_as_c be used here?
#            decl = node.ast.gen_decl(as_c=True, name=fmtdict.C_name_typedef,
#                                     arg_lang="c_type")
            decl = DeclStr(arg_lang="c_type", name=fmtdict.C_name_typedef).gen_decl(node.ast)
            C_code = [decl + ";"]
            C_force = None

        output.append("")
        if options.literalinclude:
            output.append("// start typedef " + node.name)
        append_format(output, "// typedef {namespace_scope}{class_scope}{typedef_name}", fmtdict)
        self._create_splicer(node.name, output, C_code, C_force)
        if options.literalinclude:
            output.append("// end typedef " + node.name)
            
    def wrap_enum(self, node):
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

        fmt_enum = node.fmtdict
        fmtmembers = node._fmtmembers

        output.append("")
        append_format(output, "//  {namespace_scope}{enum_name}", fmt_enum)
        append_format(output, "enum {C_enum_type} {{+", fmt_enum)
        if "c" in node.splicer:
            C_code = None
            C_force = node.splicer["c"]
        else:
            C_code = []
            C_force = None
            for member in ast.members:
                fmt_id = fmtmembers[member.name]
                if member.value is not None:
                    append_format(C_code, "{C_enum_member} = {C_value},", fmt_id)
                else:
                    append_format(C_code, "{C_enum_member},", fmt_id)
            C_code[-1] = C_code[-1][:-1]  # Avoid trailing comma for older compilers
        self._create_splicer(node.name, output, C_code, C_force)
        append_format(output, "-}};", fmt_enum)

    def build_proto_list(self, fmt, stmts_blk, proto_list):
        """Find prototype based on c_arg_decl in fc_statements.

        Parameters
        ----------
        fmt - util.Scope
            Format dictionary (fmt_arg or fmt_result).
        stmts_blk  - typemap.CStmts or util.Scope.
        proto_list - list
            Prototypes are appended to list.
        """
        if stmts_blk.c_arg_decl is not None:
            for arg in stmts_blk.c_arg_decl:
                append_format(proto_list, arg, fmt)
        elif stmts_blk.intent == "function":
            # Functions do not pass an argument by default.
            pass
        else:
            proto_list.append(fmt.c_proto_decl)

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

        if intent_blk.c_pre_call:
            need_wrapper = True
            # pre_call.append('// intent=%s' % intent)
            for line in intent_blk.c_pre_call:
                append_format(pre_call, line, fmt)

        if intent_blk.c_post_call:
            need_wrapper = True
            for line in intent_blk.c_post_call:
                append_format(post_call, line, fmt)

        return need_wrapper

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
        
    def wrap_function(self, wlang, cls, node):
        """Wrap a C++ function with C.

        Args:
            wlang - "c" or "f"
            cls  - ast.ClassNode or None for functions.
            node - ast.FunctionNode.
        """
        options = node.options
        cursor = self.cursor
        func_cursor = cursor.push_node(node)

        fmtlang = "fmt" + wlang

        self.log.write("C {0} {1.declgen}\n".format(
            wlang, node))

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
        while CXX_node._PTR_C_CXX_index is not None:
            CXX_node = self.newlibrary.function_index[CXX_node._PTR_C_CXX_index]
        CXX_ast = CXX_node.ast

        # C return type
        ast = node.ast
        declarator = ast.declarator
        C_subprogram = declarator.get_subprogram()
        r_attrs = declarator.attrs
        r_bind = get_func_bind(node, wlang)
        r_meta = r_bind.meta
        result_typemap = ast.typemap

        # self.impl_typedef_nodes.update(node.gen_headers_typedef) Python 3.6
        self.impl_typedef_nodes.update(node.gen_headers_typedef.items())
        header_typedef_nodes = OrderedDict()
        if ast.template_arguments:
            for targ in ast.template_arguments:
                header_typedef_nodes[targ.typemap.name] = targ.typemap
        else:
            header_typedef_nodes[result_typemap.name] = result_typemap

        stmt_indexes = []
        fmt_result= fmtargs["+result"][fmtlang]
        result_stmt = r_bind.stmt
        func_cursor.stmt = result_stmt
        stmt_indexes.append(result_stmt.index)
        if r_bind.fstmts:
            stmt_indexes.append(r_bind.fstmts)
        any_cfi = False
        if r_meta["api"] == 'cfi':
            any_cfi = True

        stmt_need_wrapper = result_stmt.c_need_wrapper

        self.fill_c_result(wlang, cls, node, result_stmt, fmt_result, CXX_ast, r_meta)

        self.c_helper.update(node.helpers.get("c", {}))
        
        stmts_comments = []
        if options.debug:
            if node._generated_path:
                stmts_comments.append("// Generated by %s" % " - ".join(
                    node._generated_path))
            stmts_comments.append(
                "// ----------------------------------------")
            if options.debug_index:
                stmts_comments.append("// Index:     {}".format(node._function_index))
            c_decl = gen_decl_noparams(ast)
            stmts_comments.append("// Function:  " + c_decl)
            self.document_stmts(stmts_comments, ast, result_stmt.name)
        
        notimplemented = result_stmt.notimplemented
        proto_list = []  # arguments for wrapper prototype
        proto_tail = []  # extra arguments at end of call
        call_list = []  # arguments to call function
        final_code = []
        return_code = []

        setup_this = []
        pre_call = []  # list of temporary variable declarations
        post_call = []

        if cls:
            # Add 'this' argument
            need_wrapper = True
            is_ctor = CXX_ast.declarator.is_ctor
            is_static = "static" in ast.storage
            if is_ctor:
                pass
            else:
                is_const = declarator.func_const
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
                            "Wapped class {} is not a shadow type"
                            .format(cls_typemap.name)
                        )
                    append_format(
                        setup_this,
                        "{c_const}{namespace_scope}{cxx_type} *{CXX_this} =\t "
                        "{cast_static}{c_const}{namespace_scope}{cxx_type} *{cast1}{c_var}->addr{cast2};",
                        fmt_func,
                    )
        # end if cls

        #    c_var      - argument to C function  (wrapper function)
        #    c_var_trim - variable with trimmed length of c_var
        #    c_var_len  - variable with length of c_var
        #    cxx_var    - argument to C++ function  (wrapped function).
        #                 Usually same as c_var but may be a new local variable
        #                 or the function result variable.

        # --- Loop over function parameters
        for arg in ast.declarator.params:
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name
            fmt_arg = fmtargs[arg_name][fmtlang]
            arg_bind = get_arg_bind(node, arg, wlang)
            c_attrs = declarator.attrs
            c_meta = arg_bind.meta
            if c_meta["api"] == 'cfi':
                any_cfi = True

            arg_typemap = arg.typemap
            self.header_impl.add_typemap_list(arg_typemap.impl_header)
            
            typemaps = statements.collect_arg_typemaps(arg)
            for ntypemap in typemaps:
#                self.header_impl.add_typemap_list(ntypemap.impl_header)
                header_typedef_nodes[ntypemap.name] = ntypemap

            hidden = c_meta["hidden"]

            arg_stmt = arg_bind.stmt
            func_cursor.stmt = arg_stmt
            stmt_indexes.append(arg_stmt.index)

            if arg_typemap.is_enum:
                # enums use the ci_type field.
                # make sure a wrapper is written, and make sure a
                # a C and C bufferify functions are created.
                need_wrapper = True
                stmt_indexes.append(wlang)
            
            self.fill_c_arg(wlang, cls, node, arg, arg_stmt, fmt_arg, c_meta, pre_call)
            self.c_helper.update(node.helpers.get("c", {}))

            notimplemented = notimplemented or arg_stmt.notimplemented
            if options.debug:
                stmts_comments.append(
                    "// ----------------------------------------")
                c_decl = gen_decl(arg)
                stmts_comments.append("// Argument:  " + c_decl)
                self.document_stmts(stmts_comments, arg, arg_stmt.name)

            if not hidden:
                self.build_proto_list(
                    fmt_arg,
                    arg_stmt,
                    proto_list,
                )

            need_wrapper = self.add_code_from_statements(
                fmt_arg, arg_stmt, pre_call, post_call, need_wrapper
            )
            stmt_need_wrapper = stmt_need_wrapper or arg_stmt.c_need_wrapper

            # Collect arguments to pass to wrapped function.
            if arg_stmt.c_arg_call:
                for arg_call in arg_stmt.c_arg_call:
                    append_format(call_list, arg_call, fmt_arg)
            elif arg_stmt.cxx_local_var == "scalar":
                if declarator.is_pointer():
                    call_list.append("&" + fmt_arg.cxx_var)
                else:
                    call_list.append(fmt_arg.cxx_var)
            elif arg_stmt.cxx_local_var == "pointer":
                if declarator.is_pointer():
                    call_list.append(fmt_arg.cxx_var)
                else:
                    call_list.append("*" + fmt_arg.cxx_var)
            elif declarator.is_reference():
                # reference to scalar  i.e. double &max
                # void tutorial::getMinMax(int &min);
                # wrapper(int *min) {
                #   tutorial::getMinMax(*min);
                #}
                call_list.append("*" + fmt_arg.cxx_var)
            else:
                call_list.append(fmt_arg.cxx_var)

        # --- End loop over function parameters
        func_cursor.arg = None
        func_cursor.stmt = result_stmt

        self.build_proto_list(
            fmt_result,
            result_stmt,
            proto_list,
        )
        
        if call_list:
            fmt_result.C_call_list = ",\t ".join(call_list)
        fmt_result.C_call_function = wformat(
            "{CXX_this_call}{function_name}"
            "{CXX_template}(\t{C_call_list})", fmt_result)

        if len(proto_list) + len(proto_tail) == 0:
            proto_list.append("void")
        fmt_result.C_prototype = options.get(
            "C_prototype", ",\t ".join(proto_list + proto_tail)
        )

        # generate the C body
        post_call_pattern = []
        if node.C_error_pattern is not None:
            if wlang == "f":
                suffix = "buf"
            else:
                suffix = None
            C_error_pattern = statements.compute_name(
                [node.C_error_pattern, suffix])
            if C_error_pattern in self.patterns:
                need_wrapper = True
                post_call_pattern.append("// C_error_pattern")
                append_format(
                    post_call_pattern,
                    self.patterns[C_error_pattern],
                    fmt_result,
                )

        if result_stmt.c_call:
            raw_call_code = result_stmt.c_call
            need_wrapper = True
        elif C_subprogram == "subroutine":
            raw_call_code = ["{C_call_function};"]
        else:
            if result_stmt.cxx_local_var is None:
                pass
            elif result_stmt.cxx_local_var == "result":
                pass
            else:
                # A C++ var is created by pre_call.
                # Assign to it directly. ex c_function_shadow_scalar
                fmt_result.cxx_addr = ""
                fmt_result.cxx_rv_decl = "*" + fmt_result.cxx_var
            
            raw_call_code = ["{cxx_rv_decl} =\t {C_call_function};"]
            # Return result from function
            converter, lang = fcfmt.find_result_converter(
                wlang, self.language, result_typemap)
            if result_stmt.c_return_type == "void":
                # Do not return C++ result in C wrapper.
                # Probably assigned to an argument.
                pass
            elif len(result_stmt.c_post_call):
                # c_var is created by the c_post_call clause or
                # it may be passed in as an argument.
                # For example, with struct and shadow.
                pass
            elif converter is not None:
                # Make intermediate c_var value if a conversion
                # is required i.e. not the same as cxx_var.
                fmt_result.c_rv_decl = gen_arg_as_c(
                    CXX_ast, name=fmt_result.c_var, add_params=False, lang=lang)
                fmt_result.c_val = wformat(converter, fmt_result)
                append_format(
                    return_code, "{c_rv_decl} =\t {c_val};", fmt_result
                )
            self.set_cxx_nonconst_ptr(ast, fmt_result)
                
            self.header_impl.add_typemap_list(result_typemap.impl_header)

        need_wrapper = self.add_code_from_statements(
            fmt_result, result_stmt, pre_call, post_call, need_wrapper
        )

        call_code = []
        for line in raw_call_code:
            append_format(call_code, line, fmt_result)

        if result_stmt.c_final:
            need_wrapper = True
            final_code.append("{+")
            final_code.append("// final")
            for line in result_stmt.c_final:
                append_format(final_code, line, fmt_result)
            final_code.append("-}")

        if result_stmt.c_return_type == "void":
            raw_return_code = []
        elif result_stmt.c_return:
            raw_return_code = result_stmt.c_return
        elif C_subprogram == "function":
            # Note: A C function may be converted into a Fortran subroutine
            # subprogram when the result is returned in an argument.
            fmt_result.c_get_value = statements.compute_return_prefix(ast)
            raw_return_code = ["return {c_get_value}{c_var};"]
        else:
            # XXX - No return for void statements.
            # XXX - Put on an option?
#            raw_return_code = ["return;"]
            raw_return_code = []
        for line in raw_return_code:
            append_format(return_code, line, fmt_result)

        splicer_list = ["c"]
        if wlang == "f":
            splicer_list.append("buf")
        splicer_name = statements.compute_name(splicer_list)
        if splicer_name in node.splicer:
            need_wrapper = True
            C_force = node.splicer[splicer_name]
            C_code = None
        else:
            # copy-out values, clean up
            C_force = None
            C_code = pre_call + call_code + post_call_pattern + \
                     post_call + final_code + return_code

        signature = ":".join(stmt_indexes)
        if options.debug_index:
            stmts_comments.append("// Signature: " + signature)

        if node.C_fortran_generic:
            # Use a previously generated C wrapper
            need_wrapper = False

        need_wrapper = need_wrapper or stmt_need_wrapper
        if wlang == "c":
            node.wrap.signature_c = signature
        elif wlang == "f":
            node.wrap.signature_f = signature

        if need_wrapper:
            impl = []
            if options.doxygen and node.doxygen:
                self.write_doxygen(impl, node.doxygen)
            if node.cpp_if:
                impl.append("#" + node.cpp_if)
            impl.extend(stmts_comments)

            if wlang == "f":
                if node.C_signature != signature:
                    mmm = get_func_bind(node, "f").meta
                    if mmm["intent"] not in ["getter", "setter"]:
                        if any_cfi:
                            fmt_result.f_c_suffix = fmt_func.C_cfi_suffix
                        else:
                            fmt_result.f_c_suffix = fmt_func.C_bufferify_suffix
            node.eval_template("C_name", fmt=fmt_result)
            node.eval_template("F_C_name", fmt=fmt_result)
            if "C_name" in node.user_fmt:
                # XXX - this needs to distinguish between wlang
                fmt_result.C_name = node.user_fmt["C_name"]
            
            if options.literalinclude:
                append_format(impl, "// start {C_name}", fmt_result)
            append_format(
                impl, "{C_return_type} {C_name}(\t{C_prototype})", fmt_result
            )
            impl.append("{+")
            impl.extend(setup_this)
            sname = wformat("{function_name}{function_suffix}{f_c_suffix}{template_suffix}",
                            fmt_result)
            self._create_splicer(sname, impl, C_code, C_force)
            impl.append("-}")
            if options.literalinclude:
                append_format(impl, "// end {C_name}", fmt_result)
            if node.cpp_if:
                impl.append("#endif  // " + node.cpp_if)

            if node.C_signature == signature:
                # Use the wrapper which has already been written
                pass
                
            elif notimplemented:
                self.impl.append("")
                self.impl.append("#if 0")
                self.impl.append("! Not Implemented")
                self.impl.extend(impl)
                self.impl.append("#endif")
            else:
                self.impl.append("")
                self.impl.extend(impl)

                self.header_typedef_nodes.update(header_typedef_nodes.items()) # Python 3.6
                self.header_proto_c.append("")
                if node.cpp_if:
                    self.header_proto_c.append("#" + node.cpp_if)
                append_format(
                    self.header_proto_c,
                    "{C_return_type} {C_name}(\t{C_prototype});",
                    fmt_result,
                )
                if node.cpp_if:
                    self.header_proto_c.append("#endif")
                node.C_signature = signature

        else:
            # There is no C wrapper, have Fortran call the function directly.
            fmt_result.C_name = node.ast.declarator.name
            # Needed to create interface for C only wrappers.
            node.eval_template("F_C_name", fmt=fmt_result)

        if wlang == "f":
            if "F_C_name" in node.user_fmt:
                fmt_result.F_C_name = node.user_fmt["F_C_name"]
            
        cursor.pop_node(node)

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

    def find_idtor(self, ast, ntypemap, fmt, intent_blk, meta):
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

        declarator = ast.declarator
        from_stmt = False
        if meta["owner"]:
            owner = meta["owner"]
        elif intent_blk.owner:
            owner = intent_blk.owner
            from_stmt = True
        else:
            owner = default_owner

        free_pattern = meta["free_pattern"]
        if owner == "library":
            # Library owns memory, do not let user release.
            pass
        elif not declarator.is_pointer() and not from_stmt:
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
