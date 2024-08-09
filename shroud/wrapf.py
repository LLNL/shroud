# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Generate Fortran bindings for C++ code.


Variables prefixes used by generated code:
SHAPE_ Array variable with shape for use with c_f_pointer.

"""
from __future__ import print_function
from __future__ import absolute_import

import collections
import copy
import os
import re

from . import error
from . import declast
from .declstr import gen_decl, gen_decl_noparams
from . import fcfmt
from . import statements
from . import todict
from . import typemap
from . import whelpers
from . import util
from .statements import get_func_bind, get_arg_bind
from .util import wformat, append_format

default_arg_template = """if (present({f_var})) then
+{fc_var} = {f_var}-
else
+{fc_var} = {default_value}-
endif"""

# force : boolean
#    Create generic interface even if only one function.
# functions : list
#    List of function nodes in generic interface.
GenericFunction = collections.namedtuple("GenericTuple", ["force", "cls", "functions"])
                

class Wrapf(util.WrapperMixin, fcfmt.FillFormat):
    """Generate Fortran bindings.
    """

    def __init__(self, newlibrary, config, splicers):
        self.newlibrary = newlibrary
        self.symtab = newlibrary.symtab
        self.patterns = newlibrary.patterns
        self.config = config
        self.log = config.log
        self._init_splicer(splicers)
        self.comment = "!"
        self.cont = " &"
        self.linelen = newlibrary.options.F_line_length
        self.doxygen_begin = "!>"
        self.doxygen_cont = "!!"
        self.doxygen_end = "!<"
        self.file_list = []
        self.shared_helper = config.fc_shared_helpers  # Shared between Fortran and C.
        self.cursor = error.get_cursor()
        ModuleInfo.newlibrary = newlibrary

    def wrap_library(self):
        fmt_library = self.newlibrary.fmtdict
        fmt_library.F_result_clause = ""
        fmt_library.F_pure_clause = ""
        fmt_library.F_C_result_clause = ""
        fmt_library.F_C_pure_clause = ""

        node = self.newlibrary.wrap_namespace
        fileinfo = ModuleInfo(node)
        self.wrap_namespace(node, fileinfo, top=True)

#        for info in self.file_list:
#            self.write_module(info)
        self.write_c_helper()

    def wrap_namespace(self, node, fileinfo, top=False):
        """Wrap a library or namespace.

        Parameters
        ----------
        node : ast.LibraryNode, ast.NamespaceNode
        fileinfo : ModuleInfo
        top  : True if library module, else namespace module.
        """
        options = node.options
        self.wrap_class_method_option(node.functions, fileinfo)

        self._push_splicer("class")
        for cls in node.classes:
            cls_options = cls.options
            if not cls.wrap.fortran:
                continue
            fileinfo.begin_class()

            # how to decide module name, module per class
            #            module_name = cls.options.setdefault('module_name', name.lower())
            if cls.wrap_as == "struct":
                self.wrap_struct(cls, fileinfo)
            else:
                self.wrap_class(cls, fileinfo)
        self._pop_splicer("class")

        if node.functions or node.typedefs or node.enums:
            fileinfo.begin_class()  # clear out old class info
            node.F_module_dependencies = []

            self.wrap_enums(node, fileinfo)
            self.wrap_typedefs(node, fileinfo)

            self._push_splicer("function")
            self.wrap_functions(None, node.functions, fileinfo)
            self._pop_splicer("function")

        do_write = top or not node.options.F_flatten_namespace
        if do_write:
            self._create_splicer("additional_functions", fileinfo.impl, blank=True)
            self._create_splicer("additional_declarations", fileinfo.user_declarations, blank=True)

        if top:
            # have one namespace level, then replace name each time
            self._push_splicer("namespace")
            self._push_splicer("XXX") # placer holder
        for ns in node.namespaces:
            if not ns.wrap.fortran:
                continue
            if ns.options.F_flatten_namespace:
                self.wrap_namespace(ns, fileinfo)
            else:
                # Skip file component in scope_file for splicer name.
                self._update_splicer_top("::".join(ns.scope_file[1:]))
                nsinfo = ModuleInfo(ns)
                self.wrap_namespace(ns, nsinfo)
        if top:
            self._pop_splicer("XXX")  # This name will not match since it is replaced.
            self._pop_splicer("namespace")
        else:
            # restore namespace splicer
            self._update_splicer_top("::".join(node.scope_file[1:]))

        if do_write:
            self.write_module(fileinfo)

    def wrap_class_method_option(self, functions, fileinfo):
        """Gather up info for option.class_method.
        """
        for node in functions:
            options = node.options
            if not node.wrap.fortran:
                continue
            if not options.class_method:
                continue
            fmt_func = node.fmtdict
            type_bound_part = fileinfo.method_type_bound_part.setdefault(
                options.class_method, [])
            if fmt_func.F_name_function == fmt_func.F_name_impl:
                append_format(type_bound_part,
                              "procedure :: {F_name_impl}",
                              fmt_func)
            else:
                append_format(type_bound_part,
                              "procedure :: {F_name_function} => {F_name_impl}",
                              fmt_func)

    def wrap_struct(self, node, fileinfo):
        """A struct must be bind(C)-able. i.e. all POD.
        No methods.

        Args:
            node - ast.ClassNode
            fileinfo - ModuleInfo
        """
        self.log.write("class {0.name}\n".format(node))
        ntypemap = node.typemap

        options = node.options
        fmt_class = node.fmtdict

        fmt_class.F_derived_name = ntypemap.f_derived_type

        # type declaration
        output = fileinfo.f_type_decl
        output.append("")
        self._push_splicer(fmt_class.cxx_class)

        output.append("")
        if options.literalinclude:
            output.append("! start derived-type " +
                          fmt_class.F_derived_name)
        append_format(output, "type, bind(C) :: {F_derived_name}+", fmt_class)
        for var in node.variables:
            ast = var.ast
            declarator = ast.declarator
            ntypemap = ast.typemap
            if declarator.is_indirect():
                append_format(output, "type(C_PTR) :: {variable_name}", var.fmtdict)
                self.set_f_module(fileinfo.module_use,
                                  "iso_c_binding", "C_PTR")
            else:
                output.append(gen_arg_as_fortran(ast, local=True))
                self.update_f_module(
                    fileinfo.module_use,
                    ntypemap.i_module or ntypemap.f_module,
                    var.fmtdict
                )  # XXX - self.module_imports?
        append_format(output, "-end type {F_derived_name}", fmt_class)
        if options.literalinclude:
            output.append("! end derived-type " +
                          fmt_class.F_derived_name)
        self._pop_splicer(fmt_class.cxx_class)

    def wrap_class(self, node, fileinfo):
        """Wrap a class for Fortran.

        Args:
            node - ast.ClassNode.
            fileinfo - ModuleInfo
        """
        cursor = self.cursor
        cursor.push_phase("Wrapf.wrap_class")
        cursor.push_node(node)

        self.log.write("class {}\n".format(node.name_instantiation or node.name))

        options = node.options
        fmt_class = node.fmtdict

        fmt_class.F_derived_name = node.typemap.f_derived_type
        fmt_class.f_capsule_data_type = node.typemap.f_capsule_data_type

        # wrap methods
        self._push_splicer(fmt_class.cxx_class)
        self._create_splicer("module_use", fileinfo.use_stmts)

        self.wrap_enums(node, fileinfo)
        self.wrap_typedefs(node, fileinfo)

        if node.cpp_if:
            fileinfo.impl.append("#" + node.cpp_if)
        if node.cpp_if:
            fileinfo.c_interface.append("#" + node.cpp_if)
        self._push_splicer("method")
        self.wrap_functions(node, node.functions, fileinfo)
        self._pop_splicer("method")
        if node.cpp_if:
            fileinfo.c_interface.append("#endif")

        if not node.baseclass:
            # subclasses share these functions.
            self.write_object_get_set(node, fileinfo)
        self._create_splicer("additional_functions", fileinfo.impl, blank=True)
        self._pop_splicer(fmt_class.cxx_class)

        # type declaration
        self._push_splicer(fmt_class.cxx_class)
        # XXX - make members private later, but not now for debugging.

        # One capsule type per class.
        # Necessary since a type in one module may be used by another module.
        f_type_decl = fileinfo.f_type_decl
        f_type_decl.append("")
        if node.cpp_if:
            f_type_decl.append("#" + node.cpp_if)
        fileinfo.add_f_helper(["capsule_data_helper"], fmt_class)

        if options.literalinclude:
            f_type_decl.append("! start derived-type " +
                               fmt_class.F_derived_name)
        if node.baseclass:
            # Only single inheritance supported.
            # Base class already contains F_derived_member.
            fmt_class.F_derived_member_base = node.baseclass[0][2].typemap.f_derived_type
        elif options.class_baseclass:
            # Used with wrap_struct_as=class.
            baseclass = node.parent.ast.unqualified_lookup(options.class_baseclass)
            if not baseclass:
                cursor.warning("Unknown class '{}' in option.class_baseclass".format(options.class_baseclass))
                fmt_class.F_derived_member_base = "===>F_derived_member_base<==="
            else:
                fmt_class.F_derived_member_base = baseclass.typemap.f_derived_type
        if fmt_class.F_derived_member_base:
            append_format(
                f_type_decl,
                "type, extends({F_derived_member_base}) :: {F_derived_name}+",
                fmt_class,
            )
        else:
            append_format(
                f_type_decl,
                "type {F_derived_name}\n+"
                "type({F_capsule_data_type}) :: {F_derived_member}",
                fmt_class,
            )
        self.set_f_module(
            fileinfo.module_use, "iso_c_binding", "C_PTR", "C_NULL_PTR"
        )
        self._create_splicer("component_part", f_type_decl)
        f_type_decl.append("-contains+")
        f_type_decl.extend(fileinfo.type_bound_part)
        if node.name in fileinfo.method_type_bound_part:
            # option.class_method methods with wrap_struct_as=class.
            f_type_decl.extend(fileinfo.method_type_bound_part[node.name])

        # Look for generics
        # splicer to extend generic
        #        self._push_splicer('generic')
        for key in sorted(fileinfo.f_type_generic.keys()):
            force, cls, methods = fileinfo.f_type_generic[key]
            if force or len(methods) > 1:

                # Look for any cpp_if declarations
                any_cpp_if = False
                for function in methods:
                    if function.cpp_if:
                        any_cpp_if = True
                        break

                if any_cpp_if:
                    # If using cpp, add a generic line for each function
                    # to avoid conditional/continuation problems.
                    for function in methods:
                        if function.cpp_if:
                            f_type_decl.append("#" + function.cpp_if)
                        f_type_decl.append(
                            "generic :: {} => {}".format(
                                key, function.fmtdict.F_name_function
                            )
                        )
                        if function.cpp_if:
                            f_type_decl.append("#endif")
                else:
                    parts = ["generic :: ", key, " => "]
                    for function in methods:
                        parts.append(function.fmtdict.F_name_function)
                        parts.append(", ")
                    del parts[-1]
                    f_type_decl.append("\t".join(parts))
        #                    self._create_splicer(key, self.f_type_decl)
        #        self._pop_splicer('generic')

        self._create_splicer("type_bound_procedure_part", f_type_decl)
        append_format(f_type_decl, "-end type {F_derived_name}", fmt_class)
        if options.literalinclude:
            f_type_decl.append("! end derived-type " +
                               fmt_class.F_derived_name)
        if node.cpp_if:
            f_type_decl.append("#endif")

        self._pop_splicer(fmt_class.cxx_class)
        if node.cpp_if:
            fileinfo.impl.append("#endif")


        # overload operators
        if node.cpp_if:
            fileinfo.operator_impl.append("#" + node.cpp_if)
        self.overload_compare(
            node, fileinfo,
            fmt_class,
            ".eq.",
            fmt_class.F_name_scope + "eq",
            wformat(
                "c_associated"
                "(a%{F_derived_member}%addr, b%{F_derived_member}%addr)",
                fmt_class,
            ),
        )
        #        self.overload_compare(fmt_class, '==', fmt_class.F_name_scope + 'eq', None)
        self.overload_compare(
            node, fileinfo,
            fmt_class,
            ".ne.",
            fmt_class.F_name_scope + "ne",
            wformat(
                ".not. c_associated"
                "(a%{F_derived_member}%addr, b%{F_derived_member}%addr)",
                fmt_class,
            ),
        )
        if node.cpp_if:
            fileinfo.operator_impl.append("#endif")

    #        self.overload_compare(fmt_class, '/=', fmt_class.F_name_scope + 'ne', None)
        cursor.pop_node(node)
        cursor.pop_phase("Wrapf.wrap_class")

    def wrap_typedefs(self, node, fileinfo):
        """Wrap all typedefs in a splicer block.

        Do not create typedef parameter for class or struct.
        The TYPE will be renamed instead.

        Args:
            node - ast.ClassNode, ast.LibraryNode
            fileinfo - ModuleInfo
        """
        self._push_splicer("typedef")
        for typ in node.typedefs:
            if typ.ast.typemap.sgroup in ["shadow", "struct"]:
                continue
            self.wrap_typedef(typ, fileinfo)
        self._pop_splicer("typedef")

    def wrap_typedef(self, node, fileinfo):
        """Wrap a typedef declaration.

        Simple typedefs are mapped to a parameter for the
        corresponding kind.

        Args:
            node - ast.TypedefNode.
            fileinfo - ModuleInfo
        """
        options = node.options
        fmtdict = node.fmtdict
        self.log.write("typedef {0.name}\n".format(node))

        declarator = node.ast.declarator

        if "f" in node.splicer:
            F_code = None
            F_force = node.splicer["f"]
        elif declarator.is_function_pointer():
            # Create an abstract interface
            self.add_abstract_interface(node, node.ast, fileinfo, name=node.name)
            return
        else:
            F_code = ["integer, parameter :: {} = {}".format(
                node.fmtdict.F_name_typedef, node.f_kind)]
            F_force = None
            
        # Any USE statements for typedef value (ex. C_INT)
        self.update_f_module(fileinfo.module_use, node.f_module, fmtdict)
        
        output = fileinfo.typedef_impl
        output.append("")
        if options.literalinclude:
            output.append("! start typedef " + node.name)
        append_format(output, "! typedef {namespace_scope}{class_scope}{typedef_name}", fmtdict)
        self._create_splicer(node.name, output, F_code, F_force)
        if options.literalinclude:
            output.append("! end typedef " + node.name)

    def wrap_enums(self, node, fileinfo):
        """Wrap all enums in a splicer block

        Args:
            node - ast.ClassNode, ast.LibraryNode
            fileinfo - ModuleInfo
        """
        self._push_splicer("enum")
        for enum in node.enums:
            self.wrap_enum(None, enum, fileinfo)
        self._pop_splicer("enum")

    def wrap_enum(self, cls, node, fileinfo):
        """Wrap an enumeration.
        Create an integer parameter for each member.

        Args:
            cls - ast.ClassNode
            node - ast.EnumNode
            fileinfo - ModuleInfo
        """
        options = node.options
        ast = node.ast
        output = fileinfo.enum_impl

        fmt_enum = node.fmtdict
        fmtmembers = node._fmtmembers

        output.append("")
        if node.ast.scope:
            append_format(output, "!  enum " + node.ast.scope + " {namespace_scope}{enum_name}", fmt_enum)
        else:
            append_format(output, "!  enum {namespace_scope}{enum_name}", fmt_enum)
            append_format(
                output,
                "integer, parameter :: {F_name_enum} = {F_enum_kind}",
                fmt_enum
            )
        if "f" in node.splicer:
            F_code = None
            F_force = node.splicer["f"]
        else:
            F_code = []
            F_force = None
            for member in ast.members:
                fmt_id = fmtmembers[member.name]
                append_format(
                    F_code,
                    "integer({F_name_enum}), parameter :: {F_enum_member} = {F_value}",
                    fmt_id,
                )
        self._create_splicer(node.name, output, F_code, F_force)
        self.set_f_module(fileinfo.module_use, "iso_c_binding", fmt_enum.F_enum_kind)

    def write_object_get_set(self, node, fileinfo):
        """Write get and set methods for instance pointer.

        Args:
            node - ast.ClassNode.
            fileinfo - ModuleInfo
        """
        options = node.options
        fmt_class = node.fmtdict
        impl = fileinfo.impl
        type_bound_part = fileinfo.type_bound_part
        fmt = util.Scope(fmt_class)

        # getter
        if fmt.F_name_instance_get:
            fmt.F_name_api = fmt_class.F_name_instance_get
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            append_format(type_bound_part,
                          "procedure :: {F_name_function} => {F_name_impl}",
                          fmt)

            append_format(
                impl,
                """
! Return pointer to C++ memory.
function {F_name_impl}({F_this}) result (cxxptr)+
use iso_c_binding, only: C_PTR
class({F_derived_name}), intent(IN) :: {F_this}
type(C_PTR) :: cxxptr
cxxptr = {F_this}%{F_derived_member}%addr
-end function {F_name_impl}""",
                fmt,
            )

        # setter
        if fmt_class.F_name_instance_set:
            fmt.F_name_api = fmt_class.F_name_instance_set
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            append_format(type_bound_part,
                          "procedure :: {F_name_function} => {F_name_impl}",
                          fmt)

            # XXX - release existing pointer?
            append_format(
                impl,
                """
subroutine {F_name_impl}({F_this}, {F_derived_member})+
use iso_c_binding, only: C_PTR
class({F_derived_name}), intent(INOUT) :: {F_this}
type(C_PTR), intent(IN) :: {F_derived_member}
{F_this}%{F_derived_member}%addr = {F_derived_member}
{F_this}%{F_derived_member}%idtor = 0
-end subroutine {F_name_impl}""",
                fmt,
            )

        # associated
        if fmt_class.F_name_associated:
            fmt.F_name_api = fmt_class.F_name_associated
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            append_format(type_bound_part,
                          "procedure :: {F_name_function} => {F_name_impl}",
                          fmt)

            append_format(
                impl,
                """
function {F_name_impl}({F_this}) result (rv)+
use iso_c_binding, only: c_associated
class({F_derived_name}), intent(IN) :: {F_this}
logical rv
rv = c_associated({F_this}%{F_derived_member}%addr)
-end function {F_name_impl}""",
                fmt,
            )

        if options.F_auto_reference_count:
            # assign
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            append_format(type_bound_part,
                          "procedure :: {F_name_impl}",
                          fmt)
            append_format(type_bound_part,
                          "generic :: assignment(=) => {F_name_impl}",
                          fmt)
            append_format(
                impl,
                """
subroutine {F_name_impl}(lhs, rhs)+
use iso_c_binding, only : c_associated, c_f_pointer
class({F_derived_name}), intent(INOUT) :: lhs
class({F_derived_name}), intent(IN) :: rhs

lhs%{F_derived_ptr} = rhs%{F_derived_ptr}
if (c_associated(lhs%{F_derived_ptr})) then+
call c_f_pointer(lhs%{F_derived_ptr}, lhs%{F_derived_member})
lhs%{F_derived_member}%refcount = lhs%{F_derived_member}%refcount + 1
-else+
nullify(lhs%{F_derived_member})
-endif
-end subroutine {F_name_impl}""",
                fmt,
            )

        if options.F_auto_reference_count:
            # final
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            type_bound_part.append("final :: %s" % fmt.F_name_impl)

            append_format(
                impl,
                """
subroutine {F_name_impl}({F_this})+
use iso_c_binding, only : c_associated, C_BOOL, C_NULL_PTR
type({F_derived_name}), intent(INOUT) :: {F_this}
interface+
subroutine array_destructor(ptr, gc)\t bind(C, name="{C_memory_dtor_function}")+
use iso_c_binding, only : C_BOOL, C_INT, C_PTR
implicit none
type(C_PTR), value, intent(IN) :: ptr
logical(C_BOOL), value, intent(IN) :: gc
-end subroutine array_destructor
-end interface
if (c_associated({F_this}%{F_derived_ptr})) then+
call array_destructor({F_this}%{F_derived_ptr}, .true._C_BOOL)
{F_this}%{F_derived_ptr} = C_NULL_PTR
nullify({F_this}%{F_derived_member})
-endif
-end subroutine {F_name_impl}""",
                fmt,
            )

    def overload_compare(self, node, fileinfo, fmt_class, operator, procedure, predicate):
        """ Overload .eq. and .eq.

        Args:
            node - ast.ClassNode
            fileinfo - ModuleInfo
            fmt_class -
            operator - ".eq.", ".ne."
            procedure -
            predicate -
        """
        fmt = util.Scope(fmt_class)
        fmt.procedure = procedure
        fmt.predicate = predicate

        ops = fileinfo.operator_map.setdefault(operator, [])
        ops.append((node, procedure))

        if predicate is None:
            # .eq. and == use same function
            return

        operator = fileinfo.operator_impl
        append_format(
            operator,
            """
function {procedure}(a,b) result (rv)+
use iso_c_binding, only: c_associated
type({F_derived_name}), intent(IN) ::a,b
logical :: rv
if ({predicate}) then+
rv = .true.
-else+
rv = .false.
-endif
-end function {procedure}""",
            fmt,
        )

    def locate_c_function(self, node):
        """Look for C routine to wrap.
         Usually the same node unless it is a generated.

        The C wrapper will not be the same as the Fortran wrapper when
        there are generated function involved.
        """
        C_node = node
        while C_node._PTR_F_C_index is not None:
            assert C_node._PTR_F_C_index != C_node._function_index
            C_node = self.newlibrary.function_index[C_node._PTR_F_C_index]
        node.C_node = C_node

    def wrap_functions(self, cls, functions, fileinfo):
        """Wrap functions in list

        Wrapping involves both a C interface and a Fortran wrapper.
        For some generic functions there may be single C method with
        multiple Fortran wrappers.

        Create Fortran wrappers first.  If no real work to do,
        F_C_name will be updated to call the C function directly.

        Parameters
        ----------
        cls : ClassNode or None
        functions : list of ast.FunctionNode
        fileinfo : ModuleInfo
        """
        cursor = self.cursor

        # Find which C functions are called.
        for node in functions:
            if node.wrap.fortran:
                self.locate_c_function(node)

        cursor.push_phase("Wrapf.wrap_function_impl")
        for node in functions:
            if node.wrap.fortran:
                self.log.write("Fortran {0.declgen}\n".format(
                    node))
                self.wrap_function_impl(cls, node, fileinfo)
        cursor.pop_phase("Wrapf.wrap_function_impl")

        cursor.push_phase("Wrapf.wrap_function_interface")
        for node in functions:
            wrap = node.wrap
            if wrap.c and wrap.signature_c != wrap.signature_f:
                self.log.write("C-interface c {0.declgen}\n".format(
                    node))
                self.wrap_function_interface("c", cls, node, fileinfo)
            if wrap.fortran and node.options.F_create_bufferify_function:
                self.log.write("C-interface f {0.declgen}\n".format(
                    node))
                self.wrap_function_interface("f", cls, node, fileinfo)
        cursor.pop_phase("Wrapf.wrap_function_interface")

    def add_stmt_declaration(self, stmts, arg_f_decl, arg_f_names, fmt):
        """Add declarations from fc_statements.

        Return True if f_arg_decl found.
        """
        found = False
        if stmts.f_arg_decl:
            found = True
            for line in stmts.f_arg_decl:
                append_format(arg_f_decl, line, fmt)
        if stmts.f_arg_name:
            for aname in stmts.f_arg_name:
                append_format(arg_f_names, aname, fmt)
        return found

    def add_stmt_var(self, group, lst, fmt):
        """Add a variable fc_statements to lst.

        Return True if lines where added to lst.
        """
        if not group:
            return False
        for line in group:
            append_format(lst, line, fmt)
        return True

    def add_f_module_from_stmts(self, stmt, modules, fmt):
        """Add USE/IMPORT statements defined in stmt.

        Parameters
        ----------
        stmt : Scope
        modules : dict
            Indexed as [module][symbol]
        fmt : Scope
        """
        if stmt.f_module:
            self.update_f_module(modules, stmt.f_module, fmt)

    def add_i_module_from_stmts(self, stmt, modules, imports, fmt):
        """Add USE/IMPORT statements defined in stmt for interface.

        Parameters
        ----------
        stmt : Scope
        modules : dict
            Indexed as [module][symbol]
        imports : dict
            Indexed as [symbol]
        fmt : Scope
        """
        if stmt.i_module:
            self.update_f_module(modules, stmt.i_module, fmt)
        if stmt.i_import:
            for name in stmt.i_import:
                iname = wformat(name, fmt)
                imports[iname] = True

    def update_f_module(self, modules, f_module, fmt):
        """Aggragate the information from f_module into modules.

        sort_module_info deals with IMPORT vs USE.

        Parameters
        ----------
        modules : dictionary of dictionaries:
            modules['iso_c_bindings']['C_INT'] = True
        imports: dict
            If the module name is '--import--', add to imports.
            Useful for interfaces.
        f_module : a dictionary of lists:
            dict(iso_c_binding=['C_INT'])
        fmt : Scope
        """
        if f_module is not None:
            for mname, only in f_module.items():
                mname = wformat(mname, fmt)
                if mname == "__line__":
                    continue
                module = modules.setdefault(mname, {})
                if only:  # Empty list means no ONLY clause
                    for oname in only:
                        wname = wformat(oname, fmt)
                        module[wname] = True

    def update_f_module_helper(self, modules, f_module):
        """Aggragate the information from helper["modules"] into modules.

        Parameters
        ----------
        modules : dictionary of dictionaries:
            modules['iso_c_bindings']['C_INT'] = True
        f_module : a dictionary of lists:
            dict(iso_c_binding=['C_INT'])
        """
        if f_module is None:
            return
        for mname, only in f_module.items():
            module = modules.setdefault(mname, {})
            for oname in only:
                module[oname] = True

    def set_f_module(self, modules, mname, *only):
        """Add a module to modules.

        Args:
            modules -
            mname -
            only -
        """
        module = modules.setdefault(mname, {})
        if only:  # Empty list means no ONLY clause
            for oname in only:
                module[oname] = True

    def sort_module_info(self, modules, module_name, imports=None):
        """Return USE statements based on modules.
        Save any names which must be imported in imports to be used with
        interface blocks.

        Parameters
        ----------
        modules : dict
            { 'module name': {'symbol': True}}
        module_name : str
            The name of current module. Use IMPORT when necessary.
        imports : dict or None
            Updated with symbols from modules[module_name].
        """
        arg_f_use = []
        for mname in sorted(modules):
            only = modules[mname]
            if mname == module_name:
                if imports is not None:
                    imports.update(only)
            else:
                if only:
                    snames = sorted(only.keys())
                    arg_f_use.append(
                        "use %s, only : %s" % (mname, ", ".join(snames))
                    )
                else:
                    arg_f_use.append("use %s" % mname)
        return arg_f_use

    def dump_generic_interfaces(self, fileinfo):
        """Generate code for generic interfaces into self.generic_interface

        Args:
            fileinfo - ModuleInfo
        """
        # Look for generic interfaces
        # splicer to extend generic
        self._push_splicer("generic")
        iface = fileinfo.generic_interface
        f_function_generic = fileinfo.f_function_generic
        for key in sorted(f_function_generic.keys()):
            force, cls, generics = f_function_generic[key]
            if force or len(generics) > 1:
                self._push_splicer(key)

                # Promote cpp_if to interface scope if all are identical.
                # Useful for fortran_generic.
                iface_cpp_if = generics[0].cpp_if
                if iface_cpp_if is not None:
                    for node in generics:
                        if node.cpp_if != iface_cpp_if:
                            iface_cpp_if = None
                            break

                literalinclude = False
                for node in generics:
                    if node.options.literalinclude:
                        literalinclude = True
                        break

                iface.append("")
                if cls and cls.cpp_if:
                    iface.append("#" + cls.cpp_if)
                if iface_cpp_if:
                    iface.append("#" + iface_cpp_if)
                if literalinclude:
                    iface.append("! start generic interface " + key)
                iface.append("interface " + key)
                iface.append(1)
                if iface_cpp_if:
                    for node in generics:
                        iface.append("module procedure " + node.fmtdict.F_name_impl)
                else:
                    for node in generics:
                        if node.cpp_if:
                            iface.append("#" + node.cpp_if)
                            iface.append("module procedure " + node.fmtdict.F_name_impl)
                            iface.append("#endif")
                        else:
                            iface.append("module procedure " + node.fmtdict.F_name_impl)
                iface.append(-1)
                iface.append("end interface " + key)
                if literalinclude:
                    iface.append("! end generic interface " + key)
                if iface_cpp_if:
                    iface.append("#endif")
                if cls and cls.cpp_if:
                    iface.append("#endif")
                self._pop_splicer(key)
        self._pop_splicer("generic")

    def add_abstract_interface(self, node, arg, fileinfo, fmt_arg=None, name=None):
        """Record an abstract interface.

        Function pointers are converted to abstract interfaces.
        The interface is named after the function and the argument.

        If from a typedef, then name argument will be set.

        Args:
            node -  ast.TypedefNode  ast.FunctionNode
            arg -   declast.Declaration
            fileinfo - ModuleInfo
        """
        if fmt_arg:
            fmt = fmt_arg
        else:
            fmt = node.fmtdict
        if name is None:
            fmt_tmp = util.Scope(fmt)
            fmt_tmp.argname = arg.declarator.user_name
            # argname used in F_abstract_interface_subroutine_template
            name = wformat(
                node.options.F_abstract_interface_subprogram_template, fmt_tmp
            )
        fmt.f_abstract_interface = name
        entry = fileinfo.f_abstract_interface.get(name)
        if entry is None:
            meta = get_arg_bind(node, arg, "f").meta
            fileinfo.f_abstract_interface[name] = (node, fmt, arg, meta["fptr"])
        return name

    def dump_abstract_interfaces(self, fileinfo):
        """Generate code for abstract interfaces

        Args:
            fileinfo - ModuleInfo
        """
        self._push_splicer("abstract")
        if len(fileinfo.f_abstract_interface) > 0:
            iface = fileinfo.abstract_interface
            if not self.newlibrary.options.literalinclude2:
                iface.append("")
                iface.append("abstract interface")
                iface.append(1)

            for key in sorted(fileinfo.f_abstract_interface.keys()):
                node, fmt, arg, fptr = fileinfo.f_abstract_interface[key]
                options = node.options
                subprogram = arg.declarator.get_subprogram()
                iface.append("")
                arg_f_names = []
                arg_c_decl = []
                modules = {}  # indexed as [module][variable]
                imports = {}  # indexed as [symbol]
                fmt_result = fptr._bind["f"]["+result"].fmtdict
                for i, param in enumerate(fptr.ast.declarator.params):
                    bind = get_arg_bind(fptr, param, "f")
                    meta = bind.meta
                    stmts = bind.stmt
                    fmt_arg = bind.fmtdict
                    # See also build_arg_list_interface
                    if stmts.i_arg_decl is not None:
                        # Use explicit declaration from CStmt, both must exist.
                        for name in stmts.i_arg_names:
                            append_format(arg_f_names, name, fmt_arg)
                        for arg in stmts.i_arg_decl:
                            append_format(arg_c_decl, arg, fmt_arg)
                        self.add_i_module_from_stmts(stmts, modules, imports, fmt_arg)
                    else:
                        # XXX - convert the others later
                        name = fmt_arg.f_abstract_name
                        intent = meta["intent"]
                        arg_f_names.append(name)
                        arg_c_decl.append(bind_c(param, modules, intent=intent, name=name))

                        arg_typemap = param.typemap
                        self.update_f_module(
                            modules,
                            arg_typemap.i_module or arg_typemap.f_module,
                            fmt
                        )

                if subprogram == "function":
                    arg_c_decl.append(bind_c(fptr.ast,
                        modules, name=key, is_result=True, is_callback=True,
                        params=None))
                arguments = ",\t ".join(arg_f_names)
                if options.literalinclude:
                    iface.append("! start abstract " + key)
                if self.newlibrary.options.literalinclude2:
                    iface.append("abstract interface+")
                iface.append(
                    "{} {}({}) bind(C)".format(subprogram, key, arguments)
                )
                iface.append(1)
                arg_f_use = self.sort_module_info(modules, fmt.F_module_name, imports)
                iface.extend(arg_f_use)
                if imports:
                    iface.append("import :: " + ",\t ".join(sorted(imports.keys())))
                iface.append("implicit none")
                iface.extend(arg_c_decl)
                iface.append(-1)
                iface.append("end {} {}".format(subprogram, key))
                if self.newlibrary.options.literalinclude2:
                    iface.append("-end interface")
                if options.literalinclude:
                    iface.append("! end abstract " + key)
            if not self.newlibrary.options.literalinclude2:
                iface.append(-1)
                iface.append("")
                iface.append("end interface")
        self._pop_splicer("abstract")

    def build_arg_list_interface(
        self,
        node, fileinfo,
        fmt, meta,
        ast,
        stmts_blk,
        modules,
        imports,
        arg_c_names,
        arg_c_decl,
    ):
        """Build the Fortran interface for a c wrapper function.

        Args:
            node -
            fileinfo - ModuleInfo
            fmt -
            ast - declast.Declaration
               node.ast for subprograms
               node.declarator.params[n] for parameters
            stmts_blk - typemap.CStmts or util.Scope
            modules - Build up USE statement.
            imports - Build up IMPORT statement.
            arg_c_names - Names of arguments to subprogram.
            arg_c_decl  - Declaration for arguments.
        """
        if stmts_blk.i_arg_decl is not None:
            # Use explicit declaration from CStmt, both must exist.
            for name in stmts_blk.i_arg_names:
                append_format(arg_c_names, name, fmt)
            for arg in stmts_blk.i_arg_decl:
                append_format(arg_c_decl, arg, fmt)
            if not meta["assumedtype"]:
                self.add_i_module_from_stmts(stmts_blk, modules, imports, fmt)
        elif stmts_blk.intent == "function":
            # Functions do not pass arguments by default.
            pass
        else:
            declarator = ast.declarator
            name = declarator.user_name
            attrs = declarator.attrs
            ntypemap = declarator.typemap
            arg_c_names.append(name)
            # argument declarations
#            if meta["assumedtype"]:
#                append_format(arg_c_decl, "{f_type}{f_intent_attr} :: {i_var}{i_dimension}", fmt)
                # Dimension must be assumed shape or assumed rank
#            elif "external" in attrs:
#                # EXTERNAL is not compatible with BIND(C)
#                arg_c_decl.append("external :: {}".format(name))
#            elif ntypemap.base == "procedure":
#                if "funptr" in attrs:
#                    arg_c_decl.append(
#                        "type(C_FUNPTR), value :: {}".format(name)
#                    )
#                    self.set_f_module(modules, "iso_c_binding", "C_FUNPTR")
#                else:
#                    # abstract interface already created via typedef
#                    arg_c_decl.append(
#                        "procedure({}) :: {}".format(fmt.f_kind, name)
#                    )
#                    imports[fmt.f_kind] = True
#            elif declarator.is_function_pointer():
#                if "funptr" in attrs:
#                    arg_c_decl.append(
#                        "type(C_FUNPTR), value :: {}".format(name)
#                    )
#                    self.set_f_module(modules, "iso_c_binding", "C_FUNPTR")
#                else:
#                    absiface = fmt.f_abstract_interface
#                    arg_c_decl.append(
#                        "procedure({}) :: {}".format(absiface, name)
#                    )
#                    imports[absiface] = True
#            elif declarator.is_array() > 1:
#                # Treat too many pointers as a type(C_PTR)
#                # and let the wrapper sort it out.
#                # 'char **' uses c_in_char** as a special case.
#                append_format(arg_c_decl,
#                              "type(C_PTR), intent({f_intent}) :: {i_var}", fmt)
#                self.set_f_module(modules, "iso_c_binding", "C_PTR")
#            else:
#                arg_c_decl.append(bind_c(ast, modules, meta["intent"]))
#                arg_typemap = ast.typemap
#                self.update_f_module(
#                    modules,
#                    arg_typemap.i_module or arg_typemap.f_module,
#                    fmt
#                )

    def wrap_function_interface(self, wlang, cls, node, fileinfo):
        """Write Fortran interface for C function.

        Args:
            cls  - ast.ClassNode or None for functions
            node - ast.FunctionNode
            fileinfo - ModuleInfo

        Wrapping involves both a C interface and a Fortran wrapper.
        For some generic functions there may be single C method with
        multiple Fortran wrappers.
        XXX - xlf does not allow this.
        """
        if node.C_fortran_generic:
            return

        cursor = self.cursor
        func_cursor = cursor.push_node(node)
        options = node.options

        ast = node.ast
        declarator = ast.declarator
        subprogram = declarator.get_subprogram()
        result_typemap = ast.typemap
        is_pure = declarator.attrs.get("pure", None)
        func_is_const = declarator.func_const

        r_bind = get_func_bind(node, wlang)
        r_meta = r_bind.meta
        result_api = r_meta["api"]
        sintent = r_meta["intent"]
        
        # find subprogram type
        # compute first to get order of arguments correct.
        fmt_result = r_bind.fmtdict
        result_stmt = r_bind.stmt
        func_cursor.stmt = result_stmt
        self.fill_interface_result(cls, node, r_bind)
            
        stmts_comments = []
        if options.debug:
            if node._generated_path:
                stmts_comments.append("! Generated by %s" % " - ".join(
                    node._generated_path))
            stmts_comments.append(
                "! ----------------------------------------")
            c_decl = gen_decl_noparams(ast)
            if options.debug_index:
                stmts_comments.append("! Index:     {}".format(node._function_index))
            stmts_comments.append("! Function:  " + c_decl)
            self.document_stmts(stmts_comments, ast, result_stmt.name)

        notimplemented = result_stmt.notimplemented
        arg_c_names = []  # argument names for functions
        arg_c_decl = []  # declaraion of argument names
        modules = {}  # indexed as [module][variable]
        imports = {}  # indexed as [name]

        if cls:
            is_ctor = declarator.is_ctor
            is_static = "static" in ast.storage
            if is_ctor or is_static:
                pass
            else:
                # Add 'this' argument
                arg_c_names.append(fmt_result.C_this)
                if sintent == "dtor":
                    # dtor will modify C_this to set addr to nullptr.
                    line = "type({F_capsule_data_type}), intent(INOUT) :: {C_this}"
                else:
                    line = "type({F_capsule_data_type}), intent(IN) :: {C_this}"
                append_format(arg_c_decl, line, fmt_result)
                imports[fmt_result.F_capsule_data_type] = True

        args_all_in = True  # assume all arguments are intent(in)
        for arg in ast.declarator.params:
            # default argument's intent
            # XXX look at const, ptr
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_bind = get_arg_bind(node, arg, wlang)
            fmt_arg = arg_bind.fmtdict
            arg_stmt = arg_bind.stmt
            func_cursor.stmt = arg_stmt
            self.fill_interface_arg(cls, node, arg, arg_bind)
            
            attrs = declarator.attrs
            meta = arg_bind.meta
            if meta["hidden"]:
                continue
            intent = meta["intent"]
            if intent != "in":
                args_all_in = False
            notimplemented = notimplemented or arg_stmt.notimplemented

            if options.debug:
                stmts_comments.append(
                    "! ----------------------------------------")
                c_decl = gen_decl(arg)
                stmts_comments.append("! Argument:  " + c_decl)
                self.document_stmts(stmts_comments, arg, arg_stmt.name)
            self.build_arg_list_interface(
                node, fileinfo,
                fmt_arg, meta,
                arg,
                arg_stmt,
                modules,
                imports,
                arg_c_names,
                arg_c_decl,
            )
        # --- End loop over function parameters
        func_cursor.arg = None
        func_cursor.stmt = result_stmt

        self.build_arg_list_interface(
            node, fileinfo,
            fmt_result, r_meta,
            ast,
            result_stmt,
            modules,
            imports,
            arg_c_names,
            arg_c_decl,
        )

        # Filter out non-pure functions.
        if result_typemap.base == "shadow":
            # Functions which return shadow classes are not pure
            # since the result argument will be assigned to.
            pass
        elif fmt_result.F_C_subprogram == "function" and (
            is_pure or (func_is_const and args_all_in)
        ):
            fmt_result.F_C_pure_clause = "pure "

        fmt_result.F_C_arguments = options.get(
            "F_C_arguments", ",\t ".join(arg_c_names)
        )

        if fmt_result.F_C_subprogram == "function":
            if result_stmt.i_result_decl is not None:
                for arg in result_stmt.i_result_decl:
                    append_format(arg_c_decl, arg, fmt_result)
                self.add_i_module_from_stmts(result_stmt, modules, imports, fmt_result)
            elif result_stmt.c_return_type:
                # Return type changed by user.
                c_return_type = wformat(result_stmt.c_return_type, fmt_result)
                ntypemap = self.symtab.lookup_typemap(c_return_type)
                if ntypemap is None:
                    cursor.warning("Unknown type in c_return_type: {}".format(c_return_type))
                else:
                    arg_c_decl.append("{} :: {}".format(ntypemap.f_type, fmt_result.F_result))
                    self.update_f_module(modules, ntypemap.f_module, fmt_result)
            else:
                arg_c_decl.append(bind_c(ast, modules, is_result=True, name=fmt_result.F_result))
                self.update_f_module(
                    modules,
                    result_typemap.i_module or result_typemap.f_module,
                    fmt_result
                )

        arg_f_use = self.sort_module_info(
            modules, fmt_result.F_module_name, imports
        )

        c_interface = []
        if node.cpp_if:
            c_interface.append("#" + node.cpp_if)
        c_interface.extend(stmts_comments)
        if options.literalinclude:
            append_format(c_interface, "! start {F_C_name}", fmt_result)
        if self.newlibrary.options.literalinclude2:
            c_interface.append("interface+")
        c_interface.append(
            wformat(
                "\r{F_C_pure_clause}{F_C_subprogram} {F_C_name}"
                "(\t{F_C_arguments}){F_C_result_clause}"
                '\fbind(C, name="{C_name}")',
                fmt_result,
            )
        )
        c_interface.append(1)
        c_interface.extend(arg_f_use)
        if imports:
            c_interface.append("import :: " + ",\t ".join(sorted(imports.keys())))
        c_interface.append("implicit none")
        c_interface.extend(arg_c_decl)
        c_interface.append(-1)
        c_interface.append(wformat("end {F_C_subprogram} {F_C_name}", fmt_result))
        if self.newlibrary.options.literalinclude2:
            c_interface.append("-end interface")
        if options.literalinclude:
            append_format(c_interface, "! end {F_C_name}", fmt_result)
        if node.cpp_if:
            c_interface.append("#endif")

        code = fileinfo.c_interface
        if notimplemented:
            if options.debug:
                # Include interface which would of been generated
                code.append("")
                code.append("#if 0")
                code.append("! Not Implemented")
                code.extend(c_interface)
                code.append("#endif")
        else:
            code.append("")
            code.extend(c_interface)
            
        cursor.pop_node(node)

    def build_arg_list_impl(
        self,
        fileinfo,
        fmt,
        c_ast,
        f_ast,
        arg_typemap,
        stmts_blk,
        modules,
        arg_c_call,
        need_wrapper,
    ):
        """
        Build up code to call C wrapper.
        This includes arguments to the function in arg_c_call.
        modules may also be updated.

        Add call arguments from stmts_blk if defined,
        This is used to override the C function arguments and used
        for cases like pointers and raw/pointer/allocatable.
        Otherwise, generate from c_ast.

        Args:
            fileinfo - ModuleInfo
            fmt -
            c_ast - Abstract Syntax Tree from parser, declast.Declaration
            f_ast - Abstract Syntax Tree from parser, declast.Declaration
            arg_typemap - typemap of resolved argument  i.e. int from vector<int>
            stmts_blk - typemap.FStmts, fc_statements block.
            modules - Build up USE statement.
            arg_c_call - Arguments to C wrapper.

        return need_wrapper
        A wrapper will be needed if there is meta data.
        """
        if stmts_blk.f_arg_call is not None:
            for arg in stmts_blk.f_arg_call:
                append_format(arg_c_call, arg, fmt)
        elif stmts_blk.intent == "function":
            # Functions do not pass arguments by default.
            pass
        elif arg_typemap.f_to_c:
            need_wrapper = True
            append_format(arg_c_call, arg_typemap.f_to_c, fmt)
        # XXX            elif f_ast and (c_ast.typemap is not f_ast.typemap):
        elif f_ast and (c_ast.typemap.name != f_ast.typemap.name):
            # Used with fortran_generic
            need_wrapper = True
            append_format(arg_c_call, arg_typemap.f_cast, fmt)
            self.update_f_module(modules, arg_typemap.f_module, fmt)
        else:
            arg_c_call.append(fmt.fc_var)
        return need_wrapper

    def add_code_from_statements(
        self,
        need_wrapper,
        fmt,
        intent_blk,
        modules,
        declare=None,
        pre_call=None,
        post_call=None,
    ):
        """Add pre_call and post_call code blocks.
        Also record the helper functions they need.
        Look for blocks 'declare', 'pre_call', 'post_call'.

        Args:
            need_wrapper -
            fmt -
            intent_blk -
            modules -
            declare -
            pre_call -
            post_call -

        return need_wrapper
        A wrapper is needed if code is added.
        """
        self.add_f_module_from_stmts(intent_blk, modules, fmt)

        if declare is not None and intent_blk.f_declare:
            need_wrapper = True
            for line in intent_blk.f_declare:
                append_format(declare, line, fmt)

        if pre_call is not None and intent_blk.f_pre_call:
            need_wrapper = True
            for line in intent_blk.f_pre_call:
                append_format(pre_call, line, fmt)

        if post_call is not None and intent_blk.f_post_call:
            need_wrapper = True
            for line in intent_blk.f_post_call:
                append_format(post_call, line, fmt)

        # this catches stuff like a bool to logical conversion which
        # requires the wrapper
        need_wrapper = need_wrapper or intent_blk.f_need_wrapper
        return need_wrapper

    def wrap_function_impl(self, cls, node, fileinfo):
        """Wrap implementation of Fortran function.

        Args:
            cls - ast.ClassNode.
            node - ast.FunctionNode.
            fileinfo - ModuleInfo
        """
        cursor = self.cursor
        func_cursor = cursor.push_node(node)
        options = node.options

        # Assume that the C function can be called directly via an interface.
        # If the wrapper does any work, then set need_wraper to True
        need_wrapper = options["F_force_wrapper"]
        if node._overloaded:
            # need wrapper for generic interface
            need_wrapper = True

        C_node = node.C_node  # C wrapper to call.

        # Fortran return type
        ast = node.ast
        declarator = ast.declarator
        result_typemap = ast.typemap

        r_bind = get_func_bind(node, "f")
        r_meta = r_bind.meta
        sintent = r_meta["intent"]
        fmt_result = r_bind.fmtdict
        if C_node is node:
            fmt_result.F_C_call = C_node._bind["f"]["+result"].fmtdict.F_C_name
        else:
            # node is generated, ex fortran_generic
            # while C_node is the real function
            fmt_result.F_C_call = C_node._bind["c"]["+result"].fmtdict.F_C_name
        result_stmt = r_bind.stmt
        func_cursor.stmt = result_stmt
        self.fill_fortran_result(cls, node, r_bind)

        subprogram = fmt_result.F_subprogram
        C_subprogram = subprogram
        if result_stmt.c_return_type == "void":
            # Convert C wrapper from function to subroutine.
            C_subprogram = "subroutine"
            need_wrapper = True
            # XXX - reset result_typemap based on c_return_type?

        fileinfo.apply_helpers_from_stmts(node)

        stmts_comments = []
        if options.debug:
            if node._generated_path:
                stmts_comments.append("! Generated by %s" % " - ".join(
                    node._generated_path))
            stmts_comments.append(
                "! ----------------------------------------")
            f_decl = gen_decl_noparams(ast)
            if options.debug_index:
                stmts_comments.append("! Index:     {}".format(node._function_index))
            stmts_comments.append("! Function:  " + f_decl)
            self.document_stmts(stmts_comments, ast, result_stmt.name)
            c_decl = gen_decl_noparams(C_node.ast)
            if f_decl != c_decl:
                stmts_comments.append("! Function:  " + c_decl)

        arg_c_call = []  # arguments to C function
        arg_f_names = []  # arguments in subprogram statement
        arg_f_decl = []  # Fortran variable declarations
        declare = []
        optional = []
        pre_call = []
        call = []
        post_call = []
        modules = {}  # indexed as [module][symbol]
        imports = {}

        if cls:
            need_wrapper = True
            is_ctor = declarator.is_ctor
            is_static = "static" in ast.storage
            if is_ctor or is_static:
                pass
            else:
                # Add 'this' argument
                arg_f_names.append(fmt_result.F_this)
                arg_f_decl.append(
                    wformat("class({F_derived_name}) :: {F_this}", fmt_result)
                )
                # could use {f_to_c} but I'd rather not hide the shadow class
                arg_c_call.append(
                    wformat("{F_this}%{F_derived_member}", fmt_result)
                )
        else:
            is_ctor = False
            is_static = False

        # Fortran and C arguments may have different types (fortran generic)
        #
        # f_var - argument to Fortran function (wrapper function)
        # fc_var - argument to C function (wrapped function)
        #
        c_args = C_node.ast.declarator.params
        c_index = -1  # index into c_args
        for f_arg in ast.declarator.params:
            func_cursor.arg = f_arg

            f_declarator = f_arg.declarator
            f_attrs = f_declarator.attrs
            c_index += 1
            c_arg = c_args[c_index]

            optattr = False
            
            arg_bind = get_arg_bind(node, f_arg, "f")
            fmt_arg = arg_bind.fmtdict
            arg_stmt = arg_bind.stmt
            arg_meta = arg_bind.meta
            if arg_meta["hidden"]:
                # Argument is not passed into Fortran.
                # hidden value is only in the C wrapper.
                continue
            
            func_cursor.stmt = arg_stmt
            arg_typemap = self.fill_fortran_arg(
                cls, node, C_node, f_arg, c_arg, arg_bind)

            fileinfo.apply_helpers_from_stmts(node)
            
            implied = f_attrs.get("implied", None)
            pass_obj = f_attrs.get("pass", None)

            if arg_typemap.base == "procedure":
                # typedef function pointer
                do_use = False
                fmt_arg.f_abstract_interface = arg_typemap.f_kind
            elif f_declarator.is_function_pointer():
                do_use = False
                if "funptr" not in f_attrs:
                    absiface = self.add_abstract_interface(node, f_arg, fileinfo, fmt_arg)
            elif arg_stmt.f_module:
                do_use = False
            else:
                do_use = True

            if arg_meta["ftrim_char_in"]:
                # Pass NULL terminated string to C.
                arg_f_decl.append(
                    "character(len=*), intent(IN) :: {}".format(fmt_arg.f_var)
                )
                arg_f_names.append(fmt_arg.f_var)
                arg_c_call.append("trim({})//C_NULL_CHAR".format(fmt_arg.f_var))
                self.set_f_module(modules, "iso_c_binding", "C_NULL_CHAR")
                need_wrapper = True
                continue
            elif arg_meta["assumedtype"]:
                # Passed directly to C as a 'void *'
                arg_f_decl.append(
                    "type(*) :: {}".format(fmt_arg.f_var)
                )
                arg_f_names.append(fmt_arg.f_var)
                arg_c_call.append(fmt_arg.f_var)
                continue
            elif implied:
                # implied is computed then passed to C++.
                fmt_arg.pre_call_intent, intermediate, f_helper = ftn_implied(
                    implied, node, f_arg)
                if intermediate:
                    fmt_arg.fc_var = "SH_" + fmt_arg.f_var
                    arg_f_decl.append(gen_arg_as_fortran(f_arg,
                        name=fmt_arg.fc_var, local=True, bindc=True))
                    append_format(pre_call, "{fc_var} = {pre_call_intent}", fmt_arg)
                    arg_c_call.append(fmt_arg.fc_var)
                else:
                    arg_c_call.append(fmt_arg.pre_call_intent)
                for helper in f_helper.split():
                    fileinfo.f_helper[helper] = True
                self.update_f_module(modules, f_arg.typemap.f_module, fmt_arg)
                need_wrapper = True
                continue

            if arg_meta["optional"]:
                fmt_arg.default_value = f_arg.declarator.init
                optattr = True
            if arg_stmt.f_arg_decl:
                # Explicit declarations from fc_statements.
                self.add_stmt_declaration(
                    arg_stmt, arg_f_decl, arg_f_names, fmt_arg)
                self.add_f_module_from_stmts(arg_stmt, modules, fmt_arg)
            else:
                # Generate declaration from argument.
                intent = arg_bind.meta["intent"]
                arg_f_decl.append(gen_arg_as_fortran(f_arg,
                    intent=intent, pass_obj=pass_obj, optional=optattr))
                arg_f_names.append(fmt_arg.f_var)

            if options.debug:
                stmts_comments.append(
                    "! ----------------------------------------")
                f_decl = gen_decl(f_arg)
                stmts_comments.append("! Argument:  " + f_decl)
                self.document_stmts(stmts_comments, f_arg, arg_stmt.name)
                c_decl = gen_decl(c_arg)
                if f_decl != c_decl:
                    stmts_comments.append("! Argument:  " + c_decl)

            if do_use:
                # XXX - function pointers confuse this code
                # XXX - it adds a USE for the function pointers's return type.
                self.update_f_module(modules, arg_typemap.f_module, fmt_arg)

            # Now C function arguments
            # May have different types, like generic
            # or different attributes, like adding +len to string args
            arg_typemap = c_arg.typemap

            # Create a local variable for C if necessary.
            # The local variable fc_var is used in fc_statements. 
            if optattr:
                fmt_arg.fc_var = "SH_" + fmt_arg.f_var
                declare.append(
                    "{} {}".format(
                        arg_typemap.i_type or arg_typemap.f_type,
                        fmt_arg.fc_var,
                    )
                )
                # XXX - Reusing c_local_var logic, would have issues with bool
                append_format(optional, default_arg_template, fmt_arg)

            need_wrapper = self.build_arg_list_impl(
                fileinfo,
                fmt_arg,
                c_arg,
                f_arg,
                arg_typemap,
                arg_stmt,
                modules,
                arg_c_call,
                need_wrapper,
            )

            need_wrapper = self.add_code_from_statements(
                need_wrapper,
                fmt_arg,
                arg_stmt,
                modules,
                declare,
                pre_call,
                post_call,
            )
        # --- End loop over function parameters
        #####
        func_cursor.arg = None
        func_cursor.stmt = result_stmt

        # Add function result argument.
        need_wrapper = self.build_arg_list_impl(
            fileinfo,
            fmt_result,
            C_node.ast,
            ast,
            result_typemap,
            result_stmt,
            modules,
            arg_c_call,
            need_wrapper,
        )
        found_arg_decl_ret = self.add_stmt_declaration(
            result_stmt, arg_f_decl, arg_f_names, fmt_result)

        # Declare function return value after arguments
        # since arguments may be used to compute return value
        # (for example, string lengths).
        # Unless explicitly set by FStmts.f_arg_decl
        if subprogram == "function":
            # if func_is_const:
            #     fmt_result.F_pure_clause = 'pure '
            if not found_arg_decl_ret:
                # result_as_arg or None
                # local=True will add any character len attributes
                # e.g.  CHARACTER(LEN=30)
                arg_f_decl.append(
                    gen_arg_as_fortran(ast, name=fmt_result.F_result, local=True)
                )

            if ast.declarator.is_indirect() < 2:
                # If more than one level of indirection, will return
                # a type(C_PTR).  i.e. int ** same as void *.
                # So do not add type's f_module.
                self.update_f_module(modules, result_typemap.f_module, fmt_result)

        if node.options.class_ctor:
            # Generic constructor for C "class" (wrap_struct_as=class).
            clsnode = node.lookup_class(node.options.class_ctor)
            fmt_result.F_name_generic = clsnode.fmtdict.F_derived_name
            fileinfo.f_function_generic.setdefault(
                fmt_result.F_name_generic, GenericFunction(True, cls, [])
            ).functions.append(node)
        elif options.F_create_generic:
            # if return type is templated in C++,
            # then do not set up generic since only the
            # return type may be different (ex. getValue<T>())
            if is_ctor:
                # ctor generic do not get added as derived type generic.
                # Always create a generic, even if only one function.
                fileinfo.f_function_generic.setdefault(
                    fmt_result.F_name_generic, GenericFunction(True, cls, [])
                ).functions.append(node)
            else:
                if cls:
                    fileinfo.f_type_generic.setdefault(
                        fmt_result.F_name_generic, GenericFunction(False, cls, [])
                    ).functions.append(node)
                # If from a fortran_generic list, create generic interface.
                if node._generated == "fortran_generic":
                    force = True
                else:
                    force = False
                fileinfo.f_function_generic.setdefault(
                    fmt_result.F_name_scope + fmt_result.F_name_generic,
                    GenericFunction(force, cls, [])).functions.append(node)
        if cls:
            # Add procedure to derived type
            type_bound_part = fileinfo.type_bound_part
            if node.cpp_if:
                type_bound_part.append("#" + node.cpp_if)
            if is_static:
                append_format(type_bound_part,
                              "procedure, nopass :: {F_name_function} => {F_name_impl}",
                              fmt_result)
            elif not is_ctor:
                append_format(type_bound_part,
                              "procedure :: {F_name_function} => {F_name_impl}",
                              fmt_result)
            if node.cpp_if:
                type_bound_part.append("#endif")

        # use tabs to insert continuations
        if arg_c_call:
            fmt_result.F_arg_c_call = ",\t ".join(arg_c_call)
        fmt_result.F_arguments = options.get(
            "F_arguments", ",\t ".join(arg_f_names)
        )

        # body of function
        # XXX sname = fmt_result.F_name_impl
        sname = fmt_result.F_name_function
        F_force = None
        F_code = None
        call_list = []
        if "f" in node.splicer:
            need_wrapper = True
            F_force = node.splicer["f"]
        elif result_stmt.f_call:
            call_list = result_stmt.f_call
        elif C_subprogram == "function":
            call_list = ["{F_result} = {F_C_call}({F_arg_c_call})"]
        else:
            # XXX - statements should set this explicitly
            call_list = ["call {F_C_call}({F_arg_c_call})"]

        for line in call_list:
            append_format(call, line, fmt_result)
        if C_subprogram == "function":
            need_wrapper = self.add_code_from_statements(
                need_wrapper,
                fmt_result,
                result_stmt,
                modules,
                declare,
                pre_call,
                post_call,
            )
        elif "f" in node.fstatements:
            # Result is an argument.
            need_wrapper = self.add_code_from_statements(
                need_wrapper,
                fmt_result,
                node.fstatements["f"],
                modules,
                declare,
                pre_call,
                post_call,
            )
        else:
            need_wrapper = self.add_code_from_statements(
                need_wrapper,
                fmt_result,
                result_stmt,
                modules,
                declare,
                pre_call,
                post_call,
            )
            
        arg_f_use = self.sort_module_info(modules, fmt_result.F_module_name)

        if need_wrapper or options.debug:
            impl = []
            if node.cpp_if:
                impl.append("#" + node.cpp_if)
            impl.extend(stmts_comments)
            if options.doxygen and node.doxygen:
                self.write_doxygen(impl, node.doxygen)
            if options.literalinclude:
                append_format(impl, "! start {F_name_impl}", fmt_result)
            append_format(
                impl,
                "\r{F_subprogram} {F_name_impl}(\t"
                "{F_arguments}){F_result_clause}",
                fmt_result,
            )
            impl.append(1)
            impl.extend(arg_f_use)
            impl.extend(arg_f_decl)
            if F_code is None:
                F_code = declare + optional + pre_call + call + post_call
            self._create_splicer(sname, impl, F_code, F_force)
            impl.append(-1)
            append_format(impl, "end {F_subprogram} {F_name_impl}", fmt_result)
            if options.literalinclude:
                append_format(impl, "! end {F_name_impl}", fmt_result)
            if node.cpp_if:
                impl.append("#endif")

        if need_wrapper:
            fileinfo.impl.append("")
            fileinfo.impl.extend(impl)
        else:            
            # Call the C function directly via bind(C)
            # by changing the name of the F_C_name function.
            C_node._bind["f"]["+result"].fmtdict.F_C_name = fmt_result.F_name_impl
            if options.debug:
                # Include wrapper which would of been generated.
                fileinfo.impl.append("")
                fileinfo.impl.append("#if 0")
                fileinfo.impl.append("! Only the interface is needed")
                fileinfo.impl.extend(impl)
                fileinfo.impl.append("#endif")
        cursor.pop_node(node)

    def _gather_helper_code(self, name, done, fileinfo):
        """Add code from helpers.

        First recursively process dependent_helpers
        to add code in order.

        Args:
            name - Name of helper.
            done - Dictionary of previously processed helpers.
            fileinfo - ModuleInfo
        """
        if name in done:
            return  # avoid recursion
        done[name] = True

        helper_info = whelpers.FHelpers[name]
        if "dependent_helpers" in helper_info:
            for dep in helper_info["dependent_helpers"]:
                # check for recursion
                self._gather_helper_code(dep, done, fileinfo)

        lines = helper_info.get("derived_type", None)
        if lines:
            fileinfo.helper_derived_type.append(lines)

        lines = helper_info.get("interface", None)
        if lines:
            fileinfo.interface_lines.append(lines)

        lines = helper_info.get("source", None)
        if lines:
            fileinfo.helper_source.append(lines)

        mods = helper_info.get("modules", None)
        if mods:
            self.update_f_module_helper(fileinfo.module_use, mods)

        if "private" in helper_info:
            if not self.private_lines:
                self.private_lines.append("")
            self.private_lines.append(
                "private " + ", ".join(helper_info["private"])
            )

    def gather_helper_code(self, fileinfo):
        """Gather up all helpers requested and insert code into output.

        Add in sorted order.  However, first process dependent_helpers
        to add code in order.
        Helpers are duplicated in each module as needed.

        Args:
            fileinfo - ModuleInfo
        """
        done = {}  # Avoid duplicates by keeping track of what's been gathered.
        for name in sorted(fileinfo.f_helper.keys()):
            self._gather_helper_code(name, done, fileinfo)

        # Accumulate all C helpers for later processing.
        self.shared_helper.update(fileinfo.c_helper)

    def write_module(self, fileinfo):
        """ Write Fortran wrapper module.
        This may be for a library or a class.

        Args:
            fileinfo - ModuleInfo
        """
        node = fileinfo.node
        options = node.options
        fmt_node = node.fmtdict
        fname = fmt_node.F_impl_filename
        module_name = fmt_node.F_module_name

        output = []

        # Added headers used with Fortran preprocessor.
        for hdr in self.newlibrary.fortran_header:
            if hdr[0] == "<":
                output.append("#include %s" % hdr)
            else:
                output.append('#include "%s"' % hdr)

        self.gather_helper_code(fileinfo)

        if options.doxygen:
            self.write_doxygen_file(output, fname, node)
        self._create_splicer("file_top", output)

        output.append("module %s" % module_name)
        output.append(1)

        ntypemap = self.newlibrary.file_code.get(fname)
        if ntypemap:
            self.update_f_module(fileinfo.module_use,
                                 ntypemap.f_module, fmt_node)

        # Write use statments (classes use iso_c_binding C_PTR)
        arg_f_use = self.sort_module_info(fileinfo.module_use, module_name)
        output.extend(arg_f_use)

        self._create_splicer("module_use", output)
        output.append("implicit none")
        self._create_splicer("module_top", output, blank=True)

        output.extend(fileinfo.helper_derived_type)

        output.extend(fileinfo.enum_impl)
        output.extend(fileinfo.typedef_impl)

        # XXX output.append('! splicer push class')
        output.extend(fileinfo.f_type_decl)
        # XXX  output.append('! splicer pop class')

        # Interfaces for operator overloads
        if fileinfo.operator_map:
            ops = sorted(fileinfo.operator_map)
            for op in ops:
                output.append("")
                output.append("interface operator (%s)" % op)
                output.append(1)
                for fcn, opfcn in fileinfo.operator_map[op]:
                    if fcn.cpp_if:
                        output.append("#" + fcn.cpp_if)
                    output.append("module procedure %s" % opfcn)
                    if fcn.cpp_if:
                        output.append("#endif")
                output.append(-1)
                output.append("end interface")

        self.dump_abstract_interfaces(fileinfo)
        self.dump_generic_interfaces(fileinfo)

        fileinfo.write_module(output)
        output.append("end module %s" % module_name)

        self.config.ffiles.append(
            os.path.join(self.config.c_fortran_dir, fname)
        )
        self.write_output_file(fname, self.config.c_fortran_dir, output)

    def write_c_helper(self):
        """ Write C helper functions that will be used by the wrappers.
        """
        pass

######################################################################

class ToImplied(todict.PrintNode):
    """Convert implied expression to Fortran wrapper code.

    expression has already been checked for errors by generate.check_implied.
    Convert functions:
      size  -  PyArray_SIZE
    """

    def __init__(self, expr, func, arg):
        """
        Args:
            expr -
            func -
            arg -
        """
        super(ToImplied, self).__init__()
        self.expr = expr
        self.func = func
        self.arg = arg
        # If True, create an intermediate variable.
        # Helps with debugging, and implies a type conversion of the expression
        # to the C function argument's type.
        self.intermediate = True
        self.helper = ""  # blank delimited string of Fortran helper

    def visit_Identifier(self, node):
        argname = node.name
        if argname == "true":
            return ".TRUE._C_BOOL"
        elif argname == "false":
            return ".FALSE._C_BOOL"
        elif node.args is None:
            return argname
        # Look for functions
        elif argname == "size":
            # size(arg)
            # This expected to be assigned to a C_INT or C_LONG
            # add KIND argument to the size intrinsic
            self.intermediate = True
            argname = node.args[0].name
            arg_typemap = self.arg.typemap
            if len(node.args) > 1:
                dim = "{},".format(todict.print_node(node.args[1]))
            else:
                dim = ""
            return "size({},{}kind={})".format(argname, dim, arg_typemap.f_kind)
        elif argname == "type":
            # type(arg)
            self.intermediate = True
            self.helper = "type_defines"
            argname = node.args[0].name
            typearg = self.func.ast.declarator.find_arg_by_name(argname)
            arg_typemap = typearg.typemap
            return arg_typemap.sh_type
        elif argname == "len":
            # len(arg)
            self.intermediate = True
            argname = node.args[0].name
            arg_typemap = self.arg.typemap
            return "len({},kind={})".format(argname, arg_typemap.f_kind)
        elif argname == "len_trim":
            # len_trim(arg)
            self.intermediate = True
            argname = node.args[0].name
            arg_typemap = self.arg.typemap
            return "len_trim({},kind={})".format(argname, arg_typemap.f_kind)
        else:
            return self.param_list(node)


def ftn_implied(expr, func, arg):
    """Convert string to Fortran code.

    Args:
        expr -
        func -
        arg -
    """
    node = declast.ExprParser(expr).expression()
    visitor = ToImplied(expr, func, arg)
    return visitor.visit(node), visitor.intermediate, visitor.helper

######################################################################

class ModuleInfo(object):
    """Contains information to create a Fortran module.

    Generated lines are accumulated by this class.
    A single C declaration may need to add code in several places
    in the generated Fortran.
    """
    newlibrary = None
    def __init__(self, node):
        self.node = node
        self.module_use = {}  # Use statements for a module
        self.use_stmts = []
        self.typedef_impl = []
        self.enum_impl = []
        self.f_type_decl = []
        self.method_type_bound_part = {}
        self.c_interface = []
        self.abstract_interface = []
        self.generic_interface = []
        self.user_declarations = []
        self.impl = []  # implementation, after contains
        self.operator_impl = []
        self.operator_map = {}  # list of function names by operator
        # {'.eq.': [ 'abc', 'def'] }
        self.f_function_generic = {}  # look for generic functions
        self.f_abstract_interface = {}

        self.c_helper = {}
        self.f_helper = {}
        self.helper_derived_type = []
        self.helper_source = []
        self.private_lines = []
        self.interface_lines = []

    def begin_class(self):
        self.f_type_generic = {}  # look for generic methods
        self.type_bound_part = []

    def write_module(self, output):
        output.extend(self.abstract_interface)

        if self.c_interface:
            if not self.newlibrary.options.literalinclude2:
                output.append("")
                output.append("interface+")
                output.extend(self.c_interface)
                output.append("-end interface")
            else:
                output.extend(self.c_interface)
                
        output.extend(self.generic_interface)

        output.extend(self.private_lines)
        output.extend(self.interface_lines)
        output.extend(self.user_declarations)

        output.append(-1)
        output.append("")
        output.append("contains")
        output.append(1)

        output.extend(self.impl)

        output.extend(self.operator_impl)

        output.extend(self.helper_source)

        output.append(-1)
        output.append("")
        
    def apply_helpers_from_stmts(self, node):
        self.c_helper.update(node.helpers.get("c", {}))
        self.f_helper.update(node.helpers.get("f", {}))

    def add_f_helper(self, helpers, fmt):
        """Add a list of Fortran helpers"""
        fcfmt.add_f_helper(self.f_helper, helpers, fmt)
        
######################################################################

def locate_c_function(library, node):
    """Look for C routine to wrap.
         Usually the same node unless it is a generated.
    
        The C wrapper will not be the same as the Fortran wrapper when
        there are generated function involved.
        """
    C_node = node
    while C_node._PTR_F_C_index is not None:
        assert C_node._PTR_F_C_index != C_node._function_index
        C_node = library.function_index[C_node._PTR_F_C_index]
    node.C_node = C_node

######################################################################

def bind_c(declaration, modules, intent=None, is_result=False,
           is_callback=False, **kwargs):
    """Generate an argument used with the bind(C) interface from Fortran.

    Args:
        intent - Explicit intent 'in', 'inout', 'out'.
                 Defaults to None to use intent from attrs.
        is_callback - Abstract interface for callbacks.
                 A function which returns a pointer must
                 use type(C_PTR).

        name   - Set name explicitly, else self.name.
    """
    # XXX - callers should not need to set modules directly,
    #       this routine should set modules.
    t = []
    declarator = declaration.declarator
    attrs = declarator.attrs
    if is_result and "typedef" in declaration.storage:
        ntypemap = declarator.typemap
    else:
        ntypemap = declaration.typemap
    basedef = ntypemap

    typ = ntypemap.i_type or ntypemap.f_type
    if typ is None:
        error.cursor.warning("Type {} has no value for f_type or i_type".format(ntypemap.name))
        return "===> {} <===".format(ntypemap.name)
    if is_callback and declarator.is_indirect():
        typ = "type(C_PTR)"
        modules.setdefault("iso_c_binding", {})["C_PTR"] = True
    t.append(typ)
    if basedef.base == "procedure":
        # dummy procedure can not have intent or value.
        pass
    else:
        append_fortran_value(declaration, t, is_result)
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
        decl.append(declarator.user_name)

    if ntypemap.base == "string":
        decl.append("(*)")
    elif "dimension" in attrs:
        # Any dimension is changed to assumed-size.
        decl.append("(*)")
    elif int(attrs.get("rank",0)) > 0:
        # Any dimension is changed to assumed-size.
        decl.append("(*)")
    return "".join(decl)

######################################################################

def append_fortran_value(declaration, t, is_result=False):
    declarator = declaration.declarator
    attrs = declarator.attrs
    if is_result:
        pass
    elif attrs.get("value", False):
        t.append("value")
    else:
        is_ptr = declarator.is_indirect()
        if is_ptr:
            if declaration.typemap.name == "void":
                # This causes Fortran to dereference the C_PTR
                # Otherwise a void * argument becomes void **
                if len(declarator.pointer) == 1:
                    t.append("value")     # void *
        else:
            if declaration.typemap.sgroup in["char", "string"]:
                pass
            else:
                t.append("value")

def gen_arg_as_fortran(
    declaration,
    intent=None,
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
    declarator = declaration.declarator
    attrs = declarator.attrs
    ntypemap = declaration.typemap

    is_allocatable = False
    is_pointer = False
    deref = attrs.get("deref", None)
    if deref == "allocatable":
        is_allocatable = True
    elif deref == "pointer":
        is_pointer = True

    if ntypemap.base == "string":
        if "len" in attrs and local:
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
        t.append(ntypemap.i_type or ntypemap.f_type)
    else:
        t.append(ntypemap.f_type)

    if not local:  # must be dummy argument
        append_fortran_value(declaration, t)
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
        decl.append(declaration.declarator.user_name)

    dimension = attrs.get("dimension")
    rank = attrs.get("rank")
    if rank is not None:
        rank = int(rank)
        decl.append(fcfmt.fortran_ranks[rank])
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

