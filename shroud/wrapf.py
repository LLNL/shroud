# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Generate Fortran bindings for C++ code.


Variables prefixes used by generated code:
SHPTR_  Fortran pointer, {F_pointer}
SHAPE_ Array variable with shape for use with c_f_pointer.

"""
from __future__ import print_function
from __future__ import absolute_import

import collections
import copy
import os
import re

from . import declast
from . import todict
from . import typemap
from . import whelpers
from . import util
from .util import wformat, append_format

# convert rank to f_assumed_shape.
fortran_ranks = [
    "",
    "(:)",
    "(:,:)",
    "(:,:,:)",
    "(:,:,:,:)",
    "(:,:,:,:,:)",
    "(:,:,:,:,:,:)",
    "(:,:,:,:,:,:,:)",
]

GenericFunction = collections.namedtuple("GenericTuple", ["force", "functions"])

class Wrapf(util.WrapperMixin):
    """Generate Fortran bindings.
    """

    def __init__(self, newlibrary, config, splicers):
        self.newlibrary = newlibrary
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
        self.shared_helper = config.shared_helpers  # All accumulated helpers
        ModuleInfo.newlibrary = newlibrary

    _default_buf_args = ["arg"]

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

        Args:
            node - ast.LibraryNode, ast.NamespaceNode
            top  - True if library module, else namespace module.
        """
        options = node.options

        self._push_splicer("class")
        for cls in node.classes:
            if not cls.options.wrap_fortran:
                continue
            fileinfo.begin_class()

            # how to decide module name, module per class
            #            module_name = cls.options.setdefault('module_name', name.lower())
            if cls.as_struct:
                self.wrap_struct(cls, fileinfo)
            else:
                self.wrap_class(cls, fileinfo)
        self._pop_splicer("class")

        if node.functions or node.enums:
            fileinfo.begin_class()  # clear out old class info
            node.F_module_dependencies = []

            self.wrap_enums(node, fileinfo)

            self._push_splicer("function")
            for function in node.functions:
                self.wrap_function(None, function, fileinfo)
            self._pop_splicer("function")

        do_write = top or not node.options.F_flatten_namespace
        if do_write:
            c_interface = fileinfo.c_interface
            c_interface.append("")
            if self.newlibrary.options.literalinclude2:
                c_interface.append("interface+")
            self._create_splicer("additional_interfaces", c_interface)
            fileinfo.impl.append("")
            self._create_splicer("additional_functions", fileinfo.impl)

        if top:
            # have one namespace level, then replace name each time
            self._push_splicer("namespace")
            self._push_splicer("XXX") # placer holder
        for ns in node.namespaces:
            if not ns.options.wrap_fortran:
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
            ntypemap = ast.typemap
            if ast.is_indirect():
                append_format(output, "type(C_PTR) :: {variable_name}", var.fmtdict)
                self.set_f_module(fileinfo.module_use,
                                  "iso_c_binding", "C_PTR")
            else:
                output.append(ast.gen_arg_as_fortran())
                self.update_f_module(
                    fileinfo.module_use, {},
                    ntypemap.f_c_module or ntypemap.f_module
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

        self.log.write("class {1.name}\n".format(self, node))

        options = node.options
        fmt_class = node.fmtdict

        fmt_class.F_derived_name = node.typemap.f_derived_type

        # wrap methods
        self._push_splicer(fmt_class.cxx_class)
        self._create_splicer("module_use", fileinfo.use_stmts)

        self.wrap_enums(node, fileinfo)

        if node.cpp_if:
            fileinfo.impl.append("#" + node.cpp_if)
        if node.cpp_if:
            fileinfo.c_interface.append("#" + node.cpp_if)
        self._push_splicer("method")
        for method in node.functions:
            self.wrap_function(node, method, fileinfo)
        self._pop_splicer("method")
        if node.cpp_if:
            fileinfo.c_interface.append("#endif")

        self.write_object_get_set(node, fileinfo)
        fileinfo.impl.append("")
        self._create_splicer("additional_functions", fileinfo.impl)
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
        if options.literalinclude:
            f_type_decl.append("! start derived-type " +
                               fmt_class.F_capsule_data_type)
        append_format(
            f_type_decl,
            "type, bind(C) :: {F_capsule_data_type}\n+"
            "type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory\n"
            "integer(C_INT) :: idtor = 0       ! index of destructor\n"
            "-end type {F_capsule_data_type}",
            fmt_class,
        )
        if options.literalinclude:
            f_type_decl.append("! end derived-type " +
                               fmt_class.F_capsule_data_type)
        self.set_f_module(
            fileinfo.module_use, "iso_c_binding", "C_PTR", "C_INT", "C_NULL_PTR"
        )

        append_format(
            f_type_decl,
            "\n"
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

        # Look for generics
        # splicer to extend generic
        #        self._push_splicer('generic')
        for key in sorted(fileinfo.f_type_generic.keys()):
            force, methods = fileinfo.f_type_generic[key]
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
        if node.cpp_if:
            f_type_decl.append("#endif")

        fileinfo.c_interface.append("")
        self._create_splicer("additional_interfaces", fileinfo.c_interface)

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

    def wrap_enums(self, node, fileinfo):
        """Wrap all enums in a splicer block

        Args:
            node - ast.EnumNode
            fileinfo - ModuleInfo
        """
        self._push_splicer("enum")
        for node in node.enums:
            self.wrap_enum(None, node, fileinfo)
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
        for member in ast.members:
            fmt_id = fmtmembers[member.name]
            append_format(
                output,
                "integer(C_INT), parameter :: {F_enum_member} = {F_value}",
                fmt_id,
            )
        self.set_f_module(fileinfo.module_use, "iso_c_binding", "C_INT")

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
        fmt.underscore_name = fmt_class.F_name_instance_get
        if fmt.underscore_name:
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            type_bound_part.append(
                "procedure :: %s => %s" % (fmt.F_name_function, fmt.F_name_impl)
            )

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
        fmt.underscore_name = fmt_class.F_name_instance_set
        if fmt.underscore_name:
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            type_bound_part.append(
                "procedure :: %s => %s" % (fmt.F_name_function, fmt.F_name_impl)
            )

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
        fmt.underscore_name = fmt_class.F_name_associated
        if fmt.underscore_name:
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            type_bound_part.append(
                "procedure :: %s => %s" % (fmt.F_name_function, fmt.F_name_impl)
            )

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
            fmt.underscore_name = fmt_class.F_name_assign
            fmt.F_name_function = wformat(options.F_name_function_template, fmt)
            fmt.F_name_impl = wformat(options.F_name_impl_template, fmt)

            type_bound_part.append("procedure :: %s" % fmt.F_name_impl)
            type_bound_part.append(
                "generic :: assignment(=) => %s" % fmt.F_name_impl
            )

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
            fmt.underscore_name = fmt_class.F_name_final
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

    def wrap_function(self, cls, node, fileinfo):
        """
        Wrapping involves both a C interface and a Fortran wrapper.
        For some generic functions there may be single C method with
        multiple Fortran wrappers.

        Args:
            cls  - ast.ClassNode or None for functions
            node - ast.FunctionNode
            fileinfo - ModuleInfo
        """
        if cls:
            cls_function = "method"
        else:
            cls_function = "function"

        options = node.options
        wrap = []
        if options.wrap_c:
            wrap.append("C-interface")
        if options.wrap_fortran:
            wrap.append("Fortran")
        if not wrap:
            return

        self.log.write(", ".join(wrap))
        self.log.write(" {0} {1.declgen}\n".format(cls_function, node))

        # Create fortran wrappers first.
        # If no real work to do, call the C function directly.
        if options.wrap_fortran:
            self.wrap_function_impl(cls, node, fileinfo)
        if options.wrap_c:
            self.wrap_function_interface(cls, node, fileinfo)

    def update_f_module(self, modules, imports, f_module):
        """aggragate the information from f_module into modules.
        modules is a dictionary of dictionaries:
            modules['iso_c_bindings']['C_INT'] = True
        f_module is a dictionary of lists:
            dict(iso_c_binding=['C_INT'])

        If the module name is '--import--', add to imports.
        Useful for interfaces.

        Args:
            modules -
            imports -
            f_module -
        """
        if f_module is not None:
            for mname, only in f_module.items():
                if mname == "__line__":
                    continue
                if mname == "--import--":
                    for oname in only:
                        imports[oname] = True
                else:
                    module = modules.setdefault(mname, {})
                    if only:  # Empty list means no ONLY clause
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

        Args:
            modules -
            module_name -
            imports -
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
            force, generics = f_function_generic[key]
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
                if iface_cpp_if:
                    iface.append("#" + iface_cpp_if)
                if literalinclude:
                    iface.append("! start interface " + key)
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
                    iface.append("! end interface " + key)
                if iface_cpp_if:
                    iface.append("#endif")
                self._pop_splicer(key)
        self._pop_splicer("generic")

    def add_abstract_interface(self, node, arg, fileinfo):
        """Record an abstract interface.

        Function pointers are converted to abstract interfaces.
        The interface is named after the function and the argument.

        Args:
            node -
            arg -
            fileinfo - ModuleInfo
        """
        fmt = util.Scope(node.fmtdict)
        fmt.argname = arg.name
        name = wformat(
            node.options.F_abstract_interface_subprogram_template, fmt
        )
        entry = fileinfo.f_abstract_interface.get(name)
        if entry is None:
            fileinfo.f_abstract_interface[name] = (node, fmt, arg)
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
                node, fmt, arg = fileinfo.f_abstract_interface[key]
                ast = node.ast
                subprogram = arg.get_subprogram()
                iface.append("")
                arg_f_names = []
                arg_c_decl = []
                modules = {}  # indexed as [module][variable]
                imports = {}
                for i, param in enumerate(arg.params):
                    name = param.name
                    if name is None:
                        fmt.index = str(i)
                        name = wformat(
                            node.options.F_abstract_interface_argument_template,
                            fmt,
                        )
                    arg_f_names.append(name)
                    arg_c_decl.append(param.bind_c(name=name))

                    arg_typemap, specialize = typemap.lookup_c_statements(
                        param
                    )
                    self.update_f_module(
                        modules,
                        imports,
                        arg_typemap.f_c_module or arg_typemap.f_module,
                    )

                if subprogram == "function":
                    arg_c_decl.append(ast.bind_c(name=key, params=None))
                arguments = ",\t ".join(arg_f_names)
                if node.options.literalinclude:
                    iface.append("! start abstract " + key)
                if self.newlibrary.options.literalinclude2:
                    iface.append("abstract interface+")
                iface.append(
                    "{} {}({}) bind(C)".format(subprogram, key, arguments)
                )
                iface.append(1)
                arg_f_use = self.sort_module_info(modules, None)
                iface.extend(arg_f_use)
                if imports:
                    iface.append("import :: " + ", ".join(sorted(imports.keys())))
                iface.append("implicit none")
                iface.extend(arg_c_decl)
                iface.append(-1)
                iface.append("end {} {}".format(subprogram, key))
                if self.newlibrary.options.literalinclude2:
                    iface.append("-end interface")
                if node.options.literalinclude:
                    iface.append("! end abstract " + key)
            if not self.newlibrary.options.literalinclude2:
                iface.append(-1)
                iface.append("")
                iface.append("end interface")
        self._pop_splicer("abstract")

    def build_arg_list_interface(
        self,
        node, fileinfo,
        fmt,
        ast,
        intent_blk,
        buf_args,
        modules,
        imports,
        arg_c_names,
        arg_c_decl,
        name=None,
        intent=None,
    ):
        """Build the Fortran interface for a c wrapper function.

        Args:
            node -
            fileinfo - ModuleInfo
            fmt -
            ast - Abstract Syntax Tree from parser
               node.ast for subprograms
               node.params[n] for parameters
            intent_blk - typemap.CStmts or util.Scope
            buf_args - List of arguments/metadata to add.
            modules - Build up USE statement.
            imports - Build up IMPORT statement.
            arg_c_names - Names of arguments to subprogram.
            arg_c_decl  - Declaration for arguments.
            name    - name to override ast.name (shadow only).
            intent  - override attrs["intent"] (shadow only).
        """
        attrs = ast.attrs

        # Add implied buffer arguments to prototype
        for buf_arg in buf_args:
            if buf_arg == "arg":
                arg_c_names.append(ast.name)
                # argument declarations
                if attrs["assumedtype"]:
                    if attrs["rank"]:
                        arg_c_decl.append(
                            "type(*) :: {}(*)".format(ast.name)
                        )
                    elif attrs["dimension"]:
                        arg_c_decl.append(
                            "type(*) :: {}({})".format(
                                ast.name, attrs["dimension"])
                        )
                    else:
                        arg_c_decl.append(
                            "type(*) :: {}".format(ast.name)
                        )
                elif ast.is_function_pointer():
                    absiface = self.add_abstract_interface(node, ast, fileinfo)
                    arg_c_decl.append(
                        "procedure({}) :: {}".format(absiface, ast.name)
                    )
                    imports[absiface] = True
                elif ast.is_array() > 1:
                    # Treat too many pointers as a type(C_PTR)
                    # and let the wrapper sort it out.
                    # 'char **' uses c_char_**_in as a special case.
                    intent = ast.attrs["intent"].upper()
                    arg_c_decl.append(
                        "type(C_PTR), intent({}) :: {}".format(
                            intent, name or ast.name))
                    self.set_f_module(modules, "iso_c_binding", "C_PTR")
                else:
                    arg_c_decl.append(ast.bind_c())
                    arg_typemap = ast.typemap
                    if ast.template_arguments:
                        # If a template, use its type
                        arg_typemap = ast.template_arguments[0].typemap
                    self.update_f_module(
                        modules, imports,
                        arg_typemap.f_c_module or arg_typemap.f_module
                    )
                continue
            elif buf_arg == "shadow":
                # Do not use const or value in declaration
                # Function result arguments explicitly set to intent(out).
                arg_c_names.append(name or ast.name)
                arg_c_decl.append("{}, intent({}){} :: {}".format(
                    ast.typemap.f_c_type,
                    (intent or ast.attrs["intent"]).upper(),
                    ", value" if attrs["value"] else "",
                    name or ast.name))
                self.update_f_module(
                    modules, imports,
                    ast.typemap.f_c_module or ast.typemap.f_module
                )
                continue
            elif buf_arg == "arg_decl":
                # Use explicit declaration from CStmt.
                arg_c_names.append(name or ast.name)
                for arg in intent_blk.f_arg_decl:
                    arg_c_decl.append(arg.format(
                        c_var=name or ast.name,
                    ))
                if intent_blk.f_module:
                    self.update_f_module(
                        modules, imports, intent_blk.f_module)
                continue

            buf_arg_name = attrs[buf_arg]
            if buf_arg_name is None:
                raise RuntimeError(
                    "attr {} is missing from attrs for {}".format(
                        buf_arg, node.declgen
                    )
                )
            elif buf_arg == "size":
                arg_c_names.append(buf_arg_name)
                arg_c_decl.append(
                    "integer(C_LONG), value, intent(IN) :: %s" % buf_arg_name
                )
                self.set_f_module(modules, "iso_c_binding", "C_LONG")
            elif buf_arg == "capsule":
                arg_c_names.append(buf_arg_name)
                arg_c_decl.append(
                    "type(%s), intent(INOUT) :: %s"
                    % (fmt.F_capsule_data_type, buf_arg_name)
                )
                fileinfo.add_f_helper("capsule_data_helper", fmt)
                imports[fmt.F_capsule_data_type] = True
            elif buf_arg == "context":
                if ast.attrs["_is_result"]:
                    intent = "OUT"
                else:
                    intent = "INOUT"
                arg_c_names.append(buf_arg_name)
                arg_c_decl.append(
                    "type(%s), intent(%s) :: %s"
                    % (fmt.F_array_type, intent, buf_arg_name)
                )
                fileinfo.add_f_helper("array_context", fmt)
#                self.set_f_module(modules, 'iso_c_binding', fmt.F_array_type)
                imports[fmt.F_array_type] = True
            elif buf_arg == "len_trim":
                arg_c_names.append(buf_arg_name)
                arg_c_decl.append(
                    "integer(C_INT), value, intent(IN) :: %s" % buf_arg_name
                )
                self.set_f_module(modules, "iso_c_binding", "C_INT")
            elif buf_arg == "len":
                arg_c_names.append(buf_arg_name)
                arg_c_decl.append(
                    "integer(C_INT), value, intent(IN) :: %s" % buf_arg_name
                )
                self.set_f_module(modules, "iso_c_binding", "C_INT")
            else:
                raise RuntimeError(
                    "build_arg_list_interface: unhandled case {}".format(
                        buf_arg
                    )
                )

    def wrap_function_interface(self, cls, node, fileinfo):
        """Write Fortran interface for C function.

        Args:
            cls  - ast.ClassNode or None for functions
            node - ast.FunctionNode
            fileinfo - ModuleInfo

        Wrapping involves both a C interface and a Fortran wrapper.
        For some generic functions there may be single C method with
        multiple Fortran wrappers.
        """
        options = node.options
        fmt_func = node.fmtdict
        fmt = util.Scope(fmt_func)

        ast = node.ast
        subprogram = node.C_subprogram
        result_typemap = node.C_result_typemap
        generated_suffix = node.generated_suffix
        is_ctor = ast.is_ctor()
        is_pure = ast.attrs["pure"]
        is_static = False
        func_is_const = ast.func_const

        arg_c_names = []  # argument names for functions
        arg_c_decl = []  # declaraion of argument names
        modules = {}  # indexed as [module][variable]
        imports = {}  # indexed as [name]
        stmts_comments = []

        # find subprogram type
        # compute first to get order of arguments correct.
        if subprogram == "subroutine":
            fmt.F_C_subprogram = "subroutine"
        else:
            fmt.F_C_subprogram = "function"
            fmt.F_C_result_clause = "\fresult(%s)" % fmt.F_result

        if cls:
            is_static = "static" in ast.storage
            if is_ctor or is_static:
                pass
            else:
                # Add 'this' argument
                arg_c_names.append(fmt.C_this)
                append_format(
                    arg_c_decl,
                    "type({F_capsule_data_type}), intent(IN) :: {C_this}",
                    fmt,
                )
                imports[fmt.F_capsule_data_type] = True

        sgroup = result_typemap.sgroup
        spointer = ast.get_indirect_stmt()
        c_stmts = ["c", sgroup, spointer, "result", node.generated_suffix]
        c_result_blk = typemap.lookup_fc_stmts(c_stmts)
        c_result_blk = typemap.lookup_local_stmts(
            ["c", node.generated_suffix], c_result_blk, node)
        if options.debug:
            stmts_comments.append(
                "! ----------------------------------------")
            c_decl = ast.gen_decl(params=None)
            stmts_comments.append("! Function:  " + c_decl)
            self.document_stmts(
                stmts_comments, typemap.compute_name(c_stmts),
                c_result_blk.name)

        if c_result_blk.return_type:
            # Change a subroutine into function.
            fmt.F_C_subprogram = "function"
            fmt.F_C_result_clause = "\fresult(%s)" % fmt_func.F_result

        self.build_arg_list_interface(
            node, fileinfo,
            fmt_func,
            ast,
            c_result_blk,
            c_result_blk.buf_args,
            modules,
            imports,
            arg_c_names,
            arg_c_decl,
        )

        args_all_in = True  # assume all arguments are intent(in)
        for arg in ast.params:
            # default argument's intent
            # XXX look at const, ptr
            arg_typemap = arg.typemap
            sgroup = arg_typemap.sgroup
            arg_typemap, specialize = typemap.lookup_c_statements(arg)
            fmt.c_var = arg.name

            attrs = arg.attrs
            intent = attrs["intent"] or "inout"
            if intent != "in":
                args_all_in = False
            deref_attr = attrs["deref"]
            cdesc = "cdesc" if attrs["cdesc"] is not None else None

            spointer = arg.get_indirect_stmt()
            if attrs["_is_result"]:
                c_stmts = ["c", sgroup, spointer, "result",
                           generated_suffix, deref_attr]
            else:
                c_stmts = ["c", sgroup, spointer, intent,
                           arg.stmts_suffix, deref_attr, cdesc]
            c_stmts.extend(specialize)
            c_intent_blk = typemap.lookup_fc_stmts(c_stmts)
            if options.debug:
                stmts_comments.append(
                    "! ----------------------------------------")
                c_decl = arg.gen_decl()
                stmts_comments.append("! Argument:  " + c_decl)
                self.document_stmts(
                    stmts_comments, typemap.compute_name(c_stmts),
                    c_intent_blk.name)
            self.build_arg_list_interface(
                node, fileinfo,
                fmt,
                arg,
                c_intent_blk,
                c_intent_blk.buf_args or self._default_buf_args,
                modules,
                imports,
                arg_c_names,
                arg_c_decl,
            )

        if subprogram == "function":
            self.build_arg_list_interface(
                node, fileinfo,
                fmt_func,
                ast,
                c_result_blk,
                c_result_blk.buf_extra,
                modules,
                imports,
                arg_c_names,
                arg_c_decl,
                name=fmt_func.F_result_capsule,
                intent="out",
            )
        # Filter out non-pure functions.
        if result_typemap.base == "shadow":
            # Functions which return shadow classes are not pure
            # since the result argument will be assigned to.
            pass
        elif "context" in c_result_blk.buf_args:
            # The function context argument will be assigned to.
            pass
        elif subprogram == "function" and (
            is_pure or (func_is_const and args_all_in)
        ):
            fmt.F_C_pure_clause = "pure "

        fmt.F_C_arguments = options.get(
            "F_C_arguments", ",\t ".join(arg_c_names)
        )

        if fmt.F_C_subprogram == "function":
            return_deref_attr = ast.attrs["deref"]
            if c_result_blk.f_result_decl:
                for arg in c_result_blk.f_result_decl:
                    arg_c_decl.append(arg.format(c_var=fmt.F_result))
                if c_result_blk.f_module:
                    self.update_f_module(
                        modules, imports, c_result_blk.f_module)
            elif c_result_blk.return_cptr:
                arg_c_decl.append("type(C_PTR) %s" % fmt.F_result)
                self.set_f_module(modules, "iso_c_binding", "C_PTR")
            elif c_result_blk.return_type:
                # Return type changed by user.
                ntypemap = typemap.lookup_type(c_result_blk.return_type)
                arg_c_decl.append("{} {}".format(ntypemap.f_type, fmt.F_result))
                self.update_f_module(modules, imports,
                                     ntypemap.f_module)
            elif return_deref_attr in ["pointer", "allocatable", "raw"]:
                arg_c_decl.append("type(C_PTR) %s" % fmt.F_result)
                self.set_f_module(modules, "iso_c_binding", "C_PTR")
            else:
                arg_c_decl.append(ast.bind_c(name=fmt.F_result))
                self.update_f_module(
                    modules,
                    imports,
                    result_typemap.f_c_module or result_typemap.f_module,
                )

        arg_f_use = self.sort_module_info(
            modules, fmt_func.F_module_name, imports
        )

        c_interface = fileinfo.c_interface
        c_interface.append("")

        if node.cpp_if:
            c_interface.append("#" + node.cpp_if)
        c_interface.extend(stmts_comments)
        if options.literalinclude:
            append_format(c_interface, "! start {F_C_name}", fmt)
        if self.newlibrary.options.literalinclude2:
            c_interface.append("interface+")
        c_interface.append(
            wformat(
                "\r{F_C_pure_clause}{F_C_subprogram} {F_C_name}"
                "(\t{F_C_arguments}){F_C_result_clause}"
                '\fbind(C, name="{C_name}")',
                fmt,
            )
        )
        c_interface.append(1)
        c_interface.extend(arg_f_use)
        if imports:
            c_interface.append("import :: " + ", ".join(sorted(imports.keys())))
        c_interface.append("implicit none")
        c_interface.extend(arg_c_decl)
        c_interface.append(-1)
        c_interface.append(wformat("end {F_C_subprogram} {F_C_name}", fmt))
        if self.newlibrary.options.literalinclude2:
            c_interface.append("-end interface")
        if options.literalinclude:
            append_format(c_interface, "! end {F_C_name}", fmt)
        if node.cpp_if:
            c_interface.append("#endif")

    def build_arg_list_impl(
        self,
        fileinfo,
        fmt,
        c_ast,
        f_ast,
        arg_typemap,
        f_intent_blk,
        buf_args,
        modules,
        imports,
        arg_f_decl,
        arg_c_call,
        need_wrapper,
    ):
        """
        Build up code to call C wrapper.
        This includes arguments to the function in arg_c_call 
        and any additional declarations for local variables in arg_f_decl.
        modules and imports may also be updated.

        Add call arguments from f_intent_blk if defined,
        This is used to override the C function arguments and used
        for cases like pointers and raw/pointer/allocatable.
        Otherwise, Use buf_args from c_intent_blk.

        Args:
            fileinfo - ModuleInfo
            fmt -
            c_ast - Abstract Syntax Tree from parser, declast.Declaration
            f_ast - Abstract Syntax Tree from parser, declast.Declaration
            arg_typemap - typemap of resolved argument  i.e. int from vector<int>
            f_intent_blk - typemap.FStmts, fc_statements block.
            buf_args - List of arguments/metadata to add.
            modules - Build up USE statement.
            imports - Build up IMPORT statement.
            arg_f_decl - Additional Fortran declarations for local variables.
            arg_c_call - Arguments to C wrapper.

        return need_wrapper
        A wrapper will be needed if there is meta data.
        """
        if f_intent_blk.arg_c_call:
            for arg in f_intent_blk.arg_c_call:
                append_format(arg_c_call, arg, fmt)
            return need_wrapper

        c_attrs = c_ast.attrs

        # Add any buffer arguments
        for buf_arg in buf_args:
            if buf_arg in ["arg", "arg_decl"]:
                # Attributes   None=skip, True=use default, else use value
                if arg_typemap.f_to_c:
                    need_wrapper = True
                    append_format(arg_c_call, arg_typemap.f_to_c, fmt)
                # XXX            elif f_ast and (c_ast.typemap is not f_ast.typemap):
                elif f_ast and (c_ast.typemap.name != f_ast.typemap.name):
                    # Used with fortran_generic
                    need_wrapper = True
                    append_format(arg_c_call, arg_typemap.f_cast, fmt)
                    self.update_f_module(modules, imports,
                                         arg_typemap.f_module)
                else:
                    arg_c_call.append(fmt.c_var)
                continue
            elif buf_arg == "shadow":
                # Pass down the pointer to {F_capsule_data_type}
                need_wrapper = True
                append_format(arg_c_call, "{f_var}%{F_derived_member}", fmt)
                continue

            need_wrapper = True
            if buf_arg == "size":
                append_format(arg_c_call, "size({f_var}, kind=C_LONG)", fmt)
                self.set_f_module(modules, "iso_c_binding", "C_LONG")
            elif buf_arg == "capsule":
                append_format(
                    arg_f_decl, "type({F_capsule_type}) :: {c_var_capsule}", fmt
                )
                # Pass F_capsule_data_type field to C++.
                arg_c_call.append(fmt.c_var_capsule + "%mem")
                fileinfo.add_f_helper("capsule_data_helper", fmt)
            elif buf_arg == "context":
                append_format(
                    arg_f_decl, "type({F_array_type}) :: {c_var_context}", fmt
                )
                arg_c_call.append(fmt.c_var_context)
                #                self.set_f_module(modules, 'iso_c_binding', fmt.F_array_type)
                if c_attrs["dimension"]:
                    fmt.c_var_dimension = c_attrs["dimension"]
                fileinfo.add_f_helper("array_context", fmt)
            elif buf_arg == "len_trim":
                append_format(arg_c_call, "len_trim({f_var}, kind=C_INT)", fmt)
                self.set_f_module(modules, "iso_c_binding", "C_INT")
            elif buf_arg == "len":
                append_format(arg_c_call, "len({f_var}, kind=C_INT)", fmt)
                self.set_f_module(modules, "iso_c_binding", "C_INT")
            else:
                raise RuntimeError(
                    "build_arg_list_impl: unhandled case {}".format(buf_arg)
                )
        return need_wrapper

    def add_code_from_statements(
        self,
        need_wrapper,
        fileinfo,
        fmt,
        intent_blk,
        modules,
        imports,
        declare=None,
        pre_call=None,
        post_call=None,
    ):
        """Add pre_call and post_call code blocks.
        Also record the helper functions they need.
        Look for blocks 'declare', 'pre_call', 'post_call'.

        Args:
            need_wrapper -
            fileinfo - ModuleInfo
            fmt -
            intent_blk -
            modules -
            imports -
            declare -
            pre_call -
            post_call -

        return need_wrapper
        A wrapper is needed if code is added.
        """
        self.update_f_module(modules, imports, intent_blk.f_module)

        if declare is not None and intent_blk.declare:
            need_wrapper = True
            for line in intent_blk.declare:
                append_format(declare, line, fmt)

        if pre_call is not None and intent_blk.pre_call:
            need_wrapper = True
            for line in intent_blk.pre_call:
                append_format(pre_call, line, fmt)

        if post_call is not None and intent_blk.post_call:
            need_wrapper = True
            for line in intent_blk.post_call:
                append_format(post_call, line, fmt)

        if intent_blk.c_helper:
            fileinfo.add_c_helper(intent_blk.c_helper, fmt)
        if intent_blk.f_helper:
            fileinfo.add_f_helper(intent_blk.f_helper, fmt)
        return need_wrapper

    def set_fmt_fields(self, cls, fcn, f_ast, c_ast, fmt, modules, fileinfo,
                       is_result=False,
                       ntypemap=None):
        """
        Set format fields for ast.
        Used with arguments and results.

        Args:
            cls   - ast.ClassNode or None of enclosing class.
            fcn   - ast.FunctionNode of calling function.
            f_ast - declast.Declaration - Fortran argument
            c_ast - declast.Declaration - C argument
                  Abstract Syntax Tree of argument or result
            fmt - format dictionary
        """
        c_attrs = c_ast.attrs
        typemap.assign_buf_variable_names(c_attrs, fmt)

        if is_result:
#            ntypemap = ntypemap
            # XXX - looked up in parent
            pass
        else:
            fmt.f_intent = c_attrs["intent"].upper()
            
            ntypemap = f_ast.typemap
            if c_ast.template_arguments:
                # If a template, use its type
                ntypemap = c_ast.template_arguments[0].typemap
                fmt.cxx_T = ntypemap.name
        fmt.f_type = ntypemap.f_type
        fmt.sh_type = ntypemap.sh_type
        
        if c_attrs["context"]:
            if not fmt.c_var_context:
                fmt.c_var_context = "FIXME"

        f_attrs = f_ast.attrs
        dim = f_attrs["dimension"]
        rank = f_attrs["rank"]
        if rank is not None:
            fmt.rank = str(rank)
            if rank == 0:
                fmt.size = "1"
            else:
                fmt.size = wformat("size({f_var})", fmt)
                fmt.f_assumed_shape = fortran_ranks[rank]
        elif dim:
            visitor = ToDimension(cls, fcn, fmt)
            visitor.visit(f_ast.metaattrs["dimension"])
            rank = visitor.rank
            fmt.rank = str(rank)
            if rank > 0:
                fmt.f_assumed_shape = fortran_ranks[rank]
                fmt.f_array_allocate = "(" + ",".join(visitor.shape) + ")"
                if c_ast.attrs["context"]:
                    fmt.f_array_shape = wformat(
                        ", {c_var_context}%shape(1:{rank})", fmt)

        return ntypemap

    def wrap_function_impl(self, cls, node, fileinfo):
        """Wrap implementation of Fortran function.

        Args:
            cls - ast.ClassNode.
            node - ast.FunctionNode.
            fileinfo - ModuleInfo
        """
        options = node.options
        fmt_func = node.fmtdict

        # Assume that the C function can be called directly via an interface.
        # If the wrapper does any work, then set need_wraper to True
        need_wrapper = options["F_force_wrapper"]
        if node._overloaded:
            # need wrapper for generic interface
            need_wrapper = True

        # Look for C routine to wrap.
        # Usually the same node unless it is a generated.
        C_node = node
        generated = []
        if C_node._generated:
            generated.append(C_node._generated)
        while C_node._PTR_F_C_index is not None:
            C_node = self.newlibrary.function_index[C_node._PTR_F_C_index]
            if C_node._generated:
                generated.append(C_node._generated)
        #  #This is no longer true with the result as an argument
        #        if len(node.params) != len(C_node.params):
        #            raise RuntimeError("Argument mismatch between Fortran and C functions")

        fmt_func.F_C_call = C_node.fmtdict.F_C_name
        fmtargs = C_node._fmtargs

        # Fortran return type
        result_type = node.F_return_type
        subprogram = node.F_subprogram
        result_typemap = node.F_result_typemap
        C_subprogram = C_node.C_subprogram
        generated_suffix = C_node.generated_suffix
        ast = node.ast
        is_ctor = ast.is_ctor()
        is_dtor = ast.is_dtor()
        is_static = False

        arg_c_call = []  # arguments to C function
        arg_f_names = []  # arguments in subprogram statement
        arg_f_decl = []  # Fortran variable declarations
        declare = []
        pre_call = []
        call = []
        post_call = []
        modules = {}  # indexed as [module][variable]
        imports = {}
        stmts_comments = []

        if subprogram == "subroutine":
            fmt_result = fmt_func
            if is_dtor:
                f_stmts = ["f", "shadow", "dtor"]
                c_stmts = ["c", "shadow", "dtor"]
            else:
                f_stmts = ["f", "subroutine"]
                c_stmts = ["c"]
        else:
            fmt_result0 = node._fmtresult
            fmt_result = fmt_result0.setdefault("fmtf", util.Scope(fmt_func))
            fmt_result.f_var = fmt_func.F_result
            fmt_result.cxx_type = result_typemap.cxx_type # used with helpers
            fmt_func.F_result_clause = "\fresult(%s)" % fmt_func.F_result
            self.set_fmt_fields(cls, C_node, ast, C_node.ast, fmt_result,
                                modules, fileinfo, True, result_typemap)
            sgroup = result_typemap.sgroup
            spointer = C_node.ast.get_indirect_stmt()
            return_deref_attr = ast.attrs["deref"]
            if is_ctor:
                f_stmts = ["f", "shadow", "ctor"]
                c_stmts = ["c", "shadow", "ctor"]
            else:
                sintent = "result"
                f_stmts = ["f", sgroup, spointer, "result", return_deref_attr,
                           ast.attrs["owner"]]
                c_stmts = ["c", sgroup, spointer, "result", generated_suffix]
        fmt_func.F_subprogram = subprogram

        f_result_blk = typemap.lookup_fc_stmts(f_stmts)
        f_result_blk = typemap.lookup_local_stmts("f", f_result_blk, node)
        # Useful for debugging.  Requested and found path.
        fmt_result.stmt0 = typemap.compute_name(f_stmts)
        fmt_result.stmt1 = f_result_blk.name

        c_result_blk = typemap.lookup_fc_stmts(c_stmts)
        c_result_blk = typemap.lookup_local_stmts(
            ["c", generated_suffix], c_result_blk, node)
        fmt_result.stmtc0 = typemap.compute_name(c_stmts)
        fmt_result.stmtc1 = c_result_blk.name

        if options.debug:
            stmts_comments.append(
                "! ----------------------------------------")
            f_decl = ast.gen_decl(params=None)
            stmts_comments.append("! Function:  " + f_decl)
            stmts_comments.append("! " + f_decl)
            self.document_stmts(
                stmts_comments, fmt_result.stmt0, fmt_result.stmt1)
            c_decl = C_node.ast.gen_decl(params=None)
            if f_decl != c_decl:
                stmts_comments.append("! Function:  " + c_decl)
            self.document_stmts(
                stmts_comments, fmt_result.stmtc0, fmt_result.stmtc1)

        if f_result_blk.result:
            # Change a subroutine into function.
            fmt_func.F_subprogram = "function"
            fmt_func.F_result = f_result_blk.result
            fmt_func.F_result_clause = "\fresult(%s)" % fmt_func.F_result
        
        # this catches stuff like a bool to logical conversion which
        # requires the wrapper
        need_wrapper = need_wrapper or f_result_blk.need_wrapper
        
        if cls:
            need_wrapper = True
            is_static = "static" in ast.storage
            if is_ctor or is_static:
                pass
            else:
                # Add 'this' argument
                arg_f_names.append(fmt_func.F_this)
                arg_f_decl.append(
                    wformat("class({F_derived_name}) :: {F_this}", fmt_func)
                )
                # could use {f_to_c} but I'd rather not hide the shadow class
                arg_c_call.append(
                    wformat("{F_this}%{F_derived_member}", fmt_func)
                )

        # Function result.
        if C_node.F_subprogram == "function":
            need_wrapper = self.build_arg_list_impl(
                fileinfo,
                fmt_result,
                C_node.ast,
                ast,
                result_typemap,
                f_result_blk,
                c_result_blk.buf_args,
                modules,
                imports,
                arg_f_decl,
                arg_c_call,
                need_wrapper,
            )

        # Fortran and C arguments may have different types (fortran generic)
        #
        # f_var - argument to Fortran function (wrapper function)
        # c_var - argument to C function (wrapped function)
        #
        # May be one more argument to C function than Fortran function
        # (the result)
        #
        f_args = ast.params
        f_index = -1  # index into f_args
        for c_arg in C_node.ast.params:
            arg_name = c_arg.name
            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg = fmt_arg0.setdefault("fmtf", util.Scope(fmt_func))
            fmt_arg.f_var = arg_name
            fmt_arg.c_var = arg_name
            fmt_arg.F_pointer = "SHPTR_" + arg_name

            c_attrs = c_arg.attrs
            hidden = c_attrs["hidden"]
            intent = c_attrs["intent"]
            cdesc = "cdesc" if c_attrs["cdesc"] is not None else None

            if c_arg.template_arguments:
                specialize = [c_arg.template_arguments[0].typemap.sgroup]
            else:
                specialize = []
            
            # string C functions may have their results copied
            # into an argument passed in, F_string_result_as_arg.
            # Or the wrapper may provide an argument in the Fortran API
            # to hold the result.
            is_f_arg = True  # assume C and Fortran arguments match
            if c_attrs["_is_result"]:
                if not fmt_func.F_string_result_as_arg:
                    # It is not in the Fortran API
                    is_f_arg = False
                    fmt_arg.c_var = fmt_func.F_result
                    fmt_arg.f_var = fmt_func.F_result
                    need_wrapper = True
            if not is_f_arg:
                # Pass result as an argument to the C++ function.
                f_arg = c_arg
            else:
                # An argument to the C and Fortran function
                f_index += 1
                f_arg = f_args[f_index]
            arg_typemap = self.set_fmt_fields(
                cls, C_node, f_arg, c_arg, fmt_arg, modules, fileinfo)
            f_attrs = f_arg.attrs
                
            c_sgroup = c_arg.typemap.sgroup
            c_spointer = c_arg.get_indirect_stmt()
            c_deref_attr = c_attrs["deref"]
            f_sgroup = f_arg.typemap.sgroup
            f_spointer = f_arg.get_indirect_stmt()
            f_deref_attr = f_attrs["deref"]
            if c_attrs["_is_result"]:
                # This argument is the C function result
                c_stmts = ["c", c_sgroup, c_spointer, "result", generated_suffix, c_deref_attr]
#XXX            f_stmts = ["f", f_sgroup, f_spointer, "result", return_deref_attr]  # + generated_suffix
                f_stmts = ["f", f_sgroup, f_spointer, "result", f_deref_attr]  # + generated_suffix

            else:
                c_stmts = ["c", c_sgroup, c_spointer, intent, c_arg.stmts_suffix, cdesc]  # e.g. buf
                f_stmts = ["f", f_sgroup, f_spointer, intent, f_deref_attr, cdesc]
            c_stmts.extend(specialize)
            f_stmts.extend(specialize)

            f_intent_blk = typemap.lookup_fc_stmts(f_stmts)
            c_intent_blk = typemap.lookup_fc_stmts(c_stmts)

            if is_f_arg:
                implied = f_attrs["implied"]

                if c_arg.ftrim_char_in:
                    # Pass NULL terminated string to C.
                    arg_f_decl.append(
                        "character(len=*), intent(IN) :: {}".format(f_arg.name)
                    )
                    arg_f_names.append(fmt_arg.f_var)
                    arg_c_call.append("trim({})//C_NULL_CHAR".format(f_arg.name))
                    self.set_f_module(modules, "iso_c_binding", "C_NULL_CHAR")
                    need_wrapper = True
                    continue
                elif c_attrs["assumedtype"]:
                    # Passed directly to C as a 'void *'
                    arg_f_decl.append(
                        "type(*) :: {}".format(f_arg.name)
                    )
                    arg_f_names.append(fmt_arg.f_var)
                    arg_c_call.append(f_arg.name)
                    continue
                elif f_arg.is_function_pointer():
                    absiface = self.add_abstract_interface(node, f_arg, fileinfo)
                    if c_attrs["external"]:
                        # external is similar to assumed type, in that it will
                        # accept any function.  But external is not allowed
                        # in bind(C), so make sure a wrapper is generated.
                        arg_f_decl.append("external :: {}".format(f_arg.name))
                        need_wrapper = True
                    else:
                        arg_f_decl.append(
                            "procedure({}) :: {}".format(absiface, f_arg.name)
                        )
                    arg_f_names.append(fmt_arg.f_var)
                    arg_c_call.append(f_arg.name)
                    # function pointers are pass thru without any other action
                    continue
                elif implied:
                    # implied is computed then passed to C++.
                    fmt_arg.pre_call_intent, intermediate, f_helper = ftn_implied(
                        implied, node, f_arg)
                    if intermediate:
                        fmt_arg.c_var = "SH_" + fmt_arg.f_var
                        arg_f_decl.append(f_arg.gen_arg_as_fortran(
                            name=fmt_arg.c_var, local=True, bindc=True))
                        append_format(pre_call, "{c_var} = {pre_call_intent}", fmt_arg)
                        arg_c_call.append(fmt_arg.c_var)
                    else:
                        arg_c_call.append(fmt_arg.pre_call_intent)
                    for helper in f_helper.split():
                        fileinfo.f_helper[helper] = True
                    self.update_f_module(modules, imports, f_arg.typemap.f_module)
                    need_wrapper = True
                    continue
                elif hidden:
                    # Argument is not passed into Fortran.
                    # hidden value is returned from C++.
                    arg_f_decl.append(f_arg.gen_arg_as_fortran(local=True, bindc=True))
                    need_wrapper = True
                elif f_intent_blk.arg_decl:
                    # Explicit declarations from fc_statements.
                    for line in f_intent_blk.arg_decl:
                        append_format(arg_f_decl, line, fmt_arg)
                    if f_result_blk.arg_name:
                        for aname in f_result_blk.arg_name:
                            append_format(arg_f_names, aname, fmt_result)
                    else:
                        arg_f_names.append(fmt_arg.f_var)
                else:
                    # Generate declaration from argument.
                    arg_f_decl.append(f_arg.gen_arg_as_fortran())
                    arg_f_names.append(fmt_arg.f_var)

            # Useful for debugging.  Requested and found path.
            fmt_arg.stmt0 = typemap.compute_name(f_stmts)
            fmt_arg.stmt1 = f_intent_blk.name
            fmt_arg.stmtc0 = typemap.compute_name(c_stmts)
            fmt_arg.stmtc1 = c_intent_blk.name

            if options.debug:
                stmts_comments.append(
                    "! ----------------------------------------")
                f_decl = f_arg.gen_decl()
                stmts_comments.append("! Argument:  " + f_decl)
                self.document_stmts(
                    stmts_comments, fmt_arg.stmt0, fmt_arg.stmt1)
                c_decl = c_arg.gen_decl()
                if f_decl != c_decl:
                    stmts_comments.append("! Argument:  " + c_decl)
                self.document_stmts(
                    stmts_comments, fmt_arg.stmtc0, fmt_arg.stmtc1)

            self.update_f_module(modules, imports, arg_typemap.f_module)

            # Now C function arguments
            # May have different types, like generic
            # or different attributes, like adding +len to string args
            arg_typemap, specialize = typemap.lookup_c_statements(c_arg)

            # Create a local variable for C if necessary.
            # The local variable c_var is used in fc_statements. 
            have_c_local_var = f_intent_blk.c_local_var
            if have_c_local_var:
                fmt_arg.c_var = "SH_" + fmt_arg.f_var
                declare.append(
                    "{} {}".format(
                        arg_typemap.f_c_type or arg_typemap.f_type,
                        fmt_arg.c_var,
                    )
                )

            need_wrapper = self.build_arg_list_impl(
                fileinfo,
                fmt_arg,
                c_arg,
                f_arg,
                arg_typemap,
                f_intent_blk,
                c_intent_blk.buf_args or self._default_buf_args,
                modules,
                imports,
                arg_f_decl,
                arg_c_call,
                need_wrapper,
            )

            need_wrapper = self.add_code_from_statements(
                need_wrapper, fileinfo,
                fmt_arg,
                f_intent_blk,
                modules,
                imports,
                declare,
                pre_call,
                post_call,
            )
            # End parameters loop.
        #####

        if subprogram == "function":
            need_wrapper = self.build_arg_list_impl(
                fileinfo,
                fmt_result,
                C_node.ast, #c_arg,
                ast, # f_arg,
                result_typemap,
                f_result_blk,
                c_result_blk.buf_extra,
                modules,
                imports,
                arg_f_decl,
                arg_c_call,
                need_wrapper,
            )

        # Declare function return value after arguments
        # since arguments may be used to compute return value
        # (for example, string lengths).
        return_deref_attr = ast.attrs["deref"]
        if subprogram == "function":
            # if func_is_const:
            #     fmt_func.F_pure_clause = 'pure '
            if f_result_blk.arg_decl:
                # Explicit declarations from fc_statements.
                for line in f_result_blk.arg_decl:
                    append_format(arg_f_decl, line, fmt_result)
                if f_result_blk.arg_name:
                    for aname in f_result_blk.arg_name:
                        append_format(arg_f_names, aname, fmt_result)
            elif return_deref_attr in ["allocatable", "pointer"]:
                if result_typemap.base == "vector":
                    ntypemap = ast.template_arguments[0].typemap
                else:
                    ntypemap = result_typemap
                f_type = ntypemap.f_type
                arg_f_decl.append("{}, {} :: {}{}".format(
                    f_type, return_deref_attr,
                    fmt_result.f_var, fmt_result.f_assumed_shape))
            else:
                # result_as_arg or None
                # local=True will add any character len attributes
                # e.g.  CHARACTER(LEN=30)
                arg_f_decl.append(
                    ast.gen_arg_as_fortran(name=fmt_result.F_result, local=True)
                )

            self.update_f_module(modules, imports, result_typemap.f_module)

        if options.F_create_generic:
            # if return type is templated in C++,
            # then do not set up generic since only the
            # return type may be different (ex. getValue<T>())
            if is_ctor:
                # ctor generic do not get added as derived type generic.
                # Always create a generic, even if only one function.
                fileinfo.f_function_generic.setdefault(
                    fmt_func.F_name_generic, GenericFunction(True, [])
                ).functions.append(node)
            elif cls:
                fileinfo.f_type_generic.setdefault(
                    fmt_func.F_name_generic, GenericFunction(False, [])
                ).functions.append(node)
            else:
                fileinfo.f_function_generic.setdefault(
                    fmt_func.F_name_scope + fmt_func.F_name_generic,
                    GenericFunction(False, [])).functions.append(node)
        if cls:
            # Add procedure to derived type
            type_bound_part = fileinfo.type_bound_part
            if node.cpp_if:
                type_bound_part.append("#" + node.cpp_if)
            if is_static:
                type_bound_part.append(
                    "procedure, nopass :: %s => %s"
                    % (fmt_func.F_name_function, fmt_func.F_name_impl)
                )
            elif not is_ctor:
                type_bound_part.append(
                    "procedure :: %s => %s"
                    % (fmt_func.F_name_function, fmt_func.F_name_impl)
                )
            if node.cpp_if:
                type_bound_part.append("#endif")

        # use tabs to insert continuations
        if arg_c_call:
            fmt_func.F_arg_c_call = ",\t ".join(arg_c_call)
        fmt_func.F_arguments = options.get(
            "F_arguments", ",\t ".join(arg_f_names)
        )

        # body of function
        # XXX sname = fmt_func.F_name_impl
        sname = fmt_func.F_name_function
        F_force = None
        F_code = None
        call_list = []
        if "f" in node.splicer:
            need_wrapper = True
            F_force = node.splicer["f"]
        elif f_result_blk.call:
            call_list = f_result_blk.call
        elif C_subprogram == "function":
            call_list = ["{F_result} = {F_C_call}({F_arg_c_call})"]
        else:
            call_list = ["call {F_C_call}({F_arg_c_call})"]

        for line in call_list:
            append_format(call, line, fmt_result)
        if C_subprogram == "function":
            need_wrapper = self.add_code_from_statements(
                need_wrapper, fileinfo,
                fmt_result,
                f_result_blk,
                modules,
                imports,
                declare,
                pre_call,
                post_call,
            )
        elif "f" in node.fstatements:
            # Result is an argument.
            need_wrapper = self.add_code_from_statements(
                need_wrapper, fileinfo,
                fmt_result,
                node.fstatements["f"],
                modules,
                imports,
                declare,
                pre_call,
                post_call,
            )
            
        arg_f_use = self.sort_module_info(modules, fmt_func.F_module_name)

        if need_wrapper:
            impl = fileinfo.impl
            impl.append("")
            if node.cpp_if:
                impl.append("#" + node.cpp_if)
            if options.debug:
                if generated:
                    impl.append("! Generated by %s" % " - ".join(generated))
                if options.debug_index:
                    impl.append("! function_index=%d" % node._function_index)
                impl.extend(stmts_comments)
            if options.doxygen and node.doxygen:
                self.write_doxygen(impl, node.doxygen)
            if options.literalinclude:
                append_format(impl, "! start {F_name_impl}", fmt_func)
            append_format(
                impl,
                "\r{F_subprogram} {F_name_impl}(\t"
                "{F_arguments}){F_result_clause}",
                fmt_func,
            )
            impl.append(1)
            impl.extend(arg_f_use)
            impl.extend(arg_f_decl)
            if F_code is None:
                F_code = declare + pre_call + call + post_call
            self._create_splicer(sname, impl, F_code, F_force)
            impl.append(-1)
            append_format(impl, "end {F_subprogram} {F_name_impl}", fmt_func)
            if options.literalinclude:
                append_format(impl, "! end {F_name_impl}", fmt_func)
            if node.cpp_if:
                impl.append("#endif")
        else:
            fmt_func.F_C_name = fmt_func.F_name_impl

    def _gather_helper_code(self, name, done, fileinfo):
        """Add code from helpers.

        First recursively process dependent_helpers
        to add code in order.

        Args:
            name -
            done -
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
            self.update_f_module(
                fileinfo.module_use, {}, mods
            )  # XXX self.module_imports

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
        done = {}  # avoid duplicates
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

        fileinfo.finish()
        output = []
        self.gather_helper_code(fileinfo)

        if options.doxygen:
            self.write_doxygen_file(output, fname, node)
        self._create_splicer("file_top", output)

        output.append("module %s" % module_name)
        output.append(1)

        # Write use statments (classes use iso_c_binding C_PTR)
        arg_f_use = self.sort_module_info(fileinfo.module_use, module_name)
        output.extend(arg_f_use)

        self._create_splicer("module_use", output)
        output.append("implicit none")
        output.append("")
        self._create_splicer("module_top", output)

        output.extend(fileinfo.helper_derived_type)

        output.extend(fileinfo.enum_impl)

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

class ToDimension(todict.PrintNode):
    """Convert dimension expression to Fortran wrapper code.

    1) double * out +intent(out) +deref(allocatable)+dimension(size(in))
    Allocate array before it is passed to C library which will write 
    to it.

    """

    def __init__(self, cls, fcn, fmt):
        """
        Args:
            cls  - ast.ClassNode or None
            fcn  - ast.FunctionNode of calling function.
            fmt  - util.Scope
        """
        super(ToDimension, self).__init__()
        self.cls = cls
        self.fcn = fcn
        self.fmt = fmt

        self.rank = 0
        self.shape = []
        self.need_helper = False

    def visit_list(self, node):
        # list of dimension expressions
        self.rank = len(node)
        for dim in node:
            sh = self.visit(dim)
            self.shape.append(sh)

    def visit_Identifier(self, node):
        argname = node.name
        # Look for Fortran intrinsics
        if argname == "size" and node.args:
            # size(in)
            return self.param_list(node) # function
        # Look for members of class/struct.
        elif self.cls is not None and argname in self.cls.map_name_to_node:
            # This name is in the same class as the dimension.
            # Make name relative to the class.
            self.need_helper = True
            member = self.cls.map_name_to_node[argname]
            if member.may_have_args():
                if node.args is None:
                    print("{} must have arguments".format(argname))
                else:
                    return "obj->{}({})".format(
                        argname, self.comma_list(node.args))
            else:
                if node.args is not None:
                    print("{} must not have arguments".format(argname))
                else:
                    return "obj->{}".format(argname)
        else:
            if self.fcn.ast.find_arg_by_name(argname) is None:
                self.need_helper = True
            if node.args is None:
                return argname  # variable
            else:
                return self.param_list(node) # function
        return "--??--"

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
            return "size({},kind={})".format(argname, arg_typemap.f_kind)
        elif argname == "type":
            # type(arg)
            self.intermediate = True
            self.helper = "ShroudTypeDefines"
            argname = node.args[0].name
            typearg = self.func.ast.find_arg_by_name(argname)
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

    """
    newlibrary = None
    def __init__(self, node):
        self.node = node
        self.module_use = {}  # Use statements for a module
        self.use_stmts = []
        self.enum_impl = []
        self.f_type_decl = []
        self.c_interface = []
        self.abstract_interface = []
        self.generic_interface = []
        self.impl = []  # implementation, after contains
        self.operator_impl = []
        self.operator_map = {}  # list of function names by operator
        # {'.eq.': [ 'abc', 'def'] }
        if not self.newlibrary.options.literalinclude2:
            self.c_interface.append("")
            self.c_interface.append("interface")
            self.c_interface.append(1)
        self.f_function_generic = {}  # look for generic functions
        self.f_abstract_interface = {}

        self.c_helper = {}
        self.f_helper = {}
        self.helper_derived_type = []
        self.helper_source = []
        self.private_lines = []
        self.interface_lines = []

    def finish(self):
        self.c_interface.append(-1)
        self.c_interface.append("end interface")

    def begin_class(self):
        self.f_type_generic = {}  # look for generic methods
        self.type_bound_part = []

    def write_module(self, output):
        output.extend(self.abstract_interface)
        output.extend(self.c_interface)
        output.extend(self.generic_interface)

        output.extend(self.private_lines)
        output.extend(self.interface_lines)

        output.append(-1)
        output.append("")
        output.append("contains")
        output.append(1)

        output.extend(self.impl)

        output.extend(self.operator_impl)

        output.extend(self.helper_source)

        output.append(-1)
        output.append("")

    def add_c_helper(self, helpers, fmt):
        """Add a list of C helpers."""
        c_helper = wformat(helpers, fmt)
        for helper in c_helper.split():
            self.c_helper[helper] = True

    def add_f_helper(self, helpers, fmt):
        """Add a list of Fortran helpers."""
        f_helper = wformat(helpers, fmt)
        for helper in f_helper.split():
            self.f_helper[helper] = True
        
