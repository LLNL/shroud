# Copyright Shroud Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Fill the fmtdict for C and Fortran wrappers.
"""

from . import error
from .declstr import gen_arg_as_c, gen_arg_as_cxx
from . import todict
from . import statements
from . import typemap
from . import util
from .util import wformat, append_format

import collections

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

maplang = dict(f="ci_type", c="c_type")

class FillFormat(object):
    """Loop over Nodes and fill fmt dictionaries.
    Used for Fortran and C wrappers.

    Creates the nested dictionary structure:
    _bind = {
      'lang': {
        '+result': BindArg
           .fmtdict = Scope(_fmtfunc)
        'arg1': BindArg
           .fmtdict = Scope(_fmtfunc)
      }
    }
    """
    def __init__(self, newlibrary):
        self.newlibrary = newlibrary
        self.language = newlibrary.language
        self.cursor = error.get_cursor()

    def fmt_library(self):
        self.fmt_namespace(self.newlibrary.wrap_namespace)
        fmt_assignment(self.newlibrary)

    def fmt_namespace(self, node):
        cursor = self.cursor
        
        cursor.push_phase("FillFormat typedef")
        for typ in node.typedefs:
            self.fmt_typedefs(typ)
        cursor.pop_phase("FillFormat typedef")

        for cls in node.classes:
            if cls.wrap_as == "struct":
                cursor.push_phase("FillFormat class struct")
                self.wrap_struct(cls)
                cursor.pop_phase("FillFormat class struct")
            else:
                self.wrap_class(cls)
                
            cursor.push_phase("FillFormat class function")
            self.fmt_functions(cls, cls.functions)
            cursor.pop_phase("FillFormat class function")

        cursor.push_phase("FillFormat function")
        self.fmt_functions(None, node.functions)
        cursor.pop_phase("FillFormat function")

        for ns in node.namespaces:
            self.fmt_namespace(ns)

    def wrap_struct(self, node):
        if not node.wrap.fortran:
            return
        for var in node.variables:
            bind = statements.fetch_var_bind(var, "f")
            fmt = statements.set_bind_fmtdict(bind, node.fmtdict)
            set_f_var_format(var, bind)

    def wrap_class(self, node):
        options = node.options
        fmt_class = node.fmtdict
        cls_cursor = self.cursor.push_node(node)

        for typ in node.typedefs:
            self.fmt_typedefs(typ)

        if node.baseclass:
            # Only single inheritance supported.
            # Base class already contains F_derived_member.
            # ['pubic', 'ClassName', declast.CXXClass]
            fmt_class.F_derived_member_base = node.baseclass[0][2].typemap.f_derived_type
            fmt_class.baseclass = statements.BaseClassFormat(node.baseclass[0][2])
        elif options.class_baseclass:
            # Used with wrap_struct_as=class.
            baseclass = node.parent.ast.unqualified_lookup(options.class_baseclass)
            if not baseclass:
                self.cursor.warning("Unknown class '{}' in option.class_baseclass".format(options.class_baseclass))
                fmt_class.F_derived_member_base = "===>F_derived_member_base<==="
            else:
                fmt_class.F_derived_member_base = baseclass.typemap.f_derived_type
        self.cursor.pop_node(node)
            
    def fmt_typedefs(self, node):
        self.cursor.push_node(node)
        fmt = node.fmtdict
        if node.wrap.c:
            node.reeval_template("C_name_typedef")
        else:
            # language=c, use original name.
            fmt.C_name_typedef = fmt.typedef_name
        if node.wrap.fortran:
            node.reeval_template("F_name_typedef")
        
        fmt.update(node.user_fmt)
        typemap.fill_typedef_typemap(node, node.user_fields)

        if node.wrap.fortran:
            if node.ast.declarator.is_function_pointer():
                meta = statements.fetch_typedef_bind(node, "f").meta
                fptr = meta["fptr"]
                self.fmt_function_pointer("f", fptr)

                # Get the function's type. Not the typedef's type
                # which will be "procedure(name)".
                # i_type will be used in an abstract interface.
                declarator = fptr.ast.declarator
                subprogram = declarator.get_subprogram()
                if subprogram == "function":
                    r_bind = statements.get_func_bind(fptr, "f")
                    ntypemap = declarator.typemap
                    fmt = r_bind.fmtdict
                    fmt.i_type = ntypemap.i_type or ntypemap.f_type
                    # XXX - To be changed to i_module, i_kind
                    fmt.i_module_name = ntypemap.f_module_name
                    fmt.i_kind = ntypemap.f_kind
        self.cursor.pop_node(node)

    def fmt_functions(self, cls, functions):
        for node in functions:
            if node.wrap.c:
                self.fmt_function("c", cls, node)
            if node.wrap.fortran:
                self.fmt_function("f", cls, node)

    def fmt_function(self, wlang, cls, node):
        cursor = self.cursor
        func_cursor = cursor.push_node(node)

        options = node.options
        fmt_func = node.fmtdict
        arglist = []
        setattr(fmt_func, "{}_arglist".format(wlang), arglist)

        bind_result = statements.fetch_func_bind(node, wlang)
        fmt_result = statements.set_bind_fmtdict(bind_result, fmt_func)

        if wlang == "f":
            if node.options.class_ctor:
                # Generic constructor for C "class" (wrap_struct_as=class).
                clsnode = node.lookup_class(node.options.class_ctor)
                fmt_func.F_name_generic = clsnode.fmtdict.F_derived_name
            elif options.F_create_generic:
                if node.ast.declarator.is_ctor:
                    fmt_func.F_name_generic = fmt_func.F_derived_name
                else:
                    node.reeval_template("F_name_generic")
            node.reeval_template("F_name_impl")
            node.reeval_template("F_name_function")
        fmt_func.update(node.user_fmt)

        result_stmt = bind_result.stmt
        func_cursor.stmt = result_stmt
        set_share_function_format(node, bind_result, wlang)
        func_cursor.stmt = None
        arglist.append(bind_result.fmtdict)

        # --- Loop over function parameters
        for arg in node.ast.declarator.params:
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name

            bind_arg = statements.fetch_arg_bind(node, arg, wlang)
            fmt_arg = statements.set_bind_fmtdict(bind_arg, fmt_func)
            arg_stmt = bind_arg.stmt
            func_cursor.stmt = arg_stmt
            arglist.append(fmt_arg)

            set_f_arg_format(node, arg, bind_arg, wlang)
            if wlang == "f":
                if arg.declarator.is_function_pointer():
                    fptr = bind_arg.meta["fptr"]
                    self.fmt_function_pointer(wlang, fptr)
        # --- End loop over function parameters
        func_cursor.arg = None
        func_cursor.stmt = None
            
        cursor.pop_node(node)

    def fmt_function_pointer(self, wlang, node):
        """
        Set fmt.f_abstract_name for arguments.
        """
        cursor = self.cursor
        func_cursor = cursor.push_node(node)

        fmt_func = node.fmtdict
        bind_result = statements.fetch_func_bind(node, wlang)
        fmt_result = statements.set_bind_fmtdict(bind_result, fmt_func)

        self.fill_interface_result(None, node, bind_result)

        # --- Loop over function parameters
        fmt_name = util.Scope(fmt_func)
        for i, arg in enumerate(node.ast.declarator.params):
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name
            
            if arg_name is None:
                fmt_name.index = str(i)
                arg_name = wformat(
                    node.options.F_abstract_interface_argument_template,
                    fmt_name,
                )

            bind_arg = statements.fetch_arg_bind(node, arg, wlang) #, arg_name)
            fmt_arg = statements.set_bind_fmtdict(bind_arg, fmt_func)
            arg_stmt = bind_arg.stmt
            func_cursor.stmt = arg_stmt

            if wlang == "f":
                set_f_arg_format(node, arg, bind_arg, wlang)
                fmt_arg.f_abstract_name = arg_name
                fmt_arg.f_var = arg_name
                fmt_arg.i_var = arg_name

                # XXX - fill_interface_arg
                self.set_fmt_fields_iface(arg, bind_arg, arg.typemap)
                self.set_fmt_fields_dimension(None, node, arg, bind_arg)

                
        # --- End loop over function parameters
        func_cursor.arg = None
        func_cursor.stmt = None
            
        cursor.pop_node(node)

    def fill_c_result(self, wlang, cls, node, CXX_ast, bind):
        ast = node.ast
        declarator = ast.declarator
        C_subprogram = declarator.get_subprogram()
        result_typemap = ast.typemap

        result_stmt = bind.stmt
        fmt_result = bind.fmtdict

        if C_subprogram != "subroutine":
            fmt_result.idtor = "0"  # no destructor
            fmt_result.c_var = fmt_result.C_local + fmt_result.C_result
            fmt_result.c_type = result_typemap.c_type
            fmt_result.cxx_type = result_typemap.cxx_type
            if result_typemap.ci_type:
                fmt_result.ci_type = result_typemap.ci_type
            converter, lang = find_result_converter(
                wlang, self.language, result_typemap)
            if ast.template_arguments:
                statements.set_template_fields(ast, fmt_result)
            if converter is None:
                # C and C++ are compatible
                fmt_result.cxx_var = fmt_result.c_var
            else:
                fmt_result.c_abstract_decl = gen_arg_as_c(
                    ast, name=False, add_params=False)
                fmt_result.cxx_abstract_decl = gen_arg_as_cxx(
                    ast, name=False, add_params=False, as_ptr=True)
                fmt_result.cxx_var = fmt_result.CXX_local + fmt_result.C_result

            if ast.const:
                fmt_result.c_const = "const "
            else:
                fmt_result.c_const = ""

            compute_cxx_deref(CXX_ast, fmt_result)
            set_c_function_format(node, bind)

        if result_stmt.c_return_type:
            # Override return type.
            fmt_result.C_return_type = wformat(
                result_stmt.c_return_type, fmt_result)
        else:
            fmt_result.C_return_type = gen_arg_as_c(
                ast, name=False, add_params=False, lang=maplang[wlang])
        self.name_temp_vars(fmt_result.C_result, bind, "c")
        self.apply_helpers_from_stmts(node, bind)
        statements.apply_fmtdict_from_stmts(bind)
        self.find_idtor(result_typemap, bind)
        self.set_fmt_fields_c(wlang, cls, node, ast, result_typemap, bind, True)

    def fill_c_arg(self, wlang, cls, node, arg, bind, pre_call):
        declarator = arg.declarator
        arg_name = declarator.user_name
        arg_typemap = arg.typemap

        arg_stmt = bind.stmt
        fmt_arg = bind.fmtdict

        fmt_arg.c_var = arg_name
        # XXX - order issue - c_var must be set before name_temp_vars,
        #       but set by set_fmt_fields
        self.name_temp_vars(arg_name, bind, "c")
        self.set_fmt_fields_c(wlang, cls, node, arg, arg_typemap, bind, False)
        self.apply_helpers_from_stmts(node, bind)
        statements.apply_fmtdict_from_stmts(bind)

        # prototype:  vector<int> -> int *
        converter, lang = find_arg_converter(wlang, self.language, arg_typemap)
        
        fmt_arg.c_abstract_decl = gen_arg_as_c(
            arg, name=False, add_params=False)
        fmt_arg.cxx_abstract_decl = gen_arg_as_cxx(
            arg, name=False, add_params=False, as_ptr=True)
        if converter is None:
            # Compatible
            fmt_arg.cxx_var = fmt_arg.c_var
        elif arg_stmt.c_pre_call:
            # statements have explicit code
            pass
        else:
            # convert C argument to C++
            fmt_arg.cxx_var = fmt_arg.CXX_local + fmt_arg.c_var
            fmt_arg.cxx_val = wformat(converter, fmt_arg)
            fmt_arg.cxx_decl = gen_arg_as_cxx(arg,
                name=fmt_arg.cxx_var,
                add_params=False,
                as_ptr=True
            )
            append_format(
                pre_call, "{cxx_decl} =\t {cxx_val};", fmt_arg
            )

        compute_cxx_deref(arg, fmt_arg)
        self.set_cxx_nonconst_ptr(arg, fmt_arg)
        self.find_idtor(arg_typemap, bind)

    def fill_interface_result(self, cls, node, bind):
        ast = node.ast
        declarator = ast.declarator
        result_typemap = ast.typemap
        result_stmt = bind.stmt
        fmt_result = bind.fmtdict

        self.name_temp_vars(fmt_result.C_result, bind, "c", "i")

        subprogram = declarator.get_subprogram()
        if subprogram == "function":
            fmt_result.i_var = fmt_result.i_result_var
            fmt_result.f_var = fmt_result.i_result_var
            self.set_fmt_fields_iface(ast, bind, result_typemap)
            self.set_fmt_fields_dimension(cls, node, ast, bind)
            set_f_function_format(node, bind, subprogram)

        if result_stmt.i_result_var == "as-subroutine":
            subprogram = "subroutine"
        elif result_stmt.i_result_var is not None:
            subprogram = "function"
        elif result_stmt.c_return_type == "void":
            # Change a function into a subroutine.
            subprogram = "subroutine"
        elif result_stmt.c_return_type:
            # Change a subroutine into function
            # or change the return type.
            subprogram = "function"
        fmt_result.i_subprogram = subprogram

        statements.apply_fmtdict_from_stmts(bind)

        # Compute after stmt.fmtdict is evaluated.
        if subprogram == "function":
            if result_stmt.i_result_var:
                fmt_result.i_result_var = wformat(
                    result_stmt.i_result_var, fmt_result)
            fmt_result.i_result_clause = "\fresult(%s)" % fmt_result.i_result_var
        
    def fill_interface_arg(self, cls, node, arg, bind):
        declarator = arg.declarator
        arg_name = declarator.user_name
        fmt_arg = bind.fmtdict
        arg_typemap = arg.typemap

        fmt_arg.i_var = arg_name
        fmt_arg.f_var = arg_name
        self.set_fmt_fields_iface(arg, bind, arg_typemap)
        self.set_fmt_fields_dimension(cls, node, arg, bind)
        self.name_temp_vars(arg_name, bind, "c", "i")
        statements.apply_fmtdict_from_stmts(bind)

    def fill_fortran_function(self, cls, node):
        """Sets format fields for the 'pass' argument of the function.
        """
        if cls is None:
            return
        ast = node.ast
        declarator = ast.declarator
        fmt_func = node.fmtdict

    def fill_fortran_result(self, cls, node, bind):
        ast = node.ast
        declarator = ast.declarator
        result_typemap = ast.typemap
        result_stmt = bind.stmt
        fmt_result = bind.fmtdict
        C_node = node.C_node  # C wrapper to call.

        self.name_temp_vars(fmt_result.C_result, bind, "f")

        subprogram = declarator.get_subprogram()
        if subprogram == "function":
            fmt_result.f_var = fmt_result.f_result_var
            fmt_result.fc_var = fmt_result.f_result_var
        
        if result_stmt.f_result_var == "as-subroutine":
            subprogram = "subroutine"
        elif result_stmt.f_result_var is not None:
            subprogram = "function"
        fmt_result.F_subprogram = subprogram

        self.set_fmt_fields_f(cls, C_node, ast, C_node.ast, bind,
                              subprogram, result_typemap)
        self.set_fmt_fields_dimension(cls, C_node, ast, bind)
        self.apply_helpers_from_stmts(node, bind)
        set_f_function_format(node, bind, subprogram)
        statements.apply_fmtdict_from_stmts(bind)

        # Compute after stmt.fmtdict is evaluated.
        if subprogram == "function":
            if result_stmt.f_result_var:
                fmt_result.f_result_var = wformat(
                    result_stmt.f_result_var, fmt_result)
            fmt_result.f_var = fmt_result.f_result_var
            fmt_result.fc_var = fmt_result.f_result_var
            fmt_result.f_result_clause = "\fresult(%s)" % fmt_result.f_result_var

    def fill_fortran_arg(self, cls, node, C_node, f_arg, c_arg, bind):
        arg_name = f_arg.declarator.user_name
        fmt_arg = bind.fmtdict

        fmt_arg.f_var = arg_name
        fmt_arg.fc_var = arg_name
        self.name_temp_vars(arg_name, bind, "f")
        arg_typemap = self.set_fmt_fields_f(cls, C_node, f_arg, c_arg, bind)
        self.set_fmt_fields_dimension(cls, C_node, f_arg, bind)
        self.apply_helpers_from_stmts(node, bind)
        statements.apply_fmtdict_from_stmts(bind)
        return arg_typemap
    
    def name_temp_vars(self, rootname, bind, lang, prefix=None):
        """Compute names of temporary C variables.

        Create statements c_temps/c_local and f_temps/f_local variables.

        lang - "c", "f"
        prefix - "c", "f", "i"
        """
        stmts = bind.stmt
        fmt = bind.fmtdict

        if prefix is None:
            prefix = lang

        names = stmts.get(lang + "_temps", None)
        if names is not None:
            for name in names:
                setattr(fmt,
                        "{}_var_{}".format(prefix, name),
                        "{}{}_{}".format(fmt.c_temp, rootname, name))

        if prefix == "i":
            # Interfaces have no local variables
            return

        names = stmts.get(lang + "_local", None)
        if names is not None:
            for name in names:
                setattr(fmt,
                        "{}_local_{}".format(prefix, name),
                        "{}{}_{}".format(fmt.C_local, rootname, name))
                if name == "cxx":
                    # Enable cxx_nonconst_ptr to continue to work
                    fmt.cxx_var = fmt.c_local_cxx

    def set_fmt_fields_c(self, wlang, cls, fcn, ast, ntypemap, bind, is_func):
        """
        Set format fields for ast.
        Used with arguments and results.

        Args:
            wlang    - 
            cls      - ast.ClassNode or None of enclosing class.
            fcn      - ast.FunctionNode of calling function.
            ast      - declast.Declaration
            ntypemap - typemap.Typemap
            bind     - statements.BindArg
            is_func  - True if function.
        """
        declarator = ast.declarator
        meta = bind.meta
        fmt = bind.fmtdict
        if is_func:
            rootname = fmt.C_result
        else:
            rootname = declarator.user_name
            if ast.const:
                fmt.c_const = "const "
            else:
                fmt.c_const = ""
            compute_c_deref(ast, fmt)
            fmt.c_type = find_arg_type(wlang, ntypemap) #ntypemap.c_type + "xxx"
            fmt.cxx_type = ntypemap.cxx_type
            if ntypemap.ci_type:
                fmt.ci_type = ntypemap.ci_type
            fmt.idtor = "0"

            if ntypemap.base != "shadow" and ast.template_arguments:
                statements.set_template_fields(ast, fmt)
            
            if meta["blanknull"]:
                # Argument to helper ShroudStrAlloc via attr[blanknull].
                fmt.c_blanknull = "1"
        
        if meta["dim_ast"]:
            if cls is not None:
                parent = cls
                class_context = wformat("{CXX_this}->", fmt)
            elif fcn.struct_parent:
                # struct_parent is set in add_var_getter_setter
                parent = fcn.struct_parent
                class_context = wformat("{CXX_this}->", fmt)
            else:
                parent = None
                class_context = ""
            visitor = ToDimensionC(parent, fcn, fmt, class_context)
            visitor.visit(meta["dim_ast"])
            fmt.rank = str(visitor.rank)
            if fmt.rank != "assumed":
                meta["dim_shape"] = visitor.shape

        if meta["len"]:
            fmt.attr_len = meta["len"]

    def set_fmt_fields_iface(self, ast, bind, ntypemap):
        """Set format fields for interface.

        Transfer info from Typemap to fmt for use by statements.

        Parameters
        ----------
        ast : declast.Declaration
        bind : statements.BindArg
        ntypemap : typemap.Typemap
            The typemap has already resolved template arguments.
            For example, std::vector<int>.  ntypemap will be 'int'.
        """
        meta = bind.meta
        fmt = bind.fmtdict

        if meta["assumedtype"]:
            fmt.f_type = "type(*)"
            fmt.i_type = "type(*)"
        elif ntypemap.base == "string":
            if meta["len"] and meta["intent"] == "function":
                # Declare local variable for function result
                fmt.f_type = "character(len={})".format(meta["len"])
            elif meta["deref"] == "allocatable":
                fmt.f_type = "character(len=:)"
            else:
                fmt.f_type = ntypemap.f_type
            fmt.i_type = ntypemap.i_type
        else:
            # Some types such as vector will not have a default since it
            # depends on the template arguments.
            if ntypemap.f_type:
                fmt.f_type = ntypemap.f_type
            i_type = ntypemap.i_type or ntypemap.f_type
            if i_type:
                fmt.i_type = i_type
        if ntypemap.i_module_name:
            fmt.i_module_name = ntypemap.i_module_name
            if ntypemap.i_kind:
                fmt.i_kind = ntypemap.i_kind
        elif ntypemap.f_module_name is not None:
            fmt.i_module_name = ntypemap.f_module_name
            if ntypemap.f_kind:
                fmt.i_kind = ntypemap.f_kind

        if ntypemap.f_module_name:
            fmt.f_module_name = ntypemap.f_module_name
        if ntypemap.f_kind:
            fmt.f_kind = ntypemap.f_kind
        if ntypemap.f_capsule_data_type:
            fmt.f_capsule_data_type = ntypemap.f_capsule_data_type
        if ntypemap.f_derived_type:
            fmt.f_derived_type = ntypemap.f_derived_type

    def set_fmt_fields_f(self, cls, fcn, f_ast, c_ast, bind,
                         subprogram=None,
                         ntypemap=None):
        """
        Set format fields for ast.
        Used with arguments and results.

        f_ast and c_ast may be different for fortran_generic.

        Parameters
        ----------
        cls : ast.ClassNode or None of enclosing class.
        fcn : ast.FunctionNode of calling function.
        f_ast : declast.Declaration - Fortran argument
        c_ast : declast.Declaration - C argument
              Abstract Syntax Tree of argument or result
        bind : statements.BindArg
        subprogram : str
        ntypemap : typemap.Typemap
        """
        c_attrs = c_ast.declarator.attrs
        fmt = bind.fmtdict

        if subprogram == "subroutine":
            # XXX - no need to set f_type
            pass
        elif subprogram == "function":
            # XXX this also gets set for subroutines
            pass
        else:
            ntypemap = f_ast.typemap
        if ntypemap.sgroup != "shadow" and c_ast.template_arguments:
            statements.set_template_fields(c_ast, fmt)
        if subprogram != "subroutine":
            self.set_fmt_fields_iface(c_ast, bind, ntypemap)
            if "pass" in c_attrs:
                # Used with wrap_struct_as=class for passed-object dummy argument.
                fmt.f_type = ntypemap.f_class
        return ntypemap

    def set_fmt_fields_dimension(self, cls, fcn, f_ast, bind):
        """Set fmt fields based on dimension attribute.

        f_assumed_shape is used in both implementation and interface.

        Parameters
        ----------
        cls : ast.ClassNode or None of enclosing class.
        fcn : ast.ClassNode for struct or ast.FunctionNode of calling function.
        f_ast : declast.Declaration
        bind: statements.BindArg
        """
        f_attrs = f_ast.declarator.attrs
        meta = bind.meta
        fmt = bind.fmtdict
        dim = meta["dim_ast"]
        rank = meta["rank"]
        if meta["dimension"] == "..":   # assumed-rank
            fmt.i_dimension = "(..)"
            fmt.f_dimension = "(..)"
            fmt.f_assumed_shape = "(..)"
        elif rank is not None:
            fmt.rank = str(rank)
            if rank == 0:
                # Assigned to cdesc to pass metadata to C wrapper.
                fmt.f_size = "1"
            else:
                fmt.f_size = wformat("size({f_var})", fmt)
                fmt.f_assumed_shape = fortran_ranks[rank]
                fmt.f_dimension = fortran_ranks[rank]
                fmt.i_dimension = "(*)"
        elif dim:
            visitor = ToDimension(cls, fcn, fmt)
            visitor.visit(dim)
            rank = visitor.rank
            fmt.rank = str(rank)
            if rank != "assumed" and rank > 0:
                fmt.f_assumed_shape = fortran_ranks[rank]
                fmt.i_dimension = "(*)"
                if meta["deref"] in ["allocatable","pointer"]:
                    fmt.f_dimension = fmt.f_assumed_shape
                elif visitor.compute_shape:
                    fmt.f_dimension = fmt.f_assumed_shape
                else:
                    fmt.f_dimension = "(" + ",".join(visitor.shape) + ")"

    def apply_helpers_from_stmts(self, node, bind):
        """
        Parameters:
          node - ast.FunctionNode
          bind - statements.BindArg
        """
        stmt = bind.stmt
        fmt = bind.fmtdict
        node_helpers = node.fcn_helpers.setdefault("fc", {})
        add_fc_helper(node_helpers, stmt.helper, fmt)

def fmt_assignment(library):
    """
    Create fmtdict for assignment overloads.
    """
    for assign in library.assign_operators:
        lhs = assign.lhs
        rhs = assign.rhs
        options = assign.lhs.options
        fmt_lhs = assign.lhs.fmtdict
        fmt_rhs = assign.rhs.fmtdict
        fmt = util.Scope(fmt_lhs)
        assign.fmtdict = fmt
        iface_import = {}

        fmt.cxx_type_lhs = lhs.typemap.cxx_type
        fmt.cxx_type_rhs = rhs.typemap.cxx_type
        fmt.c_type_lhs = lhs.typemap.c_type
        fmt.c_type_rhs = rhs.typemap.c_type
        fmt.f_derived_type_lhs = lhs.typemap.f_derived_type
        fmt.f_derived_type_rhs = rhs.typemap.f_derived_type
        fmt.f_capsule_data_type_lhs = lhs.typemap.f_capsule_data_type
        fmt.f_capsule_data_type_rhs = rhs.typemap.f_capsule_data_type
        iface_import[fmt.f_capsule_data_type_lhs] = True
        iface_import[fmt.f_capsule_data_type_rhs] = True

        fmt.function_suffix = "_" + fmt_rhs.cxx_class
        fmt.F_name_api = fmt_lhs.F_name_assign
        fmt.F_name_assign_api = wformat(options.F_name_impl_template, fmt)
        fmt.f_interface_import = ",\t ".join(iface_import.keys())

        fmt.C_name_api = fmt_lhs.C_name_assign
        fmt.C_name_assign_api = wformat(options.C_name_template, fmt)

        # XXX - Need to compute, otherwise it will leak.
        fmt.idtor = "0"
        
def add_fc_helper(node_helpers, helpers, fmt):
    """Add a list of Fortran and C helpers.
    Add format variable  for use by pre_call and post_call.
    C:       fmt.c_helper_{c_fmtname}
    Fortran: fmt.f_helper_{f_fmtname}
    """
    for f_helper in helpers:
        helper = wformat(f_helper, fmt)
        helper_info = statements.lookup_fc_helper(helper, "helper")
        if helper_info.name != "h_mixin_unknown":
            node_helpers[helper] = True
            fmtname = helper_info.f_fmtname
            if fmtname:
                setattr(fmt, "f_helper_" + helper, fmtname)
            fmtname = helper_info.c_fmtname
            if fmtname:
                setattr(fmt, "c_helper_" + helper, fmtname)


######################################################################

class ToDimensionC(todict.PrintNode):
    """Convert dimension expression to C wrapper code.

    expression has already been checked for errors by generate.check_implied.
    Convert functions:
      size  -  PyArray_SIZE
    """

    def __init__(self, cls, fcn, fmt, context):
        """
        cls is the class which contains fcn.  It may also be the
        struct associated with a getter.  It will be used to find
        variable names used in dimension expression.

        Args:
            cls  - ast.ClassNode or None
            fcn  - ast.FunctionNode of calling function.
            fmt  - util.Scope
            context - how to access Identifiers in cls.
                      Different for function arguments and
                      class/struct members.

        """
        super(ToDimensionC, self).__init__()
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
            arg = self.fcn.ast.declarator.find_arg_by_name(argname)
            if arg:
                declarator = arg.declarator
                if "hidden" in declarator.attrs:
                    # (int *arg +intent(out)+hidden)
                    # c_out_native_*_hidden creates a local scalar.
                    deref = ''
                elif declarator.is_indirect():
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
        raise RuntimeError("fcfmt.py: Detected assumed-rank dimension")

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
        self.compute_shape = False

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
            self.compute_shape = True
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
                    self.compute_shape = True
                    return "obj->{}({})".format(
                        argname, self.comma_list(node.args))
            else:
                if node.args is not None:
                    print("{} must not have arguments".format(argname))
                else:
                    self.compute_shape = True
                    return "obj->{}".format(argname)
        else:
            if self.fcn.ast.declarator.find_arg_by_name(argname) is None:
                self.need_helper = True
            if node.args is None:
                return argname  # variable
            else:
                self.compute_shape = True
                return self.param_list(node) # function
        return "--??--"

    def visit_AssumedRank(self, node):
        # (..)
        self.rank = "assumed"
        return "===assumed-rank==="
        error.get_cursor().warning("Detected assumed-rank dimension")
                
######################################################################

def set_share_function_format(node, bind, wlang):
    """
    node  - ast.FunctionNode
    bind  - statements.BindArg
    wlang - str
    """
    fmt = bind.fmtdict
    meta = bind.meta

    fmt.stmt_name = bind.stmt.name
    fmt.typemap = node.ast.typemap
    fmt.gen = FormatGen(node, node.ast, bind, wlang)
    
def set_c_function_format(node, bind):
    fmt = bind.fmtdict
    meta = bind.meta

    if meta["funcarg"]:
        name = meta["funcarg"]
        fmt.f_var = name
        fmt.i_var = name
        fmt.i_result_var = name
#        fmt.c_var = name  # not being propagated to other uses of result
#        fmt.c_result_var = name
    
def set_f_function_format(node, bind, subprogram):
    fmt = bind.fmtdict
    meta = bind.meta

    if meta["deref"] == "allocatable":
        fmt.f_deref_attr = ", allocatable"
    elif meta["deref"] == "pointer":
        fmt.f_deref_attr = ", pointer"
    if meta["funcarg"]:
        name = meta["funcarg"]
        fmt.f_var = name
        fmt.i_var = name
        fmt.i_result_var = name
#        fmt.c_var = name
#        fmt.c_result_var = name
    
def set_f_arg_format(node, arg, bind, wlang):
    """
    node  - ast.FunctionNode
    arg   - declast.Declaration
    bind  - statements.BindArg
    wlang - str
    """
    fmt = bind.fmtdict
    meta = bind.meta

    fmt.stmt_name = bind.stmt.name
    fmt.typemap = arg.declarator.typemap
    fmt.gen = FormatGen(node, arg, bind, wlang)
    
    intent = meta["intent"].upper()
    if intent == "SETTER":
        intent = "IN"
    if intent != "NONE":
        fmt.f_intent = intent
        fmt.f_intent_attr = ", intent({})".format(fmt.f_intent)
        fmt.i_intent = intent
        fmt.i_intent_attr = ", intent({})".format(fmt.i_intent)

    if meta["optional"]:
        fmt.f_optional_attr = ", optional"
    if meta["value"]:
        fmt.f_value_attr = ", value"
    if meta["deref"] == "allocatable":
        fmt.f_deref_attr = ", allocatable"
    elif meta["deref"] == "pointer":
        fmt.f_deref_attr = ", pointer"

def set_f_var_format(var, bind):
    """Set format fields for variable (in a struct).

    Transfer info from Typemap to fmt for use by statements.

    Parameters
    ----------
    var  : ast.VariableNode
    bind : statements.BindArg
    """
    meta = bind.meta
    fmt = bind.fmtdict
    declarator = var.ast.declarator
    ntypemap = declarator.typemap

    fmt.i_var = declarator.name

    if declarator.is_indirect():
        fmt.i_type = "type(C_PTR)"
        fmt.i_module_name = "iso_c_binding"
        fmt.i_kind = "C_PTR"
    else:
        i_type = ntypemap.i_type or ntypemap.f_type
        if i_type:
            fmt.i_type = i_type
        if ntypemap.i_module_name:
            fmt.i_module_name = ntypemap.i_module_name
            if ntypemap.i_kind:
                fmt.i_kind = ntypemap.i_kind
        elif ntypemap.f_module_name is not None:
            fmt.i_module_name = ntypemap.f_module_name
            if ntypemap.f_kind:
                fmt.i_kind = ntypemap.f_kind

        if meta["len"]:
            fmt.i_type = "character(len={})".format(meta["len"])

        if declarator.array:
            decl = ["("]
            # Convert to column-major order.
            for dim in reversed(declarator.array):
                decl.append(todict.print_node(dim))
                decl.append(",")
            decl[-1] = ")"
            fmt.i_dimension = "".join(decl)

def compute_c_deref(arg, fmt):
    """Compute format fields to dereference C argument."""
    if arg.declarator.is_indirect(): #pointer():
        fmt.c_deref = "*"
        fmt.c_member = "->"
        fmt.c_addr = ""
    else:
        fmt.c_deref = ""
        fmt.c_member = "."
        fmt.c_addr = "&"

def compute_cxx_deref(arg, fmt):
    """Compute format fields to dereference C++ variable."""
    if arg.declarator.is_pointer():
#        fmt.cxx_deref = "*"
        fmt.cxx_member = "->"
        fmt.cxx_addr = ""
    else:
#        fmt.cxx_deref = ""
        fmt.cxx_member = "."
        fmt.cxx_addr = "&"

def find_arg_type(wlang, ntypemap):
    if wlang == "f":
        return ntypemap.ci_type or ntypemap.c_type
    else:
        return ntypemap.c_type
        
def find_arg_converter(wlang, language, ntypemap):
    """Find converter for an argument.
    Convert from the C wrapper to the C++ library type.
    The bufferify function (wlang == "f") will use ci_type.
    There is no ci_to_cxx since it would be identical to c_to_cxx.
    """
    if wlang == "f":
        converter = ntypemap.c_to_cxx
        lang = "ci_type"
    elif language == "c":
        converter = None
        lang = "c_type"
    else:
        converter = ntypemap.c_to_cxx
        lang = "c_type"
    return (converter, lang)

def find_result_converter(wlang, language, ntypemap):
    """Find converter for a result (or out argument).
    Convert from the C++ library type to the C wrapper.
    The bufferify function (wlang == "f") will use ci_type.
    """
    if wlang == "f":
        converter = ntypemap.cxx_to_ci or ntypemap.cxx_to_c
        lang = "ci_type"
    elif language == "c":
        converter = None
        lang = "c_type"
    else:
        converter = ntypemap.cxx_to_c
        lang = "c_type"
    return (converter, lang)

######################################################################

StateTuple = collections.namedtuple("StateType", "ast fmtdict language wlang")

class NonConst(object):
    """Return a non-const pointer to argument"""
    def __init__(self, state):
        self.state = state

    def __compute(self, name):
        arg = self.state.ast
        fmt = util.Scope(self.state.fmtdict)
        if arg.declarator.is_pointer():
            fmt.cxx_addr = ""
        else:
            fmt.cxx_addr = "&"
        # This convoluted eval is to get the proper error message
        # if name does not exist.
#        fmt.cxx_var = wformat("{{{}}}".format(name), self.state.fmtdict)
        fmt.cxx_var = self.state.fmtdict.get(name)
        if fmt.cxx_var is None:
            print("Missing name in nonconst.{}".format(name))
            return "===>nonconst.{}<===".format(name)
        if self.state.language == "c":
            if arg.const:
                value = wformat(
                    "({typemap.cxx_type} *) {cxx_addr}{cxx_var}", fmt)
            else:
                value = wformat(
                    "{cxx_addr}{cxx_var}", fmt)
        elif arg.const:
            # cast away constness
            value = wformat("const_cast<{cxx_type} *>\t({cxx_addr}{cxx_var})", fmt)
        else:
            value = wformat("{cxx_addr}{cxx_var}", fmt)
        return value

    def __getattr__(self, name):
        return self.__compute(name)

    def __str__(self):
        return self.__compute("cxx_var")

class FormatCdecl(object):
    """
    Return the C declaration from the ast.
    """
    def __init__(self, state):
        self.state = state

    def __getattr__(self, name):
        """If name is in fmtdict, use it. Else use name directly"""
        varname = self.state.fmtdict.get(name) or "===>{}<===".format(name)
        decl = gen_arg_as_c(self.state.ast, name=varname)
        return decl

    def __str__(self):
        #### Abstract declarator as C
        decl = gen_arg_as_c(self.state.ast, name=False)
        return decl
            
class FormatCXXdecl(object):
    """
    Return the original declaration from the ast.
    """
    def __init__(self, state):
        self.state = state

    def __getattr__(self, name):
        """If name is in fmtdict, use it. Else use name directly"""
        varname = self.state.fmtdict.get(name) or "===>{}<===".format(name)
        decl = gen_arg_as_cxx(self.state.ast,
                              with_template_args=True,
                              name=varname)
        return decl

    def __str__(self):
        decl = self.state.ast.to_string_declarator(abstract=True)
        return decl

class FormatCXXresult(object):
    """
    Return the original declaration from the ast without function parameters.
    """
    def __init__(self, state):
        self.state = state

    def __getattr__(self, name):
        """If name is in fmtdict, use it. Else use name directly"""
        varname = self.state.fmtdict.get(name) or "===>{}<===".format(name)
        decl = gen_arg_as_cxx(self.state.ast,
                              with_template_args=True,
                              add_params=False,  # Required for function results
                              name=varname)
        return decl

    def __str__(self):
        decl = self.state.ast.to_string_declarator(abstract=True)
        return decl

class FormatCIdecl(object):
    """
    Return a declaration used by the C interface.
    The main purpose to to use the correct C type when
    passing an enumeration.
    """
    def __init__(self, state):
        self.state = state

    def __getattr__(self, name):
        """If name is in fmtdict, use it. Else use name directly"""
        varname = self.state.fmtdict.get(name) or "===>{}<===".format(name)
        decl = gen_arg_as_c(self.state.ast, lang=maplang[self.state.wlang])
        return decl

    def __str__(self):
        decl = self.state.ast.to_string_declarator(abstract=True)
        return decl
            
class FormatGen(object):
    """
    An instance is added to the format dictionary for every AST
    as "gen". It is used to generate fields while processing
    statements.

      "{gen.cdecl}"
    """

    def __init__(self, func, ast, bind, wlang):
        """
        func - ast.FunctionNode
        ast  - declast.Declaration
        """
        self.language = func.get_language()
        self.ast     = ast
        self.bind    = bind
        self.fmtdict = bind.fmtdict
        state = self.state = StateTuple(ast, bind.fmtdict, self.language, wlang)
        self._cache = {}

        self.nonconst_addr = NonConst(state)
        self.cdecl = FormatCdecl(state)
        self.cxxdecl = FormatCXXdecl(state)
        self.cxxresult = FormatCXXresult(state)
        self.cidecl = FormatCIdecl(state)

    @property
    def c_to_cxx(self):
        return wformat(self.state.ast.typemap.c_to_cxx, self.state.fmtdict)

    @property
    def tester(self):
        return "tester"
        
    @property
    def name(self):
        return self.state.ast.declarator.user_name

    ##########
    @property
    def f_allocate_shape(self):
        """Shape to use with ALLOCATE statement from cdesc variable.
        Blank if scalar.
        """
        f_var_cdesc = self.fmtdict.get("f_var_cdesc", "===>f_var_cdesc<===")
        rank = int(self.fmtdict.get("rank", 0))
        if rank == 0:
            value = ""
        else:
            value = "(" + ",".join(
                ["{0}%shape({1})".format(f_var_cdesc, r)
                 for r in range(1, rank+1)]) + ")"
        return value

    @property
    def c_f_pointer_shape(self):
        """Shape for C_F_POINTER intrinsic from cdesc variable.
        Blank for scalars.
        """
        f_var_cdesc = self.fmtdict.get("f_var_cdesc", "===>f_var_cdesc<===")
        rank = self.fmtdict.get("rank", "0")
        if int(rank) == 0:
            value = ""
        else:
            value = ",\t {0}%shape(1:{1})".format(f_var_cdesc, rank)
        return value

    @property
    def f_cdesc_shape(self):
        """Assign variable shape to cdesc in Fortran using SHAPE intrinsic.
        This will be passed to C wrapper.
        Blank for scalars.
        """
        fmtdict = self.fmtdict
        f_var = fmtdict.get("f_var", "===>f_var<===")
        f_var_cdesc = fmtdict.get("f_var_cdesc", "===>f_var_cdesc<===")
        rank = fmtdict.get("rank", "0")
        if int(rank) == 0:
            value = ""
        else:
            value = "\n{0}%shape(1:{1}) = shape({2})".format(f_var_cdesc, rank, f_var)
        return value

    ##########
    @property
    def c_dimension_size(self):
        """Compute size of array from dimension attribute.
        "1" if scalar.
        """
        shape = self.bind.meta.get("dim_shape")
        if shape is None:
            return "1"
        fmtdim = ["({})".format(dim) for dim in shape]
        value = "*".join(fmtdim)
        return value

    @property
    def c_array_shape(self):
        """Assign array shape to a cdesc variable in C.
        Blank if scalar.
        """
        shape = self.bind.meta.get("dim_shape")
        if shape is None:
            return ""
        c_var_cdesc = self.fmtdict.get("c_var_cdesc",
                                       "===>c_var_cdesc<===")
        fmtshape = []
        for i, dim in enumerate(shape):
            fmtshape.append("{}->shape[{}] = {};".format(
                c_var_cdesc, i, dim))
        value = "\n" + "\n".join(fmtshape)
        return value

    @property
    def c_array_size(self):
        """Return expression to compute the size of an array.
        c_array_shape must be used first to define c_var_cdesc->shape.
        "1" if scalar.
        """
        shape = self.bind.meta.get("dim_shape")
        if shape is None:
            return "1"
        c_var_cdesc = self.fmtdict.get("c_var_cdesc",
                                       "===>c_var_cdesc<===")
        fmtsize = []
        for i, dim in enumerate(shape):
            fmtsize.append("{}->shape[{}]".format(
                c_var_cdesc, i, dim))
        value = "*\t".join(fmtsize)
        return value

    @property
    def c_extents_decl(self):
        """Define the shape in local variable extents
        in a CFI_index_t variable.
        Blank if scalar.
        """
        shape = self.bind.meta.get("dim_shape")
        if shape is None:
            return ""
        c_local_extents = self.fmtdict.get("c_local_extents",
                                           "===>c_local_extents<===")
        value = "CFI_index_t {0}[] = {{{1}}};\n".format(
            c_local_extents, ",\t ".join(shape))
        return value

    @property
    def c_extents_use(self):
        """Return variable name of extents of CFI array.
        NULL if scalar.
        """
        shape = self.bind.meta.get("dim_shape")
        if shape is None:
            return "NULL"
        c_local_extents = self.fmtdict.get("c_local_extents",
                                           "===>c_local_extents<===")
        return c_local_extents

    @property
    def c_lower_use(self):
        """Return variable name of lower bounds of CFI array
        from helper lower_bounds_CFI.
        NULL if scalar.
        """
        shape = self.bind.meta.get("dim_shape")
        if shape is None:
            return "NULL"
        helper = self.fmtdict.get("c_helper_lower_bounds_CFI",
                                  "===>c_helper_lower_bounds_CFI<===")
        return helper

    ##########
    def __str__(self):
        """  "{gen}" returns the name"""
        return self.name

    #@functools.cached_property
