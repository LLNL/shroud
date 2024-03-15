# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Fill the fmtdict for C and Fortran wrappers.
"""

from . import error
from . import todict
from . import statements
from . import util
from . import whelpers
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

class FillFormat(object):
    """Loop over Nodes and fill fmt dictionaries.
    """
    def __init__(self, newlibrary):
        self.newlibrary = newlibrary
        self.language = newlibrary.language
        self.cursor = error.get_cursor()

    def fmt_library(self):
        self.fmt_namespace(self.newlibrary.wrap_namespace)

    def fmt_namespace(self, node):
        cursor = self.cursor
        
        for cls in node.classes:
            cursor.push_phase("FillFormat class function")
            self.fmt_functions(cls, cls.functions)
            cursor.pop_phase("FillFormat class function")

        cursor.push_phase("FillFormat function")
        self.fmt_functions(None, node.functions)
        cursor.pop_phase("FillFormat function")

        for ns in node.namespaces:
            self.fmt_namespace(ns)

    def fmt_functions(self, cls, functions):
        for node in functions:
            if node.wrap.c:
                self.fmt_function("c", cls, node)
            if node.wrap.fortran:
                self.fmt_function("f", cls, node)

    def fmt_function(self, wlang, cls, node):
        cursor = self.cursor
        func_cursor = cursor.push_node(node)

        fmtlang = "fmt" + wlang

        fmt_func = node.fmtdict
        fmtargs = node._fmtargs
        fmt_arg0 = fmtargs.setdefault("+result", {})
        fmt_result = fmt_arg0.setdefault(fmtlang, util.Scope(fmt_func))

        bind = node._bind.setdefault(wlang, {})
        bind_result = bind.setdefault("+result", statements.BindArg())

        if wlang == "f":
            node.eval_template("F_name_impl")
            node.eval_template("F_name_function")
            node.eval_template("F_name_generic")

        ast = node.ast

        result_stmt = bind_result.stmt
        func_cursor.stmt = result_stmt
        fmt_result.stmt_name = result_stmt.name
        func_cursor.stmt = None

        # --- Loop over function parameters
        for arg in ast.declarator.params:
            func_cursor.arg = arg
            declarator = arg.declarator
            arg_name = declarator.user_name

            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg = fmt_arg0.setdefault(fmtlang, util.Scope(fmt_func))
            bind_arg = statements.fetch_arg_bind(node, arg, wlang)
            arg_stmt = bind_arg.stmt
            func_cursor.stmt = arg_stmt
            fmt_arg.stmt_name = arg_stmt.name

        # --- End loop over function parameters
        func_cursor.arg = None
        func_cursor.stmt = None
            
        cursor.pop_node(node)

    def fill_c_result(self, cls, node, result_stmt, fmt_result, CXX_ast, meta):
        ast = node.ast
        declarator = ast.declarator
        C_subprogram = declarator.get_subprogram()
        result_typemap = ast.typemap

        if C_subprogram != "subroutine":
            fmt_result.idtor = "0"  # no destructor
            fmt_result.c_var = fmt_result.C_local + fmt_result.C_result
            fmt_result.c_type = result_typemap.c_type
            fmt_result.cxx_type = result_typemap.cxx_type
            fmt_result.sh_type = result_typemap.sh_type
            fmt_result.cfi_type = result_typemap.cfi_type
            if ast.template_arguments:
                fmt_result.cxx_T = ','.join([str(targ) for targ in ast.template_arguments])
            if result_stmt.cxx_local_var == "result":
                # C result is passed in as an argument. Create local C++ name.
                fmt_result.cxx_var = fmt_result.CXX_local + fmt_result.C_result
            elif self.language == "c":
                fmt_result.cxx_var = fmt_result.c_var
            elif result_typemap.cxx_to_c is None:
                # C and C++ are compatible
                fmt_result.cxx_var = fmt_result.c_var
            else:
                fmt_result.cxx_var = fmt_result.CXX_local + fmt_result.C_result

            if ast.const:
                fmt_result.c_const = "const "
            else:
                fmt_result.c_const = ""

            fmt_result.cxx_rv_decl = CXX_ast.gen_arg_as_cxx(
                name=fmt_result.cxx_var, params=None, continuation=True
            )

            compute_cxx_deref(
                CXX_ast, result_stmt.cxx_local_var, fmt_result)

        if result_stmt.c_return_type:
            # Override return type.
            fmt_result.C_return_type = wformat(
                result_stmt.c_return_type, fmt_result)
        else:
            fmt_result.C_return_type = ast.gen_arg_as_c(
                name=None, params=None, continuation=True
            )
            
        self.name_temp_vars(fmt_result.C_result, result_stmt, fmt_result, "c")
        self.apply_c_helpers_from_stmts(node, result_stmt, fmt_result)
        statements.apply_fmtdict_from_stmts(result_stmt, fmt_result)
        self.find_idtor(node.ast, result_typemap, fmt_result, result_stmt, meta)
        self.set_fmt_fields_c(cls, node, ast, result_typemap, fmt_result, meta, True)

    def fill_c_arg(self, cls, node, arg, arg_stmt, fmt_arg, meta):
        declarator = arg.declarator
        arg_name = declarator.user_name
        arg_typemap = arg.typemap  # XXX - look up vector
        arg_typemap, junk = statements.lookup_c_statements(arg)
           
        fmt_arg.c_var = arg_name
        # XXX - order issue - c_var must be set before name_temp_vars,
        #       but set by set_fmt_fields
        self.name_temp_vars(arg_name, arg_stmt, fmt_arg, "c")
        self.set_fmt_fields_c(cls, node, arg, arg_typemap, fmt_arg, meta, False)
        self.apply_c_helpers_from_stmts(node, arg_stmt, fmt_arg)
        statements.apply_fmtdict_from_stmts(arg_stmt, fmt_arg)

        if arg_stmt.cxx_local_var:
            # Explicit conversion must be in pre_call.
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
        compute_cxx_deref(arg, arg_stmt.cxx_local_var, fmt_arg)
        self.set_cxx_nonconst_ptr(arg, fmt_arg)
        self.find_idtor(arg, arg_typemap, fmt_arg, arg_stmt, meta)

    def fill_interface_result(self, cls, node, bind, fmt_result):
        ast = node.ast
        declarator = ast.declarator
        subprogram = declarator.get_subprogram()
        result_typemap = ast.typemap
        result_stmt = bind.stmt

        if subprogram == "subroutine":
            fmt_result.F_C_subprogram = "subroutine"
        else:
            fmt_result.F_C_subprogram = "function"
            fmt_result.F_C_result_clause = "\fresult(%s)" % fmt_result.F_result
            fmt_result.i_var = fmt_result.F_result
            fmt_result.f_var = fmt_result.F_result
            fmt_result.f_intent = "OUT"
            fmt_result.c_type = result_typemap.c_type  # c_return_type
            fmt_result.f_type = result_typemap.f_type
            self.set_fmt_fields_iface(node, ast, bind, fmt_result,
                                      fmt_result.F_result, result_typemap,
                                      "function")
            self.set_fmt_fields_dimension(cls, node, ast, fmt_result, bind)

        if result_stmt.c_return_type == "void":
            # Change a function into a subroutine.
            fmt_result.F_C_subprogram = "subroutine"
            fmt_result.F_C_result_clause = ""
        elif result_stmt.c_return_type:
            # Change a subroutine into function
            # or change the return type.
            fmt_result.F_C_subprogram = "function"
            fmt_result.F_C_result_clause = "\fresult(%s)" % fmt_result.F_result
        if result_stmt.i_result_var:
            fmt_result.F_result = wformat(
                result_stmt.i_result_var, fmt_result)
            fmt_result.F_C_result_clause = "\fresult(%s)" % fmt_result.F_result
        self.name_temp_vars(fmt_result.C_result, result_stmt, fmt_result, "c", "i")
        statements.apply_fmtdict_from_stmts(result_stmt, fmt_result)

    def fill_interface_arg(self, cls, node, arg, bind, fmt_arg):
        declarator = arg.declarator
        arg_name = declarator.user_name
        arg_stmt = bind.stmt

        arg_typemap, junk = statements.lookup_c_statements(arg)
        fmt_arg.i_var = arg_name
        fmt_arg.f_var = arg_name
        self.set_fmt_fields_iface(node, arg, bind, fmt_arg, arg_name, arg_typemap)
        self.set_fmt_fields_dimension(cls, node, arg, fmt_arg, bind)
        self.name_temp_vars(arg_name, arg_stmt, fmt_arg, "c", "i")
        statements.apply_fmtdict_from_stmts(arg_stmt, fmt_arg)

    def fill_fortran_result(self, cls, node, bind, fmt_result):
        ast = node.ast
        declarator = ast.declarator
        result_typemap = ast.typemap
        result_stmt = bind.stmt
        C_node = node.C_node  # C wrapper to call.

        subprogram = declarator.get_subprogram()
        if result_stmt.f_result == "subroutine":
            subprogram = "subroutine"
        elif result_stmt.f_result is not None:
            subprogram = "function"
            fmt_result.F_result = result_stmt.f_result
        if subprogram == "function":
            fmt_result.f_var = fmt_result.F_result
            fmt_result.fc_var = fmt_result.F_result
            fmt_result.F_result_clause = "\fresult(%s)" % fmt_result.F_result
        fmt_result.F_subprogram = subprogram
        
        self.name_temp_vars(fmt_result.C_result, result_stmt, fmt_result, "f")
        self.set_fmt_fields_f(cls, C_node, ast, C_node.ast, bind, fmt_result,
                              subprogram, result_typemap)
        self.set_fmt_fields_dimension(cls, C_node, ast, fmt_result, bind)
        self.apply_helpers_from_stmts(node, result_stmt, fmt_result)
        statements.apply_fmtdict_from_stmts(result_stmt, fmt_result)

    def fill_fortran_arg(self, cls, node, C_node, f_arg, c_arg, bind, fmt_arg):
        arg_name = f_arg.declarator.user_name
        arg_stmt = bind.stmt

        fmt_arg.f_var = arg_name
        fmt_arg.fc_var = arg_name
        self.name_temp_vars(arg_name, arg_stmt, fmt_arg, "f")
        arg_typemap = self.set_fmt_fields_f(cls, C_node, f_arg, c_arg, bind, fmt_arg)
        self.set_fmt_fields_dimension(cls, C_node, f_arg, fmt_arg, bind)
        self.apply_helpers_from_stmts(node, arg_stmt, fmt_arg)
        statements.apply_fmtdict_from_stmts(arg_stmt, fmt_arg)
        return arg_typemap
    
    def name_temp_vars(self, rootname, stmts, fmt, lang, prefix=None):
        """Compute names of temporary C variables.

        Create stmts.temps and stmts.local variables.

        lang - "c", "f"
        prefix - "c", "f", "i"
        """
        if prefix is None:
            prefix = lang

        names = stmts.get(lang + "_temps", None)
        if names is not None:
            for name in names:
                setattr(fmt,
                        "{}_var_{}".format(prefix, name),
                        "{}{}_{}".format(fmt.c_temp, rootname, name))
        names = stmts.get(lang + "_local", None)
        if names is not None:
            for name in names:
                setattr(fmt,
                        "{}_local_{}".format(prefix, name),
                        "{}{}_{}".format(fmt.C_local, rootname, name))

    def set_fmt_fields_c(self, cls, fcn, ast, ntypemap, fmt, meta, is_func):
        """
        Set format fields for ast.
        Used with arguments and results.

        Args:
            cls      - ast.ClassNode or None of enclosing class.
            fcn      - ast.FunctionNode of calling function.
            ast      - declast.Declaration
            ntypemap - typemap.Typemap
            fmt      - scope.Util
            meta     -
            is_func  - True if function.
        """
        declarator = ast.declarator
        if is_func:
            rootname = fmt.C_result
        else:
            rootname = declarator.user_name
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

            if ntypemap.base != "shadow" and ast.template_arguments:
                fmt.cxx_T = ','.join([str(targ) for targ in ast.template_arguments])
            
            if meta["blanknull"]:
                # Argument to helper ShroudStrAlloc via attr[blanknull].
                fmt.c_blanknull = "1"
        
        attrs = declarator.attrs
        
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
                fmtdim = []
                for dim in visitor.shape:
                    fmtdim.append(dim)
                if fmtdim:
                    # Multiply dimensions together to get size.
                    fmt.c_array_size2 = "*\t".join(fmtdim)

                if hasattr(fmt, "c_var_cdesc"):
                    # array_type is assumed to be c_var_cdesc.
                    # Assign each rank of dimension.
                    fmtshape = []
                    fmtsize = []
                    for i, dim in enumerate(visitor.shape):
                        fmtshape.append("{}->shape[{}] = {};".format(
                            fmt.c_var_cdesc, i, dim))
                        fmtsize.append("{}->shape[{}]".format(
                            fmt.c_var_cdesc, i, dim))
                    fmt.c_array_shape = "\n" + "\n".join(fmtshape)
                    if fmtsize:
                        # Multiply extents together to get size.
                        fmt.c_array_size = "*\t".join(fmtsize)
                if hasattr(fmt, "c_var_extents"):
                    # Used with CFI_establish
                    fmtextent = []
                    for i, dim in enumerate(visitor.shape):
                        fmtextent.append("{}[{}] = {};\n".format(
                            fmt.c_var_extents, i, dim))
                    fmt.c_temp_extents_decl = (
                        "CFI_index_t {0}[{1}];\n{2}".
                        format(fmt.c_var_extents, fmt.rank,
                               "".join(fmtextent)))
                    # Used with CFI_setpointer to set lower bound to 1.
                    fmt.c_temp_lower_decl = (
                        "CFI_index_t {0}[{1}] = {{{2}}};\n".
                        format(fmt.c_var_lower, fmt.rank,
                               ",".join(["1" for x in range(visitor.rank)])))
                    fmt.c_temp_extents_use = fmt.c_var_extents
                    fmt.c_temp_lower_use = fmt.c_var_lower

        if "len" in attrs:
            fmt.c_char_len = attrs["len"];
                
    def set_fmt_fields_iface(self, fcn, ast, bind, fmt, rootname,
                             ntypemap, subprogram=None):
        """Set format fields for interface.

        Transfer info from Typemap to fmt for use by statements.

        Parameters
        ----------
        fcn : ast.FunctionNode
        ast : declast.Declaration
        fmt : util.Scope
        rootname : str
        ntypemap : typemap.Typemap
            The typemap has already resolved template arguments.
            For example, std::vector<int>.  ntypemap will be 'int'.
        subprogram : str
            "function" or "subroutine" or None
        """
        attrs = ast.declarator.attrs
        meta = bind.meta

        if subprogram == "subroutine":
            pass
        elif subprogram == "function":
            # XXX this also gets set for subroutines
            fmt.f_intent = "OUT"
        else:
            fmt.f_intent = meta["intent"].upper()
            if fmt.f_intent == "SETTER":
                fmt.f_intent = "IN"
        
        fmt.f_type = ntypemap.f_type
        fmt.sh_type = ntypemap.sh_type
        if ntypemap.f_kind:
            fmt.f_kind = ntypemap.f_kind
        if ntypemap.f_capsule_data_type:
            fmt.f_capsule_data_type = ntypemap.f_capsule_data_type
        if ntypemap.f_derived_type:
            fmt.f_derived_type = ntypemap.f_derived_type
        if ntypemap.f_module_name:
            fmt.f_type_module = ntypemap.f_module_name

    def set_fmt_fields_f(self, cls, fcn, f_ast, c_ast, bind, fmt,
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
        fmt : format dictionary
        subprogram : str
        ntypemap : typemap.Typemap
        """
        c_attrs = c_ast.declarator.attrs

        if subprogram == "subroutine":
            # XXX - no need to set f_type and sh_type
            rootname = fmt.C_result
        elif subprogram == "function":
            # XXX this also gets set for subroutines
            rootname = fmt.C_result
        else:
            ntypemap = f_ast.typemap
            rootname = c_ast.declarator.user_name
        if ntypemap.sgroup != "shadow" and c_ast.template_arguments:
            # XXX - need to add an argument for each template arg
            ntypemap = c_ast.template_arguments[0].typemap
            fmt.cxx_T = ','.join([str(targ) for targ in c_ast.template_arguments])
        if subprogram != "subroutine":
            self.set_fmt_fields_iface(fcn, c_ast, bind, fmt, rootname,
                                      ntypemap, subprogram)
            if "pass" in c_attrs:
                # Used with wrap_struct_as=class for passed-object dummy argument.
                fmt.f_type = ntypemap.f_class
        return ntypemap

    def set_fmt_fields_dimension(self, cls, fcn, f_ast, fmt, bind):
        """Set fmt fields based on dimension attribute.

        f_assumed_shape is used in both implementation and interface.

        Parameters
        ----------
        cls : ast.ClassNode or None of enclosing class.
        fcn : ast.FunctionNode of calling function.
        f_ast : declast.Declaration
        fmt: util.Scope
        """
        f_attrs = f_ast.declarator.attrs
        meta = bind.meta
        dim = meta["dim_ast"]
        rank = meta["rank"]
        if meta["dimension"] == "..":   # assumed-rank
            fmt.i_dimension = "(..)"
            fmt.f_assumed_shape = "(..)"
        elif rank is not None:
            fmt.rank = str(rank)
            if rank == 0:
                # Assigned to cdesc to pass metadata to C wrapper.
                fmt.size = "1"
                if hasattr(fmt, "f_var_cdesc"):
                    fmt.f_cdesc_shape = ""
            else:
                fmt.size = wformat("size({f_var})", fmt)
                fmt.f_assumed_shape = fortran_ranks[rank]
                fmt.i_dimension = "(*)"
                if hasattr(fmt, "f_var_cdesc"):
                    fmt.f_cdesc_shape = wformat("\n{f_var_cdesc}%shape(1:{rank}) = shape({f_var})", fmt)
        elif dim:
            visitor = ToDimension(cls, fcn, fmt)
            visitor.visit(dim)
            rank = visitor.rank
            fmt.rank = str(rank)
            if rank != "assumed" and rank > 0:
                fmt.f_assumed_shape = fortran_ranks[rank]
                # XXX use f_var_cdesc since shape is assigned in C
                fmt.f_array_allocate = "(" + ",".join(visitor.shape) + ")"
                if hasattr(fmt, "f_var_cdesc"):
                    # XXX kludge, name is assumed to be f_var_cdesc.
                    fmt.f_cdesc_shape = wformat("\n{f_var_cdesc}%shape(1:{rank}) = shape({f_var})", fmt)
                    # XXX - maybe avoid {rank} with: {f_var_cdes}(:rank({f_var})) = shape({f_var})
                    fmt.f_array_allocate = "(" + ",".join(
                        ["{0}%shape({1})".format(fmt.f_var_cdesc, r)
                         for r in range(1, rank+1)]) + ")"
                    fmt.f_array_shape = wformat(
                        ",\t {f_var_cdesc}%shape(1:{rank})", fmt)

        if "len" in f_attrs:
            fmt.f_char_len = "len=%s" % f_attrs["len"];
        elif hasattr(fmt, "f_var_cdesc"):
            if meta["deref"] == "allocatable":
                # Use elem_len from the C wrapper.
                fmt.f_char_type = wformat("character(len={f_var_cdesc}%elem_len) ::\t ", fmt)

    def apply_c_helpers_from_stmts(self, node, stmt, fmt):
        node_helpers = node.helpers.setdefault("c", {})
        add_c_helper(node_helpers, stmt.c_helper, fmt)

    def apply_helpers_from_stmts(self, node, stmt, fmt):
        node_helpers = node.helpers.setdefault("c", {})
        add_c_helper(node_helpers, stmt.c_helper, fmt)
        node_helpers = node.helpers.setdefault("f", {})
        add_f_helper(node_helpers, stmt.f_helper, fmt)
        
def add_c_helper(node_helpers, helpers, fmt):
    """Add a list of C helpers."""
    for c_helper in helpers:
        helper = wformat(c_helper, fmt)
        if helper not in whelpers.CHelpers:
            error.get_cursor().warning("No such c_helper '{}'".format(helper))
        else:
            node_helpers[helper] = True
            name = whelpers.CHelpers[helper].get("name")
            if name:
                setattr(fmt, "c_helper_" + helper, name)

def add_f_helper(node_helpers, helpers, fmt):
    """Add a list of Fortran helpers.
    Add fmt.fhelper_X for use by pre_call and post_call.
    """
    for f_helper in helpers:
        helper = wformat(f_helper, fmt)
        if helper not in whelpers.FHelpers:
            error.get_cursor().warning("No such f_helper '{}'".format(helper))
        else:
            node_helpers[helper] = True
            name = whelpers.FHelpers[helper].get("name")
            if name:
                setattr(fmt, "f_helper_" + helper, name)


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
            if self.fcn.ast.declarator.find_arg_by_name(argname) is None:
                self.need_helper = True
            if node.args is None:
                return argname  # variable
            else:
                return self.param_list(node) # function
        return "--??--"

    def visit_AssumedRank(self, node):
        # (..)
        self.rank = "assumed"
        return "===assumed-rank==="
        error.get_cursor().warning("Detected assumed-rank dimension")
                
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
    elif arg.declarator.is_indirect(): #pointer():
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
    elif arg.declarator.is_pointer():
#        fmt.cxx_deref = "*"
        fmt.cxx_member = "->"
        fmt.cxx_addr = ""
    else:
#        fmt.cxx_deref = ""
        fmt.cxx_member = "."
        fmt.cxx_addr = "&"
