# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Fill the fmtdict for C and Fortran wrappers.
"""

from . import error
from . import todict
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
            for func in cls.functions:
                self.fmt_function(cls, func)
            cursor.pop_phase("FillFormat class function")

        cursor.push_phase("FillFormat function")
        for func in node.functions:
            self.fmt_function(None, func)
        cursor.pop_phase("FillFormat function")

        for ns in node.namespaces:
            self.fmt_namespace(ns)

    def fmt_function(self, cls, node):
        cursor = self.cursor
        func_cursor = cursor.push_node(node)
        if node.wrap.c:
            node.eval_template("C_name")
            node.eval_template("F_C_name")
        if node.wrap.fortran:
            node.eval_template("F_name_impl")
            node.eval_template("F_name_function")
            node.eval_template("F_name_generic")

        if not node.wrap.fortran:
            cursor.pop_node(node)
            return
        cursor.pop_node(node)
        return  # <--- work in progress

        locate_c_function(self.newlibrary, node)
            
        fmt_func = node.fmtdict
        fmt_func = util.Scope(node.fmtdict)
        node.fmtdict2 = fmt_func  # XXX - migration

        C_node = node.C_node  # C wrapper to call.

        fmt_func.F_C_call = C_node.fmtdict.F_C_name
        fmtargs = C_node._fmtargs

        ast = node.ast
        declarator = ast.declarator
        subprogram = declarator.get_subprogram()
        result_typemap = ast.typemap

        r_attrs = declarator.attrs
        r_meta = declarator.metaattrs
        sintent = r_meta["intent"]
        fmt_result = node._fmtresult.setdefault("fmtf2", util.Scope(fmt_func))
        result_stmt = statements.lookup_fc_function(node)
        
        # wrap c
        result_api = r_meta["api"]
        result_is_const = ast.const
        sintent = r_meta["intent"]
        
        if subprogram == "subroutine":
            # C wrapper
            fmt_pattern = fmt_func
            # interface
            fmt_func.F_C_subprogram = "subroutine"
        else:
            ## C wrapper
            fmt_result.idtor = "0"  # no destructor
            fmt_result.c_var = fmt_result.C_local + fmt_result.C_result
            fmt_result.c_type = result_typemap.c_type
            fmt_result.cxx_type = result_typemap.cxx_type
            fmt_result.sh_type = result_typemap.sh_type
            fmt_result.cfi_type = result_typemap.cfi_type
            if ast.template_arguments:
                fmt_result.cxx_T = ','.join([str(targ) for targ in ast.template_arguments])
#                for targ in ast.template_arguments:
#                    header_typedef_nodes[targ.typemap.name] = targ.typemap
#            else:
#                header_typedef_nodes[result_typemap.name] = result_typemap
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

            if result_is_const:
                fmt_result.c_const = "const "
            else:
                fmt_result.c_const = ""

            fmt_func.cxx_rv_decl = CXX_ast.gen_arg_as_cxx(
                name=fmt_result.cxx_var, params=None, continuation=True
            )

 #           compute_cxx_deref(
 #               CXX_ast, result_stmt.cxx_local_var, fmt_result)
 #           fmt_pattern = fmt_result
            ## F impl
            fmt_result.f_var = fmt_func.F_result
            fmt_result.fc_var = fmt_func.F_result
            fmt_func.F_result_clause = "\fresult(%s)" % fmt_func.F_result
            ## interface
            fmt_func.F_C_subprogram = "function"
            fmt_func.F_C_result_clause = "\fresult(%s)" % fmt_func.F_result
            fmt_result.i_var = fmt_func.F_result
            fmt_result.f_var = fmt_func.F_result
            fmt_result.f_intent = "OUT"
            fmt_result.f_type = result_typemap.f_type
            self.set_fmt_fields_iface(node, ast, fmt_result,
                                      fmt_func.F_result, result_typemap,
                                      "function")
            self.set_fmt_fields_dimension(cls, node, ast, fmt_result)

        ## C wrapper
#        result_stmt = statements.lookup_local_stmts(
#            ["c", result_api], result_stmt, node)
#        func_cursor.stmt = result_stmt
#        self.name_temp_vars(fmt_result.C_result, result_stmt, fmt_result, "c")
#        self.add_c_helper(result_stmt.c_helper, fmt_result)
        statements.apply_fmtdict_from_stmts(result_stmt, fmt_result)
        
        ## Fortran impl
        fmt_func.F_subprogram = subprogram
        result_stmt = statements.lookup_local_stmts("f", result_stmt, node)
        fmt_result.stmtf = result_stmt.name
        func_cursor.stmt = result_stmt

        self.name_temp_vars(fmt_func.C_result, result_stmt, fmt_result, "f")
        self.set_fmt_fields(cls, C_node, ast, C_node.ast, fmt_result,
                            subprogram, result_typemap)
        self.set_fmt_fields_dimension(cls, C_node, ast, fmt_result)
#        fileinfo.apply_helpers_from_stmts(result_stmt, fmt_result)
        statements.apply_fmtdict_from_stmts(result_stmt, fmt_result)

        # interface
        r_attrs = ast.declarator.attrs
        r_meta = ast.declarator.metaattrs
        result_api = r_meta["api"]
        sintent = r_meta["intent"]
        
        result_stmt = statements.lookup_local_stmts(
            ["c", result_api], result_stmt, node)
        

        f_args = ast.declarator.params
        f_index = -1  # index into f_args
        for c_arg in C_node.ast.declarator.params:
            func_cursor.arg = c_arg
            arg_name = c_arg.declarator.user_name
            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg = fmt_arg0.setdefault("fmtf2", util.Scope(fmt_func))

            ## C wrapper
            arg = c_arg
            declarator = arg.declarator
            arg_name = declarator.user_name
            c_attrs = declarator.attrs
            c_meta = declarator.metaattrs

            arg_typemap = arg.typemap  # XXX - look up vector
            sgroup = arg_typemap.sgroup

#            self.header_impl.add_typemap_list(arg_typemap.impl_header)
                    
            arg_typemap, specialize = statements.lookup_c_statements(arg)
#            header_typedef_nodes[arg_typemap.name] = arg_typemap
            cxx_local_var = ""
            sapi = c_meta["api"]

            # regular argument (not function result)
            arg_call = arg
            spointer = declarator.get_indirect_stmt()
            if c_attrs["hidden"] and node._generated:
                sapi = "hidden"
            stmts = ["f", c_meta["intent"], sgroup, spointer,
                     sapi, c_meta["deref"], c_attrs["owner"]] + specialize
            arg_stmt = statements.lookup_fc_stmts(stmts)
            func_cursor.stmt = arg_stmt
            fmt_arg.c_var = arg_name
            # XXX - order issue - c_var must be set before name_temp_vars,
            #       but set by set_fmt_fields
#            self.name_temp_vars(arg_name, arg_stmt, fmt_arg, "c")
#            self.set_fmt_fields(cls, node, arg, arg_typemap, fmt_arg, False)
#            self.add_c_helper(arg_stmt.c_helper, fmt_arg)
            statements.apply_fmtdict_from_stmts(arg_stmt, fmt_arg)

            if arg_stmt.cxx_local_var:
                # Explicit conversion must be in pre_call.
                cxx_local_var = arg_stmt.cxx_local_var
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
#                append_format(
#                    pre_call, "{cxx_decl} =\t {cxx_val};", fmt_arg
#                )
#            compute_cxx_deref(arg, cxx_local_var, fmt_arg)

            ## F impl
            fmt_arg.f_var = arg_name
            fmt_arg.fc_var = arg_name

            c_declarator = c_arg.declarator
            c_attrs = c_declarator.attrs
            c_meta = c_declarator.metaattrs
#            hidden = c_attrs["hidden"]
            intent = c_meta["intent"]
#            optattr = False

            junk, specialize = statements.lookup_c_statements(c_arg)
            
            # An argument to the C and Fortran function
            f_index += 1
            f_arg = f_args[f_index]
#            f_declarator = f_arg.declarator
#            f_name = f_declarator.user_name
#            f_attrs = f_declarator.attrs

            c_sgroup = c_arg.typemap.sgroup
            c_spointer = c_declarator.get_indirect_stmt()
            # Pass metaattrs["api"] to both Fortran and C (i.e. "buf").
            # Fortran need to know how the C function is being called.
            f_stmts = ["f", intent, c_sgroup, c_spointer, c_meta["api"],
                       c_meta["deref"], c_attrs["owner"]]
            f_stmts.extend(specialize)

            arg_stmt = statements.lookup_fc_stmts(f_stmts)
            func_cursor.stmt = arg_stmt
            self.name_temp_vars(arg_name, arg_stmt, fmt_arg, "f")
            arg_typemap = self.set_fmt_fields(
                cls, C_node, f_arg, c_arg, fmt_arg)
            self.set_fmt_fields_dimension(cls, C_node, f_arg, fmt_arg)
 #           fileinfo.apply_helpers_from_stmts(arg_stmt, fmt_arg)
            statements.apply_fmtdict_from_stmts(arg_stmt, fmt_arg)

            ## interface
            fmt_arg.i_var = arg_name
            fmt_arg.f_var = arg_name
            self.set_fmt_fields_iface(node, c_arg, fmt_arg, arg_name, arg_typemap)
            self.set_fmt_fields_dimension(cls, node, c_arg, fmt_arg)
        # --- End loop over function parameters
        #####
        func_cursor.arg = None
        func_cursor.stmt = result_stmt

        cursor.pop_node(node)

# From util.py
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

    def set_fmt_fields_iface(self, fcn, ast, fmt, rootname,
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
        meta = ast.declarator.metaattrs

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

    def set_fmt_fields(self, cls, fcn, f_ast, c_ast, fmt,
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
        c_meta = c_ast.declarator.metaattrs

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
            self.set_fmt_fields_iface(fcn, c_ast, fmt, rootname,
                                      ntypemap, subprogram)
            if c_attrs["pass"]:
                # Used with wrap_struct_as=class for passed-object dummy argument.
                fmt.f_type = ntypemap.f_class
        return ntypemap

    def set_fmt_fields_dimension(self, cls, fcn, f_ast, fmt):
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
        f_meta = f_ast.declarator.metaattrs
        dim = f_meta["dimension"]
        rank = f_attrs["rank"]
        if f_meta["assumed-rank"]:
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

        if f_attrs["len"]:
            fmt.f_char_len = "len=%s" % f_attrs["len"];
        elif hasattr(fmt, "f_var_cdesc"):
            if f_attrs["deref"] == "allocatable":
                # Use elem_len from the C wrapper.
                fmt.f_char_type = wformat("character(len={f_var_cdesc}%elem_len) ::\t ", fmt)

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

                
