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

        cursor.pop_node(node)

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

    def set_fmt_fields_c(self, cls, fcn, ast, ntypemap, fmt, is_func):
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
            
            if ast.blanknull:
                # Argument to helper ShroudStrAlloc via attr[blanknull].
                fmt.c_blanknull = "1"
        
        attrs = declarator.attrs
        meta = declarator.metaattrs
        
        if meta["dimension"]:
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
            visitor.visit(meta["dimension"])
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

        if attrs["len"]:
            fmt.c_char_len = attrs["len"];
                
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

    def set_fmt_fields_f(self, cls, fcn, f_ast, c_ast, fmt,
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
                if declarator.attrs["hidden"]:
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
        raise RuntimeError("wrapc.py: Detected assumed-rank dimension")

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
