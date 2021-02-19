# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
"""

from . import util

# The tree of c and fortran statements.
cf_tree = {}
default_scopes = dict()

def lookup_c_statements(arg):
    """Look up the c_statements for an argument.
    If the argument type is a template, look for
    template specialization.

    Args:
        arg -
    """
    arg_typemap = arg.typemap

    specialize = []
    if arg.template_arguments:
        arg_typemap = arg.template_arguments[0].typemap
        specialize.append(arg_typemap.sgroup)
    return arg_typemap, specialize

def lookup_fc_stmts(path):
    return lookup_stmts_tree(cf_tree, path)
        
def compute_name(path, char="_"):
    """
    Compute a name from a list of components.
    Blank entries are filtered out.

    Used to find C_error_pattern.
    
    Args:
        path  - list of name components.
    """
    work = [ part for part in path if part ] # skip empty components
    return char.join(work)

def lookup_local_stmts(path, parent, node):
    """Look in node.fstatements for additional statements.
    XXX - Only used with result.
    mode - "update", "replace"

    Args:
        path   - list of path components ["c", "buf"]
        parent - parent Scope.
        node   - FunctionNode.
    """
    name = compute_name(path)
    blk = node.fstatements.get(name, None)
    if blk:
        mode = blk.get("mode", "update")
        if mode == "update":
            blk.reparent(parent)
            return blk
    return parent

def create_buf_variable_names(options, blk, attrs):
    """Turn on attribute for buf_arg if defined in blk.
    """
    for buf_arg in blk.buf_args:
        if attrs[buf_arg] is not None and \
           attrs[buf_arg] is not True:
            # None - Not set.
            # True - Do not override user specified variable name.
            pass
        elif buf_arg in ["size", "capsule", "context",
                         "len_trim", "len"]:
            attrs[buf_arg] = True

def set_buf_variable_names(options, attrs, c_var):
    """Set attribute name from option template.
    XXX - make sure they don't conflict with other names.
    """
    if attrs["size"] is True:
        attrs["size"] = options.C_var_size_template.format(
            c_var=c_var
        )
    if attrs["capsule"] is True:
        attrs["capsule"] = options.C_var_capsule_template.format(
            c_var=c_var
        )
    if attrs["owner"] == "caller" and \
       attrs["deref"] == "pointer" \
              and attrs["capsule"] is None:
        attrs["capsule"] = options.C_var_capsule_template.format(
            c_var=c_var
        )
    if attrs["context"] is True:
        attrs["context"] = options.C_var_context_template.format(
            c_var=c_var
        )
    if attrs["cdesc"] is True:
        # XXX - not sure about future of cdesc and difference with context.
        attrs["context"] = options.C_var_context_template.format(
            c_var=c_var
        )
    if attrs["len_trim"] is True:
        attrs["len_trim"] = options.C_var_trim_template.format(
            c_var=c_var
        )
    if attrs["len"] is True:
        attrs["len"] = options.C_var_len_template.format(
            c_var=c_var
        )

def assign_buf_variable_names(attrs, fmt):
    """
    Transfer names from attribute to fmt.
    """
    if attrs["capsule"]:
        fmt.c_var_capsule = attrs["capsule"]
    if attrs["context"]:
        fmt.c_var_context = attrs["context"]
    if attrs["len"]:
        fmt.c_var_len = attrs["len"]
    if attrs["len_trim"]:
        fmt.c_var_trim = attrs["len_trim"]
    if attrs["size"]:
        fmt.c_var_size = attrs["size"]
            

def compute_return_prefix(arg, local_var):
    """Compute how to access variable: dereference, address, as-is"""
    if local_var == "scalar":
        if arg.is_pointer():
            return "&"
        else:
            return ""
    elif local_var == "pointer":
        if arg.is_pointer():
            return ""
        else:
            return "*"
    elif local_var == "funcptr":
        return ""
    elif arg.is_reference():
        # Convert a return reference into a pointer.
        return "&"
    else:
        return ""


def update_statements_for_language(language):
    """Preprocess statements for lookup.

    Update statements for c or c++.
    Fill in cf_tree.

    Parameters
    ----------
    language : str
        "c" or "c++"
    """
    update_for_language(fc_statements, language)
    update_stmt_tree(fc_statements, cf_tree, default_stmts)
    

def update_for_language(stmts, lang):
    """
    Move language specific entries to current language.

    stmts=[
      dict(
        name='foo_bar',
        c_declare=[],
        cxx_declare=[],
      ),
      ...
    ]

    For lang==c,
      foo_bar["declare"] = foo_bar["c_declare"]
    """
    for item in stmts:
        for clause in [
                "impl_header",
                "cxx_local_var",
                "declare",
                "post_parse",
                "pre_call",
                "post_call",
                "cleanup",
                "fail",
        ]:
            specific = lang + "_" + clause
            if specific in item:
                # XXX - maybe make sure clause does not already exist.
                item[clause] = item[specific]


def update_stmt_tree(stmts, tree, defaults):
    """Update tree by adding stmts.  Each key in stmts is split by
    underscore then inserted into tree to form nested dictionaries to
    the values from stmts.  The end key is named _node, since it is
    impossible to have an intermediate element with that name (since
    they're split on underscore).

    Implement "base" field.  Base must be defined before use.

    Add "_key" to tree to aid debugging.

    Each typemap is converted into a Scope instance with the parent
    based on the language (c or f) and added as "scope" field.
    This additional layer of indirection is needed to implement base.

    stmts = [
       {name="c_native_in",}           # value1
       {name="c_native_out",}          # value2
       {name="c_native_pointer_out",}  # value3
       {name="c_string_in",}           # value4
    ]
    tree = {
      "c": {
         "native": {
           "in": {"_node":value1},
           "out":{"_node":value2},
           "pointer":{
             "out":{"_node":value3},
           },
         },
         "string":{
           "in": {"_node":value4},
         },
      },
    }

    """
    # Convert defaults into Scope nodes.
    for key, node in defaults.items():
        default_scopes[key] = node()

    # Index by name to find alias, base, mixin.
    nodes = {}
    for node in stmts:
        # node is a dict.
        if "name" not in node:
            raise RuntimeError("Missing name in statements: {}".
                               format(str(node)))
        if node["name"] in nodes:
            raise RuntimeError("Duplicate key in statements: {}".
                               format(node["name"]))
        nodes[node["name"]] = node

    for node in stmts:
        key = node["name"]
        step = tree
        steps = key.split("_")
        label = []
        for part in steps:
            step = step.setdefault(part, {})
            label.append(part)
            step["_key"] = "_".join(label)
#        if "alias" in node:
#            step['_node'] = nodes[node["alias"]]
        if "base" in node:
            step['_node'] = node
            scope = util.Scope(nodes[node["base"]]["scope"])
            scope.update(node)
            node["scope"] = scope
        else:
            step['_node'] = node
            scope = util.Scope(default_scopes[steps[0]])
            if "mixin" in node:
                for mpart in node["mixin"]:
                    scope.update(nodes[mpart])
            scope.update(node)
            node["scope"] = scope
#    print_tree(tree)

def print_tree(tree, indent=""):
    """Print statements search tree.
    Intermediate nodes are prefixed with --.
    Useful for debugging.
    """
    parts = tree.get('_key', 'root').split('_')
    if "_node" in tree:
        #        final = '' # + tree["_node"]["scope"].name + '-'
        print("{}{} -- {}".format(indent, parts[-1], tree.get('_key', '??')))
    else:
        print("{}{}".format(indent, parts[-1]))
    indent += '  '
    for key in sorted(tree.keys()):
        if key == '_node':
            continue
        if key == 'scope':
            continue
        if key == '_key':
            continue
        value = tree[key]
        if isinstance(value, dict):
            print_tree(value, indent)

def lookup_stmts_tree(tree, path):
    """
    Lookup path in statements tree.
    Look for longest path which matches.
    Used to find specific cases first, then fall back to general.
    ex path = ['result', 'allocatable']
         Finds 'result_allocatable' if it exists, else 'result'.
    If not found, return an empty dictionary.

    path typically consists of:
      in, out, inout, result
      generated_clause - buf
      deref - allocatable

    Args:
        tree  - dictionary of nested dictionaries
        path  - list of name components.
                Blank entries are ignored.
    """
    found = default_scopes[path[0]]
    work = []
    step = tree
    for part in path:
        if not part:
            # skip empty parts
            continue
        if part not in step:
            continue
        step = step[part]
        if "_node" in step:
            # Path ends here.
            found = step["_node"]["scope"]
#    if not isinstance(found, util.Scope):
#        raise RuntimeError
    return found


class CStmts(object):
    """C Statements.
    arg_call    - List of arguments passed to C function.

    Used with buf_args = "arg_decl".
    c_arg_decl  - Add C declaration to C wrapper with buf_args=arg_decl
    f_arg_decl  - Add Fortran declaration to Fortran wrapper interface block
                  with buf_args=arg_decl.
    f_result_decl - Declaration for function result.
    f_module    - Add module info to interface block.
    """
    def __init__(self,
        name="c_default",
        buf_args=[], buf_extra=[],
        iface_header=[],
        impl_header=[],
        c_helper="", c_local_var=None,
        cxx_local_var=None,
        arg_call=[],
        pre_call=[], call=[], post_call=[], final=[], ret=[],
        destructor_name=None,
        owner="library",
        return_type=None, return_cptr=False,
        c_arg_decl=[],
        f_arg_decl=[],
        f_result_decl=[],
        f_module=None,
    ):
        self.name = name
        self.buf_args = buf_args
        self.buf_extra = buf_extra
        self.iface_header = iface_header
        self.impl_header = impl_header
        self.c_helper = c_helper
        self.c_local_var = c_local_var
        self.cxx_local_var = cxx_local_var

        self.pre_call = pre_call
        self.call = call
        self.arg_call = arg_call
        self.post_call = post_call
        self.final = final
        self.ret = ret

        self.destructor_name = destructor_name
        self.owner = owner
        self.return_type = return_type
        self.return_cptr = return_cptr
        self.c_arg_decl = c_arg_decl
        self.f_arg_decl = f_arg_decl
        self.f_result_decl = f_result_decl
        self.f_module = f_module

class FStmts(object):
    """Fortran Statements.

    """
    def __init__(self,
        name="f_default",
        c_helper="",
        c_local_var=None,
        f_helper="", f_module=None,
        need_wrapper=False,
        arg_name=None,
        arg_decl=None,
        arg_c_call=None,
        declare=[], pre_call=[], call=[], post_call=[],
        result=None,  # name of result variable
    ):
        self.name = name
        self.c_helper = c_helper
        self.c_local_var = c_local_var
        self.f_helper = f_helper
        self.f_module = f_module

        self.need_wrapper = need_wrapper
        self.arg_name = arg_name        # Names in subprogram list.
        self.arg_decl = arg_decl        # argument/result declaration
        self.arg_c_call = arg_c_call    # argument to C function.
        self.declare = declare          # local declaration
        self.pre_call = pre_call
        self.call = call
        self.post_call = post_call
        self.result = result

# Define class for nodes in tree based on their first entry.
# c_native_*_in uses 'c'.
default_stmts = dict(
    c=CStmts,
    f=FStmts,
)
                
        

# language   "c" 
# sgroup     "native", "string", "char"
# spointer   "pointer" ""
# intent     "in", "out", "inout", "result"
# generated  "buf"
# attribute  "allocatable"
#
# language   "f"
# sgroup     "native", "string", "char"
# spointer   "pointer" ""
# intent     "in", "out", "inout", "result"
# deref      "allocatable", "pointer"

fc_statements = [
    dict(
        name="f_bool_in",
        c_local_var=True,
        pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
    ),
    dict(
        name="f_bool_out",
        c_local_var=True,
        post_call=["{f_var} = {c_var}  ! coerce to logical"],
    ),
    dict(
        name="f_bool_inout",
        c_local_var=True,
        pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
        post_call=["{f_var} = {c_var}  ! coerce to logical"],
    ),
    dict(
        name="f_bool_result",
        # The wrapper is needed to convert bool to logical
        need_wrapper=True
    ),

    dict(
        # A C function with a 'int *' argument passes address of array
        name="f_native_*_in_raw",
        # same as "f_void_*",
        arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["C_LOC"]),
        arg_c_call=["C_LOC({f_var})"],
    ),

    dict(
        # double * out +intent(out) +deref(allocatable)+dimension(size(in)),
        # Allocate array then pass to C wrapper.
        name="f_native_*_out_allocatable",
        arg_decl=[
            "{f_type}, intent({f_intent}), allocatable :: {f_var}{f_assumed_shape}",
        ],
        pre_call=[
            "allocate({f_var}{f_array_allocate})",
        ],
    ),
    
    dict(
        # Any array of pointers.  Assumed to be non-contiguous memory.
        # All Fortran can do is treat as a type(C_PTR).
        name="c_native_**_in",
        buf_args=["arg_decl"],
        c_arg_decl=[
            "{cxx_type} **{cxx_var}",
        ],
        f_arg_decl=[
            "type(C_PTR), intent(IN), value :: {c_var}",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        # double **count _intent(out)+dimension(ncount)
        name="c_native_**_out_buf",
        buf_args=["context"],
        c_helper="ShroudTypeDefines",
        pre_call=[
            "{c_const}{cxx_type} *{cxx_var};",
        ],
        arg_call=["&{cxx_var}"],
        post_call=[
            "{c_var_context}->cxx.addr  = {cxx_nonconst_ptr};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.base = {cxx_var};",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = sizeof({cxx_type});",
            "{c_var_context}->rank = {rank};"
            "{c_array_shape}",
            "{c_var_context}->size = {c_array_size};",
        ],
        # XXX - similar to c_native_*_result_buf
    ),
    dict(
        name="c_native_*&_out_buf",
        base="c_native_**_out_buf",
        arg_call=["{cxx_var}"],
    ),
    dict(
        # deref(pointer)
        # A C function with a 'int **' argument associates it
        # with a Fortran pointer to a scalar.
        name="f_XXX_native_**_out",
        arg_decl=[
            "{f_type}, intent({f_intent}), pointer :: {f_var}",
        ],
        f_module=dict(iso_c_binding=["C_PTR", "c_f_pointer"]),
        declare=[
            "type(C_PTR) :: {F_pointer}",
        ],
        arg_c_call=["{F_pointer}"],
        post_call=[
            "call c_f_pointer({F_pointer}, {f_var})",
        ],
    ),
    dict(
        # deref(pointer)
        # A C function with a 'int **' argument associates it
        # with a Fortran pointer.
        name="f_native_**_out",
        arg_decl=[
            "{f_type}, intent({f_intent}), pointer :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        post_call=[
            "call c_f_pointer({c_var_context}%base_addr, {f_var}{f_array_shape})",
        ],
    ),
    dict(
        # Make argument type(C_PTR) from 'int **'
        name="f_native_**_out_raw",
        arg_decl=[
            "type(C_PTR), intent({f_intent}) :: {f_var}",
        ],
        declare=[
            "type({F_array_type}) {c_var_context}",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_c_call=["{c_var_context}"],
        # This post_call block will set need_wrapper=True
        # No real need for F_array_type since C_PTR can be passed directly
        # but c_native_**_out_buf uses buf_args=context.
        # XXX - maybe use c_native_**_out_buf_raw
        post_call=[
            "{f_var} = {c_var_context}%base_addr",
        ],
    ),
    dict(
        name="f_native_*&_out",
        base="f_native_**_out",
    ),

    # XXX only in buf?
    # Used with intent IN, INOUT, and OUT.
#    c_native_pointer_cdesc=dict(
    dict(
        name="c_native_*_cdesc",
        buf_args=["context"],
#        c_helper="ShroudTypeDefines",
        c_pre_call=[
            "{cxx_type} * {c_var} = {c_var_context}->addr.base;",
        ],
        cxx_pre_call=[
#            "{cxx_type} * {c_var} = static_cast<{cxx_type} *>\t"
#            "({c_var_context}->addr.base);",
            "{cxx_type} * {c_var} = static_cast<{cxx_type} *>\t"
            "(const_cast<void *>({c_var_context}->addr.base));",
        ],
    ),
#    f_native_pointer_cdesc=dict(
    dict(
        name="f_native_*_cdesc",
        # TARGET required for argument to C_LOC.
        arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_helper="ShroudTypeDefines",
        f_module=dict(iso_c_binding=["C_LOC"]),
#        initialize=[
        pre_call=[
            "{c_var_context}%base_addr = C_LOC({f_var})",
            "{c_var_context}%type = {sh_type}",
            "! {c_var_context}%elem_len = C_SIZEOF()",
#            "{c_var_context}%size = size({f_var})",
            "{c_var_context}%size = {size}",
            "{c_var_context}%rank = {rank}",
            # This also works with scalars since (1:0) is a zero length array.
            "{c_var_context}%shape(1:{rank}) = shape({f_var})",
        ],
    ),
    dict(
        name="f_native_*_in_cdesc",
        base="f_native_*_cdesc",
    ),
    dict(
        name="f_native_*_out_cdesc",
        base="f_native_*_cdesc",
    ),

########################################
# void *
    dict(
        name="f_void_*_in",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR), intent(IN) :: {f_var}",
        ],
    ),
    dict(
        # return a type(C_PTR)
        name="f_void_*_result",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    dict(
        name="c_void_*_cdesc",
        base="c_native_*_cdesc",
    ),
    dict(
        name="f_void_*_cdesc",
        base="f_native_*_cdesc",
    ),    

########################################
# void **
    dict(
        # Treat as an assumed length array in Fortran interface.
        name='c_void_**_in',
        buf_args=["arg_decl"],
        c_arg_decl=[
            "void **{c_var}",
        ],
        f_arg_decl=[
            "type(C_PTR), intent(IN) :: {c_var}{f_c_dimension}",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name="f_void_**_in",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR), intent(IN) :: {f_var}{f_assumed_shape}",
        ],
    ),
    dict(
        # XXX - intent as a format string
        name="f_void_**_out",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR), intent(OUT) :: {f_var}",
        ],
    ),
    
    # Function has a result with deref(allocatable).
    #
    #    C wrapper:
    #       Add context argument for result
    #       Fill in values to describe array.
    #
    #    Fortran:
    #        c_step1(context)
    #        allocate(Fout(len))
    #        c_step2(context, Fout, size(len))
    #
    #        c_step1(context)
    #        call c_f_pointer(c_ptr, f_ptr, shape)
    dict(
        name="c_native_*_result_buf",
        buf_args=["context"],
        c_helper="ShroudTypeDefines",
        post_call=[
            "{c_var_context}->cxx.addr  = {cxx_nonconst_ptr};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.base = {cxx_var};",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = sizeof({cxx_type});",
            "{c_var_context}->rank = {rank};"
            "{c_array_shape}",
            "{c_var_context}->size = {c_array_size};",
        ],
        return_cptr=True,
    ),
    dict(
        name="f_native_*_result_allocatable",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_type}",
        f_module=dict(iso_c_binding=["C_PTR"]),
        declare=[
            "type(C_PTR) :: {F_pointer}",
        ],
        call=[
            "{F_pointer} = {F_C_call}({F_arg_c_call})",
        ],
        post_call=[
            # XXX - allocate scalar
            "allocate({f_var}({c_var_dimension}))",
            "call {hnamefunc0}({c_var_context}, {f_var}, size({f_var}, kind=C_SIZE_T))",
        ],
    ),

    # f_pointer_shape may be blank for a scalar, otherwise it
    # includes a leading comma.
    dict(
        name="f_native_*_result_pointer",
        f_module=dict(iso_c_binding=["C_PTR", "c_f_pointer"]),
        declare=[
            "type(C_PTR) :: {F_pointer}",
        ],
        call=[
            "{F_pointer} = {F_C_call}({F_arg_c_call})",
        ],
        post_call=[
            "call c_f_pointer({F_pointer}, {F_result}{f_array_shape})",
        ],
    ),
    dict(
        # +deref(pointer) +owner(caller)
        name="f_native_*_result_pointer_caller",
        f_helper="capsule_helper",
        f_module=dict(iso_c_binding=["C_PTR", "c_f_pointer"]),
        arg_name=["{c_var_capsule}"],
        arg_decl=[
            "{f_type}, pointer :: {f_var}{f_assumed_shape}",
            "type({F_capsule_type}), intent(OUT) :: {c_var_capsule}",
        ],
        declare=[
            "type(C_PTR) :: {F_pointer}",
        ],
        call=[
            "{F_pointer} = {F_C_call}({F_arg_c_call})",
        ],
        post_call=[
            "call c_f_pointer({F_pointer}, {F_result}{f_array_shape})",
            "{c_var_capsule}%mem = {c_var_context}%cxx",
        ],
    ),
    dict(
        name="f_native_*_result_raw",
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    dict(
        # int **func(void)
        # regardless of deref value.
        name="f_native_**_result",
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    
    dict(
        name="f_native_&_result",
        base="f_native_*_result_pointer",   # XXX - change base to &?
    ),

    dict(
        name="f_native_*_result_scalar",
        # avoid catching f_native_*_result
    ),


    ########################################
    # char arg
    dict(
        name="c_char_scalar_in",
        buf_args=["arg_decl"],
        c_arg_decl=[
            "char {c_var}",
        ],
        f_arg_decl=[
            "character(kind=C_CHAR), value, intent(IN) :: {c_var}",
        ],
        f_module=dict(iso_c_binding=["C_CHAR"]),
    ),
    dict(
        name="f_char_scalar_in",
        # By default the declaration is character(LEN=*).
        arg_decl=[
            "character, value, intent(IN) :: {f_var}",
        ],
    ),
    dict(
        name="c_char_scalar_result",
        f_result_decl=[
            "character(kind=C_CHAR) :: {c_var}",
        ],
        f_module=dict(iso_c_binding=["C_CHAR"]),
    ),
    dict(
        name="c_char_scalar_result_buf",
        buf_args=["arg", "len"],
        c_impl_header=["<string.h>"],
        cxx_impl_header=["<cstring>"],
        post_call=[
            "{stdlib}memset({c_var}, ' ', {c_var_len});",
            "{c_var}[0] = {cxx_var};",
        ],
    ),
    
    dict(
        name="c_char_*_result",
        return_cptr=True,
    ),
    dict(
        name="c_char_*_in_buf",
        buf_args=["arg", "len_trim"],
        cxx_local_var="pointer",
        c_helper="ShroudStrAlloc ShroudStrFree",
        pre_call=[
            "char * {cxx_var} = ShroudStrAlloc(\t"
            "{c_var},\t {c_var_trim},\t {c_var_trim});",
        ],
        post_call=[
            "ShroudStrFree({cxx_var});"
        ],
    ),
    dict(
        name="c_char_*_out_buf",
        buf_args=["arg", "len"],
        c_helper="ShroudStrBlankFill",
        post_call=[
            "ShroudStrBlankFill({c_var}, {c_var_len});"
        ],
    ),
    dict(
        name="c_char_*_inout_buf",
        buf_args=["arg", "len_trim", "len"],
        cxx_local_var="pointer",
        c_helper="ShroudStrAlloc ShroudStrCopy ShroudStrFree",
        pre_call=[
            "char * {cxx_var} = ShroudStrAlloc(\t"
            "{c_var},\t {c_var_len},\t {c_var_trim});",
        ],
        post_call=[
            # nsrc=-1 will call strlen({cxx_var})
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var},\t -1);",
            "ShroudStrFree({cxx_var});",
        ],
    ),
    dict(
        name="c_char_*_result_buf",
        buf_args=["arg", "len"],
        c_helper="ShroudStrCopy",
        post_call=[
            # nsrc=-1 will call strlen({cxx_var})
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var},\t -1);",
        ],
    ),
    dict(
        name="c_char_*_result_buf_allocatable",
        buf_args=["context"],
        c_helper="ShroudTypeDefines",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        post_call=[
            "{c_var_context}->cxx.addr = {cxx_nonconst_ptr};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.ccharp = {cxx_var};",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = {cxx_var} == {nullptr} ? 0 : {stdlib}strlen({cxx_var});",
            "{c_var_context}->size = 1;",
            "{c_var_context}->rank = 0;",
        ],
    ),

    dict(
        # char *func() +deref(raw)
        name="f_char_*_result_raw",
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    #####
    dict(
        # Treat as an assumed length array in Fortran interface.
        name='c_char_**_in',
        buf_args=["arg_decl"],
        c_arg_decl=[
            "char **{c_var}",
        ],
        f_arg_decl=[
            "type(C_PTR), intent(IN) :: {c_var}(*)",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name='c_char_**_in_buf',
        # arg_decl - argument is char *, not char **.
        buf_args=["arg_decl", "size", "len"],
        c_helper="ShroudStrArrayAlloc ShroudStrArrayFree",
        cxx_local_var="pointer",
        pre_call=[
            "char **{cxx_var} = ShroudStrArrayAlloc("
            "{c_var},\t {c_var_size},\t {c_var_len});",
        ],
        post_call=[
            "ShroudStrArrayFree({cxx_var}, {c_var_size});",
        ],

        c_arg_decl=[
            "char *{c_var}",
        ],
        f_arg_decl=[
            "character(kind=C_CHAR), intent(IN) :: {c_var}(*)",
        ],
        f_module=dict(iso_c_binding=["C_CHAR"]),
    ),
    #####
    dict(
        name="f_char_*_result_allocatable",
        need_wrapper=True,
        c_helper="copy_string",
        f_helper="copy_string",
        arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        post_call=[
            "allocate(character(len={c_var_context}%elem_len):: {f_var})",
            "call {hnamefunc0}({c_var_context}, {f_var}, {c_var_context}%elem_len)",
        ],
    ),
    dict(
        name="f_char_scalar_result_allocatable",
        base="f_char_*_result_allocatable",
    ),

    dict(
        name="c_string_in",
        cxx_local_var="scalar",
        pre_call=["{c_const}std::string {cxx_var}({c_var});"],
    ),
    dict(
        name="c_string_out",
        cxx_impl_header=["<cstring>"],
        # #- pre_call=[
        # #-     'int {c_var_trim} = strlen({c_var});',
        # #-     ],
        cxx_local_var="scalar",
        pre_call=["{c_const}std::string {cxx_var};"],
        post_call=[
            # This may overwrite c_var if cxx_val is too long
            "strcpy({c_var}, {cxx_var}{cxx_member}c_str());"
        ],
    ),
    dict(
        name="c_string_inout",
        cxx_impl_header=["<cstring>"],
        cxx_local_var="scalar",
        pre_call=["{c_const}std::string {cxx_var}({c_var});"],
        post_call=[
            # This may overwrite c_var if cxx_val is too long
            "strcpy({c_var}, {cxx_var}{cxx_member}c_str());"
        ],
    ),
    dict(
        name="c_string_in_buf",
        buf_args=["arg", "len_trim"],
        cxx_local_var="scalar",
        pre_call=[
            (
                "{c_const}std::string "
                "{cxx_var}({c_var}, {c_var_trim});"
            )
        ],
    ),
    dict(
        name="c_string_out_buf",
        buf_args=["arg", "len"],
        c_helper="ShroudStrCopy",
        cxx_local_var="scalar",
        pre_call=["std::string {cxx_var};"],
        post_call=[
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var}{cxx_member}data(),"
            "\t {cxx_var}{cxx_member}size());"
        ],
    ),
    dict(
        name="c_string_inout_buf",
        buf_args=["arg", "len_trim", "len"],
        c_helper="ShroudStrCopy",
        cxx_local_var="scalar",
        pre_call=["std::string {cxx_var}({c_var}, {c_var_trim});"],
        post_call=[
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var}{cxx_member}data(),"
            "\t {cxx_var}{cxx_member}size());"
        ],
    ),
    dict(
        name="c_string_result",
        # cxx_to_c creates a pointer from a value via c_str()
        # The default behavior will dereference the value.
        ret=[
            "return {c_var};",
        ],
        return_cptr=True,
    ),
    dict(
        name="c_string_result_buf",
        buf_args=["arg", "len"],
        c_helper="ShroudStrCopy",
        post_call=[
            "if ({cxx_var}{cxx_member}empty()) {{+",
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {nullptr},\t 0);",
            "-}} else {{+",
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var}{cxx_member}data(),"
            "\t {cxx_var}{cxx_member}size());",
            "-}}",
        ],
    ),

    # std::string
    dict(
        name="c_string_scalar_in",
        buf_args=["arg_decl"],
        c_arg_decl=[
            # Argument is a pointer while std::string is a scalar.
            # C++ compiler will convert to std::string when calling function.
            "char *{c_var}",
        ],
        f_arg_decl=[
            # Remove VALUE added by c_default
            "character(kind=C_CHAR), intent(IN) :: {c_var}(*)",
        ],
        f_module=dict(iso_c_binding=["C_CHAR"]),
    ),
    dict(
        name="c_string_scalar_in_buf",
        base="c_string_scalar_in",
        buf_args=["arg_decl", "len_trim"],
        cxx_local_var="scalar",
        pre_call=[
            "std::string {cxx_var}({c_var}, {c_var_trim});",
        ],
        call=[
            "{cxx_var}",
        ],
    ),
    dict(
        name="f_string_scalar_in",  # pairs with c_string_scalar_in_buf
        need_wrapper=True,
        buf_args=["arg", "len"],
        arg_decl=[
            # Remove VALUE added by f_default
            "character(len=*), intent(IN) :: {f_var}",
        ],
    ),
    
    # Uses a two part call to copy results of std::string into a
    # allocatable Fortran array.
    #    c_step1(context)
    #    allocate(character(len=context%elem_len): Fout)
    #    c_step2(context, Fout, context%elem_len)
    # only used with bufferifed routines and intent(out) or result
    # std::string * function()
    dict(
        name="c_string_result_buf_allocatable",
        # pass address of string and length back to Fortran
        buf_args=["context"],
        c_helper="ShroudStrToArray",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        post_call=[
            "ShroudStrToArray({c_var_context}, {cxx_addr}{cxx_var}, {idtor});",
        ],
    ),

    # Since 'c_string_scalar_result_buf_allocatable' exists,
    # must set an alias for c_string_scalar.
    # No need to allocate a local copy since the string is copied
    # into a Fortran variable before the string is deleted.
    dict(
        name="c_string_scalar_result_buf",
        base="c_string_result_buf",
    ),
    
    # std::string function()
    # Must allocate the std::string then assign to it via cxx_rv_decl.
    # This allows the std::string to outlast the function return.
    # The Fortran wrapper will ALLOCATE memory, copy then delete the string.
    dict(
        name="c_string_scalar_result_buf_allocatable",
        # pass address of string and length back to Fortran
        buf_args=["context"],
        cxx_local_var="pointer",
        c_helper="ShroudStrToArray",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        pre_call=[
            "std::string * {cxx_var} = new std::string;",
        ],
        destructor_name="new_string",
        destructor=[
            "std::string *cxx_ptr = \treinterpret_cast<std::string *>(ptr);",
            "delete cxx_ptr;",
        ],
        post_call=[
            "ShroudStrToArray({c_var_context}, {cxx_var}, {idtor});",
        ],
    ),
    
    # similar to f_char_scalar_result_allocatable
    dict(
        name="f_string_scalar_result_allocatable",
        need_wrapper=True,
        c_helper="copy_string",
        f_helper="copy_string",
        arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        post_call=[
            "allocate(character(len={c_var_context}%elem_len):: {f_var})",
            "call {hnamefunc0}({c_var_context}, {f_var}, {c_var_context}%elem_len)",
        ],
    ),
    dict(
        name="f_string_*_result_allocatable",
        base="f_string_scalar_result_allocatable",
    ),
    dict(
        name="f_string_&_result_allocatable",
        base="f_string_scalar_result_allocatable",
    ),
    
    
    dict(
        name="c_vector_in_buf",
        buf_args=["arg", "size"],
        cxx_local_var="scalar",
        pre_call=[
            (
                "{c_const}std::vector<{cxx_T}> "
                "{cxx_var}({c_var}, {c_var} + {c_var_size});"
            )
        ],
    ),
    # cxx_var is always a pointer to a vector
    dict(
        name="c_vector_out_buf",
        buf_args=["context"],
        cxx_local_var="pointer",
        c_helper="ShroudTypeDefines",
        pre_call=[
            "{c_const}std::vector<{cxx_T}>"
            "\t *{cxx_var} = new std::vector<{cxx_T}>;"
        ],
        post_call=[
            # Return address and size of vector data.
            "{c_var_context}->cxx.addr  = {cxx_var};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.base = {cxx_var}->empty()"
            " ? {nullptr} : &{cxx_var}->front();",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = sizeof({cxx_T});",
            "{c_var_context}->size = {cxx_var}->size();",
            "{c_var_context}->rank = 1;",
            "{c_var_context}->shape[0] = {c_var_context}->size;",
        ],
        destructor_name="std_vector_{cxx_T}",
        destructor=[
            "std::vector<{cxx_T}> *cxx_ptr ="
            " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    dict(
        name="c_vector_inout_buf",
        buf_args=["arg", "size", "context"],
        cxx_local_var="pointer",
        c_helper="ShroudTypeDefines",
        pre_call=[
            "std::vector<{cxx_T}> *{cxx_var} = \tnew std::vector<{cxx_T}>\t("
            "\t{c_var}, {c_var} + {c_var_size});"
        ],
        post_call=[
            # Return address and size of vector data.
            "{c_var_context}->cxx.addr  = {cxx_var};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.base = {cxx_var}->empty()"
            " ? {nullptr} : &{cxx_var}->front();",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = sizeof({cxx_T});",
            "{c_var_context}->size = {cxx_var}->size();",
            "{c_var_context}->rank = 1;",
            "{c_var_context}->shape[0] = {c_var_context}->size;",
        ],
        destructor_name="std_vector_{cxx_T}",
        destructor=[
            "std::vector<{cxx_T}> *cxx_ptr ="
            " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    # Almost same as intent_out_buf.
    dict(
        name="c_vector_result_buf",
        buf_args=["context"],
        cxx_local_var="pointer",
        c_helper="ShroudTypeDefines",
        pre_call=[
            "{c_const}std::vector<{cxx_T}>"
            "\t *{cxx_var} = new std::vector<{cxx_T}>;"
        ],
        post_call=[
            # Return address and size of vector data.
            "{c_var_context}->cxx.addr  = {cxx_var};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.base = {cxx_var}->empty()"
            " ? {nullptr} : &{cxx_var}->front();",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = sizeof({cxx_T});",
            "{c_var_context}->size = {cxx_var}->size();",
            "{c_var_context}->rank = 1;",
            "{c_var_context}->shape[0] = {c_var_context}->size;",
        ],
        destructor_name="std_vector_{cxx_T}",
        destructor=[
            "std::vector<{cxx_T}> *cxx_ptr ="
            " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    #                dict(
    #                    name="c_vector_result_buf",
    #                    buf_args=['arg', 'size'],
    #                    c_helper='ShroudStrCopy',
    #                    post_call=[
    #                        'if ({cxx_var}.empty()) {{+',
    #                        'ShroudStrCopy({c_var}, {c_var_len},'
    #                        '{nullptr}, 0);',
    #                        '-}} else {{+',
    #                        'ShroudStrCopy({c_var}, {c_var_len},'
    #                        '\t {cxx_var}{cxx_member}data(),'
    #                        '\t {cxx_var}{cxx_member}size());',
    #                        '-}}',
    #                    ],
    #                ),
    
    # Specialize for vector<string>.
    dict(
        name="c_vector_in_buf_string",
        buf_args=["arg", "size", "len"],
        c_helper="ShroudLenTrim",
        cxx_local_var="scalar",
        pre_call=[
            "std::vector<{cxx_T}> {cxx_var};",
            "{{+",
            "{c_const}char * BBB = {c_var};",
            "std::vector<{cxx_T}>::size_type",
            "+{c_temp}i = 0,",
            "{c_temp}n = {c_var_size};",
            "-for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{+",
            "{cxx_var}.push_back("
            "std::string(BBB,ShroudLenTrim(BBB, {c_var_len})));",
            "BBB += {c_var_len};",
            "-}}",
            "-}}",
        ],
    ),
    dict(
        name="c_vector_out_buf_string",
        buf_args=["arg", "size", "len"],
        c_helper="ShroudLenTrim",
        cxx_local_var="scalar",
        pre_call=["{c_const}std::vector<{cxx_T}> {cxx_var};"],
        post_call=[
            "{{+",
            "char * BBB = {c_var};",
            "std::vector<{cxx_T}>::size_type",
            "+{c_temp}i = 0,",
            "{c_temp}n = {c_var_size};",
            "{c_temp}n = std::min({cxx_var}.size(),{c_temp}n);",
            "-for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{+",
            "ShroudStrCopy("
            "BBB, {c_var_len},"
            "\t {cxx_var}[{c_temp}i].data(),"
            "\t {cxx_var}[{c_temp}i].size());",
            "BBB += {c_var_len};",
            "-}}",
            "-}}",
        ],
    ),
    dict(
        name="c_vector_inout_buf_string",
        buf_args=["arg", "size", "len"],
        cxx_local_var="scalar",
        pre_call=[
            "std::vector<{cxx_T}> {cxx_var};",
            "{{+",
            "{c_const}char * BBB = {c_var};",
            "std::vector<{cxx_T}>::size_type",
            "+{c_temp}i = 0,",
            "{c_temp}n = {c_var_size};",
            "-for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{+",
            "{cxx_var}.push_back"
            "(std::string(BBB,ShroudLenTrim(BBB, {c_var_len})));",
            "BBB += {c_var_len};",
            "-}}",
            "-}}",
    ],
        post_call=[
            "{{+",
            "char * BBB = {c_var};",
            "std::vector<{cxx_T}>::size_type",
            "+{c_temp}i = 0,",
            "{c_temp}n = {c_var_size};",
            "-{c_temp}n = std::min({cxx_var}.size(),{c_temp}n);",
            "for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{+",
            "ShroudStrCopy(BBB, {c_var_len},"
            "\t {cxx_var}[{c_temp}i].data(),"
            "\t {cxx_var}[{c_temp}i].size());",
            "BBB += {c_var_len};",
            "-}}",
            "-}}",
        ],
    ),
    #                    dict(
    #                        name="c_vector_result_buf_string",
    #                        c_helper='ShroudStrCopy',
    #                        post_call=[
    #                            'if ({cxx_var}.empty()) {{+',
    #                            'std::memset({c_var}, \' \', {c_var_len});',
    #                            '-}} else {{+',
    #                            'ShroudStrCopy({c_var}, {c_var_len}, '
    #                            '\t {cxx_var}{cxx_member}data(),'
    #                            '\t {cxx_var}{cxx_member}size());',
    #                            '-}}',
    #                        ],
    #                    ),
    # copy into user's existing array
    dict(
        name="f_vector_out",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "call {hnamefunc0}(\t{c_var_context},\t {f_var},\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        name="f_vector_inout",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "call {hnamefunc0}(\t{c_var_context},\t {f_var},\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        name="f_vector_result",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "call {hnamefunc0}(\t{c_var_context},\t {f_var},\t size({f_var},kind=C_SIZE_T))"
        ],
    ),
    # copy into allocated array
    dict(
        name="f_vector_out_allocatable",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "allocate({f_var}({c_var_context}%size))",
            "call {hnamefunc0}(\t{c_var_context},\t {f_var},\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        name="f_vector_inout_allocatable",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "if (allocated({f_var})) deallocate({f_var})",
            "allocate({f_var}({c_var_context}%size))",
            "call {hnamefunc0}(\t{c_var_context},\t {f_var},\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    # Similar to f_vector_out_allocatable but must declare result variable.
    # Always return a 1-d array.
    dict(
        name="f_vector_result_allocatable",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "allocate({f_var}({c_var_context}%size))",
            "call {hnamefunc0}(\t{c_var_context},\t {f_var},\t size({f_var},kind=C_SIZE_T))",
        ],
    ),

    # Pass in a pointer to a shadow object via buf_args.
    # Extract pointer to C++ instance.
    # convert C argument into a pointer to C++ type.
    dict(
        name="c_shadow_in",
        buf_args=["shadow"],
        cxx_local_var="pointer",
        pre_call=[
            "{c_const}{cxx_type} * {cxx_var} =\t "
            "{cast_static}{c_const}{cxx_type} *{cast1}{c_var}{c_member}addr{cast2};",
        ],
    ),
    dict(
        name="c_shadow_inout",
        base="c_shadow_in",
    ),
    dict(
        name="c_shadow_scalar_in",
        base="c_shadow_in",
    ),
    # Return a C_capsule_data_type.
    dict(
        name="c_shadow_result",
        buf_extra=["shadow"],
        c_local_var="pointer",
        post_call=[
            "{shadow_var}->addr = {cxx_nonconst_ptr};",
            "{shadow_var}->idtor = {idtor};",
        ],
        ret=[
            "return {shadow_var};",
        ],
        return_type="{c_type} *",
        return_cptr=True,
    ),
    dict(
        name="c_shadow_scalar_result",
        # Return a instance by value.
        # Create memory in pre_call so it will survive the return.
        # owner="caller" sets idtor flag to release the memory.
        # c_local_var is passed in via buf_extra=shadow.
        buf_extra=["shadow"],
        cxx_local_var="pointer",
        c_local_var="pointer",
        owner="caller",
        pre_call=[
            "{cxx_type} * {cxx_var} = new {cxx_type};",
        ],
        post_call=[
            "{shadow_var}->addr = {cxx_nonconst_ptr};",
            "{shadow_var}->idtor = {idtor};",
        ],
        ret=[
            "return {shadow_var};",
        ],
        return_type="{c_type} *",
        return_cptr=True,
    ),
    dict(
        name="f_shadow_result",
        need_wrapper=True,
        f_module=dict(iso_c_binding=["C_PTR"]),
        declare=[
            "type(C_PTR) :: {F_result_ptr}",
        ],
        call=[
            # The C function returns a pointer.
            # Save in a type(C_PTR) variable.
            "{F_result_ptr} = {F_C_call}({F_arg_c_call})"
        ],
    ),
    dict(
        name="c_shadow_ctor",
        buf_extra=["shadow"],
        cxx_local_var="pointer",
        call=[
            "{cxx_type} *{cxx_var} =\t new {cxx_type}({C_call_list});",
            "{shadow_var}->addr = static_cast<{c_const}void *>(\t{cxx_var});",
            "{shadow_var}->idtor = {idtor};",
        ],
        ret=[
            "return {shadow_var};",
        ],
        return_type="{c_type} *",
        owner="caller",
    ),
    dict(
        name="c_shadow_scalar_ctor",
        base="c_shadow_ctor",
    ),
    dict(
        name="f_shadow_ctor",
        base="f_shadow_result",
    ),
    dict(
        # NULL in stddef.h
        name="c_shadow_dtor",
        c_impl_header=["<stddef.h>"],
        cxx_impl_header=["<cstddef>"],
        call=[
            "delete {CXX_this};",
            "{C_this}->addr = {nullptr};",
        ],
        return_type="void",
    ),

    dict(
        # Used with in, out, inout
        # C pointer -> void pointer -> C++ pointer
        name="c_struct",
        cxx_cxx_local_var="pointer", # cxx_local_var only used with C++
        cxx_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} = \tstatic_cast<{c_const}{cxx_type} *>\t(static_cast<{c_const}void *>(\t{c_addr}{c_var}));",
        ],
    ),
    dict(
        name="c_struct_result",
        # C++ pointer -> void pointer -> C pointer
        c_local_var="pointer",
        cxx_post_call=[
            "{c_const}{c_type} * {c_var} = \tstatic_cast<{c_const}{c_type} *>(\tstatic_cast<{c_const}void *>(\t{cxx_addr}{cxx_var}));",
        ],
    ),
    dict(
        name="f_struct_scalar_result",
        # Needed to differentiate from f_struct_pointer_result.
    ),
    dict(
        name="f_struct_*_result",
        base="f_native_*_result_pointer",
    ),
]
