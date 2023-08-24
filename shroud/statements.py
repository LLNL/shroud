# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
"""
import yaml

try:
    # XXX - python 3.7
    import importlib.resources
    def read_yaml_resource(name):
        fp = importlib.resources.open_binary('shroud', name)
        stmts = yaml.safe_load(fp)
        return stmts
except ImportError:
    from pkg_resources import resource_filename
    def read_yaml_resource(name):
        fp = open(resource_filename('shroud', name), 'rb')
        stmts = yaml.safe_load(fp)
        return stmts

from . import util

# The tree of c and fortran statements.
cf_tree = {}
fc_dict = {} # dictionary of Scope of all expanded fc_statements.
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
        specialize.append('targ')
        # XXX currently only the first template argument is processed.
        targ = arg.template_arguments[0]
        arg_typemap = targ.typemap
        specialize.append(arg_typemap.sgroup)
        spointer = targ.declarator.get_indirect_stmt()
        specialize.append(spointer)
    return arg_typemap, specialize

def template_stmts(ast):
    """Create statement labels for template arguments.
    targ_int_scalar

    Parameters
    ----------
    ast : declast.Declaration
    """
    specialize = []
    if ast.template_arguments:
        specialize.append('targ')
        # XXX currently only the first template argument is processed.
        targ = ast.template_arguments[0]
        arg_typemap = targ.typemap
        specialize.append(arg_typemap.sgroup)
        spointer = targ.declarator.get_indirect_stmt()
        specialize.append(spointer)
    return specialize

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


def assign_buf_variable_names(attrs, meta, options, fmt, rootname):
    """
    Transfer names from attribute to fmt.
    """
    # XXX - make sure they don't conflict with other names.
    if meta["capsule"]:
        fmt.c_var_capsule = options.C_var_capsule_template.format(
            c_var=rootname)
    if attrs["cdesc"]:
        # XXX - c_var_cdesc is set via Stmts.temps=["cdesc"]
        # XXX   not sure if this is needed still.
        fmt.c_var_cdesc2 = options.C_var_context_template.format(
            c_var=rootname)


def compute_return_prefix(arg, local_var):
    """Compute how to access variable: dereference, address, as-is"""
    if local_var == "scalar":
        if arg.is_pointer():
            return "&"
        else:
            return ""
    elif local_var == "pointer":
        if arg.declarator.is_pointer():
            return ""
        else:
            return "*"
    elif local_var == "funcptr":
        return ""
    elif arg.declarator.is_reference():
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
    stmts = read_yaml_resource('fc-statements.yaml')
    fc_statements.extend(stmts)
    
    update_for_language(fc_statements, language)
    update_stmt_tree(fc_statements, fc_dict, cf_tree, default_stmts)
    

def update_for_language(stmts, lang):
    """
    Move language specific entries to current language.

    stmts=[
      dict(
        name='foo_bar',
        lang_c=dict(
          declare=[],
        ),
        lang_cxx=dict(
          declare=[],
        ),
      ),
      ...
    ]

    For lang==c,
      foo_bar.update(foo_bar["lang_c"])
    """
    specific = "lang_" + lang
    for item in stmts:
        if specific in item:
            item.update(item[specific])


def compute_stmt_permutations(out, parts):
    """Expand parts which have multiple values

    Ex: parts = 
      [['c'], ['in', 'out', 'inout'], ['native'], ['*'], ['cfi']]
    Three entries will be appended to out:
      ['c', 'in', 'native', '*', 'cfi']
      ['c', 'out', 'native', '*', 'cfi']
      ['c', 'inout', 'native', '*', 'cfi']

    Parameters
    ----------
    out : list
        Results are appended to the list.
    parts :
    """
    tmp = []
    for i, part in enumerate(parts):
        if isinstance(part, list):
            if len(part) == 1:
                tmp.append(part[0])
            else:
                for expand in part:
                    compute_stmt_permutations(
                        out, tmp + [expand] + parts[i+1:])
                break
        else:
            tmp.append(part)
    else:
        out.append(tmp)
                

def add_statement_to_tree(tree, nodes, node_stmts, node, steps):
    """Add node to tree.

    Parameters
    ----------
    tree : dict
        The accumulated tree.
    nodes : dict
        Scopes indexed by name to implement 'base'.
    node_stmts : dict
        nodes indexed by name to implement 'mixin'.
    node : dict
        A 'statements' dict from fc_statement to add.
    steps : list of str
        ['c', 'native', '*', 'in', 'cfi']
    """
    step = tree
    label = []
    for part in steps:
        step = step.setdefault(part, {})
        label.append(part)
        step["_key"] = "_".join(label)
    if "base" in node:
        step['_node'] = node
        scope = util.Scope(nodes[node["base"]])
    else:
        step['_node'] = node
        scope = util.Scope(default_scopes[steps[0]])
    if "mixin" in node:
        for mpart in node["mixin"]:
            scope.update(node_stmts[mpart])
    scope.update(node)
    step["_stmts"] = scope
    name = step["_key"]
    # Name scope using variant name (ex in/out/inout).
    scope.name = name
    nodes[name] = scope
    node_stmts[name] = node
    return scope
        
def update_stmt_tree(stmts, nodes, tree, defaults):
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
       {name="c_in_native",}           # value1
       {name="c_out_native",}          # value2
       {name="c_out_native_pointer",}  # value3
       {name="c_in_string",}           # value4
    ]
    tree = {
      "c": {
        "in": {
           "native": {"_node":value1},
           "string": {"_node":value4},
         },
         "out": {
           "native":{"_node":value2},
           "pointer":{
             "out":{"_node":value3},
           },
         },
      },
    }

    Parameters
    ----------
    stmts : dict
    nodes : dict
        Created Scope members for 'base'.
    tree : dict
    defaults: dict
    """
    # Convert defaults into Scope nodes.
    for key, node in defaults.items():
        default_scopes[key] = node

    # Index by name to find alias, base, mixin.
    node_stmts = {} # Dict from fc_statements for 'mixin'.
    nodes.clear()   # Allow function to be called multiple times.
    for node in stmts:
        # node is a dict.
        if "name" not in node:
            raise RuntimeError("Missing name in statements: {}".
                               format(str(node)))

    for node in stmts:
        key = node["name"]
        steps = key.split("_")
        substeps = []
        for part in steps:
            subparts = part.split("/")
            substeps.append(subparts)

        expanded = []
        compute_stmt_permutations(expanded, substeps)

        for namelst in expanded:
            name = "_".join(namelst)
            if name in nodes:
                raise RuntimeError("Duplicate key in statements: {}".
                                   format(name))
            stmt = add_statement_to_tree(tree, nodes, node_stmts, node, namelst)
            stmt.intent = namelst[1]

            # check for consistency
            if key[0] == "c":
                if (stmt.c_arg_decl is not None or
                    stmt.f_c_arg_decl is not None or
                    stmt.f_c_arg_names is not None):
                    err = False
                    for field in ["c_arg_decl", "f_c_arg_decl", "f_c_arg_names"]:
                        fvalue = stmt.get(field)
                        if fvalue is None:
                            err = True
                            print("Missing", field, "in", node["name"])
                        elif not isinstance(fvalue, list):
                            err = True
                            print(field, "must be a list in", node["name"])
                    if (stmt.c_arg_decl is None or
                        stmt.f_c_arg_decl is None or
                        stmt.f_c_arg_names is None):
                        print("c_arg_decl, f_c_arg_decl and f_c_arg_names must all exist")
                        err = True
                    if err:
                        raise RuntimeError("Error with fields")
                    length = len(stmt.c_arg_decl)
                    if any(len(lst) != length for lst in [stmt.f_c_arg_decl, stmt.f_c_arg_names]):
                        raise RuntimeError(
                            "c_arg_decl, f_c_arg_decl and f_c_arg_names "
                            "must all be same length in {}".format(node["name"]))
            

def write_cf_tree(fp):
    """Write out statements tree.

    Parameters
    ----------
    fp : file
    """
    lines = []
    print_tree_index(cf_tree, lines)
    fp.writelines(lines)
    print_tree_statements(fp, fc_dict, default_stmts)


def print_tree_index(tree, lines, indent=""):
    """Print statements search tree index.
    Intermediate nodes are prefixed with --.
    Useful for debugging.

    Parameters
    ----------
    fp : file
    lines : list
        list of output lines
    indent : str
        indention for recursion.
    """
    parts = tree.get('_key', 'root').split('_')
    if "_node" in tree:
        #        final = '' # + tree["_node"]["scope"].name + '-'
        origname = tree["_node"]["name"]
        lines.append("{}{} -- {}\n".format(indent, parts[-1], origname))
    else:
        lines.append("{}{}\n".format(indent, parts[-1]))
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
            print_tree_index(value, lines, indent)


def print_tree_statements(fp, statements, defaults):
    """Print expanded statements.

    Statements may not have all values directly defined since 'base'
    and 'mixin' brings in other values.  This will dump the values as
    used by Shroud.

    Statements
    ----------
    fp : file
    statements : dict
    defaults : dict

    """
    # Convert Scope into a dictionary for YAML.
    # Add all non-null values from the default dict.
    yaml.SafeDumper.ignore_aliases = lambda *args : True
    complete = {}
    for name in sorted(statements.keys()):
        root = name.split("_", 1)[0]
        base = defaults[root]
        value = statements[name]
        all = {}
        for key in base.__dict__.keys():
            if key[0] == "_":
                continue
            if value[key]:
                all[key] = value[key]
        for key in ["lang_c", "lang_cxx"]:
            val = value.get(key, None)
            if val:
                all[key] = val
        complete[name] = all
    yaml.safe_dump(complete, fp)
            
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
            found = step["_stmts"]
#    if not isinstance(found, util.Scope):
#        raise RuntimeError
    return found


# C Statements.
#  intent      - Set from name.
#  c_arg_call  - List of arguments passed to C/C++ library function.
#
#  c_arg_decl  - Add C declaration to C wrapper.
#                Empty list is no arguments, None is default argument.
#  f_c_arg_decl - Add Fortran declaration to Fortran wrapper interface block.
#                Empty list is no arguments, None is default argument.
#  f_c_arg_names - Empty list is no arguments
#  f_c_result_decl - Declaration for function result.
#                  Can be an empty list to override default.
#  f_module    - Add module info to interface block.
CStmts = util.Scope(
    None,
    name="c_default",
    intent=None,
    iface_header=[],
    impl_header=[],
    c_helper="", c_local_var=None,
    cxx_local_var=None,
    c_arg_call=[],
    c_pre_call=[],
    call=[],
    c_post_call=[],
    final=[],
    ret=[],
    destructor_name=None,
    owner="library",
    return_type=None,

    c_arg_decl=None,
    f_c_arg_names=None,
    f_c_arg_decl=None,

    f_c_result_decl=None,
    f_c_result_var=None,
    f_c_module=None,
    f_c_module_line=None,
    f_c_import=None,
    temps=None,
    local=None,

    notimplemented=False,
)

# Fortran Statements.
FStmts = util.Scope(
    None,
    name="f_default",
    intent=None,
    c_helper="",
    c_local_var=None,
    c_result_var=None,
    f_helper="",
    f_module=None,
    f_module_line=None,
    f_import=None,
    need_wrapper=False,
    arg_name=None,
    arg_decl=None,
    arg_c_call=None,
    declare=[], pre_call=[], call=[], post_call=[],
    result=None,  # name of result variable
    temps=None,
    local=None,
)

# Define class for nodes in tree based on their first entry.
# c_native_*_in uses 'c'.
default_stmts = dict(
    c=CStmts,
    f=FStmts,
)
                
        

# language   "c", "f"
# intent     "in", "out", "inout", "function", "subroutine", "ctor", "dtor"
# sgroup     "native", "string", "char" "struct", "shadow", "bool"
# spointer   "scalar" "*" "**", "&"
# generated
#      "buf"  Pass argument and meta data like SIZE or LEN.
#      "cdesc"  Pass array_type
#      "cfi"    Pass Fortran 2018 CFI_cdesc_t   option.F_CFI
# deref      "allocatable", "pointer", "raw"
# owner      "caller"

fc_statements = [
    dict(
        # No arguments. Set necessary fields as a group.
        name="c_mixin_noargs",
        c_arg_decl=[],
        f_c_arg_decl=[],
        f_c_arg_names=[],
    ),    
    dict(
        name="c_subroutine",
        mixin=["c_mixin_noargs"],
    ),
    dict(
        name="f_subroutine",
        arg_c_call=[],
    ),

    dict(
        name="c_function",
    ),
    dict(
        name="f_function",
    ),

    ########## mixin ##########
    dict(
        name="c_mixin_function_cdesc",
        # Pass array_type as argument to contain the function result.
        c_arg_decl=[
            "{C_array_type} *{c_var_cdesc}",
        ],
        f_c_arg_decl=[
            "type({F_array_type}), intent(OUT) :: {c_var}",
        ],
        f_c_arg_names=["{c_var}"],
        f_c_import=["{F_array_type}"],
        return_type="void",  # Convert to function.
        temps=["cdesc"],
###        f_c_arg_names=["{c_var}"],
    ),
    dict(
        name="f_mixin_function_cdesc",
        f_helper="array_context",
        declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
        ],
        arg_c_call=["{c_var_cdesc}"],  # Pass result as an argument.
        temps=["cdesc"],
        need_wrapper=True,
    ),

    dict(
        # Pass function result as a capsule argument from Fortran to C.
        name="f_mixin_function_shadow_capsule",
        arg_decl=[
            "{f_type} :: {f_var}",
        ],
        arg_c_call=[
            "{f_var}%{F_derived_member}",
        ],
        need_wrapper=True,
    ),
    dict(
        # Pass function result as a capsule argument from Fortran to C.
        name="f_mixin_function_shadow_capptr",
        arg_decl=[
            "{f_type} :: {f_var}",
            "type(C_PTR) :: {F_result_ptr}",
        ],
        arg_c_call=[
            "{f_var}%{F_derived_member}",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
        c_result_var="{F_result_ptr}",
        need_wrapper=True,
    ),
    
    ##########
    # array
    dict(
        # Pass argument and size to C.
        name="f_mixin_in_array_buf",
        arg_c_call=["{f_var}", "size({f_var}, kind=C_SIZE_T)"],
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        need_wrapper=True,
    ),
    dict(
        # Pass argument, len and size to C.
        name="f_mixin_in_2d_array_buf",
        arg_decl=[
            "{f_type}, intent({f_intent}) :: {f_var}(:,:)",
        ],
        arg_c_call=["{f_var}",
                    "size({f_var}, 1, kind=C_SIZE_T)",
                    "size({f_var}, 2, kind=C_SIZE_T)"],
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        need_wrapper=True,
    ),
    dict(
        # Pass argument and size to C.
        name="c_mixin_in_array_buf",
        c_arg_decl=[
            "{cxx_type} *{c_var}",   # XXX c_type
            "size_t {c_var_size}",
        ],
        f_c_arg_names=["{c_var}", "{c_var_size}"],
        f_c_arg_decl=[
            "{f_type}, intent(IN) :: {c_var}(*)",
            "integer(C_SIZE_T), intent(IN), value :: {c_var_size}",
        ],
        f_c_module_line="iso_c_binding:{f_kind},C_SIZE_T",
        temps=["size"],
    ),
    dict(
        # Pass argument, len and size to C.
        name="c_mixin_in_2d_array_buf",
        c_arg_decl=[
            "{cxx_type} *{c_var}",   # XXX c_type
            "size_t {c_var_len}",
            "size_t {c_var_size}",
        ],
        f_c_arg_names=["{c_var}", "{c_var_len}", "{c_var_size}"],
        f_c_arg_decl=[
            "{f_type}, intent(IN) :: {c_var}(*)",
            "integer(C_SIZE_T), intent(IN), value :: {c_var_len}",
            "integer(C_SIZE_T), intent(IN), value :: {c_var_size}",
        ],
        f_c_module_line="iso_c_binding:{f_kind},C_SIZE_T",
        temps=["len", "size"],
    ),

    dict(
        # Pass argument and size to C.
        # Pass array_type to C which will fill it in.
        name="f_mixin_inout_array_cdesc",
        f_helper="array_context",
        declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
        ],
        arg_c_call=["{f_var}", "size({f_var}, kind=C_SIZE_T)", "{c_var_cdesc}"],
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        temps=["cdesc"],
    ),
    dict(
        # Pass argument and size to C.
        # Pass array_type to C which will fill it in.
        name="c_mixin_inout_array_cdesc",
        c_helper="array_context",
        c_arg_decl=[
            "{cxx_type} *{c_var}",   # XXX c_type
            "size_t {c_var_size}",
            "{C_array_type} *{c_var_cdesc}",
        ],
#        c_iface_header="<stddef.h>",
#        cxx_iface_header="<cstddef>",
        f_c_arg_names=["{c_var}", "{c_var_size}", "{c_var_cdesc}"],
        f_c_arg_decl=[
            "{f_type}, intent(IN) :: {c_var}(*)",
            "integer(C_SIZE_T), intent(IN), value :: {c_var_size}",
            "type({F_array_type}), intent(OUT) :: {c_var_cdesc}",
        ],
        f_c_import=["{F_array_type}"],
        f_c_module_line="iso_c_binding:{f_kind},C_SIZE_T",
        temps=["size", "cdesc"],
    ),

    dict(
        # Pass array_type to C which will fill it in.
        name="f_mixin_out_array_cdesc",
        f_helper="array_context",
        declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
        ],
        arg_c_call=["{c_var_cdesc}"],
        temps=["cdesc"],
    ),
    dict(
        # Pass array_type to C which will fill it in.
        name="c_mixin_out_array_cdesc",
        c_helper="array_context",
        c_arg_decl=[
            "{C_array_type} *{c_var_cdesc}",
        ],
        f_c_arg_names=["{c_var_cdesc}"],
        f_c_arg_decl=[
            "type({F_array_type}), intent(OUT) :: {c_var_cdesc}",
        ],
        f_c_import=["{F_array_type}"],
        temps=["cdesc"],
    ),
    dict(
        # cdesc - array from argument
        # out   - filled by C wrapper.
        # Used to return a pointer to a non-fortran compatiable type
        # such as std::vector or std::string.
        name="c_mixin_out_array_cdesc-and-cdesc",
        c_helper="array_context",
        c_arg_decl=[
            "{C_array_type} *{c_var_cdesc}",
            "{C_array_type} *{c_var_out}",
        ],
        f_c_arg_names=["{c_var_cdesc}", "{c_var_out}"],
        f_c_arg_decl=[
            "type({F_array_type}), intent(OUT) :: {c_var_cdesc}",
            "type({F_array_type}), intent(OUT) :: {c_var_out}",
        ],
        f_c_import=["{F_array_type}"],
        temps=["cdesc", "out"],
    ),

    dict(
        # Pass argument, size and len to C.
        name="f_mixin_in_string_array_buf",
        arg_c_call=[
            "{f_var}",
            "size({f_var}, kind=C_SIZE_T)",
            "len({f_var}, kind=C_INT)"
        ],
        f_module=dict(iso_c_binding=["C_SIZE_T", "C_INT"]),
        need_wrapper=True,
    ),
    dict(
        # Pass argument, size and len to C.
        name="c_mixin_in_string_array_buf",
        c_arg_decl=[
            "const char *{c_var}",   # XXX c_type
            "size_t {c_var_size}",
            "int {c_var_len}",
        ],
        f_c_arg_names=["{c_var}", "{c_var_size}", "{c_var_len}"],
        f_c_arg_decl=[
            "character(kind=C_CHAR), intent(IN) :: {c_var}(*)",
            "integer(C_SIZE_T), intent(IN), value :: {c_var_size}",
            "integer(C_INT), intent(IN), value :: {c_var_len}",
        ],
        f_c_module_line="iso_c_binding:C_CHAR,C_SIZE_T,C_INT",
        temps=["size", "len"],
    ),

    
    ##########
    # Return CHARACTER address and length to Fortran.
    dict(
        name="c_mixin_out_character_cdesc",
        c_arg_decl=[
            "{C_array_type} *{c_var_cdesc}",
        ],
        f_c_arg_decl=[
            "type({F_array_type}), intent(OUT) :: {c_var}",
        ],
        f_c_arg_names=["{c_var}"],
        f_c_import=["{F_array_type}"],
#        return_type="void",  # Only difference from c_mixin_function_buf
        temps=["cdesc"],
    ),

    # Pass CHARACTER and LEN to C wrapper.
    dict(
        name="f_mixin_in_character_buf",
        # Do not use arg_decl here since it does not understand +len(30) on functions.

        temps=["len"],
        declare=[
            "integer(C_INT) {c_var_len}",
        ],
        pre_call=[
            "{c_var_len} = len({f_var}, kind=C_INT)",
        ],
        arg_c_call=["{f_var}", "{c_var_len}"],

        # XXX - statements.yaml getNameErrorPattern pgi reports an error
        # Argument number 2 to c_get_name_error_pattern_bufferify: kind mismatch 
        # By breaking it out as an explicit assign, the error goes away.
        # Only difference from other uses is setting
        # function attribute +len(get_name_length())
#        arg_c_call=[
#            "{f_var}",
#            "len({f_var}, kind=C_INT)",
#        ],
        f_module=dict(iso_c_binding=["C_INT"]),
        need_wrapper=True,
    ),
    dict(
        # Used with function which pass in character argument.
        # Used with function which return a char *.
        # C wrapper will fill argument.
        name="c_mixin_in_character_buf",
        c_arg_decl=[
            "char *{c_var}",
            "int {c_var_len}",
        ],
        f_c_arg_names=["{c_var}", "{c_var_len}"],
        f_c_arg_decl=[
            "character(kind=C_CHAR), intent({f_intent}) :: {c_var}(*)",
            "integer(C_INT), value, intent(IN) :: {c_var_len}",
        ],
        f_c_module=dict(iso_c_binding=["C_CHAR", "C_INT"]),
        temps=["len"],
    ),

    ##########
    # bool
    dict(
        name="f_in_bool",
        c_local_var=True,
        pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
    ),
    dict(
        name="f_out_bool",
        c_local_var=True,
        post_call=["{f_var} = {c_var}  ! coerce to logical"],
    ),
    dict(
        name="f_inout_bool",
        c_local_var=True,
        pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
        post_call=["{f_var} = {c_var}  ! coerce to logical"],
    ),
    dict(
        name="f_function_bool",
        # The wrapper is needed to convert bool to logical
        need_wrapper=True
    ),

    ##########
    # native
    dict(
        name="f_out_native_*",
        arg_decl=[
            "{f_type}, intent({f_intent}) :: {f_var}{f_assumed_shape}",
        ],
    ),
    
    dict(
        # Any array of pointers.  Assumed to be non-contiguous memory.
        # All Fortran can do is treat as a type(C_PTR).
        name="c_in_native_**",
        c_arg_decl=[
            "{cxx_type} **{cxx_var}",
        ],
        f_c_arg_decl=[
            "type(C_PTR), intent(IN), value :: {c_var}",
        ],
        f_c_arg_names=["{c_var}"],
        f_c_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        # double **count _intent(out)+dimension(ncount)
        name="c_out_native_**_cdesc",
        mixin=["c_mixin_out_array_cdesc"],
        c_helper="ShroudTypeDefines array_context",
        c_pre_call=[
            "{c_const}{cxx_type} *{cxx_var};",
        ],
        c_arg_call=["&{cxx_var}"],
        c_post_call=[
            "{c_var_cdesc}->cxx.addr  = {cxx_nonconst_ptr};",
            "{c_var_cdesc}->cxx.idtor = {idtor};",
            "{c_var_cdesc}->addr.base = {cxx_var};",
            "{c_var_cdesc}->type = {sh_type};",
            "{c_var_cdesc}->elem_len = sizeof({cxx_type});",
            "{c_var_cdesc}->rank = {rank};"
            "{c_array_shape}",
            "{c_var_cdesc}->size = {c_array_size};",
        ],
        # XXX - similar to c_function_native_*_buf
    ),
    dict(
        name="c_out_native_*&_cdesc",
        base="c_out_native_**_cdesc",
        c_arg_call=["{cxx_var}"],
    ),
    dict(
        # deref(allocatable)
        # A C function with a 'int **' argument associates it
        # with a Fortran pointer.
        # f_out_native_**_cdesc_allocatable
        # f_out_native_*&_cdesc_allocatable
        name="f_out_native_**/*&_cdesc_allocatable",
        mixin=["f_mixin_out_array_cdesc"],
        c_helper="copy_array",
        f_helper="copy_array",
#XXX        f_helper="copy_array_{c_type}",
        arg_decl=[
            "{f_type}, intent({f_intent}), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        post_call=[
            # intent(out) ensure that it is already deallocated.
            "allocate({f_var}{f_array_allocate})",
            "call {hnamefunc0}(\t{c_var_cdesc},\t C_LOC({f_var}),\t {c_var_cdesc}%size)"#size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        # deref(pointer)
        # A C function with a 'int **' argument associates it
        # with a Fortran pointer.
        # f_out_native_**_cdesc_pointer
        # f_out_native_*&_cdesc_pointer
        name="f_out_native_**/*&_cdesc_pointer",
        mixin=["f_mixin_out_array_cdesc"],
        arg_decl=[
            "{f_type}, intent({f_intent}), pointer :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        post_call=[
            "call c_f_pointer({c_var_cdesc}%base_addr, {f_var}{f_array_shape})",
        ],
    ),
    dict(
        # Make argument type(C_PTR) from 'int ** +intent(out)+deref(raw)'
        name="f_out_native_**_raw",
        arg_decl=[
            "type(C_PTR), intent({f_intent}) :: {f_var}",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
    ),

    # Used with intent IN, INOUT, and OUT.
    dict(
        # c_in_native_*_cdesc
        # c_out_native_*_cdesc
        # c_inout_native_*_cdesc
        name="c_in/out/inout_native_*_cdesc",
        mixin=["c_mixin_out_array_cdesc"],

        c_arg_decl=[
            "{C_array_type} *{c_var_cdesc}",
        ],
        f_c_arg_names=["{c_var_cdesc}"],
        f_c_arg_decl=[
            "type({F_array_type}), intent(OUT) :: {c_var_cdesc}",
        ],
        
#        c_helper="ShroudTypeDefines",
        lang_c=dict(
            c_pre_call=[
                "{cxx_type} * {c_var} = {c_var_cdesc}->addr.base;",
            ],
        ),
        lang_cxx=dict(
            c_pre_call=[
#            "{cxx_type} * {c_var} = static_cast<{cxx_type} *>\t"
#            "({c_var_cdesc}->addr.base);",
                "{cxx_type} * {c_var} = static_cast<{cxx_type} *>\t"
                "(const_cast<void *>({c_var_cdesc}->addr.base));",
            ],
        ),
    ),
    dict(
        # f_in_native_*_cdesc
        # f_out_native_*_cdesc
        # f_inout_native_*_cdesc
        name="f_in/out/inout_native_*_cdesc",
        mixin=["f_mixin_out_array_cdesc"],
        # TARGET required for argument to C_LOC.
        arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_helper="ShroudTypeDefines array_context",
        f_module=dict(iso_c_binding=["C_LOC"]),
        pre_call=[
            "{c_var_cdesc}%base_addr = C_LOC({f_var})",
            "{c_var_cdesc}%type = {sh_type}",
            "! {c_var_cdesc}%elem_len = C_SIZEOF()",
#            "{c_var_cdesc}%size = size({f_var})",
            "{c_var_cdesc}%size = {size}",
            # Do not set shape for scalar via f_cdesc_shape
            "{c_var_cdesc}%rank = {rank}{f_cdesc_shape}",
        ],
    ),

    ########################################
    ##### hidden
    # Hidden argument will not be added for Fortran
    # or C buffer wrapper. Instead it is a local variable
    # in the C wrapper and passed to library function.
    dict(
        # c_out_native_*_hidden
        # c_inout_native_*_hidden
        name="c_out/inout_native_*_hidden",
        c_pre_call=[
            "{cxx_type} {cxx_var};",
        ],
        c_arg_call=["&{cxx_var}"],
    ),
    dict(
        # c_out_native_&_hidden
        # c_inout_native_&_hidden
        name="c_out/inout_native_&_hidden",
        c_pre_call=[
            "{cxx_type} {cxx_var};",
        ],
        c_arg_call=["{cxx_var}"],
    ),
    
    ########################################
    # void *
    dict(
        name="f_in_void_*",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR), intent(IN) :: {f_var}",
        ],
    ),
    dict(
        # return a type(C_PTR)
        name="f_function_void_*",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    dict(
        # c_in_void_*_cdesc
        # c_out_void_*_cdesc
        # c_inout_void_*_cdesc
        name="c_in/out/inout_void_*_cdesc",
        base="c_in_native_*_cdesc",
    ),
    dict(
        name="f_in/out/inout_void_*_cdesc",
        base="f_in_native_*_cdesc",
    ),    

    ########################################
    # void **
    dict(
        # Treat as an assumed length array in Fortran interface.
        # c_in_void_**
        # c_out_void_**
        # c_inout_void_**
        name='c_in/out/inout_void_**',
        c_arg_decl=[
            "void **{c_var}",
        ],
        f_c_arg_decl=[
            "type(C_PTR), intent({f_intent}) :: {c_var}{f_c_dimension}",
        ],
        f_c_arg_names=["{c_var}"],
        f_c_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name="f_in/out/inout_void_**",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR), intent({f_intent}) :: {f_var}{f_assumed_shape}",
        ],
    ),
    
    dict(
        # Works with deref allocatable and pointer.
        # c_function_native_*
        # c_function_native_&
        # c_function_native_**
        name="c_function_native_*/&/**",
        f_c_result_decl=[
            "type(C_PTR) {c_var}",
        ],
        f_c_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name="c_function_native_*_scalar",
        f_c_result_decl=[
            "{f_type} :: {c_var}",
        ],
        f_c_module_line="iso_c_binding:{f_kind}",
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
    #
    # Works with deref pointer and allocatable since Fortran
    # does that part.
    dict(
        name="c_function_native_*_cdesc",
        mixin=["c_mixin_function_cdesc"],
        c_helper="ShroudTypeDefines array_context",
        c_post_call=[
            "{c_var_cdesc}->cxx.addr  = {cxx_nonconst_ptr};",
            "{c_var_cdesc}->cxx.idtor = {idtor};",
            "{c_var_cdesc}->addr.base = {cxx_var};",
            "{c_var_cdesc}->type = {sh_type};",
            "{c_var_cdesc}->elem_len = sizeof({cxx_type});",
            "{c_var_cdesc}->rank = {rank};"
            "{c_array_shape}",
            "{c_var_cdesc}->size = {c_array_size};",
        ],
    ),
    dict(
        name="f_function_native_*_cdesc_allocatable",
        mixin=["f_mixin_function_cdesc"],
        c_helper="copy_array",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        arg_decl=[
            "{f_type}, allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        post_call=[
            # XXX - allocate scalar
            "allocate({f_var}{f_array_allocate})",
            "call {hnamefunc0}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},\t kind=C_SIZE_T))",
        ],
    ),

    dict(
        # Pointer to scalar.
        # type(C_PTR) is returned instead of a cdesc argument.
        name="f_function_native_*_pointer",
        f_module=dict(iso_c_binding=["C_PTR", "c_f_pointer"]),
        arg_decl=[
            "{f_type}, pointer :: {f_var}",
        ],
        declare=[
            "type(C_PTR) :: {c_local_ptr}",
        ],
        call=[
            "{c_local_ptr} = {F_C_call}({F_arg_c_call})",
        ],
        post_call=[
            "call c_f_pointer({c_local_ptr}, {F_result})",
        ],
        local=["ptr"],
    ),
    dict(
        # f_function_native_*_cdesc_pointer
        # f_getter_native_*_cdesc_pointer
        name="f_function/getter_native_*_cdesc_pointer",
        mixin=["f_mixin_function_cdesc"],
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        arg_decl=[
            "{f_type}, pointer :: {f_var}{f_assumed_shape}",
        ],
        post_call=[
            "call c_f_pointer({c_var_cdesc}%base_addr, {F_result}{f_array_shape})",
        ],
    ),
    dict(
        # +deref(pointer) +owner(caller)
        name="f_function_native_*_cdesc_pointer_caller",
        mixin=["f_mixin_function_cdesc"],
        f_helper="capsule_helper",
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        arg_name=["{c_var_capsule}"],
        arg_decl=[
            "{f_type}, pointer :: {f_var}{f_assumed_shape}",
            "type({F_capsule_type}), intent(OUT) :: {c_var_capsule}",
        ],
        post_call=[
            "call c_f_pointer(\t{c_var_cdesc}%base_addr,\t {F_result}{f_array_shape})",
            "{c_var_capsule}%mem = {c_var_cdesc}%cxx",
        ],
    ),
    dict(
        name="f_function_native_*_raw",
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    dict(
        # int **func(void)
        # regardless of deref value.
        name="f_function_native_**",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    
    dict(
        name="f_function_native_&",
        base="f_function_native_*_pointer",   # XXX - change base to &?
    ),
    dict(
        name="f_function_native_&_buf_pointer",
        base="f_function_native_*_pointer",   # XXX - change base to &?
        arg_decl=[
            "{f_type}, pointer :: {f_var}{f_assumed_shape}",
        ],
    ),

    dict(
        name="f_function_native_*_scalar",
        # avoid catching f_native_*
    ),


    ########################################
    # char arg
    dict(
        name="c_in_char_scalar",
        c_arg_decl=[
            "char {c_var}",
        ],
        f_c_arg_decl=[
            "character(kind=C_CHAR), value, intent(IN) :: {c_var}",
        ],
        f_c_arg_names=["{c_var}"],
        f_c_module=dict(iso_c_binding=["C_CHAR"]),
    ),
    dict(
        name="f_in_char_scalar",
        # By default the declaration is character(LEN=*).
        arg_decl=[
            "character, value, intent(IN) :: {f_var}",
        ],
    ),

#    dict(
#        This simpler version had to be replace for pgi and cray.
#        See below.
#        name="c_function_char_scalar",
#        f_c_result_decl=[
#            "character(kind=C_CHAR) :: {c_var}",
#        ],
#        f_c_module=dict(iso_c_binding=["C_CHAR"]),
#    ),
    dict(
        name="f_function_char_scalar",
        arg_c_call=["{c_var}"],  # Pass result as an argument.
    ),
    dict(
        # Pass result as an argument.
        # pgi and cray compilers have problems with functions which
        # return a scalar char.
        name="c_function_char_scalar",
        call=[
            "*{c_var} = {function_name}({C_call_list});",
        ],
        c_arg_decl=[
            "char *{c_var}",
        ],
        f_c_arg_decl=[
            "character(kind=C_CHAR), intent(OUT) :: {c_var}",
        ],
        f_c_arg_names=["{c_var}"],
        f_c_module=dict(iso_c_binding=["C_CHAR"]),
        return_type="void",  # Convert to function.
    ),
#    dict(
#        # Blank fill result.
#        name="c_XXXfunction_char_scalar_buf",
#        c_impl_header=["<string.h>"],
#        lang_cxx=dict(
#            impl_header=["<cstring>"],
#        ),
#        c_post_call=[
#            "{stdlib}memset({c_var}, ' ', {c_var_len});",
#            "{c_var}[0] = {cxx_var};",
#        ],
#    ),
    
    dict(
        name="c_function_char_*",
        f_c_result_decl=[
            "type(C_PTR) {c_var}",
        ],
        f_c_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        # NULL terminate the input string.
        # Skipped if ftrim_char_in, the terminate is done in Fortran.
        name="c_in_char_*_buf",
        mixin=["c_mixin_in_character_buf"],
        cxx_local_var="pointer",
        c_helper="ShroudStrAlloc ShroudStrFree",
        c_pre_call=[
            "char * {cxx_var} = ShroudStrAlloc(\t"
            "{c_var},\t {c_var_len},\t {c_blanknull});",
        ],
        c_post_call=[
            "ShroudStrFree({cxx_var});"
        ],
    ),
    dict(
        # f_in_char_*_buf
        # f_out_char_*_buf
        name="f_in/out_char_*_buf",
        mixin=["f_mixin_in_character_buf"],
    ),
    dict(
        name="c_out_char_*_buf",
        mixin=["c_mixin_in_character_buf"],
        c_helper="ShroudStrBlankFill",
        c_post_call=[
            "ShroudStrBlankFill({c_var}, {c_var_len});"
        ],
    ),
    dict(
        name="f_inout_char_*_buf",
        mixin=["f_mixin_in_character_buf"],
    ),
    dict(
        name="c_inout_char_*_buf",
        mixin=["c_mixin_in_character_buf"],
        cxx_local_var="pointer",
        c_helper="ShroudStrAlloc ShroudStrCopy ShroudStrFree",
        c_pre_call=[
            "char * {cxx_var} = ShroudStrAlloc(\t"
            "{c_var},\t {c_var_len},\t {c_blanknull});",
        ],
        c_post_call=[
            # nsrc=-1 will call strlen({cxx_var})
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var},\t -1);",
            "ShroudStrFree({cxx_var});",
        ],
    ),
    dict(
        # Copy result into caller's buffer.
        name="f_function_char_*_buf",
        mixin=["f_mixin_in_character_buf"],
    ),
    dict(
        # Copy result into caller's buffer.
        #  char *getname() +len(30)
        name="c_function_char_*_buf",
        cxx_local_var="result",
        mixin=["c_mixin_in_character_buf"],
        c_helper="ShroudStrCopy",
        c_post_call=[
            # nsrc=-1 will call strlen({cxx_var})
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var},\t -1);",
        ],
        return_type="void",
    ),

    dict(
        # Used with both deref allocatable and pointer.
        name="c_function_char_*_cdesc",
        mixin=["c_mixin_function_cdesc"],
        c_helper="ShroudTypeDefines",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        c_post_call=[
            "{c_var_cdesc}->cxx.addr = {cxx_nonconst_ptr};",
            "{c_var_cdesc}->cxx.idtor = {idtor};",
            "{c_var_cdesc}->addr.ccharp = {cxx_var};",
            "{c_var_cdesc}->type = {sh_type};",
            "{c_var_cdesc}->elem_len = {cxx_var} == {nullptr} ? 0 : {stdlib}strlen({cxx_var});",
            "{c_var_cdesc}->size = 1;",
            "{c_var_cdesc}->rank = 0;",
        ],
    ),

    dict(
        # char *func() +deref(raw)
        name="f_function_char_*_raw",
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    #####
    dict(
        # Treat as an assumed length array in Fortran interface.
        name='c_in_char_**',
        c_arg_decl=[
            "char **{c_var}",
        ],
        f_c_arg_decl=[
            "type(C_PTR), intent(IN) :: {c_var}(*)",
        ],
        f_c_arg_names=["{c_var}"],
        f_c_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name='f_in_char_**_buf',
        mixin=["f_mixin_in_string_array_buf"],
    ),
    dict(
        name='c_in_char_**_buf',
        mixin=["c_mixin_in_string_array_buf"],
        c_helper="ShroudStrArrayAlloc ShroudStrArrayFree",
        cxx_local_var="pointer",
        c_pre_call=[
            "char **{cxx_var} = ShroudStrArrayAlloc("
            "{c_var},\t {c_var_size},\t {c_var_len});",
        ],
        c_post_call=[
            "ShroudStrArrayFree({cxx_var}, {c_var_size});",
        ],
    ),
    #####
    dict(
        # f_function_char_scalar_cdesc_allocatable
        # f_function_char_*_cdesc_allocatable
        name="f_function_char_scalar/*_cdesc_allocatable",
        mixin=["f_mixin_function_cdesc"],
        c_helper="copy_string",
        f_helper="copy_string array_context",
        arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        post_call=[
            "allocate(character(len={c_var_cdesc}%elem_len):: {f_var})",
            "call {hnamefunc0}(\t{c_var_cdesc},\t {f_var},\t {c_var_cdesc}%elem_len)",
        ],
    ),
    dict(
        # f_function_char_scalar_cdesc_pointer
        # f_function_char_*_cdesc_pointer
        # f_function_string_scalar_cdesc_pointer
        # f_function_string_*_cdesc_pointer
        name="f_function_char/string_scalar/*_cdesc_pointer",
        mixin=["f_mixin_function_cdesc"],
        f_helper="pointer_string array_context",
        arg_decl=[
            "character(len=:), pointer :: {f_var}",
        ],
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        post_call=[
            # BLOCK is Fortran 2008
            #"block+",
            #"character(len={c_var_cdesc}%elem_len), pointer :: {c_local_s}",
            #"call c_f_pointer({c_var_cdesc}%base_addr, {c_local_s})",
            #"{f_var} => {c_local_s}",
            #"-end block",
            "call {hnamefunc0}(\t{c_var_cdesc},\t {f_var})",
        ],
    ),

    dict(
        # c_in_string_*
        # c_in_string_&
        name="c_in_string_*/&",
        cxx_local_var="scalar",
        c_pre_call=["{c_const}std::string {cxx_var}({c_var});"],
    ),
    dict(
        # c_out_string_*
        # c_out_string_&
        name="c_out_string_*/&",
        lang_cxx=dict(
            impl_header=["<cstring>"],
        ),
        # #- c_pre_call=[
        # #-     'int {c_var_trim} = strlen({c_var});',
        # #-     ],
        cxx_local_var="scalar",
        c_pre_call=["{c_const}std::string {cxx_var};"],
        c_post_call=[
            # This may overwrite c_var if cxx_val is too long
            "strcpy({c_var}, {cxx_var}{cxx_member}c_str());"
        ],
    ),
    dict(
        # c_inout_string_*
        # c_inout_string_&
        name="c_inout_string_*/&",
        lang_cxx=dict(
            impl_header=["<cstring>"],
        ),
        cxx_local_var="scalar",
        c_pre_call=["{c_const}std::string {cxx_var}({c_var});"],
        c_post_call=[
            # This may overwrite c_var if cxx_val is too long
            "strcpy({c_var}, {cxx_var}{cxx_member}c_str());"
        ],
    ),
    dict(
        name="f_in_string_*/&_buf",
        mixin=["f_mixin_in_character_buf"],
    ),
    dict(
        # c_in_string_*_buf
        # c_in_string_&_buf
        name="c_in_string_*/&_buf",
        mixin=["c_mixin_in_character_buf"],
        c_helper="ShroudLenTrim",
        cxx_local_var="scalar",
        c_pre_call=[
            "{c_const}std::string {cxx_var}({c_var},\t ShroudLenTrim({c_var}, {c_var_len}));",
        ],
    ),
    dict(
        name="f_out_string_*/&_buf",
        mixin=["f_mixin_in_character_buf"],
    ),
    dict(
        # c_out_string_*_buf
        # c_out_string_&_buf
        name="c_out_string_*/&_buf",
        mixin=["c_mixin_in_character_buf"],
        c_helper="ShroudStrCopy",
        cxx_local_var="scalar",
        c_pre_call=[
            "std::string {cxx_var};",
        ],
        c_post_call=[
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var}{cxx_member}data(),"
            "\t {cxx_var}{cxx_member}size());"
        ],
    ),
    dict(
        name="f_inout_string_*/&_buf",
        mixin=["f_mixin_in_character_buf"],
    ),
    dict(
        # c_inout_string_*_buf
        # c_inout_string_&_buf
        name="c_inout_string_*/&_buf",
        mixin=["c_mixin_in_character_buf"],
        c_helper="ShroudStrCopy ShroudLenTrim",
        cxx_local_var="scalar",
        c_pre_call=[
            "std::string {cxx_var}({c_var},\t ShroudLenTrim({c_var}, {c_var_len}));",
        ],
        c_post_call=[
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var}{cxx_member}data(),"
            "\t {cxx_var}{cxx_member}size());"
        ],
    ),
    dict(
        # c_function_string_scalar
        # c_function_string_*
        # c_function_string_&
        name="c_function_string_scalar/*/&",
        # cxx_to_c creates a pointer from a value via c_str()
        # The default behavior will dereference the value.
        ret=[
            "return {c_var};",
        ],
        f_c_result_decl=[
            "type(C_PTR) {c_var}",
        ],
        f_c_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        # No need to allocate a local copy since the string is copied
        # into a Fortran variable before the string is deleted.
        # c_function_string_scalar_buf
        # c_function_string_*_buf
        # c_function_string_&_buf
        name="c_function_string_scalar/*/&_buf",
        mixin=["c_mixin_in_character_buf"],
        f_c_arg_decl=[
            # Change to intent(OUT) from mixin.
            "character(kind=C_CHAR), intent(OUT) :: {c_var}(*)",
            "integer(C_INT), value, intent(IN) :: {c_var_len}",
        ],
        c_helper="ShroudStrCopy",
        c_post_call=[
            "if ({cxx_var}{cxx_member}empty()) {{+",
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {nullptr},\t 0);",
            "-}} else {{+",
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var}{cxx_member}data(),"
            "\t {cxx_var}{cxx_member}size());",
            "-}}",
        ],
        return_type="void",
    ),

    # std::string
    dict(
        name="f_XXXin_string_scalar",  # pairs with c_in_string_scalar_buf
        need_wrapper=True,
        mixin=["f_mixin_in_character_buf"],
        arg_decl=[
            # Remove VALUE added by f_default
            "character(len=*), intent(IN) :: {f_var}",
        ],
    ),
    dict(
        # Used with C wrapper.
        name="c_in_string_scalar",
        c_arg_decl=[
            # Argument is a pointer while std::string is a scalar.
            # C++ compiler will convert to std::string when calling function.
            "char *{c_var}",
        ],
        f_c_arg_decl=[
            # Remove VALUE added by c_default
            "character(kind=C_CHAR), intent(IN) :: {c_var}(*)",
        ],
        f_c_arg_names=["{c_var}"],
        f_c_module=dict(iso_c_binding=["C_CHAR"]),
    ),
    dict(
        name="f_in_string_scalar_buf",
        mixin=["f_mixin_in_character_buf"],
        arg_decl=[
            # Remove VALUE added by f_default
            "character(len=*), intent({f_intent}) :: {f_var}",
        ],
    ),
    dict(
        name="c_in_string_scalar_buf",
        mixin=["c_mixin_in_character_buf"],
        cxx_local_var="scalar",
        c_pre_call=[
            "int {c_local_trim} = ShroudLenTrim({c_var}, {c_var_len});",
            "std::string {cxx_var}({c_var}, {c_local_trim});",
        ],
        call=[
            "{cxx_var}",
        ],
        local=["trim"],
    ),
    
    # Uses a two part call to copy results of std::string into a
    # allocatable Fortran array.
    #    c_step1(context)
    #    allocate(character(len=context%elem_len): Fout)
    #    c_step2(context, Fout, context%elem_len)
    # only used with bufferifed routines and intent(out) or result
    # std::string * function()
    dict(
        # c_function_string_*_cdesc_allocatable
        # c_function_string_&_cdesc_allocatable
        # c_function_string_*_cdesc_pointer
        # c_function_string_&_cdesc_pointer
        name="c_function_string_*/&_cdesc_allocatable/pointer",
        mixin=["c_mixin_function_cdesc"],
        c_helper="ShroudStrToArray",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        c_post_call=[
            "ShroudStrToArray(\t{c_var_cdesc},\t {cxx_addr}{cxx_var},\t {idtor});",
        ],
    ),

    # std::string function()
    # Must allocate the std::string then assign to it via cxx_rv_decl.
    # This allows the std::string to outlast the function return.
    # The Fortran wrapper will ALLOCATE memory, copy then delete the string.
    dict(
        name="c_function_string_scalar_cdesc_allocatable",
        mixin=["c_mixin_function_cdesc"],
        cxx_local_var="pointer",
        c_helper="ShroudStrToArray",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        c_pre_call=[
            "std::string * {cxx_var} = new std::string;",
        ],
        destructor_name="new_string",
        destructor=[
            "std::string *cxx_ptr = \treinterpret_cast<std::string *>(ptr);",
            "delete cxx_ptr;",
        ],
        c_post_call=[
            "ShroudStrToArray({c_var_cdesc}, {cxx_var}, {idtor});",
        ],
    ),

    dict(
        # f_function_string_scalar_buf
        # f_function_string_*_buf
        # f_function_string_&_buf
        name="f_function_string_scalar/*/&_buf",
        mixin=["f_mixin_in_character_buf"],
    ),
    
    # similar to f_function_char_scalar_allocatable
    dict(
        # f_function_string_scalar_cdesc_allocatable
        # f_function_string_*_cdesc_allocatable
        # f_function_string_&_cdesc_allocatable
        name="f_function_string_scalar/*/&_cdesc_allocatable",
        mixin=["f_mixin_function_cdesc"],
        c_helper="copy_string",
        f_helper="copy_string array_context",
        arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        post_call=[
            "allocate(character(len={c_var_cdesc}%elem_len):: {f_var})",
            "call {hnamefunc0}({c_var_cdesc},\t {f_var},\t {c_var_cdesc}%elem_len)",
        ],
    ),
    
    ########################################
    # vector
    # Specialize for std::vector<native>
    dict(
        # c_in_vector_scalar_buf_targ_native_scalar
        # c_in_vector_*_buf_targ_native_scalar
        # c_in_vector_&_buf_targ_native_scalar
        name="c_in_vector_scalar/*/&_buf_targ_native_scalar",
        mixin=["c_mixin_in_array_buf"],
        cxx_local_var="scalar",
        c_pre_call=[
            (
                "{c_const}std::vector<{cxx_T}> "
                "{cxx_var}({c_var}, {c_var} + {c_var_size});"
            )
        ],
    ),
    # cxx_var is always a pointer to a vector
    dict(
        # c_out_vector_*_cdesc_targ_native_scalar
        # c_out_vector_&_cdesc_targ_native_scalar
        name="c_out_vector_*/&_cdesc_targ_native_scalar",
        mixin=["c_mixin_out_array_cdesc"],
        cxx_local_var="pointer",
        c_helper="ShroudTypeDefines",
        c_pre_call=[
            "{c_const}std::vector<{cxx_T}>"
            "\t *{cxx_var} = new std::vector<{cxx_T}>;"
        ],
        c_post_call=[
            # Return address and size of vector data.
            "{c_var_cdesc}->cxx.addr  = {cxx_var};",
            "{c_var_cdesc}->cxx.idtor = {idtor};",
            "{c_var_cdesc}->addr.base = {cxx_var}->empty()"
            " ? {nullptr} : &{cxx_var}->front();",
            "{c_var_cdesc}->type = {sh_type};",
            "{c_var_cdesc}->elem_len = sizeof({cxx_T});",
            "{c_var_cdesc}->size = {cxx_var}->size();",
            "{c_var_cdesc}->rank = 1;",
            "{c_var_cdesc}->shape[0] = {c_var_cdesc}->size;",
        ],
        destructor_name="std_vector_{cxx_T}",
        destructor=[
            "std::vector<{cxx_T}> *cxx_ptr ="
            " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    dict(
        # c_out_vector_*_cdesc_allocatable_targ_native_scalar
        # c_out_vector_&_cdesc_allocatable_targ_native_scalar
        name="c_out_vector_*/&_cdesc_allocatable_targ_native_scalar",
        # XXX - this mixin is not working as expected, nested mixings...
#        mixin=["c_out_vector_*_cdesc_targ_native_scalar"],
        base="c_out_vector_*_cdesc_targ_native_scalar",
    ),
    dict(
        name="c_inout_vector_cdesc_targ_native_scalar",
        mixin=["c_mixin_inout_array_cdesc"],
        cxx_local_var="pointer",
        c_helper="ShroudTypeDefines",
        c_pre_call=[
            "std::vector<{cxx_T}> *{cxx_var} = \tnew std::vector<{cxx_T}>\t("
            "\t{c_var}, {c_var} + {c_var_size});"
        ],
        c_post_call=[
            # Return address and size of vector data.
            "{c_var_cdesc}->cxx.addr  = {cxx_var};",
            "{c_var_cdesc}->cxx.idtor = {idtor};",
            "{c_var_cdesc}->addr.base = {cxx_var}->empty()"
            " ? {nullptr} : &{cxx_var}->front();",
            "{c_var_cdesc}->type = {sh_type};",
            "{c_var_cdesc}->elem_len = sizeof({cxx_T});",
            "{c_var_cdesc}->size = {cxx_var}->size();",
            "{c_var_cdesc}->rank = 1;",
            "{c_var_cdesc}->shape[0] = {c_var_cdesc}->size;",
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
        name="c_function_vector_scalar_cdesc_targ_native_scalar",
        mixin=["c_mixin_function_cdesc"],
        cxx_local_var="pointer",
        c_helper="ShroudTypeDefines",
        c_pre_call=[
            "{c_const}std::vector<{cxx_T}>"
            "\t *{cxx_var} = new std::vector<{cxx_T}>;"
        ],
        c_post_call=[
            # Return address and size of vector data.
            "{c_var_cdesc}->cxx.addr  = {cxx_var};",
            "{c_var_cdesc}->cxx.idtor = {idtor};",
            "{c_var_cdesc}->addr.base = {cxx_var}->empty()"
            " ? {nullptr} : &{cxx_var}->front();",
            "{c_var_cdesc}->type = {sh_type};",
            "{c_var_cdesc}->elem_len = sizeof({cxx_T});",
            "{c_var_cdesc}->size = {cxx_var}->size();",
            "{c_var_cdesc}->rank = 1;",
            "{c_var_cdesc}->shape[0] = {c_var_cdesc}->size;",
        ],
        destructor_name="std_vector_{cxx_T}",
        destructor=[
            "std::vector<{cxx_T}> *cxx_ptr ="
            " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    #                dict(
    #                    name="c_function_vector_buf",
    #                    c_helper='ShroudStrCopy',
    #                    c_post_call=[
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

    # Specialize for std::vector<native *>
    dict(
        # Create a vector for pointers
        name="c_in_vector_&_buf_targ_native_*",
        mixin=["c_mixin_in_2d_array_buf"],
        cxx_local_var="scalar",
        c_pre_call=[
            "std::vector<{cxx_T}> {cxx_var};",
            "for (size_t i=0; i < {c_var_size}; ++i) {{+",
            "{cxx_var}.push_back({c_var} + ({c_var_len}*i));",
            "-}}"
        ],
    ),
    dict(
        name="f_in_vector_buf_targ_native_*",
        mixin=["f_mixin_in_2d_array_buf"],
    ),
    
    # Specialize for std::vector<string>.
    dict(
        name="f_in_vector_buf_targ_string_scalar",
        mixin=["f_mixin_in_string_array_buf"],
    ),
    dict(
        # c_in_vector_scalar_buf_targ_string_scalar
        # c_in_vector_*_buf_targ_string_scalar
        # c_in_vector_&_buf_targ_string_scalar
        name="c_in_vector_scalar/*/&_buf_targ_string_scalar",
        mixin=["c_mixin_in_string_array_buf"],
        c_helper="ShroudLenTrim",
        cxx_local_var="scalar",
        c_pre_call=[
            "std::vector<{cxx_T}> {cxx_var};",
            "{{+",
            "{c_const}char * {c_local_s} = {c_var};",
            "std::vector<{cxx_T}>::size_type",
            "+{c_local_i} = 0,",
            "{c_local_n} = {c_var_size};",
            "-for(; {c_local_i} < {c_local_n}; {c_local_i}++) {{+",
            "{cxx_var}.push_back(\t"
            "std::string({c_local_s},\tShroudLenTrim({c_local_s}, {c_var_len})));",
            "{c_local_s} += {c_var_len};",
            "-}}",
            "-}}",
        ],
        local=["i", "n", "s"],
    ),
    # XXX untested [cf]_out_vector_buf_string
    dict(
        name="f_out_vector_buf_targ_string_scalar",
        mixin=["f_mixin_in_string_array_buf"],
    ),
    dict(
        name="c_out_vector_buf_targ_string_scalar",
        mixin=["c_mixin_in_string_array_buf"],
        c_helper="ShroudStrCopy",
        cxx_local_var="scalar",
        c_pre_call=["{c_const}std::vector<{cxx_T}> {cxx_var};"],
        c_post_call=[
            "{{+",
            "char * {c_local_s} = {c_var};",
            "std::vector<{cxx_T}>::size_type",
            "+{c_local_i} = 0,",
            "{c_local_n} = {c_var_size};",
            "{c_local_n} = std::min({cxx_var}.size(),{c_local_n});",
            "-for(; {c_local_i} < {c_local_n}; {c_local_i}++) {{+",
            "ShroudStrCopy("
            "{c_local_s}, {c_var_len},"
            "\t {cxx_var}[{c_local_i}].data(),"
            "\t {cxx_var}[{c_local_i}].size());",
            "{c_local_s} += {c_var_len};",
            "-}}",
            "-}}",
        ],
        local=["i", "n", "s"],
    ),
    # XXX untested [cf]_inout_vector_buf_string
    dict(
        name="f_inout_vector_buf_targ_string_scalar",
        mixin=["f_mixin_in_string_array_buf"],
    ),
    dict(
        name="c_inout_vector_buf_targ_string_scalar",
        mixin=["c_mixin_in_string_array_buf"],
        cxx_local_var="scalar",
        c_pre_call=[
            "std::vector<{cxx_T}> {cxx_var};",
            "{{+",
            "{c_const}char * {c_local_s} = {c_var};",
            "std::vector<{cxx_T}>::size_type",
            "+{c_local_i} = 0,",
            "{c_local_n} = {c_var_size};",
            "-for(; {c_local_i} < {c_local_n}; {c_local_i}++) {{+",
            "{cxx_var}.push_back"
            "(std::string({c_local_s},\tShroudLenTrim({c_local_s}, {c_var_len})));",
            "{c_local_s} += {c_var_len};",
            "-}}",
            "-}}",
        ],
        c_post_call=[
            "{{+",
            "char * {c_local_s} = {c_var};",
            "std::vector<{cxx_T}>::size_type",
            "+{c_local_i} = 0,",
            "{c_local_n} = {c_var_size};",
            "-{c_local_n} = std::min({cxx_var}.size(),{c_local_n});",
            "for(; {c_local_i} < {c_local_n}; {c_local_i}++) {{+",
            "ShroudStrCopy({c_local_s}, {c_var_len},"
            "\t {cxx_var}[{c_local_i}].data(),"
            "\t {cxx_var}[{c_local_i}].size());",
            "{c_local_s} += {c_var_len};",
            "-}}",
            "-}}",
        ],
        local=["i", "n", "s"],
    ),

    dict(
        # Pass argument and size to C.
        # Pass array_type to C which will fill it in.
        name="f_mixin_inout_char_array_cdesc",
        f_helper="array_context",
        declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
        ],
#        arg_c_call=["{f_var}", "size({f_var}, kind=C_SIZE_T)", "{c_var_cdesc}"],
        arg_c_call=["{c_var_cdesc}"],
#        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        temps=["cdesc"],
    ),

    ##########
    dict(
        # Collect information about a string argument
        name="f_mixin_str_array",
        mixin=["f_mixin_out_array_cdesc"],

        # TARGET required for argument to C_LOC.
        arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_helper="ShroudTypeDefines array_context",
        f_module=dict(iso_c_binding=["C_LOC"]),
        declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
        ],
        pre_call=[
            "{c_var_cdesc}%cxx%addr = C_LOC({f_var})",
            "{c_var_cdesc}%base_addr = C_LOC({f_var})",
            "{c_var_cdesc}%type = SH_TYPE_CHAR",
            "{c_var_cdesc}%elem_len = len({f_var})",
            "{c_var_cdesc}%size = size({f_var})",
#            "{c_var_cdesc}%size = {size}",
            # Do not set shape for scalar via f_cdesc_shape
            "{c_var_cdesc}%rank = rank({f_var}){f_cdesc_shape}",
#            "{c_var_cdesc}%rank = {rank}{f_cdesc_shape}",
        ],
        arg_c_call=["{c_var_cdesc}"],
        temps=["cdesc"],
    ),

    dict(
        name="f_out_vector_&_cdesc_targ_string_scalar",
        mixin=["f_mixin_str_array"],
    ),
    dict(
        name="c_out_vector_&_cdesc_targ_string_scalar",
        mixin=["c_mixin_out_array_cdesc"],
        c_helper="vector_string_out",
        c_pre_call=[
            "{c_const}std::vector<std::string> {cxx_var};"
        ],
        c_arg_call=["{cxx_var}"],
        c_post_call=[
            "{hnamefunc0}(\t{c_var_cdesc},\t {cxx_var});",
        ],

    ),

    ##########
    # As above but +deref(allocatable)
    # 
    dict(
        name="f_out_vector_&_cdesc_allocatable_targ_string_scalar",
        arg_decl=[
            "character({f_char_len}), intent({f_intent}), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_helper="vector_string_allocatable array_context capsule_data_helper",
        c_helper="vector_string_allocatable",
        f_module=dict(iso_c_binding=["C_LOC"]),
        declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
            "type({F_array_type}) :: {c_var_out}",
        ],
        arg_c_call=["{c_var_out}"],
        post_call=[
            "{c_var_cdesc}%size = {c_var_out}%size;",
            "{c_var_cdesc}%elem_len = {c_var_out}%elem_len",
            "allocate({f_char_type}{f_var}({c_var_cdesc}%size))",
            "{c_var_cdesc}%cxx%addr = C_LOC({f_var});",
            "{c_var_cdesc}%base_addr = C_LOC({f_var});",
            "call {hnamefunc0}({c_var_cdesc}, {c_var_out})",
        ],
        temps=["cdesc", "out"],
    ),
    dict(
        name="c_out_vector_&_cdesc_allocatable_targ_string_scalar",
        mixin=["c_mixin_out_array_cdesc"],
        c_helper="vector_string_out_len",
        c_pre_call=[
#            "std::vector<std::string> *{cxx_var} = new {cxx_type};"  XXX cxx_tye=std::string
            "std::vector<std::string> *{cxx_var} = new std::vector<std::string>;"
        ],
        c_arg_call=["*{cxx_var}"],
        c_post_call=[
            "if ({c_char_len} > 0) {{+",
            "{c_var_cdesc}->elem_len = {c_char_len};",
            "-}} else {{+",
            "{c_var_cdesc}->elem_len = {hnamefunc0}(*{cxx_var});",
            "-}}",
            "{c_var_cdesc}->size      = {cxx_var}->size();",
            "{c_var_cdesc}->cxx.addr  = {cxx_var};",
            "{c_var_cdesc}->cxx.idtor = {idtor};",
        ],
    ),

    ##########
    #                    dict(
    #                        name="c_function_vector_buf_string",
    #                        c_helper='ShroudStrCopy',
    #                        c_post_call=[
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
        name="f_in_vector_buf_targ_native_scalar",
        mixin=["f_mixin_in_array_buf"],
    ),
    dict(
        # f_out_vector_*_cdesc_targ_native_scalar
        # f_out_vector_&_cdesc_targ_native_scalar
        name="f_out_vector_*/&_cdesc_targ_native_scalar",
        mixin=["f_mixin_out_array_cdesc"],
        c_helper="copy_array",
        f_helper="copy_array",
        # TARGET required for argument to C_LOC.
        arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["C_SIZE_T", "C_LOC"]),
        post_call=[
            "call {hnamefunc0}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        name="f_inout_vector_cdesc_targ_native_scalar",
        mixin=["f_mixin_inout_array_cdesc"],
        c_helper="copy_array",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        # TARGET required for argument to C_LOC.
        arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        post_call=[
            "call {hnamefunc0}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        # XXX - This group is not tested
        name="f_function_vector_scalar_cdesc",
        c_helper="copy_array",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        # TARGET required for argument to C_LOC.
        arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        post_call=[
            "call {hnamefunc0}(\t{temp0},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))"
        ],
    ),
    # copy into allocated array
    dict(
        # f_out_vector_*_cdesc_allocatable_targ_native_scalar
        # f_out_vector_&_cdesc_allocatable_targ_native_scalar
        name="f_out_vector_*/&_cdesc_allocatable_targ_native_scalar",
        mixin=["f_mixin_out_array_cdesc"],
        c_helper="copy_array",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        # TARGET required for argument to C_LOC.
        arg_decl=[
            "{f_type}, intent({f_intent}), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        post_call=[
            "allocate({f_var}({c_var_cdesc}%size))",
            "call {hnamefunc0}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        name="f_inout_vector_cdesc_allocatable_targ_native_scalar",
        mixin=["f_mixin_inout_array_cdesc"],
        c_helper="copy_array",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        # TARGET required for argument to C_LOC.
        arg_decl=[
            "{f_type}, intent({f_intent}), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        post_call=[
            "if (allocated({f_var})) deallocate({f_var})",
            "allocate({f_var}({c_var_cdesc}%size))",
            "call {hnamefunc0}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    # Similar to f_vector_out_allocatable but must declare result variable.
    # Always return a 1-d array.
    dict(
        name="f_function_vector_scalar_cdesc_allocatable_targ_native_scalar",
        mixin=["f_mixin_function_cdesc"],
        c_helper="copy_array",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        arg_decl=[
            "{f_type}, allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        post_call=[
            "allocate({f_var}({c_var_cdesc}%size))",
            "call {hnamefunc0}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
        ],
    ),

    ##########
    # Extract pointer to C++ instance.
    # convert C argument into a pointer to C++ type.
    dict(
        name="c_mixin_shadow",
        c_arg_decl=[
            "{c_type} * {c_var}",
        ],
        f_c_arg_decl=[
            "type({f_capsule_data_type}), intent({f_intent}) :: {c_var}",
        ],
        f_c_arg_names=["{c_var}"],
        f_c_module_line="{f_c_module_line}",
    ),
    
    dict(
        name="f_in_shadow",
        arg_decl=[
            "{f_type}, intent({f_intent}) :: {f_var}",
        ],
        arg_c_call=[
            "{f_var}%{F_derived_member}",
        ],
        need_wrapper=True,
    ),
    dict(
        # c_in_shadow_scalar
        # c_inout_shadow_scalar  # XXX inout by value makes no sense.
        name="c_in/inout_shadow_scalar",
        mixin=["c_mixin_shadow"],
        c_arg_decl=[
            "{c_type} {c_var}",
        ],
        f_c_arg_decl=[
            "type({f_capsule_data_type}), intent({f_intent}), value :: {c_var}",
        ],
        cxx_local_var="pointer",
        c_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} =\t "
            "{cast_static}{c_const}{cxx_type} *{cast1}{c_var}.addr{cast2};",
        ],
    ),
    dict(
        # c_in_shadow_*
        # c_in_shadow_&
        # c_inout_shadow_*
        # c_inout_shadow_&
        name="c_in/inout_shadow_*/&",
        mixin=["c_mixin_shadow"],
        cxx_local_var="pointer",
        c_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} =\t "
            "{cast_static}{c_const}{cxx_type} *{cast1}{c_var}->addr{cast2};",
        ],
    ),

    # Return a C_capsule_data_type.
    dict(
        # c_function_shadow_*_capsule
        # c_function_shadow_&_capsule
        name="c_function_shadow_*/&_capsule",
        mixin=["c_mixin_shadow"],
        cxx_local_var="result",
        c_post_call=[
            "{c_var}->addr = {cxx_nonconst_ptr};",
            "{c_var}->idtor = {idtor};",
        ],
        return_type="void",
    ),
    dict(
        name="c_function_shadow_scalar_capsule",
        # Return a instance by value.
        # Create memory in c_pre_call so it will survive the return.
        # owner="caller" sets idtor flag to release the memory.
        # c_local_var is passed in as argument.
        mixin=["c_mixin_shadow"],
        cxx_local_var="pointer",
        c_local_var="pointer",
        owner="caller",
        c_pre_call=[
            "{cxx_type} * {cxx_var} = new {cxx_type};",
        ],
        c_post_call=[
            "{c_var}->addr = {cxx_nonconst_ptr};",
            "{c_var}->idtor = {idtor};",
        ],
        return_type="void",
    ),
    
    # Return a C_capsule_data_type.
    dict(
        # c_function_shadow_*_capptr
        # c_function_shadow_&_capptr
        name="c_function_shadow_*/&_capptr",
        mixin=["c_mixin_shadow", "c_function_shadow_*_capsule"],
        c_local_var="pointer",
        return_type=None,
        ret=[
            "return {c_var};",
        ],
        f_c_result_var="{F_result_ptr}",
        f_c_result_decl=[
            "type(C_PTR) :: {F_result_ptr}",
        ],
        f_c_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name="c_function_shadow_scalar_capptr",
        mixin=["c_mixin_shadow", "c_function_shadow_scalar_capsule"],
        return_type="{c_type} *",
        ret=[
            "return {c_var};",
        ],
        f_c_result_var="{F_result_ptr}",
        f_c_result_decl=[
            "type(C_PTR) :: {F_result_ptr}",
        ],
        f_c_module=dict(iso_c_binding=["C_PTR"]),
    ),
    
    dict(
        # f_function_shadow_scalar_capsule
        # f_function_shadow_*_capsule
        # f_function_shadow_&_capsule
        name="f_function_shadow_scalar/*/&_capsule",
        mixin=["f_mixin_function_shadow_capsule"],
    ),
    dict(
        # f_function_shadow_scalar_capptr
        # f_function_shadow_*_capptr
        # f_function_shadow_&_capptr
        name="f_function_shadow_scalar/*/&_capptr",
        mixin=["f_mixin_function_shadow_capptr"],
    ),
    dict(
        name="f_dtor",
        arg_c_call=[],
    ),
    dict(
        name="c_ctor_shadow_scalar_capsule",
        mixin=["c_mixin_shadow"],
        cxx_local_var="pointer",
        call=[
            "{cxx_type} *{cxx_var} =\t new {cxx_type}({C_call_list});",
            "{c_var}->addr = static_cast<{c_const}void *>(\t{cxx_var});",
            "{c_var}->idtor = {idtor};",
        ],
        return_type="void",
        owner="caller",
    ),
    dict(
        name="c_ctor_shadow_scalar_capptr",
        mixin=["c_mixin_shadow", "c_ctor_shadow_scalar_capsule"],
        return_type=None,
        ret=[
            "return {c_var};",
        ],
        f_c_result_var="{F_result_ptr}",
        f_c_result_decl=[
            "type(C_PTR) {F_result_ptr}",
        ],
        f_c_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name="f_ctor_shadow_scalar_capsule",
        mixin=["f_mixin_function_shadow_capsule"],
    ),
    dict(
        name="f_ctor_shadow_scalar_capptr",
        mixin=["f_mixin_function_shadow_capptr"],
    ),
    dict(
        # NULL in stddef.h
        name="c_dtor",
        mixin=["c_mixin_noargs"],
        lang_c=dict(
            impl_header=["<stddef.h>"],
        ),
        lang_cxx=dict(
            impl_header=["<cstddef>"],
        ),
        call=[
            "delete {CXX_this};",
            "{C_this}->addr = {nullptr};",
        ],
        return_type="void",
    ),

    dict(
        # Used with in, out, inout
        # C pointer -> void pointer -> C++ pointer
        # c_in_struct
        # c_out_struct
        # c_inout_struct
        name="c_in/out/inout_struct",
        lang_cxx=dict(
            cxx_local_var="pointer", # cxx_local_var only used with C++
            c_pre_call=[
                "{c_const}{cxx_type} * {cxx_var} = \tstatic_cast<{c_const}{cxx_type} *>\t(static_cast<{c_const}void *>(\t{c_addr}{c_var}));",
            ],
        ),
    ),
    dict(
        name="c_function_struct",
        # C++ pointer -> void pointer -> C pointer
        c_local_var="pointer",
        lang_cxx=dict(
            c_post_call=[
                "{c_const}{c_type} * {c_var} = \tstatic_cast<{c_const}{c_type} *>(\tstatic_cast<{c_const}void *>(\t{cxx_addr}{cxx_var}));",
            ],
        ),
    ),

    # start function_struct_scalar
    dict(
        name="f_function_struct_scalar",
        arg_c_call=["{f_var}"],
    ),
    dict(
        name="c_function_struct_scalar",
        c_arg_decl=["{c_type} *{c_var}"],
        f_c_arg_decl=["{f_type}, intent(OUT) :: {c_var}"],
        f_c_arg_names=["{c_var}"],
        f_c_import=["{f_kind}"],
        return_type="void",  # Convert to function.
        cxx_local_var="result",
        c_post_call=[
            "memcpy((void *) {c_var}, (void *) &{cxx_var}, sizeof({cxx_var}));",
        ],
    ),
    # end function_struct_scalar
    
    # Similar to c_function_native_*
    dict(
        name="c_function_struct_*",
        base="c_function_struct",
        f_c_result_decl=[
            "type(C_PTR) {c_var}",
        ],
        f_c_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name="f_function_struct_*_pointer",
        base="f_function_native_*_pointer",
    ),

    ########################################
    # getter/setter
    # getters are functions.
    #  When passed meta data, converted into a subroutine.
    # setters are first argument to subroutine.

    dict(
        # Base for all getters to avoid calling function.
        name="c_getter",
        mixin=["c_mixin_noargs"],
        call=[
            "// skip call c_getter",
        ],
    ),
    dict(
        # Not actually calling a subroutine.
        # Work is done by arg's setter.
        name="c_setter",
        mixin=["c_mixin_noargs"],
        call=[
            "// skip call c_setter",
        ],
    ),
    dict(
        # Argument to setter.
        name="c_setter_arg",
    ),
    dict(
        name="f_getter",
    ),
    dict(
        name="f_setter",
        arg_c_call=[],
    ),

    dict(
        # c_getter_native_scalar
        # c_getter_native_*
        name="c_getter_native_scalar/*",
        base="c_getter",
        ret=[
            "return {CXX_this}->{field_name};",
        ],
    ),
    dict(
        name="f_setter_native",
        arg_c_call=["{c_var}"],
        # f_setter is intended for the function, this is for an argument.
    ),
    dict(
        # c_setter_native_scalar
        # c_setter_native_*
        name="c_setter_native_scalar/*",
        base="c_setter_arg",
        c_post_call=[
            "{CXX_this}->{field_name} = val;",
        ],
    ),
    dict(
        # Similar to calling a function, but save field pointer instead.
        name="c_getter_native_*_cdesc",
        mixin=["c_getter", "c_mixin_function_cdesc"],
        c_helper="ShroudTypeDefines array_context",
        c_post_call=[
            "{c_var_cdesc}->cxx.addr  = {CXX_this}->{field_name};",
            "{c_var_cdesc}->cxx.idtor = {idtor};",
            "{c_var_cdesc}->addr.base = {CXX_this}->{field_name};",
            "{c_var_cdesc}->type = {sh_type};",
            "{c_var_cdesc}->elem_len = sizeof({cxx_type});",
            "{c_var_cdesc}->rank = {rank};"
            "{c_array_shape}",
            "{c_var_cdesc}->size = {c_array_size};",
        ],
    ),
    #####
    dict(
        # Return meta data to Fortran.
        name="c_getter_string_scalar_cdesc",
        mixin=["c_getter", "c_mixin_out_character_cdesc"],
        c_post_call=[
            "{c_var_cdesc}->addr.base = {CXX_this}->{field_name}.data();",
            "{c_var_cdesc}->type = 0; // SH_CHAR;",
            "{c_var_cdesc}->elem_len = {CXX_this}->{field_name}.size();",
            "{c_var_cdesc}->rank = 0;"
        ],
        return_type="void",  # Convert to function.
    ),
    dict(
        # Create std::string from Fortran meta data.
        name="c_setter_string_scalar_buf",
        base="c_setter_arg",
        mixin=["c_mixin_in_character_buf"],
        c_post_call=[
            "{CXX_this}->{field_name} = std::string({c_var},\t {c_var_len});",
        ],
    ),
    dict(
        # Extract meta data and pass to C.
        name="f_setter_string_scalar_buf",
        mixin=["f_mixin_in_character_buf"],
    ),
    dict(
        # Get meta data from C and allocate CHARACTER.
        name="f_getter_string_scalar_cdesc_allocatable",
        base="f_function_string_scalar_cdesc_allocatable",
    ),
    
    
    ########################################
    # CFI - Further Interoperability with C
    ########################################
    # char arg
    dict(
        # Add allocatable attribute to declaration.
        # f_function_char_scalar_cfi_allocatable
        # f_function_char_*_cfi_allocatable
        name="f_function_char_scalar/*_cfi_allocatable",
        need_wrapper=True,
        arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        arg_c_call=["{f_var}"],  # Pass result as an argument.
    ),
    dict(
        # Add allocatable attribute to declaration.
        # f_function_char_scalar_cfi_allocatable
        # f_function_char_*_cfi_allocatable
        name="f_function_char_scalar/*_cfi_pointer",
        need_wrapper=True,
        arg_decl=[
            "character(len=:), pointer :: {f_var}",
        ],
        arg_c_call=["{f_var}"],  # Pass result as an argument.
    ),
    
    dict(
        # XXX - needs a better name. function/arg
        # Function which return char * or std::string.
        name="c_mixin_function_character",
        iface_header=["ISO_Fortran_binding.h"],
        c_arg_decl=[
            "CFI_cdesc_t *{c_var_cfi}",
        ],
        f_c_arg_decl=[
            "XXX-unused character(len=*), intent({f_intent}) :: {c_var}",
        ],
        f_c_arg_names=["{c_var}"],
        temps=["cfi"],
    ),
    dict(
        # Character argument which use CFI_desc_t.
        name="c_mixin_arg_character_cfi",
        iface_header=["ISO_Fortran_binding.h"],
        cxx_local_var="pointer",
        c_arg_decl=[
            "CFI_cdesc_t *{c_var_cfi}",
        ],
        f_c_arg_decl=[
            "character(len=*), intent({f_intent}) :: {c_var}",
        ],
        f_c_arg_names=["{c_var}"],
        c_pre_call=[
            "char *{cxx_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
        ],
        temps=["cfi"],
    ),
    dict(
        # Native argument which use CFI_desc_t.
        name="c_mixin_arg_native_cfi",
        iface_header=["ISO_Fortran_binding.h"],
        cxx_local_var="pointer",
        c_arg_decl=[
            "CFI_cdesc_t *{c_var_cfi}",
        ],
        f_c_arg_decl=[
            "{f_type}, intent({f_intent}) :: {c_var}{f_assumed_shape}",
        ],
        f_c_module_line="iso_c_binding:{f_kind}",
        f_c_arg_names=["{c_var}"],
#        c_pre_call=[
#            "{c_type} *{cxx_var} = "
#            "{cast_static}{c_type} *{cast1}{c_var_cfi}->base_addr{cast2};",
#        ],
        temps=["cfi", "extents", "lower"],
    ),

    dict(
        # Allocate copy of C pointer (requires +dimension)
        name="c_mixin_native_cfi_allocatable",
        c_post_call=[
            "if ({cxx_var} != {nullptr}) {{+",
            "{c_temp_lower_decl}"
            "{c_temp_extents_decl}"
            "int SH_ret = CFI_allocate({c_var_cfi}, \t{c_temp_lower_use},"
            " \t{c_temp_extents_use}, \t0);",
            "if (SH_ret == CFI_SUCCESS) {{+",
            "{stdlib}memcpy({c_var_cfi}->base_addr, \t{cxx_var}, \t{c_var_cfi}->elem_len);",
#XXX            "{C_memory_dtor_function}({cxx_var});",
            "-}}",
            "-}}",
        ],
    ),
    dict(
        # Convert C pointer to Fortran pointer
        name="c_mixin_native_cfi_pointer",
        c_post_call=[
            "{{+",
            "CFI_CDESC_T({rank}) {c_local_fptr};",
            "CFI_cdesc_t *{c_local_cdesc} = {cast_reinterpret}CFI_cdesc_t *{cast1}&{c_local_fptr}{cast2};",
            "void *{c_local_cptr} = const_cast<{c_type} *>({cxx_var});",
            "{c_temp_extents_decl}"
            "{c_temp_lower_decl}"
            "int {c_local_err} = CFI_establish({c_local_cdesc},\t {c_local_cptr},"
            "\t CFI_attribute_pointer,\t {cfi_type},"
            "\t 0,\t {rank},\t {c_temp_extents_use});",
            "if ({c_local_err} == CFI_SUCCESS) {{+",
            "{c_local_err} = CFI_setpointer(\t{c_var_cfi},\t {c_local_cdesc},\t {c_temp_lower_use});",
            "-}}",
            "-}}",
        ],
        local=["cptr", "fptr", "cdesc", "err"],
    ),
    
    ########################################

    dict(
        # c_in_native_*_cfi
        # c_inout_native_*_cfi
        name="c_in/inout_native_*_cfi",
        mixin=[
            "c_mixin_arg_native_cfi",
        ],
        c_pre_call=[
            "{cxx_type} *{cxx_var} = "
            "{cast_static}{cxx_type} *{cast1}{c_var_cfi}->base_addr{cast2};",
        ],
    ),
    
    ########################################
    dict(
        name="c_in_char_*_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        # Null terminate string.
        c_helper="ShroudStrAlloc ShroudStrFree",
        c_pre_call=[
            "char *{c_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
            "char *{cxx_var} = ShroudStrAlloc(\t"
            "{c_var},\t {c_var_cfi}->elem_len,\t {c_blanknull});",
        ],
        c_post_call=[
            "ShroudStrFree({cxx_var});",
        ],
    ),
    dict(
        name="c_out_char_*_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        c_helper="ShroudStrBlankFill",
        c_post_call=[
            "ShroudStrBlankFill({cxx_var}, {c_var_cfi}->elem_len);"
        ],
    ),
    dict(
        name="c_inout_char_*_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        # Null terminate string.
        c_helper="ShroudStrAlloc ShroudStrCopy ShroudStrFree",
        c_pre_call=[
            "char *{c_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
            "char *{cxx_var} = ShroudStrAlloc(\t"
            "{c_var},\t {c_var_cfi}->elem_len,\t {c_blanknull});",
        ],
        c_post_call=[
            # nsrc=-1 will call strlen({cxx_var})
            "ShroudStrCopy({c_var}, {c_var_cfi}->elem_len,"
            "\t {cxx_var},\t -1);",
            "ShroudStrFree({cxx_var});",
        ],
    ),
    dict(
        # Blank fill result.
        name="c_function_char_scalar_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        lang_c=dict(
            impl_header=["<string.h>"],
        ),
        lang_cxx=dict(
            impl_header=["<cstring>"],
        ),
        cxx_local_var=None,  # replace mixin
        c_pre_call=[],         # replace mixin        
        c_post_call=[
            "char *{c_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
            "{stdlib}memset({c_var}, ' ', {c_var_cfi}->elem_len);",
            "{c_var}[0] = {cxx_var};",
        ],
    ),
    dict(
        # Copy result into caller's buffer.
        name="f_function_char_*_cfi",
        arg_c_call=["{f_var}"],
        need_wrapper=True,
    ),
    dict(
        # Copy result into caller's buffer.
        name="c_function_char_*_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        cxx_local_var="result",
        c_pre_call=[],         # undo mixin
        c_helper="ShroudStrCopy",
        c_post_call=[
            # XXX c_type is undefined
            # nsrc=-1 will call strlen({cxx_var})
            "char *{c_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
            "ShroudStrCopy({c_var}, {c_var_cfi}->elem_len,"
            "\t {cxx_var},\t -1);",
        ],
        return_type="void",  # Convert to function.
    ),
    dict(
        name="c_function_char_*_cfi_allocatable",
        mixin=[
            "c_mixin_function_character",
        ],
        return_type="void",  # Convert to function.
        f_c_arg_names=["{c_var}"],
        f_c_arg_decl=[        # replace mixin
            "character(len=:), intent({f_intent}), allocatable :: {c_var}",
        ],
        cxx_local_var=None,  # replace mixin
        c_pre_call=[],         # replace mixin
        c_post_call=[
            "if ({cxx_var} != {nullptr}) {{+",
            "int SH_ret = CFI_allocate({c_var_cfi}, \t(CFI_index_t *) 0, \t(CFI_index_t *) 0, \tstrlen({cxx_var}));",
            "if (SH_ret == CFI_SUCCESS) {{+",
            "{stdlib}memcpy({c_var_cfi}->base_addr, \t{cxx_var}, \t{c_var_cfi}->elem_len);",
            "-}}",
            "-}}",
        ],
    ),
    dict(
        name="c_function_char_*_cfi_pointer",
        mixin=[
            "c_mixin_function_character",
        ],
        return_type="void",  # Convert to function.
        f_c_arg_names=["{c_var}"],
        f_c_arg_decl=[        # replace mixin
            "character(len=:), intent({f_intent}), pointer :: {c_var}",
        ],
        cxx_local_var=None,  # replace mixin
        c_pre_call=[],         # replace mixin
        c_post_call=[
# CFI_index_t nbar[1] = {3};
#  CFI_CDESC_T(1) c_p;
#  CFI_establish((CFI_cdesc_t* )&c_p, bar, CFI_attribute_pointer, CFI_type_int,
#                nbar[0]*sizeof(int), 1, nbar);
#  CFI_setpointer(f_p, (CFI_cdesc_t *)&c_p, NULL);

            # CFI_index_t nbar[1] = {3};
            "int {c_local_err};",
            "if ({cxx_var} == {nullptr}) {{+",
            "{c_local_err} = CFI_setpointer(\t{c_var_cfi},\t {nullptr},\t {nullptr});",
            "-}} else {{+",
            "CFI_CDESC_T(0) {c_local_fptr};",
            "CFI_cdesc_t *{c_local_cdesc} = {cast_reinterpret}CFI_cdesc_t *{cast1}&{c_local_fptr}{cast2};",
            "void *{c_local_cptr} = {cxx_nonconst_ptr};",
            "size_t {c_local_len} = {stdlib}strlen({cxx_var});",
            "{c_local_err} = CFI_establish({c_local_cdesc},\t {c_local_cptr},"
            "\t CFI_attribute_pointer,\t CFI_type_char,"
            "\t {c_local_len},\t 0,\t {nullptr});",
            "if ({c_local_err} == CFI_SUCCESS) {{+",
            "{c_var_cfi}->elem_len = {c_local_cdesc}->elem_len;",  # set assumed-length
            "{c_local_err} = CFI_setpointer(\t{c_var_cfi},\t {c_local_cdesc},\t {nullptr});",
            "-}}",
            "-}}",            
        ],
        local=["cptr", "fptr", "cdesc", "len", "err"],
    ),
    
    ########################################
    # char **
    dict(
        name='c_in_char_**_cfi',
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        f_c_arg_decl=[
            "character(len=*), intent({f_intent}) :: {c_var}(:)",
        ],
        c_pre_call=[
            "char *{c_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
            "size_t {c_var_len} = {c_var_cfi}->elem_len;",
            "size_t {c_var_size} = {c_var_cfi}->dim[0].extent;",
            "char **{cxx_var} = ShroudStrArrayAlloc("
            "{c_var},\t {c_var_size},\t {c_var_len});",
        ],
        temps=["cfi", "len", "size"],

        c_helper="ShroudStrArrayAlloc ShroudStrArrayFree",
        cxx_local_var="pointer",
        c_post_call=[
            "ShroudStrArrayFree({cxx_var}, {c_var_size});",
        ],
    ),

    ########################################
    # std::string
    dict(
        # c_in_string_scalar_cfi
        # c_in_string_*_cfi
        # c_in_string_&_cfi
        name="c_in_string_scalar/*/&_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        c_helper="ShroudLenTrim",
        cxx_local_var="scalar",   # replace mixin
        c_pre_call=[
            # Get Fortran character pointer and create std::string.
            "char *{c_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
            "size_t {c_local_trim} = ShroudLenTrim({c_var}, {c_var_cfi}->elem_len);",
            "{c_const}std::string {cxx_var}({c_var}, {c_local_trim});",
        ],
        local=["trim"],
    ),
    dict(
        # c_out_string_*_cfi
        # c_out_string_&_cfi
        name="c_out_string_*/&_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        c_helper="ShroudStrCopy",
        cxx_local_var="scalar",
        c_pre_call=[
            "std::string {cxx_var};",
            "char *{c_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
        ],
        c_post_call=[
            "ShroudStrCopy({c_var},"
            "\t {c_var_cfi}->elem_len,"
            "\t {cxx_var}{cxx_member}data(),"
            "\t {cxx_var}{cxx_member}size());"
        ],
    ),
    dict(
        # c_inout_string_*_cfi
        # c_inout_string_&_cfi
        name="c_inout_string_*/&_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        c_helper="ShroudStrCopy",
        cxx_local_var="scalar",
        c_pre_call=[
            "char *{c_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
            "size_t {c_local_trim} = ShroudLenTrim({c_var}, {c_var_cfi}->elem_len);",
            "{c_const}std::string {cxx_var}({c_var}, {c_local_trim});",
        ],
        c_post_call=[
            "ShroudStrCopy({c_var},"
            "\t {c_var_cfi}->elem_len,"
            "\t {cxx_var}{cxx_member}data(),"
            "\t {cxx_var}{cxx_member}size());"
        ],
        local=["trim"],
    ),
    dict(
        # c_function_string_scalar_cfi
        # c_function_string_*_cfi
        # c_function_string_&_cfi
        name="c_function_string_scalar/*/&_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        cxx_local_var=None, # replace mixin
        c_pre_call=[],        # replace mixin
        c_helper="ShroudStrCopy",
        c_post_call=[
            "char *{c_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
            "if ({cxx_var}{cxx_member}empty()) {{+",
            "ShroudStrCopy({c_var}, {c_var_cfi}->elem_len,"
            "\t {nullptr},\t 0);",
            "-}} else {{+",
            "ShroudStrCopy({c_var}, {c_var_cfi}->elem_len,"
            "\t {cxx_var}{cxx_member}data(),"
            "\t {cxx_var}{cxx_member}size());",
            "-}}",
        ],
        return_type="void",  # Convert to function.
    ),
    # std::string * function()
    dict(
        # c_function_string_*_cfi_allocatable
        # c_function_string_&_cfi_allocatable
        name="c_function_string_*/&_cfi_allocatable",
        mixin=[
            "c_mixin_function_character",
        ],
        f_c_arg_decl=[
            "character(len=:), intent({f_intent}), allocatable :: {c_var}",
        ],
        return_type="void",  # Convert to function.
        f_c_arg_names=["{c_var}"],
        lang_c=dict(
            impl_header=["<string.h>"],
        ),
        lang_cxx=dict(
            impl_header=["<cstring>"],
        ),
        c_post_call=[
            "int SH_ret = CFI_allocate({c_var_cfi}, \t(CFI_index_t *) 0, \t(CFI_index_t *) 0, \t{cxx_var}{cxx_member}length());",
            "if (SH_ret == CFI_SUCCESS) {{+",
            "{stdlib}memcpy({c_var_cfi}->base_addr,"
            " \t{cxx_var}{cxx_member}data(),"
            " \t{cxx_var}{cxx_member}length());",
            "-}}",
        ],
    ),
    # XXX - consolidate with c_function_*_cfi_pointer?
    # XXX - via a helper to get address and length of string
    dict(
        name="c_function_string_*_cfi_pointer",
        mixin=[
            "c_mixin_function_character",
        ],
        return_type="void",  # Convert to function.
        f_c_arg_names=["{c_var}"],
        f_c_arg_decl=[        # replace mixin
            "character(len=:), intent({f_intent}), pointer :: {c_var}",
        ],
        cxx_local_var=None,  # replace mixin
        c_pre_call=[],         # replace mixin
        c_post_call=[
            "int {c_local_err};",
            "if ({cxx_var} == {nullptr}) {{+",
            "{c_local_err} = CFI_setpointer(\t{c_var_cfi},\t {nullptr},\t {nullptr});",
            "-}} else {{+",
            "CFI_CDESC_T(0) {c_local_fptr};",
            "CFI_cdesc_t *{c_local_cdesc} = {cast_reinterpret}CFI_cdesc_t *{cast1}&{c_local_fptr}{cast2};",
            "void *{c_local_cptr} = const_cast<char *>({cxx_var}{cxx_member}data());",
            "size_t {c_local_len} = {cxx_var}{cxx_member}length();",
            "{c_local_err} = CFI_establish({c_local_cdesc},\t {c_local_cptr},"
            "\t CFI_attribute_pointer,\t CFI_type_char,"
            "\t {c_local_len},\t 0,\t {nullptr});",
            "if ({c_local_err} == CFI_SUCCESS) {{+",
            "{c_var_cfi}->elem_len = {c_local_cdesc}->elem_len;",  # set assumed-length
            "{c_local_err} = CFI_setpointer(\t{c_var_cfi},\t {c_local_cdesc},\t {nullptr});",
            "-}}",
            "-}}",            
        ],
        local=["cptr", "fptr", "cdesc", "len", "err"],
    ),

    # std::string & function()
    dict(
        name="c_function_string_scalar_cfi_allocatable",
        mixin=[
            "c_mixin_function_character",
        ],
        f_c_arg_names=["{c_var}"],
        f_c_arg_decl=[        # replace mixin
            "character(len=:), intent({f_intent}), allocatable :: {c_var}",
        ],
        return_type="void",  # convert to function
        cxx_local_var=None,  # replace mixin
        c_pre_call=[],         # replace mixin
        c_post_call=[
            "int SH_ret = CFI_allocate({c_var_cfi}, \t(CFI_index_t *) 0, \t(CFI_index_t *) 0, \t{cxx_var}.length());",
            "if (SH_ret == CFI_SUCCESS) {{+",
            "{stdlib}memcpy({c_var_cfi}->base_addr, \t{cxx_var}.data(), \t{c_var_cfi}->elem_len);",
            "-}}",
        ],
        
        destructor_name="new_string",
        destructor=[
            "std::string *cxx_ptr = \treinterpret_cast<std::string *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    
    dict(
        # f_function_string_scalar_cfi
        # f_function_string_*_cfi
        # f_function_string_&_cfi
        name="f_function_string_scalar/*/&_cfi",
        # XXX - avoid calling C directly since the Fortran function
        # is returning an CHARACTER, which CFI can not do.
        # Fortran wrapper passed function result to C which fills it.
        need_wrapper=True,
        arg_c_call=["{f_var}"],
    ),
    # similar to f_char_scalar_allocatable
    dict(
        # f_function_string_scalar_cfi_allocatable
        # f_function_string_*_cfi_allocatable
        # f_function_string_&_cfi_allocatable
        name="f_function_string_scalar/*/&_cfi_allocatable",
        # XXX - avoid calling C directly since the Fortran function
        # is returning an allocatable, which CFI can not do.
        # Fortran wrapper passed function result to C which fills it.
        need_wrapper=True,
        arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        arg_c_call=["{f_var}"],
    ),
    dict(
        # f_function_string_scalar_cfi_pointer
        # f_function_string_*_cfi_pointer
        # f_function_string_&_cfi_pointer
        name="f_function_string_scalar/*/&_cfi_pointer",
        # XXX - avoid calling C directly since the Fortran function
        # is returning an pointer, which CFI can not do.
        # Fortran wrapper passed function result to C which fills it.
        need_wrapper=True,
        arg_decl=[
            "character(len=:), pointer :: {f_var}",
        ],
        arg_c_call=["{f_var}"],
    ),

    ##########
    # Pass a cdesc down to describe the memory and a capsule to hold the
    # C++ array. Copy into Fortran argument.
    # [see also f_out_vector_&_cdesc_allocatable_targ_string_scalar]
    dict(
        name="f_out_string_**_cdesc_copy",
        mixin=["f_mixin_str_array"],
    ),
    dict(
        name="c_out_string_**_cdesc_copy",
        mixin=["c_mixin_out_array_cdesc"],
        c_helper="array_string_out",
        c_pre_call=[
            "std::string *{cxx_var};"
        ],
        c_arg_call=["&{cxx_var}"],
        c_post_call=[
            "{hnamefunc0}(\t{c_var_cdesc},\t {cxx_var}, {c_array_size2});",
        ],

    ),

    dict(
        # std::string **arg+intent(out)+dimension(size)
        # Returning a pointer to a string*. However, this needs additional mapping
        # for the C interface.  Fortran calls the +api(cdesc) variant.
        name="c_out_string_**_copy",
        notimplemented=True,
    ),

    ##########
    # Pass a cdesc down to describe the memory and a capsule to hold the
    # C++ array. Allocate in fortran, fill from C.
    # [see also f_out_vector_&_cdesc_allocatable_targ_string_scalar]
    dict(
        name="f_out_string_**_cdesc_allocatable",
        arg_decl=[
            "character({f_char_len}), intent(out), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["C_LOC"]),
        declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
            "type({F_array_type}) :: {c_var_out}",
        ],
        arg_c_call=["{c_var_out}"],
        post_call=[
            "{c_var_cdesc}%size = {c_var_out}%size;",
            "{c_var_cdesc}%elem_len = {c_var_out}%elem_len;",
            "allocate({f_char_type}{f_var}({c_var_cdesc}%size))",
            "{c_var_cdesc}%cxx%addr = C_LOC({f_var});",
            "{c_var_cdesc}%base_addr = C_LOC({f_var});",
            "call {hnamefunc0}({c_var_cdesc}, {c_var_out})",
        ],
        temps=["cdesc", "out"],
        f_helper="array_string_allocatable array_context",
        c_helper="array_string_allocatable",
    ),
    dict(
        name="c_out_string_**_cdesc_allocatable",
        mixin=["c_mixin_out_array_cdesc"],
        c_helper="array_string_out_len",
        c_pre_call=[
            "std::string *{cxx_var};",
        ],
        c_arg_call=["&{cxx_var}"],
        c_post_call=[
            "{c_var_cdesc}->rank = {rank};"
            "{c_array_shape}",
            "{c_var_cdesc}->size     = {c_array_size};",
            # XXX - assume a sufficiently smart compiler will only use one clause
            #  if c_char_len is a constant.
            "if ({c_char_len} > 0) {{+",
            "{c_var_cdesc}->elem_len = {c_char_len};",
            "-}} else {{+",
            "{c_var_cdesc}->elem_len = {hnamefunc0}({cxx_var}, {c_var_cdesc}->size);",
            "-}}",
            "{c_var_cdesc}->cxx.addr  = {cxx_var};",
            "{c_var_cdesc}->cxx.idtor = 0;",  # XXX - check ownership
        ],
    ),

    dict(
        # std::string **arg+intent(out)+dimension(size)+deref(allocatable)
        # Returning a pointer to a string*. However, this needs additional mapping
        # for the C interface.  Fortran calls the +api(cdesc) variant.
        name="c_out_string_**_allocatable",
        notimplemented=True,
    ),

    ########################################
    # native
    dict(
        name="f_out_native_*_cfi_allocatable",
    ),
    dict(
        # Set Fortran pointer to point to cxx_var
        name="c_out_native_**_cfi_allocatable",
        mixin=[
            "c_mixin_arg_native_cfi",
            "c_mixin_native_cfi_allocatable",
        ],
        f_c_arg_decl=[
            "{f_type}, intent({f_intent}), allocatable :: {c_var}{f_assumed_shape}",
        ],
        c_pre_call=[
            "{c_const}{c_type} * {cxx_var};",
        ],
        c_arg_call=["&{cxx_var}"],
    ),
    dict(
        # Set Fortran pointer to point to cxx_var
        name="c_out_native_**_cfi_pointer",
        mixin=[
            "c_mixin_arg_native_cfi",
            "c_mixin_native_cfi_pointer",
        ],
        f_c_arg_decl=[
            "{f_type}, intent({f_intent}), pointer :: {c_var}{f_assumed_shape}",
        ],

        # set pointer on fortran declaration
        c_pre_call=[
            "{c_const}{c_type} * {cxx_var};",
        ],
        c_arg_call=["&{cxx_var}"],
    ),

    dict(
        # Pass result as an argument to C wrapper.
        name="f_function_native_*_cfi_allocatable",
        arg_decl=[
            "{f_type}, allocatable :: {f_var}{f_assumed_shape}",
        ],
        arg_c_call=["{f_var}"],
    ),
    dict(
        # Convert to subroutine and pass result as an argument.
        # Return an allocated copy of data.
        name="c_function_native_*_cfi_allocatable",
        mixin=[
            "c_mixin_arg_native_cfi",
            "c_mixin_native_cfi_allocatable",  # c_post_call
        ],
        f_c_arg_decl=[
            "{f_type}, intent({f_intent}), allocatable :: {c_var}{f_assumed_shape}",
        ],

        cxx_local_var="result",
        return_type="void",  # Convert to function.
    ),

    dict(
        # Pass result as an argument to C wrapper.
        name="f_function_native_*_cfi_pointer",
        arg_decl=[
            "{f_type}, pointer :: {f_var}{f_assumed_shape}",
        ],
        pre_call=[
            "nullify({f_var})",
        ],
        arg_c_call=["{f_var}"],
    ),
    dict(
        # Convert to subroutine and pass result as an argument.
        # Return Fortran pointer to data.
        name="c_function_native_*_cfi_pointer",
        mixin=[
            "c_mixin_arg_native_cfi",
            "c_mixin_native_cfi_pointer",  # c_post_call
        ],
        f_c_arg_decl=[
            "{f_type}, intent({f_intent}), pointer :: {c_var}{f_assumed_shape}",
        ],

        cxx_local_var="result",
        return_type="void",  # Convert to function.
    ),
    
]
