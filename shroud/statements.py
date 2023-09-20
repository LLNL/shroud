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

from collections import OrderedDict

# The dictionary of c and fortran statements.
fc_dict = OrderedDict() # dictionary of Scope of all expanded fc_statements.

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
    """Lookup statements for C and Fortran wrappers.

    Looked up in the dictionary instead of the tree
    so the name must match exactly.
    """
    name = compute_name(path)
    stmt = fc_dict.get(name, None)
    if stmt is None:
        # XXX - return something so code will get generated
        #  It'll be wrong but acts as a starting place.
        stmt = fc_dict.get("f_mixin_unknown")
        print("Unknown fc statement: %s" % name)
#        raise RuntimeError("Unknown fc statement: %s" % name)
    return stmt
        
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


def compute_return_prefix(arg):
    """Compute how to access variable: dereference, address, as-is"""
    if arg.declarator.is_reference():
        # Convert a return reference into a pointer.
        return "&"
    else:
        return ""


def update_fc_statements_for_language(language):
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

    check_statements(fc_statements)
    update_for_language(fc_statements, language)
    process_mixin(fc_statements, default_stmts, fc_dict)


def check_statements(stmts):
    """Check against a schema

    Checked against the raw input. Before any base or mixin.
    """
    for stmt in stmts:
        # node is a dict.
        if "name" not in stmt:
            raise RuntimeError("Missing name in statements: {}".
                               format(str(node)))
        name = stmt["name"]
        parts = name.split("_")
        if len(parts) < 2:
            raise RuntimeError("Statement name is too short")
        lang = parts[0]
        intent = parts[1]

        if lang not in ["c", "f", "fc", "lua", "py"]:
            raise RuntimeError("Statement does not start with a language code: %s" % name)

        if intent not in [
                "in", "out", "inout", "mixin",
                "function", "subroutine",
                "getter", "setter",
                "ctor", "dtor",
                "base", "descr",
                "defaulttmp", "defaultstruct", "XXXin", "test", "shared",
                "in/out/inout", "out/inout", "in/inout", "function/getter",
        ]:
            raise RuntimeError("Statement does not contain a valid intent: %s" % name)


def post_mixin_check_statement(name, stmt):
    """check for consistency.
    Called after mixin are applied.
    This makes it easer to a group to change one of
    c_arg_decl, i_arg_decl, i_arg_names.
    """
    parts = name.split("_")
    lang = parts[0]
    intent = parts[1]

    if lang in ["c", "fc"] and intent != "mixin":
        c_arg_decl = stmt.get("c_arg_decl", None)
        i_arg_decl = stmt.get("i_arg_decl", None)
        i_arg_names = stmt.get("i_arg_names", None)
        if (c_arg_decl is not None or
            i_arg_decl is not None or
            i_arg_names is not None):
            err = False
            for field in ["c_arg_decl", "i_arg_decl", "i_arg_names"]:
                fvalue = stmt.get(field)
                if fvalue is None:
                    err = True
                    print("Missing", field, "in", name)
                elif not isinstance(fvalue, list):
                    err = True
                    print(field, "must be a list in", name)
            if (c_arg_decl is None or
                i_arg_decl is None or
                i_arg_names is None):
                print("c_arg_decl, i_arg_decl and i_arg_names must all exist")
                err = True
            if err:
                raise RuntimeError("Error with fields")
            length = len(c_arg_decl)
            if any(len(lst) != length for lst in [i_arg_decl, i_arg_names]):
                raise RuntimeError(
                    "c_arg_decl, i_arg_decl and i_arg_names "
                    "must all be same length in {}".format(name))

##-    if lang in ["f", "fc"]:
##-        # Default f_arg_name is often ok.
##-        f_arg_name = stmt.get("f_arg_name", None)
##-        f_arg_decl = stmt.get("f_arg_decl", None)
##-        if f_arg_name is not None or f_arg_decl is not None:
##-            err = False
##-            for field in ["f_arg_name", "f_arg_decl"]:
##-                fvalue = stmt.get(field)
##-                if fvalue is None:
##-                    err = True
##-                    print("Missing", field, "in", name)
##-                elif not isinstance(fvalue, list):
##-                    err = True
##-                    print(field, "must be a list in", name)
##-            if (f_arg_name is None or
##-                f_arg_decl is None):
##-                print("f_arg_name and f_arg_decl must both exist")
##-                err = True
##-            if err:
##-                raise RuntimeError("Error with fields")
##-            if len(f_arg_name) != len(f_arg_decl):
##-                raise RuntimeError(
##-                    "f_arg_name and f_arg_decl "
##-                    "must all be same length in {}".format(name))
            

def process_mixin(stmts, defaults, stmtdict):
    """Return a dictionary of all statements
    names and aliases will be expanded (ex in/out/inout)
    Each dictionary will have a unique name.

    Add into dictionary.
    Add as aliases
    Add mixin into dictionary

    alias=[
        "c_function_native_*_allocatable/raw",
        "c_function_native_*/&/**_pointer",
    ],
    """
    # Apply base and mixin
    # This allows mixins to propagate
    # i.e. you can mixin a group which itself has a mixin.
    # Save by name permutations into mixins  (in/out/inout)
    mixins = OrderedDict()
    aliases = []
    for stmt in stmts:
        name = stmt["name"]
#        print("XXXXX", name)
        node = {}
        parts = name.split("_")
        if parts[1] == "mixin":
            if "base" in stmt:
                print("XXXX - mixin should not have 'base' field: ", name)
            if "alias" in stmt:
                print("XXXX - mixin should not have 'alias' field: ", name)
        if "mixin" in stmt:
            if "base" in stmt:
                print("XXXX - Groups with mixin cannot have a 'base' field ", name)
            for mixin in stmt["mixin"]:
                ### compute mixin permutations
#                parts = mixin.split("_")
#                if parts[1] != "mixin":
#                    print("XXXX - mixin must have intent 'mixin': ", name)
                if mixin not in mixins:
                    raise RuntimeError("Mixin {} not found for {}".format(mixin, name))
#                print("M    ", mixin)
                node.update(mixins[mixin])
        node.update(stmt)
        post_mixin_check_statement(name, node)
        node["orig"] = name
        out = compute_all_permutations(name)
        firstname = "_".join(out[0])
        if len(out) == 1:
            if name in mixins:
                raise RuntimeError("process_mixin: key already exists {}".format(name))
            node["name"] = name
            mixins[name] = node
        else:
            lparts = {}  # count language parts
            for part in out:
                aname = "_".join(part)
#                print("X    ", aname)
                anode = node.copy()
                anode["name"] = aname
                if aname in mixins:
                    raise RuntimeError("process_mixin: key already exists {}".format(aname))
                mixins[aname] = anode
                lparts[part[0]] = True
            # Sanity check. Otherwise defaults[lang] would be wrong.
            if len(lparts) > 1:
                raise RuntimeError("Only one language per group")

        if "alias" in stmt:
            aliases.append((firstname, stmt["alias"]))

    # Apply defaults.
    for stmt in mixins.values():
        name = stmt["name"]
        parts = name.split("_",2)
        lang = parts[0]
        intent = parts[1]
        if "base" in stmt:
            node = util.Scope(stmtdict[stmt["base"]])
        else:
            node = util.Scope(defaults[lang])
        node.update(stmt)
        node.intent = intent
        stmtdict[name] = node

    # Install with alias name.
    for name, aliases in aliases:
        node = stmtdict[name]
        for alias in aliases:
#            print("AAAA ", alias)
            aout = compute_all_permutations(alias)
            for apart in aout:
                aname = "_".join(apart)
                anode = util.Scope(node)
                if aname in stmtdict:
                    raise RuntimeError("process_mixin: alias already exists {}".format(aname))
                anode.name = aname
                anode.intent = apart[1]
                stmtdict[aname] = anode
#                print("A    ", aname)

    
def update_for_language(stmts, lang):
    """
    Move language specific entries to current language.
    lang is from YAML file, c or cxx.

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


def compute_all_permutations(key):
    """Expand parts which have multiple values

    Ex: parts = 
      [['c'], ['in', 'out', 'inout'], ['native'], ['*'], ['cfi']]
    Three entries will be returned:
      ['c', 'in', 'native', '*', 'cfi']
      ['c', 'out', 'native', '*', 'cfi']
      ['c', 'inout', 'native', '*', 'cfi']
    """
    steps = key.split("_")
    substeps = []
    for part in steps:
        subparts = part.split("/")
        substeps.append(subparts)

    expanded = []
    compute_stmt_permutations(expanded, substeps)
    return expanded


def compute_stmt_permutations(out, parts):
    """Recursively expand permutations

    ex: f_function_string_scalar/*/&_cfi_copy

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


def add_statement_to_tree(tree, node):
    """Add node to tree.

    Note: mixin are added first, then entries from node.

    Parameters
    ----------
    tree : dict
        The accumulated tree.
    node : dict
        A 'statements' dict from fc_statement to add.
    """
    steps = node["name"].split("_")
    step = tree
    label = []
    for part in steps:
        step = step.setdefault(part, {})
        label.append(part)
        step["_key"] = "_".join(label)
    step['_node'] = node

        
def update_stmt_tree(stmts):
    """Return tree by adding stmts.  Each key in stmts is split by
    underscore then inserted into tree to form nested dictionaries to
    the values from stmts.  The end key is named _node, since it is
    impossible to have an intermediate element with that name (since
    they're split on underscore).

    Add "_key" to tree to aid debugging.

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
    """
    tree = {}
    for node in stmts.values():
        add_statement_to_tree(tree, node)
    return tree


def write_cf_tree(fp):
    """Write out statements tree.

    Parameters
    ----------
    fp : file
    """
    tree = update_stmt_tree(fc_dict)
    lines = []
    print_tree_index(tree, lines)
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
        origname = tree["_node"]["orig"]
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
            if key not in value:
                print("XXX key not in value", key, value.name)
            if value[key]:
                all[key] = value[key]
        if root != "f":
            for key in ["lang_c", "lang_cxx"]:
                val = value.get(key, None)
                if val:
                    all[key] = val
#        for key in ["orig"]:
#            val = value.get(key, None)
#            if val:
#                all[key] = val
        complete[name] = all
    yaml.safe_dump(complete, fp)
            

# C Statements.
#  intent      - Set from name.
#  c_arg_call  - List of arguments passed to C/C++ library function.
#
#  c_arg_decl  - Add C declaration to C wrapper.
#                Empty list is no arguments, None is default argument.
#  c_call       - code to call the function.
#                 Ex. Will be empty for getter and setter.
#  i_arg_decl - Add Fortran declaration to Fortran wrapper interface block.
#                Empty list is no arguments, None is default argument.
#  i_arg_names - Empty list is no arguments
#  i_result_decl - Declaration for function result.
#                  Can be an empty list to override default.
#  i_module    - Add module info to interface block.
CStmts = util.Scope(
    None,
    name="c_default",
    intent=None,
    iface_header=[],
    impl_header=[],
    c_helper="",
    cxx_local_var=None,
    c_arg_call=[],
    c_pre_call=[],
    c_call=[],
    c_post_call=[],
    c_final=[],      # tested in strings.yaml, part of ownership
    c_return=[],
    c_return_type=None,
    c_temps=None,
    c_local=None,

    destructor_name=None,
    destructor=[],
    owner="library",

    c_arg_decl=None,
    i_arg_names=None,
    i_arg_decl=None,

    i_result_decl=None,
    i_result_var=None,
    i_module=None,
    i_module_line=None,
    i_import=None,

    notimplemented=False,
)

# Fortran Statements.
FStmts = util.Scope(
    None,
    name="f_default",
    intent=None,
    c_helper="",
    c_result_var=None,
    f_helper="",
    f_module=None,
    f_module_line=None,
    f_import=None,
    f_need_wrapper=False,
    f_arg_name=None,
    f_arg_decl=None,
    f_arg_call=None,
    f_declare=[],
    f_pre_call=[],
    f_call=[],
    f_post_call=[],
    f_result=None,  # name of result variable
    f_temps=None,
    f_local=None,
)

# Fortran/C Statements - both sets of defaults.
FCStmts = util.Scope(None)
FCStmts.update(CStmts._to_dict())
FCStmts.update(FStmts._to_dict())

# Define class for nodes in tree based on their first entry.
# c_native_*_in uses 'c'.
default_stmts = dict(
    c=CStmts,
    f=FCStmts,
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
        i_arg_decl=[],
        i_arg_names=[],
    ),
    dict(
        name="f_subroutine_void_scalar_capptr",
        mixin=[
            "c_mixin_noargs",
        ],
        alias=[
            "c_subroutine_void_scalar",
            "c_subroutine_void_scalar_capptr",
        ],
    ),
    dict(
        name="f_subroutine",
        mixin=[
            "c_mixin_noargs",
        ],
        alias=[
            "c_subroutine",
            "f_subroutine_void_scalar",
        ],
        f_arg_call=[],
    ),

    dict(
        name="c_function",
        alias=[
            "c_function_void_*",
        ],
    ),
    dict(
        name="f_function",
    ),
    dict(
        name="f_function_native_scalar",
        alias=[
            "c_function_native_scalar",
        ],
    ),

    ########## mixin ##########
    dict(
        # Return a C pointer directly.
        name="f_mixin_function_ptr",
        f_module=dict(iso_c_binding=["C_PTR"]),
        f_arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
        i_result_decl=[
            "type(C_PTR) {c_var}",
        ],
        i_module=dict(iso_c_binding=["C_PTR"]),
        
    ),
    dict(
        # Return a C pointer as a type(C_PTR)
        name="f_mixin_function_c-ptr",
        f_module=dict(iso_c_binding=["C_PTR", "c_f_pointer"]),
        f_arg_decl=[
            "{f_type}, pointer :: {f_var}",
        ],
        f_declare=[
            "type(C_PTR) :: {c_local_ptr}",
        ],
        f_call=[
            "{c_local_ptr} = {F_C_call}({F_arg_c_call})",
        ],
        f_post_call=[
            "call c_f_pointer({c_local_ptr}, {F_result})",
        ],
        f_local=["ptr"],

        i_result_decl=[
            "type(C_PTR) {c_var}",
        ],
        i_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        # Pass array_type as argument to contain the function result.
        name="f_mixin_function_cdesc",
        f_helper="array_context",
        f_declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
        ],
        f_arg_call=["{c_var_cdesc}"],
        f_temps=["cdesc"],
        f_need_wrapper=True,

        c_arg_decl=[
            "{C_array_type} *{c_var_cdesc}",
        ],
        i_arg_decl=[
            "type({F_array_type}), intent(OUT) :: {c_var}",
        ],
        i_arg_names=["{c_var}"],
        i_import=["{F_array_type}"],
        c_return_type="void",  # Convert to function.
        c_temps=["cdesc"],
    ),

    dict(
        # Allocate Fortran CHARACTER scalar, then fill from cdesc.
        name="f_mixin_char_cdesc_allocate",
        c_helper="ShroudStrToArray",
        f_helper="copy_string array_context",
        f_arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        f_post_call=[
            "allocate(character(len={c_var_cdesc}%elem_len):: {f_var})",
            "call {fhelper_copy_string}(\t{c_var_cdesc},\t {f_var},\t {c_var_cdesc}%elem_len)",
        ],
    ),
    
    dict(
        # Add function result to cdesc. Used with pointer and allocatable
        # c_temp cdesc already added
        # assumes mixin=["f_mixin_function_cdesc"],
        name="c_mixin_native_cdesc_fill-cdesc",
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
        # Used with both deref allocatable and pointer.
        # assumes mixin=["f_mixin_function_cdesc"],
        name="c_mixin_function_char_*_cdesc",
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
        # Pass function result as a capsule argument from Fortran to C.
        name="f_mixin_function_shadow_capsule",
        f_arg_decl=[
            "{f_type} :: {f_var}",
        ],
        f_arg_call=[
            "{f_var}%{F_derived_member}",
        ],
        f_need_wrapper=True,
    ),
    dict(
        # Pass function result as a capsule field of shadow class from Fortran to C.
        name="f_mixin_function_shadow_capptr",
        f_arg_decl=[
            "{f_type} :: {f_var}",
            "type(C_PTR) :: {F_result_ptr}",
        ],
        f_arg_call=[
            "{f_var}%{F_derived_member}",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
        c_result_var="{F_result_ptr}",
        f_need_wrapper=True,

        i_result_var="{F_result_ptr}",
        i_result_decl=[
            "type(C_PTR) :: {F_result_ptr}",
        ],
        i_module=dict(iso_c_binding=["C_PTR"]),
    ),

    # Convert function result to character argument
    dict(
        name="f_mixin_function_buf_character-arg",
        f_result="subroutine",
        f_arg_name=["{F_string_result_as_arg}"],
        f_arg_decl=[
            "character(*), intent(OUT) :: {F_string_result_as_arg}",
        ],
        f_temps=["len"],
        f_declare=[
            "integer(C_INT) {c_var_len}",
        ],
        f_pre_call=[
            "{c_var_len} = len({F_string_result_as_arg}, kind=C_INT)",
        ],
        f_arg_call=[
            "{F_string_result_as_arg}",
            "{c_var_len}",
        ],

        # fc_mixin_function-to-subroutine
        c_return_type = "void",

        c_arg_decl=[
            "char *{F_string_result_as_arg}",
            "int n{F_string_result_as_arg}",
        ],
        i_arg_names=["{F_string_result_as_arg}", "n{F_string_result_as_arg}"],
        i_arg_decl=[
            "character(kind=C_CHAR), intent(OUT) :: {F_string_result_as_arg}(*)",
            "integer(C_INT), value, intent(IN) :: n{F_string_result_as_arg}",
        ],
        i_module_line="iso_c_binding:C_CHAR,C_INT",
    ),
    
    # Convert function result to character argument
    dict(
        name="f_mixin_function_cfi_character-arg",
        f_result = "subroutine",
        f_arg_name=["{F_string_result_as_arg}"],
        f_arg_decl=[
            "character(len=*), intent(OUT) :: {F_string_result_as_arg}",
        ],
        f_arg_call=["{F_string_result_as_arg}"],
        i_arg_names=["{F_string_result_as_arg}"],
        i_arg_decl=[
            "character(len=*), intent(OUT) :: {F_string_result_as_arg}",
        ],
#        c_arg_decl  XXX - consistency check wants this set
    ),

    ### destructors
    # Each destructor must have a unique name.
    dict(
        name="c_mixin_destructor_new-string",
        destructor_name="new_string",
        destructor=[
            "std::string *cxx_ptr = \treinterpret_cast<std::string *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    
    dict(
        name="c_mixin_destructor_new-vector",
        destructor_name="std_vector_{cxx_T}",
        destructor=[
            "std::vector<{cxx_T}> *cxx_ptr ="
            " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    
    dict(
        # Default returned by lookup_fc_stmts when group is not found.
        name="f_mixin_unknown",
    ),

    ##########
    # array
    dict(
        # Pass argument and size to C.
        name="f_mixin_in_array_buf",
        f_arg_call=["{f_var}", "size({f_var}, kind=C_SIZE_T)"],
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        f_need_wrapper=True,

        c_arg_decl=[
            "{cxx_type} *{c_var}",   # XXX c_type
            "size_t {c_var_size}",
        ],
        i_arg_names=["{c_var}", "{c_var_size}"],
        i_arg_decl=[
            "{f_type}, intent(IN) :: {c_var}(*)",
            "integer(C_SIZE_T), intent(IN), value :: {c_var_size}",
        ],
        i_module_line="iso_c_binding:{f_kind},C_SIZE_T",
        c_temps=["size"],
    ),
    dict(
        # Pass argument, len and size to C.
        name="f_mixin_in_2d_array_buf",
        f_arg_decl=[
            "{f_type}, intent({f_intent}) :: {f_var}(:,:)",
        ],
        f_arg_call=["{f_var}",
                    "size({f_var}, 1, kind=C_SIZE_T)",
                    "size({f_var}, 2, kind=C_SIZE_T)"],
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        f_need_wrapper=True,

        c_arg_decl=[
            "{cxx_type} *{c_var}",   # XXX c_type
            "size_t {c_var_len}",
            "size_t {c_var_size}",
        ],
        i_arg_names=["{c_var}", "{c_var_len}", "{c_var_size}"],
        i_arg_decl=[
            "{f_type}, intent(IN) :: {c_var}(*)",
            "integer(C_SIZE_T), intent(IN), value :: {c_var_len}",
            "integer(C_SIZE_T), intent(IN), value :: {c_var_size}",
        ],
        i_module_line="iso_c_binding:{f_kind},C_SIZE_T",
        c_temps=["len", "size"],
    ),

    dict(
        # Pass argument and size to C.
        # Pass array_type to C which will fill it in.
        name="f_mixin_inout_array_cdesc",
        f_helper="array_context",
        f_declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
        ],
        f_arg_call=["{f_var}", "size({f_var}, kind=C_SIZE_T)", "{c_var_cdesc}"],
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        f_temps=["cdesc"],
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
        i_arg_names=["{c_var}", "{c_var_size}", "{c_var_cdesc}"],
        i_arg_decl=[
            "{f_type}, intent(IN) :: {c_var}(*)",
            "integer(C_SIZE_T), intent(IN), value :: {c_var_size}",
            "type({F_array_type}), intent(OUT) :: {c_var_cdesc}",
        ],
        i_import=["{F_array_type}"],
        i_module_line="iso_c_binding:{f_kind},C_SIZE_T",
        c_temps=["size", "cdesc"],
    ),

    dict(
        # Pass array_type to C which will fill it in.
        name="f_mixin_out_array_cdesc",
        f_helper="array_context",
        f_declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
        ],
        f_arg_call=["{c_var_cdesc}"],
        f_temps=["cdesc"],
    ),
    dict(
        # Pass array_type to C which will fill it in.
        name="c_mixin_out_array_cdesc",
        c_helper="array_context",
        c_arg_decl=[
            "{C_array_type} *{c_var_cdesc}",
        ],
        i_arg_names=["{c_var_cdesc}"],
        i_arg_decl=[
            "type({F_array_type}), intent(OUT) :: {c_var_cdesc}",
        ],
        i_import=["{F_array_type}"],
        c_temps=["cdesc"],
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
        i_arg_names=["{c_var_cdesc}", "{c_var_out}"],
        i_arg_decl=[
            "type({F_array_type}), intent(OUT) :: {c_var_cdesc}",
            "type({F_array_type}), intent(OUT) :: {c_var_out}",
        ],
        i_import=["{F_array_type}"],
        c_temps=["cdesc", "out"],
    ),

    dict(
        # Pass argument, size and len to C.
        name="f_mixin_in_string_array_buf",
        f_arg_call=[
            "{f_var}",
            "size({f_var}, kind=C_SIZE_T)",
            "len({f_var}, kind=C_INT)"
        ],
        f_module=dict(iso_c_binding=["C_SIZE_T", "C_INT"]),
        f_need_wrapper=True,

        c_arg_decl=[
            "const char *{c_var}",   # XXX c_type
            "size_t {c_var_size}",
            "int {c_var_len}",
        ],
        i_arg_names=["{c_var}", "{c_var_size}", "{c_var_len}"],
        i_arg_decl=[
            "character(kind=C_CHAR), intent(IN) :: {c_var}(*)",
            "integer(C_SIZE_T), intent(IN), value :: {c_var_size}",
            "integer(C_INT), intent(IN), value :: {c_var_len}",
        ],
        i_module_line="iso_c_binding:C_CHAR,C_SIZE_T,C_INT",
        c_temps=["size", "len"],
    ),

    
    ##########
    # Return CHARACTER address and length to Fortran via a cdesc.
    dict(
        name="c_mixin_out_character_cdesc",
        c_arg_decl=[
            "{C_array_type} *{c_var_cdesc}",
        ],
        i_arg_decl=[
            "type({F_array_type}), intent(OUT) :: {c_var}",
        ],
        i_arg_names=["{c_var}"],
        i_import=["{F_array_type}"],
#        c_return_type="void",  # Only difference from c_mixin_function_buf
        c_temps=["cdesc"],
    ),

    # Pass CHARACTER and LEN to C wrapper.
    dict(
        name="f_mixin_in_character_buf",
        # Do not use arg_decl here since it does not understand +len(30) on functions.

        f_temps=["len"],
        f_declare=[
            "integer(C_INT) {c_var_len}",
        ],
        f_pre_call=[
            "{c_var_len} = len({f_var}, kind=C_INT)",
        ],
        f_arg_call=["{f_var}", "{c_var_len}"],

        # XXX - statements.yaml getNameErrorPattern pgi reports an error
        # Argument number 2 to c_get_name_error_pattern_bufferify: kind mismatch 
        # By breaking it out as an explicit assign, the error goes away.
        # Only difference from other uses is setting
        # function attribute +len(get_name_length())
#        f_arg_call=[
#            "{f_var}",
#            "len({f_var}, kind=C_INT)",
#        ],
        f_module=dict(iso_c_binding=["C_INT"]),
        f_need_wrapper=True,
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
        i_arg_names=["{c_var}", "{c_var_len}"],
        i_arg_decl=[
            "character(kind=C_CHAR), intent({f_intent}) :: {c_var}(*)",
            "integer(C_INT), value, intent(IN) :: {c_var_len}",
        ],
        i_module=dict(iso_c_binding=["C_CHAR", "C_INT"]),
        c_temps=["len"],
    ),

    ##########
    # default
    dict(  # f_default
        name="f_defaulttmp",
        alias=[
            "f_in_vector_&_cdesc_targ_native_scalar",
        ],
    ),

    dict(  # c_default
        name="c_defaulttmp",
        alias=[
            "c_out_native_&*",
        ],
    ),

    dict(
        name="f_defaulttmp_native_scalar",
        alias=[
            "f_function_native_*_scalar",

            "f_in_native_scalar",
            "c_in_native_scalar",

            "f_in_native_*",
            "c_in_native_*",

            "f_in_native_&",
            "c_in_native_&",

            "f_out_char_*",
            "c_out_char_*",

            "f_out_native_&",
            "c_out_native_&",

            "f_out_native_*&_pointer",
            "c_out_native_*&_pointer",

            "f_inout_native_*",
            "c_inout_native_*",

            "f_inout_native_&",
            "c_inout_native_&",
            
            "f_out_native_**_allocatable",
            "c_out_native_**_allocatable",

            "f_out_native_**_pointer",
            "c_out_native_**_pointer",

            "f_out_native_***",
            "c_out_native_***",
            
            "f_in_char_*",
            "c_in_char_*",

            "f_inout_char_*",
            "c_inout_char_*",
            
            "f_in_void_scalar",
            "c_in_void_scalar",

            "f_out_void_*&",
            "c_out_void_*&",

            "f_in_char_*_capi",
            "c_in_char_*_capi",

            
            "f_in_unknown_scalar",
            "c_in_unknown_scalar",
        ],
    ),
    
    ##########
    # bool
    dict(
        name="f_mixin_local-logical-var",
        f_temps=["cxx"],
        f_declare=[
            "logical(C_BOOL) :: {c_var_cxx}",
        ],
        f_arg_call=[
            "{c_var_cxx}",
        ],
    ),
    dict(
        name="f_in_bool_scalar",
        mixin=["f_mixin_local-logical-var"],
        alias=[
            "c_in_bool_scalar",
        ],
        f_pre_call=["{c_var_cxx} = {f_var}  ! coerce to C_BOOL"],
    ),
    dict(
        name="f_out_bool_*",
        mixin=["f_mixin_local-logical-var"],
        alias=[
            "c_out_bool_*",
        ],
        f_post_call=["{f_var} = {c_var_cxx}  ! coerce to logical"],
    ),
    dict(
        name="f_inout_bool_*",
        mixin=["f_mixin_local-logical-var"],
        alias=[
            "c_inout_bool_*",
        ],
        f_pre_call=["{c_var_cxx} = {f_var}  ! coerce to C_BOOL"],
        f_post_call=["{f_var} = {c_var_cxx}  ! coerce to logical"],
    ),
    dict(
        name="f_function_bool_scalar",
        alias=[
            "c_function_bool_scalar",
        ],
        # The wrapper is needed to convert bool to logical
        f_need_wrapper=True
    ),

    ##########
    # native

    dict(
        # Make argument a Fortran pointer
        name="f_mixin_out_native_cdesc_pointer",
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        f_arg_decl=[
            "{f_type}, intent({f_intent}), pointer :: {f_var}{f_assumed_shape}",
        ],
        f_post_call=[
            "call c_f_pointer({c_var_cdesc}%base_addr, {f_var}{f_array_shape})",
        ],
    ),
    dict(
        # Make result a Fortran pointer
        name="f_mixin_function_native_cdesc_pointer",
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        f_arg_decl=[
            "{f_type}, pointer :: {f_var}{f_assumed_shape}",
        ],
        f_post_call=[
            "call c_f_pointer({c_var_cdesc}%base_addr, {F_result}{f_array_shape})",
        ],
    ),

    dict(
        name="f_out_native_*",
        alias=[
            "c_out_native_*",
        ],
        f_arg_decl=[
            "{f_type}, intent({f_intent}) :: {f_var}{f_assumed_shape}",
        ],
    ),
    
    dict(
        # Any array of pointers.  Assumed to be non-contiguous memory.
        # All Fortran can do is treat as a type(C_PTR).
        name="f_in_native_**",
        alias=[
            "c_in_native_**",
        ],
        c_arg_decl=[
            "{cxx_type} **{cxx_var}",
        ],
        i_arg_decl=[
            "type(C_PTR), intent(IN), value :: {c_var}",
        ],
        i_arg_names=["{c_var}"],
        i_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        # double **count _intent(out)+dimension(ncount)
        name="c_out_native_**_cdesc",
        mixin=[
            "c_mixin_out_array_cdesc",
            "c_mixin_native_cdesc_fill-cdesc",
        ],
        c_pre_call=[
            "{c_const}{cxx_type} *{cxx_var};",
        ],
        c_arg_call=["&{cxx_var}"],
    ),
    dict(
        name="f_out_native_*&_cdesc",
        mixin=[
            "f_mixin_out_array_cdesc",
            "f_mixin_out_native_cdesc_pointer",
            "c_out_native_**_cdesc",
        ],
        alias=[
            "f_out_native_*&_cdesc_pointer",
            "c_out_native_*&_cdesc_pointer",
        ],
        c_arg_call=["{cxx_var}"],
    ),
    dict(
        # deref(allocatable)
        # A C function with a 'int **' argument associates it
        # with a Fortran pointer.
        # f_out_native_*&_cdesc_allocatable
        name="f_out_native_*&_cdesc_allocatable",
        mixin=["f_mixin_out_array_cdesc"],
        c_helper="copy_array",
        f_helper="copy_array",
#XXX        f_helper="copy_array_{c_type}",
        f_arg_decl=[
            "{f_type}, intent({f_intent}), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        f_post_call=[
            # intent(out) ensure that it is already deallocated.
            "allocate({f_var}{f_array_allocate})",
            "call {fhelper_copy_array}(\t{c_var_cdesc},\t C_LOC({f_var}),\t {c_var_cdesc}%size)"#size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        # deref(allocatable)
        # A C function with a 'int **' argument associates it
        # with a Fortran pointer.
#TTT        name="f_out_native_**/*&_cdesc_allocatable",
        name="f_out_native_**_cdesc_allocatable",
        mixin=[
            "f_mixin_out_array_cdesc",
            "c_out_native_**_cdesc",
        ],
        alias=[
            "c_out_native_**_cdesc_allocatable",
        ],
        c_helper="copy_array ShroudTypeDefines array_context",
        f_helper="copy_array",
#XXX        f_helper="copy_array_{c_type}",
        f_arg_decl=[
            "{f_type}, intent({f_intent}), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        f_post_call=[
            # intent(out) ensure that it is already deallocated.
            "allocate({f_var}{f_array_allocate})",
            "call {fhelper_copy_array}(\t{c_var_cdesc},\t C_LOC({f_var}),\t {c_var_cdesc}%size)"#size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        # deref(pointer)
        # A C function with a 'int **' argument associates it
        # with a Fortran pointer.
        name="f_out_native_**_cdesc_pointer",
        mixin=[
            "f_mixin_out_array_cdesc",
            "f_mixin_out_native_cdesc_pointer",
            "c_out_native_**_cdesc",
        ],
        alias=[
            "c_out_native_**_cdesc_pointer",
        ],
    ),
    dict(
        # Make argument type(C_PTR) from 'int ** +intent(out)+deref(raw)'
        name="f_out_native_**_raw",
        alias=[
            "c_out_native_**_raw",
        ],
        f_arg_decl=[
            "type(C_PTR), intent({f_intent}) :: {f_var}",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
    ),

    # Used with intent IN, INOUT, and OUT.
    dict(
        name="f_in/out/inout_native_*_cdesc",
        mixin=[
            "f_mixin_out_array_cdesc",
            "c_mixin_out_array_cdesc",
        ],
        alias=[
            "c_in_native_*_cdesc",
            "c_out_native_*_cdesc",
            "c_inout_native_*_cdesc",

            "c_in_void_*_cdesc",
            "c_out_void_*_cdesc",
            "c_inout_void_*_cdesc",
            "f_in_void_*_cdesc",
            "f_out_void_*_cdesc",
            "f_inout_void_*_cdesc",
        ],

        # TARGET required for argument to C_LOC.
        f_arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_helper="ShroudTypeDefines array_context",
        f_module=dict(iso_c_binding=["C_LOC"]),
        f_pre_call=[
            "{c_var_cdesc}%base_addr = C_LOC({f_var})",
            "{c_var_cdesc}%type = {sh_type}",
            "! {c_var_cdesc}%elem_len = C_SIZEOF()",
#            "{c_var_cdesc}%size = size({f_var})",
            "{c_var_cdesc}%size = {size}",
            # Do not set shape for scalar via f_cdesc_shape
            "{c_var_cdesc}%rank = {rank}{f_cdesc_shape}",
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

    ########################################
    ##### hidden
    # Hidden argument will not be added for Fortran
    # or C buffer wrapper. Instead it is a local variable
    # in the C wrapper and passed to library function.
    dict(
        # f_out_native_*_hidden
        # f_inout_native_*_hidden
        name="f_out/inout_native_*_hidden",
        alias=[
            "c_out/inout_native_*_hidden",
        ],
        c_pre_call=[
            "{cxx_type} {cxx_var};",
        ],
        c_arg_call=["&{cxx_var}"],
    ),
    dict(
        # f_out_native_&_hidden
        # f_inout_native_&_hidden
        name="f_out/inout_native_&_hidden",
        alias=[
            "c_out/inout_native_&_hidden",
        ],
        c_pre_call=[
            "{cxx_type} {cxx_var};",
        ],
        c_arg_call=["{cxx_var}"],
    ),
    
    ########################################
    # void *
    dict(
        name="f_in_void_*",
        alias=[
            "c_in_void_*",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
        f_arg_decl=[
            "type(C_PTR), intent(IN) :: {f_var}",
        ],
    ),
    dict(
        # return a type(C_PTR)
        name="f_function_void_*",
        f_module=dict(iso_c_binding=["C_PTR"]),
        f_arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),

    ########################################
    # void **
    dict(
        # Treat as an assumed length array in Fortran interface.
        # f_in_void_**
        # f_out_void_**
        # f_inout_void_**
        # c_in_void_**
        # c_out_void_**
        # c_inout_void_**
        name="f_in/out/inout_void_**",
        alias=[
            "c_in/out/inout_void_**",
            "f_in_void_**_cfi",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
        f_arg_decl=[
            "type(C_PTR), intent({f_intent}) :: {f_var}{f_assumed_shape}",
        ],
        c_arg_decl=[
            "void **{c_var}",
        ],
        i_arg_decl=[
            "type(C_PTR), intent({f_intent}) :: {c_var}{i_dimension}",
        ],
        i_arg_names=["{c_var}"],
        i_module=dict(iso_c_binding=["C_PTR"]),
    ),
    
    dict(
        # Works with deref allocatable and pointer.
        # c_function_native_*
        # c_function_native_&
        name="c_function_native_*/&",
        # f_mixin_function_ptr
        alias=[
            "c_function_native_**_pointer",
        ],
        i_result_decl=[
            "type(C_PTR) {c_var}",
        ],
        i_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name="c_function_native_*_scalar",
        i_result_decl=[
            "{f_type} :: {c_var}",
        ],
        i_module_line="iso_c_binding:{f_kind}",
    ),
    dict(
        name="f_function_native_*_cdesc_allocatable",
        mixin=[
            "f_mixin_function_cdesc",
            "c_mixin_native_cdesc_fill-cdesc",
        ],
        c_helper="copy_array ShroudTypeDefines array_context",
        # XXX - use case for append to c_helper, first two from mixin
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        f_arg_decl=[
            "{f_type}, allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_post_call=[
            # XXX - allocate scalar
            "allocate({f_var}{f_array_allocate})",
            "call {fhelper_copy_array}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},\t kind=C_SIZE_T))",
        ],
    ),

    dict(
        # Pointer to scalar.
        # type(C_PTR) is returned instead of a cdesc argument.
        name="f_function_native_&",
        alias=[
            "f_function_native_*_pointer",   # XXX - change base to &?
            "c_function_native_*_pointer",
            "f_function_native_&_pointer",
            "c_function_native_&_pointer",
        ],
        # mixin f_mixin_function_c-ptr
        f_module=dict(iso_c_binding=["C_PTR", "c_f_pointer"]),
        f_arg_decl=[
            "{f_type}, pointer :: {f_var}",
        ],
        f_declare=[
            "type(C_PTR) :: {c_local_ptr}",
        ],
        f_call=[
            "{c_local_ptr} = {F_C_call}({F_arg_c_call})",
        ],
        f_post_call=[
            "call c_f_pointer({c_local_ptr}, {F_result})",
        ],
        f_local=["ptr"],

        i_result_decl=[
            "type(C_PTR) {c_var}",
        ],
        i_module=dict(iso_c_binding=["C_PTR"]),
        
    ),
    dict(
        name="f_function_native_*_cdesc_pointer",
        mixin=[
            "f_mixin_function_cdesc",
            "f_mixin_function_native_cdesc_pointer",
            "c_mixin_native_cdesc_fill-cdesc",
        ],
    ),
    dict(
        # +deref(pointer) +owner(caller)
        name="f_function_native_*_cdesc_pointer_caller",
        mixin=[
            "f_mixin_function_cdesc",
            "c_mixin_native_cdesc_fill-cdesc",
        ],
        f_helper="capsule_helper",
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        f_arg_name=["{c_var_capsule}"],
        f_arg_decl=[
            "{f_type}, pointer :: {f_var}{f_assumed_shape}",
            "type({F_capsule_type}), intent(OUT) :: {c_var_capsule}",
        ],
        f_post_call=[
            "call c_f_pointer(\t{c_var_cdesc}%base_addr,\t {F_result}{f_array_shape})",
            "{c_var_capsule}%mem = {c_var_cdesc}%cxx",
        ],
    ),
    dict(
        name="f_function_native_*_raw",
        mixin=[
            "f_mixin_function_ptr",
        ],
        alias=[
            "c_function_native_*_raw",
        ],
    ),
    dict(
        # int **func(void)
        # regardless of deref value.
        name="f_function_native_**",
        mixin=[
            "f_mixin_function_ptr",
        ],
        alias=[
            "c_function_native_**",

            "f_function_native_*_allocatable",
            "c_function_native_*_allocatable",
        ],
    ),
    
    dict(
        name="f_function_native_&_buf_pointer",
        mixin=[
            "f_function_native_&",
        ],
        f_arg_decl=[
            "{f_type}, pointer :: {f_var}{f_assumed_shape}",
        ],
    ),

    ########################################
    # char arg
    dict(
        name="f_in_char_scalar",
        alias=[
            "c_in_char_scalar",
        ],
        # By default the declaration is character(LEN=*).
        f_arg_decl=[
            "character, value, intent(IN) :: {f_var}",
        ],
        c_arg_decl=[
            "char {c_var}",
        ],
        i_arg_decl=[
            "character(kind=C_CHAR), value, intent(IN) :: {c_var}",
        ],
        i_arg_names=["{c_var}"],
        i_module=dict(iso_c_binding=["C_CHAR"]),
    ),

#    dict(
#        This simpler version had to be replace for pgi and cray.
#        See below.
#        name="c_function_char_scalar",
#        i_result_decl=[
#            "character(kind=C_CHAR) :: {c_var}",
#        ],
#        i_module=dict(iso_c_binding=["C_CHAR"]),
#    ),
    dict(
        name="f_function_char_scalar",
        alias=[
            "c_function_char_scalar",
        ],
        f_arg_call=["{c_var}"],  # Pass result as an argument.

        # Pass result as an argument.
        # pgi and cray compilers have problems with functions which
        # return a scalar char.
        c_call=[
            "*{c_var} = {function_name}({C_call_list});",
        ],
        c_arg_decl=[
            "char *{c_var}",
        ],
        i_arg_decl=[
            "character(kind=C_CHAR), intent(OUT) :: {c_var}",
        ],
        i_arg_names=["{c_var}"],
        i_module=dict(iso_c_binding=["C_CHAR"]),
        c_return_type="void",  # Convert to function.
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
        # c_function_char_*_allocatable
        # c_function_char_*_copy
        # c_function_char_*_pointer
        # c_function_char_*_raw
        name="f_function_char_*",
        alias=[
            "c_function_char_*",
            "f_function_char_*_allocatable/copy/pointer/raw",
            "c_function_char_*_allocatable/copy/pointer/raw",
        ],

        f_arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
        
        i_result_decl=[
            "type(C_PTR) {c_var}",
        ],
        i_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        # NULL terminate the input string.
        # Skipped if ftrim_char_in, the terminate is done in Fortran.
        name="f_in_char_*_buf",
        mixin=[
            "f_mixin_in_character_buf",
            "c_mixin_in_character_buf",
        ],
        alias=[
            "c_in_char_*_buf",
        ],
        c_temps=["len", "str"],
        c_helper="ShroudStrAlloc ShroudStrFree",
        c_pre_call=[
            "char * {c_var_str} = ShroudStrAlloc(\t"
            "{c_var},\t {c_var_len},\t {c_blanknull});",
        ],
        c_arg_call=["{c_var_str}"],
        c_post_call=[
            "ShroudStrFree({c_var_str});"
        ],
    ),
    dict(
        name="f_out_char_*_buf",
        mixin=[
            "f_mixin_in_character_buf",
            "c_mixin_in_character_buf",
        ],
        alias=[
            "c_out_char_*_buf",
        ],
        c_helper="ShroudStrBlankFill",
        c_post_call=[
            "ShroudStrBlankFill({c_var}, {c_var_len});"
        ],
    ),
    dict(
        name="f_inout_char_*_buf",
        mixin=[
            "f_mixin_in_character_buf",
            "c_mixin_in_character_buf",
        ],
        alias=[
            "c_inout_char_*_buf",
        ],
        c_temps=["len", "str"],
        c_helper="ShroudStrAlloc ShroudStrCopy ShroudStrFree",
        c_pre_call=[
            "char * {c_var_str} = ShroudStrAlloc(\t"
            "{c_var},\t {c_var_len},\t {c_blanknull});",
        ],
        c_arg_call=["{c_var_str}"],
        c_post_call=[
            # nsrc=-1 will call strlen({c_var_str})
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {c_var_str},\t -1);",
            "ShroudStrFree({c_var_str});",
        ],
    ),
    dict(
        # Copy result into caller's buffer.
        #  char *getname() +len(30)
        name="f_function_char_*_buf_copy",
        mixin=[
            "f_mixin_in_character_buf",
            "c_mixin_in_character_buf",
        ],
        alias=[
            "c_function_char_*_buf_copy",
        ],
        cxx_local_var="result",
        c_helper="ShroudStrCopy",
        c_post_call=[
            # nsrc=-1 will call strlen({cxx_var})
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var},\t -1);",
        ],
        c_return_type="void",
    ),

    dict(
        name="f_function_char_*_arg",
        mixin=[
            "f_mixin_function_buf_character-arg",
        ],
        alias=[
            "c_function_char_*_arg",
            "f_function_char_*_buf_arg",
            "c_function_char_*_buf_arg",
        ],

        c_helper="ShroudStrCopy",
        c_post_call=[
            # nsrc=-1 will call strlen({c_var_str})
            "ShroudStrCopy({F_string_result_as_arg}, n{F_string_result_as_arg},"
            "\t {c_var},\t -1);",
        ]
    ),

    dict(
        name="f_function_char_*_cdesc_arg",
        alias=[
            "c_function_char_*_cdesc_arg",
        ],
    ),

    dict(
        name="f_function_string_scalar_buf_arg",
        mixin=[
            "f_mixin_function_buf_character-arg",
        ],
        alias=[
            "c_function_string_scalar_buf_arg",

            "f_function_string_&_buf_arg",
            "c_function_string_&_buf_arg",
        ],
        # XXX make as a mixin.
        c_helper="ShroudStrCopy",
        c_post_call=[
            "if ({cxx_var}{cxx_member}empty()) {{+",
            "ShroudStrCopy({F_string_result_as_arg}, n{F_string_result_as_arg},"
            "\t {nullptr},\t 0);",
            "-}} else {{+",
            "ShroudStrCopy({F_string_result_as_arg}, n{F_string_result_as_arg},"
            "\t {cxx_var}{cxx_member}data(),"
            "\t {cxx_var}{cxx_member}size());",
            "-}}",
        ],
    ),

    #####
    dict(
        # Treat as an assumed length array in Fortran interface.
        name="f_in_char_**",
        alias=[
            "c_in_char_**",
        ],
        c_arg_decl=[
            "char **{c_var}",
        ],
        i_arg_decl=[
            "type(C_PTR), intent(IN) :: {c_var}(*)",
        ],
        i_arg_names=["{c_var}"],
        i_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name='f_in_char_**_buf',
        mixin=[
            "f_mixin_in_string_array_buf",
        ],
        alias=[
            'c_in_char_**_buf',
        ],
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
        name="f_function_char_scalar_cdesc_allocatable",
        mixin=[
            "f_mixin_function_cdesc",
            "f_mixin_char_cdesc_allocate",
            "c_mixin_function_char_*_cdesc",
        ],
        alias=[
            "f_function_char_*_cdesc_allocatable",
        ],
        c_helper="copy_string ShroudTypeDefines",
    ),

    # XXX note: split by char/scalar - use of ShroudStrToArray
    # allocatable - split by scalar (using new) and */& using library created memory
    dict(
        name="f_function_char_scalar_cdesc_pointer",
        mixin=[
            "f_mixin_function_cdesc",
            "c_mixin_function_char_*_cdesc",
        ],
        alias=[
            "f_function_char_*_cdesc_pointer",
            "c_function_char_*_cdesc_pointer",
        ],
        f_helper="pointer_string array_context",
        f_arg_decl=[
            "character(len=:), pointer :: {f_var}",
        ],
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        f_post_call=[
            # BLOCK is Fortran 2008
            #"block+",
            #"character(len={c_var_cdesc}%elem_len), pointer :: {c_local_s}",
            #"call c_f_pointer({c_var_cdesc}%base_addr, {c_local_s})",
            #"{f_var} => {c_local_s}",
            #"-end block",
            "call {fhelper_pointer_string}(\t{c_var_cdesc},\t {f_var})",
        ],
    ),
    dict(
        # f_function_string_scalar_cdesc_pointer
        # f_function_string_*_cdesc_pointer
        # c_function_string_*_cdesc_pointer
        # c_function_string_&_cdesc_pointer
        name="f_function_string_*_cdesc_pointer",
        mixin=[
            "f_mixin_function_cdesc",
            "c_mixin_function_char_*_cdesc",   # XXX - maybe other mixing with string
        ],
        alias=[
            "f_function_string_&_cdesc_pointer",
            "f_function_string_*/&_cdesc_pointer_caller/library",
            "c_function_string_*/&_cdesc_pointer",
            "c_function_string_*/&_cdesc_pointer_caller/library",
        ],
        f_helper="pointer_string array_context",
        f_arg_decl=[
            "character(len=:), pointer :: {f_var}",
        ],
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        f_post_call=[
            # BLOCK is Fortran 2008
            #"block+",
            #"character(len={c_var_cdesc}%elem_len), pointer :: {c_local_s}",
            #"call c_f_pointer({c_var_cdesc}%base_addr, {c_local_s})",
            #"{f_var} => {c_local_s}",
            #"-end block",
            "call {fhelper_pointer_string}(\t{c_var_cdesc},\t {f_var})",
        ],

        # TTT - replace c_function_string_*/&_cdesc_allocatable_pointer
        c_helper="ShroudStrToArray",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        c_post_call=[
            "ShroudStrToArray(\t{c_var_cdesc},\t {cxx_addr}{cxx_var},\t {idtor});",
        ],
    ),

    dict(
        # c_in_string_*
        # c_in_string_&
        name="f_in_string_*/&",
        alias=[
            "c_in_string_*",
            "c_in_string_&",
        ],
        cxx_local_var="scalar",
        c_pre_call=["{c_const}std::string {cxx_var}({c_var});"],
    ),
    dict(
        # f_out_string_*
        # f_out_string_&
        name="f_out_string_*/&",
        alias=[
            "c_out_string_*",
            "c_out_string_&",
        ],
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
        # f_inout_string_*
        # f_inout_string_&
        name="f_inout_string_*/&",
        mixin=[
            "f_mixin_in_character_buf",
        ],
        alias=[
            "c_inout_string_*/&",
        ],
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
        # f_in_string_*_buf
        # f_in_string_&_buf
        name="f_in_string_*/&_buf",
        mixin=[
            "f_mixin_in_character_buf",
            "c_mixin_in_character_buf",
        ],
        alias=[
            "c_in_string_*/&_buf",
        ],
        c_helper="ShroudLenTrim",
        cxx_local_var="scalar",
        c_pre_call=[
            "{c_const}std::string {cxx_var}({c_var},\t ShroudLenTrim({c_var}, {c_var_len}));",
        ],
    ),
    dict(
        # f_out_string_*_buf
        # f_out_string_&_buf
        name="f_out_string_*/&_buf",
        mixin=[
            "f_mixin_in_character_buf",
            "c_mixin_in_character_buf",
        ],
        alias=[
            "c_out_string_*/&_buf",
        ],
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
        # f_inout_string_*_buf
        # f_inout_string_&_buf
        name="f_inout_string_*/&_buf",
        mixin=[
            "f_mixin_in_character_buf",
            "c_mixin_in_character_buf"
        ],
        alias=[
            "c_inout_string_*/&_buf",
        ],
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
        # Fortran calling a C function without
        # any api argument - is this useful?
        # c_function_string_scalar
        # c_function_string_*
        # c_function_string_&
        name="f_shared_function_string_scalar",
        alias=[
            "f_function_string_scalar/*/&",
            "c_function_string_scalar/*/&",
            "f_function_string_*_allocatable",
            "c_function_string_*_allocatable",
            "f_function_string_*_copy",
            "c_function_string_*_copy",
            "f_function_string_*_pointer",
            "c_function_string_*_pointer",
            "f_function_string_&_allocatable",
            "c_function_string_&_allocatable",
            "f_function_string_&_copy",
            "c_function_string_&_copy",
        ],
        # cxx_to_c creates a pointer from a value via c_str()
        # The default behavior will dereference the value.
        c_return=[
            "return {c_var};",
        ],
        i_result_decl=[
            "type(C_PTR) {c_var}",
        ],
        i_module=dict(iso_c_binding=["C_PTR"]),
    ),

    # std::string
    dict(
        name="f_XXXin_string_scalar",  # pairs with c_in_string_scalar_buf
        f_need_wrapper=True,
        mixin=["f_mixin_in_character_buf"],
        f_arg_decl=[
            # Remove VALUE added by f_default
            "character(len=*), intent(IN) :: {f_var}",
        ],
    ),
    dict(
        # Used with C wrapper.
        name="f_in_string_scalar",
        alias=[
            "c_in_string_scalar",
        ],
        c_arg_decl=[
            # Argument is a pointer while std::string is a scalar.
            # C++ compiler will convert to std::string when calling function.
            "char *{c_var}",
        ],
        i_arg_decl=[
            # Remove VALUE added by c_default
            "character(kind=C_CHAR), intent(IN) :: {c_var}(*)",
        ],
        i_arg_names=["{c_var}"],
        i_module=dict(iso_c_binding=["C_CHAR"]),
    ),
    dict(
        name="f_in_string_scalar_buf",
        mixin=[
            "f_mixin_in_character_buf",
            "c_mixin_in_character_buf",
        ],
        alias=[
            "c_in_string_scalar_buf",
        ],
        f_arg_decl=[
            # Remove VALUE added by f_default
            "character(len=*), intent({f_intent}) :: {f_var}",
        ],
        cxx_local_var="scalar",
        c_pre_call=[
            "int {c_local_trim} = ShroudLenTrim({c_var}, {c_var_len});",
            "std::string {cxx_var}({c_var}, {c_local_trim});",
        ],
        c_call=[
            "XXX{cxx_var}",  # XXX - this seems wrong and is untested
        ],
        c_local=["trim"],
    ),
    
    dict(
        # f_function_string_*_cdesc_allocatable_caller
        # f_function_string_&_cdesc_allocatable_caller
        # f_function_string_*_cdesc_allocatable_library
        # f_function_string_&_cdesc_allocatable_library
        name="f_function_string_*_cdesc_allocatable",
        mixin=[
            "f_mixin_function_cdesc",
            "f_mixin_char_cdesc_allocate",
        ],
        alias=[
            "f_function_string_&_cdesc_allocatable",
            "f_function_string_*/&_cdesc_allocatable_caller/library",

        ],

        c_helper="copy_string ShroudStrToArray",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        c_post_call=[
            "ShroudStrToArray(\t{c_var_cdesc},\t {cxx_addr}{cxx_var},\t {idtor});",
        ],
    ),

    dict(
        # f_function_string_scalar_buf
        # f_function_string_*_buf
        # f_function_string_&_buf
        # f_function_string_scalar_buf_copy
        # f_function_string_*_buf_copy
        # f_function_string_&_buf_copy
        # TTT - is the buf version used?
        name="f_function_string_scalar_buf",
        mixin=[
            "f_mixin_in_character_buf",
            "c_mixin_in_character_buf",
        ],
        alias=[
            "f_function_string_*_buf",
            "f_function_string_&_buf",
            
            "f_function_string_scalar_buf_copy",
            "f_function_string_*_buf_copy",
            "f_function_string_&_buf_copy",

#            "f_function_string_scalar/*/&_buf",
#            "f_function_string_scalar/*/&_buf_copy",
            "c_function_string_scalar/*/&_buf_copy",
        ],

        i_arg_decl=[
            # Change to intent(OUT) from mixin.
            "character(kind=C_CHAR), intent(OUT) :: {c_var}(*)",
            "integer(C_INT), value, intent(IN) :: {c_var_len}",
        ],
        c_helper="ShroudStrCopy",
        # No need to allocate a local copy since the string is copied
        # into a Fortran variable before the string is deleted.
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
        c_return_type="void",
    ),

    # similar to f_function_char_scalar_allocatable
    dict(
        # f_function_string_scalar_cdesc_allocatable
        # f_function_string_scalar_cdesc_allocatable_caller
        # f_function_string_scalar_cdesc_allocatable_library
        name="f_function_string_scalar_cdesc_allocatable",
        mixin=[
            "f_mixin_function_cdesc",
            "f_mixin_char_cdesc_allocate",
            "c_mixin_destructor_new-string",
        ],
        alias=[
            "f_function_string_scalar_cdesc_allocatable_caller/library",

            # f_function_string_&_cdesc_allocatable
        ],
        c_helper="ShroudStrToArray copy_string",
        cxx_local_var="pointer",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        c_pre_call=[
            "std::string * {cxx_var} = new std::string;",
        ],
        c_post_call=[
            "ShroudStrToArray({c_var_cdesc}, {cxx_var}, {idtor});",
        ],
    ),
    
    ########################################
    # vector
    # Specialize for std::vector<native>
    dict(
        # f_in_vector_scalar_buf_targ_native_scalar
        # f_in_vector_*_buf_targ_native_scalar
        # f_in_vector_&_buf_targ_native_scalar
        name="f_in_vector_scalar/*/&_buf_targ_native_scalar",
        mixin=[
            "f_mixin_in_array_buf",
        ],
        alias=[
            "c_in_vector_scalar/*/&_buf_targ_native_scalar",
        ],
        cxx_local_var="scalar",
        c_pre_call=[
            (
                "{c_const}std::vector<{cxx_T}> "
                "{cxx_var}({c_var}, {c_var} + {c_var_size});"
            )
        ],
    ),

    dict(
        # Fill cdesc with vector information
        # Return address and size of vector data.
        name="c_mixin_vector_cdesc_fill-cdesc",
        c_post_call=[
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
    ),
    
    dict(
        name="c_mixin_out_vector_cdesc_targ_native_scalar",
        mixin=[
            "c_mixin_out_array_cdesc",
            "c_mixin_destructor_new-vector",
            "c_mixin_vector_cdesc_fill-cdesc",
        ],
        cxx_local_var="pointer",
        c_pre_call=[
            "{c_const}std::vector<{cxx_T}>"
            "\t *{cxx_var} = new std::vector<{cxx_T}>;"
        ],
    ),
    
    # copy into user's existing array
    # cxx_var is always a pointer to a vector
    dict(
        # f_out_vector_*_cdesc_targ_native_scalar
        # f_out_vector_&_cdesc_targ_native_scalar
        name="f_out_vector_*/&_cdesc_targ_native_scalar",
        mixin=[
            "f_mixin_out_array_cdesc",
            "c_mixin_out_vector_cdesc_targ_native_scalar",
        ],
        alias=[
            "c_out_vector_*/&_cdesc_targ_native_scalar",
        ],

        c_helper="copy_array ShroudTypeDefines",
        f_helper="copy_array",
        # TARGET required for argument to C_LOC.
        f_arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["C_SIZE_T", "C_LOC"]),
        f_post_call=[
            "call {fhelper_copy_array}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        name="c_inout_vector_cdesc_targ_native_scalar",
        mixin=[
            "c_mixin_inout_array_cdesc",
            "c_mixin_destructor_new-vector",
            "c_mixin_vector_cdesc_fill-cdesc",
        ],
        cxx_local_var="pointer",
        c_helper="ShroudTypeDefines",
        c_pre_call=[
            "std::vector<{cxx_T}> *{cxx_var} = \tnew std::vector<{cxx_T}>\t("
            "\t{c_var}, {c_var} + {c_var_size});"
        ],
    ),
    # Almost same as intent_out_buf.
    # Similar to f_vector_out_allocatable but must declare result variable.
    # Always return a 1-d array.
    dict(
        name="f_function_vector_scalar_cdesc_allocatable_targ_native_scalar",
        mixin=[
            "f_mixin_function_cdesc",
            "c_mixin_destructor_new-vector",
            "c_mixin_vector_cdesc_fill-cdesc",
        ],

        c_helper="copy_array ShroudTypeDefines",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        f_arg_decl=[
            "{f_type}, allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_post_call=[
            "allocate({f_var}({c_var_cdesc}%size))",
            "call {fhelper_copy_array}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
        ],
        
        cxx_local_var="pointer",
        c_pre_call=[
            "{c_const}std::vector<{cxx_T}>"
            "\t *{cxx_var} = new std::vector<{cxx_T}>;"
        ],
    ),

    # Specialize for std::vector<native *>
    dict(
        # Create a vector for pointers
        name="f_in_vector_&_buf_targ_native_*",
        mixin=[
            "f_mixin_in_2d_array_buf",
        ],
        alias=[
            "c_in_vector_&_buf_targ_native_*",
        ],
        cxx_local_var="scalar",
        c_pre_call=[
            "std::vector<{cxx_T}> {cxx_var};",
            "for (size_t i=0; i < {c_var_size}; ++i) {{+",
            "{cxx_var}.push_back({c_var} + ({c_var_len}*i));",
            "-}}"
        ],
    ),
    
    dict(
        # f_in_vector_scalar_buf_targ_string_scalar
        # f_in_vector_*_buf_targ_string_scalar
        # f_in_vector_&_buf_targ_string_scalar
        name="f_in_vector_scalar/*/&_buf_targ_string_scalar",
        mixin=[
            "f_mixin_in_string_array_buf",
        ],
        alias=[
            "c_in_vector_scalar/*/&_buf_targ_string_scalar",
        ],
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
        c_local=["i", "n", "s"],
    ),
    # XXX untested [cf]_out_vector_buf_string
    dict(
        name="f_out_vector_buf_targ_string_scalar",
        mixin=[
            "f_mixin_in_string_array_buf",
        ],

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
        c_local=["i", "n", "s"],
    ),
    # XXX untested [cf]_inout_vector_buf_string
    dict(
        name="f_inout_vector_buf_targ_string_scalar",
        mixin=[
            "f_mixin_in_string_array_buf",
        ],

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
        c_local=["i", "n", "s"],
    ),

    dict(
        # Pass argument and size to C.
        # Pass array_type to C which will fill it in.
        name="f_mixin_inout_char_array_cdesc",
        f_helper="array_context",
        f_declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
        ],
#        f_arg_call=["{f_var}", "size({f_var}, kind=C_SIZE_T)", "{c_var_cdesc}"],
        f_arg_call=["{c_var_cdesc}"],
#        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        f_temps=["cdesc"],
    ),

    ##########
    dict(
        # Collect information about a string argument
        name="f_mixin_str_array",
        mixin=["f_mixin_out_array_cdesc"],

        # TARGET required for argument to C_LOC.
        f_arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_helper="ShroudTypeDefines array_context",
        f_module=dict(iso_c_binding=["C_LOC"]),
        f_declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
        ],
        f_pre_call=[
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
        f_arg_call=["{c_var_cdesc}"],
        f_temps=["cdesc"],
    ),

    dict(
        name="f_out_vector_&_cdesc_targ_string_scalar",
        mixin=[
            "f_mixin_str_array",
            "c_mixin_out_array_cdesc",
        ],
        alias=[
            "c_out_vector_&_cdesc_targ_string_scalar",
        ],
        c_helper="vector_string_out",
        c_pre_call=[
            "{c_const}std::vector<std::string> {cxx_var};"
        ],
        c_arg_call=["{cxx_var}"],
        c_post_call=[
            "{chelper_vector_string_out}(\t{c_var_cdesc},\t {cxx_var});",
        ],
    ),

    ##########
    # As above but +deref(allocatable)
    # 
    dict(
        name="f_out_vector_&_cdesc_allocatable_targ_string_scalar",
        mixin=[
            "c_mixin_out_array_cdesc",
        ],
        alias=[
            "c_out_vector_&_cdesc_allocatable_targ_string_scalar",
        ],
        f_arg_decl=[
            "character({f_char_len}), intent({f_intent}), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_helper="vector_string_allocatable array_context capsule_data_helper",
        c_helper="vector_string_allocatable vector_string_out_len",
        f_module=dict(iso_c_binding=["C_LOC"]),
        f_declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
            "type({F_array_type}) :: {c_var_out}",
        ],
        f_arg_call=["{c_var_out}"],
        f_post_call=[
            "{c_var_cdesc}%size = {c_var_out}%size;",
            "{c_var_cdesc}%elem_len = {c_var_out}%elem_len",
            "allocate({f_char_type}{f_var}({c_var_cdesc}%size))",
            "{c_var_cdesc}%cxx%addr = C_LOC({f_var});",
            "{c_var_cdesc}%base_addr = C_LOC({f_var});",
            "call {fhelper_vector_string_allocatable}({c_var_cdesc}, {c_var_out})",
        ],
        f_temps=["cdesc", "out"],

        c_pre_call=[
#            "std::vector<std::string> *{cxx_var} = new {cxx_type};"  XXX cxx_tye=std::string
            "std::vector<std::string> *{cxx_var} = new std::vector<std::string>;"
        ],
        c_arg_call=["*{cxx_var}"],
        c_post_call=[
            "if ({c_char_len} > 0) {{+",
            "{c_var_cdesc}->elem_len = {c_char_len};",
            "-}} else {{+",
            "{c_var_cdesc}->elem_len = {chelper_vector_string_out_len}(*{cxx_var});",
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
#    dict(
#        # f_out_vector_*_cdesc_targ_native_scalar
#        # f_out_vector_&_cdesc_targ_native_scalar
#        name="f_out_vector_*/&_cdesc_targ_native_scalar",
#        mixin=["f_mixin_out_array_cdesc"],
#        c_helper="copy_array",
#        f_helper="copy_array",
#        # TARGET required for argument to C_LOC.
#        f_arg_decl=[
#            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
#        ],
#        f_module=dict(iso_c_binding=["C_SIZE_T", "C_LOC"]),
#        f_post_call=[
#            "call {hnamefunc0}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
#        ],
#    ),
    dict(
        name="f_inout_vector_&_cdesc_targ_native_scalar",
        mixin=[
            "f_mixin_inout_array_cdesc",
            "c_inout_vector_cdesc_targ_native_scalar",
        ],
        alias=[
            "c_inout_vector_&_cdesc_targ_native_scalar",
        ],
        c_helper="copy_array",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        # TARGET required for argument to C_LOC.
        f_arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_post_call=[
            "call {fhelper_copy_array}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        # XXX - This group is not tested
        name="f_function_vector_scalar_cdesc",
        c_helper="copy_array",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        # TARGET required for argument to C_LOC.
        f_arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_post_call=[
            "call {fhelper_copy_array}(\t{temp0},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))"
        ],
    ),
    # copy into allocated array
    dict(
        # f_out_vector_*_cdesc_allocatable_targ_native_scalar
        # f_out_vector_&_cdesc_allocatable_targ_native_scalar
        name="f_out_vector_*/&_cdesc_allocatable_targ_native_scalar",
        mixin=[
            "f_mixin_out_array_cdesc",
            "c_mixin_out_vector_cdesc_targ_native_scalar"
        ],
        alias=[
            "c_out_vector_*_cdesc_allocatable_targ_native_scalar",
            "c_out_vector_&_cdesc_allocatable_targ_native_scalar",
        ],
        c_helper="copy_array",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        # TARGET required for argument to C_LOC.
        f_arg_decl=[
            "{f_type}, intent({f_intent}), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_post_call=[
            "allocate({f_var}({c_var_cdesc}%size))",
            "call {fhelper_copy_array}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        name="f_inout_vector_&_cdesc_allocatable_targ_native_scalar",
        mixin=[
            "f_mixin_inout_array_cdesc",
            "c_inout_vector_cdesc_targ_native_scalar",
        ],
        alias=[
            "c_inout_vector_&_cdesc_allocatable_targ_native_scalar",
        ],
        c_helper="copy_array",
        f_helper="copy_array",
        f_module=dict(iso_c_binding=["C_LOC", "C_SIZE_T"]),
        # TARGET required for argument to C_LOC.
        f_arg_decl=[
            "{f_type}, intent({f_intent}), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_post_call=[
            "if (allocated({f_var})) deallocate({f_var})",
            "allocate({f_var}({c_var_cdesc}%size))",
            "call {fhelper_copy_array}(\t{c_var_cdesc},\t C_LOC({f_var}),\t size({f_var},kind=C_SIZE_T))",
        ],
    ),

    ##########
    # Extract pointer to C++ instance.
    # convert C argument into a pointer to C++ type.

    dict(
        name="f_mixin_shadow-arg",
        f_arg_decl=[
            "{f_type}, intent({f_intent}) :: {f_var}",
        ],
        f_arg_call=[
            "{f_var}%{F_derived_member}",
        ],
        f_need_wrapper=True,
    ),
    
    dict(
        name="c_mixin_shadow",
        c_arg_decl=[
            "{c_type} * {c_var}",
        ],
        i_arg_decl=[
            "type({f_capsule_data_type}), intent({f_intent}) :: {c_var}",
        ],
        i_arg_names=["{c_var}"],
        i_module_line="{i_module_line}",
    ),
    
#    dict(
#        name="f_in_shadow",    f_mixin_shadow-arg
#        alias=[
#            # TTT
#            "f_in_shadow_scalar",
#            "f_in_shadow_*",
##TTT            "f_in_shadow_&",
#        ],
#        f_arg_decl=[
#            "{f_type}, intent({f_intent}) :: {f_var}",
#        ],
#        f_arg_call=[
#            "{f_var}%{F_derived_member}",
#        ],
#        f_need_wrapper=True,
#    ),
    dict(
        # c_in_shadow_scalar
        # c_inout_shadow_scalar  # XXX inout by value makes no sense.
        name="f_in_shadow_scalar",
        mixin=[
            "f_mixin_shadow-arg",
            "c_mixin_shadow"
        ],
        alias=[
            "c_in_shadow_scalar",
        ],
        c_arg_decl=[
            "{c_type} {c_var}",
        ],
        i_arg_decl=[
            "type({f_capsule_data_type}), intent({f_intent}), value :: {c_var}",
        ],
        cxx_local_var="pointer",
        c_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} =\t "
            "{cast_static}{c_const}{cxx_type} *{cast1}{c_var}.addr{cast2};",
        ],
    ),

    dict(
        name="f_in_shadow_*",
        mixin=[
            "f_mixin_shadow-arg",
            "c_mixin_shadow",
        ],
        alias=[
            "c_in_shadow_*",
            "f_inout_shadow_*",
            "c_inout_shadow_*",
            "f_inout_shadow_&",
            "c_inout_shadow_&",
        ],

        cxx_local_var="pointer",
        c_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} =\t "
            "{cast_static}{c_const}{cxx_type} *{cast1}{c_var}->addr{cast2};",
        ],
    ),

    # Return a C_capsule_data_type.
    dict(
        name="f_function_shadow_*_capsule",
        mixin=[
            "f_mixin_function_shadow_capsule",
            "c_mixin_shadow",
        ],
        cxx_local_var="result",
        c_post_call=[
            "{c_var}->addr = {cxx_nonconst_ptr};",
            "{c_var}->idtor = {idtor};",
        ],
        c_return_type="void",
    ),

    # TTT new merged group
    dict(
        name="f_in_shadow_&",
        mixin=["c_mixin_shadow"],
        alias=[
            "c_in_shadow_&",
        ],
        f_arg_decl=[
            "{f_type}, intent({f_intent}) :: {f_var}",
        ],
        f_arg_call=[
            "{f_var}%{F_derived_member}",
        ],
        f_need_wrapper=True,

        cxx_local_var="pointer",
        c_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} =\t "
            "{cast_static}{c_const}{cxx_type} *{cast1}{c_var}->addr{cast2};",
        ],
    ),

    dict(
        name="c_function_shadow_scalar_capsule",
        # Return a instance by value.
        # Create memory in c_pre_call so it will survive the return.
        # owner="caller" sets idtor flag to release the memory.
        # c_local_var is passed in as argument.
        mixin=["c_mixin_shadow"],
        cxx_local_var="pointer",
        owner="caller",
        c_pre_call=[
            "{cxx_type} * {cxx_var} = new {cxx_type};",
        ],
        c_post_call=[
            "{c_var}->addr = {cxx_nonconst_ptr};",
            "{c_var}->idtor = {idtor};",
        ],
        c_return_type="void",
    ),
    
    # Return a C_capsule_data_type.
    dict(
        # f_function_shadow_*_capptr
        # f_function_shadow_&_capptr
        name="f_function_shadow_*/&_capptr",
        mixin=[
            "f_mixin_function_shadow_capptr",
            "c_mixin_shadow",
        ],
        alias=[
            "c_function_shadow_*/&_capptr",
            "f_function_shadow_*/&_capptr_caller/library",
            "c_function_shadow_*/&_capptr_caller/library",
        ],

        cxx_local_var="result",
        c_post_call=[
            "{c_var}->addr = {cxx_nonconst_ptr};",
            "{c_var}->idtor = {idtor};",
        ],
        
        c_return_type=None,
        c_return=[
            "return {c_var};",
        ],
    ),
    
    dict(
        # f_function_shadow_scalar_capptr
        # f_function_shadow_*_capptr
        # f_function_shadow_&_capptr
        name="f_function_shadow_scalar_capptr",
        mixin=[
            "f_mixin_function_shadow_capptr",
            "c_mixin_shadow",
            "c_function_shadow_scalar_capsule"
        ],
        alias=[
            "c_function_shadow_scalar_capptr",
            "f_function_shadow_scalar_capptr_targ_native_scalar",
            "c_function_shadow_scalar_capptr_targ_native_scalar",
            "f_function_shadow_scalar_capptr_caller/library",
        ],

        c_return_type="{c_type} *",
        c_return=[
            "return {c_var};",
        ],
    ),
    dict(
        name="f_ctor_shadow_scalar_capptr",
        mixin=[
            "f_mixin_function_shadow_capptr",
            "c_mixin_shadow",
        ],
        alias=[
            "c_ctor_shadow_scalar_capptr",
        ],
        c_call=[
            "{cxx_type} *{cxx_var} =\t new {cxx_type}({C_call_list});",
            "{c_var}->addr = static_cast<{c_const}void *>(\t{cxx_var});",
            "{c_var}->idtor = {idtor};",
        ],
        owner="caller",
    ),
    dict(
        # NULL in stddef.h
        name="c_dtor",
        mixin=["c_mixin_noargs"],
        alias=[
            "c_dtor_void_scalar",
        ],
        lang_c=dict(
            impl_header=["<stddef.h>"],
        ),
        lang_cxx=dict(
            impl_header=["<cstddef>"],
        ),
        c_call=[
            "delete {CXX_this};",
            "{C_this}->addr = {nullptr};",
        ],
        c_return_type="void",
    ),
    dict(
        name="f_dtor",
        mixin=[
            "c_dtor",
        ],
        alias=[
            "f_dtor_void_scalar",  # Used with interface
        ],
        f_arg_call=[],
    ),

    dict(
        # Used with in, out, inout
        # C pointer -> void pointer -> C++ pointer
        name="f_defaulttmp_struct",
        alias=[
            "f_in_struct_scalar",
            "f_in_struct_*",
            "f_in_struct_&",
            "f_out_struct_*",
            "f_out_struct_&",
            "f_inout_struct_*",
            "f_inout_struct_&",
            "c_in_struct_scalar",
            "c_in_struct_*",
            "c_in_struct_&",
            "c_out_struct_*",
            "c_out_struct_&",
            "c_inout_struct_*",
            "c_inout_struct_&",
        ],
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
        lang_cxx=dict(
            c_temps=["c"],
            c_post_call=[
                "{c_const}{c_type} * {c_var_c} = \tstatic_cast<{c_const}{c_type} *>(\tstatic_cast<{c_const}void *>(\t{cxx_addr}{cxx_var}));",
            ],
            c_return=[
                "return {c_var_c};",
            ]
        ),
    ),

    # start function_struct_scalar
    dict(
        name="f_function_struct_scalar",
        alias=[
            "c_function_struct_scalar",
        ],
        f_arg_call=["{f_var}"],

        c_arg_decl=["{c_type} *{c_var}"],
        i_arg_decl=["{f_type}, intent(OUT) :: {c_var}"],
        i_arg_names=["{c_var}"],
        i_import=["{f_kind}"],
        c_return_type="void",  # Convert to function.
        cxx_local_var="result",
        c_post_call=[
            "memcpy((void *) {c_var}, (void *) &{cxx_var}, sizeof({cxx_var}));",
        ],
    ),
    # end function_struct_scalar
    
    # Similar to c_function_native_*
    dict(
        name="f_function_struct_*_pointer",
        mixin=[
            "f_mixin_function_c-ptr",
            "c_function_struct",
        ],
        alias=[
            "c_function_struct_*_pointer",
        ],
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
        c_call=[
            "// skip call c_getter",
        ],
    ),
    dict(
        # Argument to setter.
        name="c_setter_arg",
    ),
    dict(
        name="f_getter_native_scalar",
        mixin=[
            "c_getter",
        ],
        alias=[
            "c_getter_native_scalar",
            "f_getter_native_*_pointer",
            "c_getter_native_*_pointer",
        ],
        c_return=[
            "return {CXX_this}->{field_name};",
        ],
    ),
    dict(
        name="f_setter",
        mixin=["c_mixin_noargs"],
        alias=[
            "c_setter",
            "f_setter_void_scalar",  # for interface
            "c_setter_void_scalar",
        ],
        f_arg_call=[],
        c_call=[
            "// skip call c_setter",
        ],
    ),

    dict(
        name="f_setter_native_scalar/*",
        mixin=[
            "c_setter_arg",
        ],
        alias=[
            "c_setter_native_scalar",
            "c_setter_native_*",
        ],
        f_arg_call=["{c_var}"],
        # f_setter is intended for the function, this is for an argument.
        # c_setter_native_scalar
        # c_setter_native_*
        c_post_call=[
            "{CXX_this}->{field_name} = val;",
        ],
    ),
    dict(
        # Similar to calling a function, but save field pointer instead.
        name="f_getter_native_*_cdesc_pointer",
        mixin=[
            "c_getter",
            "f_mixin_function_cdesc",
            "f_mixin_function_native_cdesc_pointer",
        ],
        # See f_function_native_*_cdesc_pointer  f_mixin_function_native_cdesc_pointer
        
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
        name="f_getter_string_scalar_cdesc_allocatable",
        mixin=[
            "f_mixin_function_cdesc",
            "f_mixin_char_cdesc_allocate",
            "c_getter",
            "c_mixin_out_character_cdesc",
        ],
        alias=[
            "c_getter_string_scalar_cdesc_allocatable",
        ],
##       c_helper="ShroudStrToArray",
        
        c_post_call=[
            "{c_var_cdesc}->addr.base = {CXX_this}->{field_name}.data();",
            "{c_var_cdesc}->type = 0; // SH_CHAR;",
            "{c_var_cdesc}->elem_len = {CXX_this}->{field_name}.size();",
            "{c_var_cdesc}->rank = 0;"
        ],
        c_return_type="void",  # Convert to function.
    ),
    dict(
        # Extract meta data and pass to C.
        # Create std::string from Fortran meta data.
        name="f_setter_string_scalar_buf",
        mixin=[
            "f_mixin_in_character_buf",            
            "c_setter_arg",
            "c_mixin_in_character_buf",
        ],
        alias=[
            "c_setter_string_scalar_buf",
        ],
        c_post_call=[
            "{CXX_this}->{field_name} = std::string({c_var},\t {c_var_len});",
        ],
    ),
    
    ########################################
    # CFI - Further Interoperability with C
    ########################################
    # char arg
    dict(
        # XXX - needs a better name. function/arg
        # Function which return char * or std::string.
        name="c_mixin_function_character",
        iface_header=["ISO_Fortran_binding.h"],
        c_arg_decl=[
            "CFI_cdesc_t *{c_var_cfi}",
        ],
        i_arg_decl=[
            "XXX-unused character(len=*), intent({f_intent}) :: {c_var}",
        ],
        i_arg_names=["{c_var}"],
        c_temps=["cfi"],
    ),
    dict(
        # Add allocatable attribute to declaration.
        # f_function_char_scalar_cfi_allocatable
        # f_function_char_*_cfi_allocatable
        name="f_function_char_*_cfi_allocatable",
        mixin=[
            "c_mixin_function_character",
        ],
        alias=[
            "f_function_char_scalar_cfi_allocatable",
        ],
        f_need_wrapper=True,
        f_arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        f_arg_call=["{f_var}"],  # Pass result as an argument.

        c_return_type="void",  # Convert to function.
        i_arg_names=["{c_var}"],
        i_arg_decl=[        # replace mixin
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
        # Add allocatable attribute to declaration.
        # f_function_char_scalar_cfi_pointer
        # f_function_char_*_cfi_pointer
        name="f_function_char_scalar_cfi_pointer",
        mixin=[
            "c_mixin_function_character",
        ],
        alias=[
            "f_function_char_*_cfi_pointer",
        ],
        f_need_wrapper=True,
        f_arg_decl=[
            "character(len=:), pointer :: {f_var}",
        ],
        f_arg_call=["{f_var}"],  # Pass result as an argument.

        c_return_type="void",  # Convert to function.
        i_arg_names=["{c_var}"],
        i_arg_decl=[        # replace mixin
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
        c_local=["cptr", "fptr", "cdesc", "len", "err"],
    ),
    
    dict(
        # Character argument which use CFI_desc_t.
        name="c_mixin_arg_character_cfi",
        iface_header=["ISO_Fortran_binding.h"],
        cxx_local_var="pointer",
        c_arg_decl=[
            "CFI_cdesc_t *{c_var_cfi}",
        ],
        i_arg_decl=[
            "character(len=*), intent({f_intent}) :: {c_var}",
        ],
        i_arg_names=["{c_var}"],
        c_pre_call=[
            "char *{cxx_var} = "
            "{cast_static}char *{cast1}{c_var_cfi}->base_addr{cast2};",
        ],
        c_temps=["cfi"],
    ),
    dict(
        # Native argument which use CFI_desc_t.
        name="c_mixin_arg_native_cfi",
        iface_header=["ISO_Fortran_binding.h"],
        cxx_local_var="pointer",
        c_arg_decl=[
            "CFI_cdesc_t *{c_var_cfi}",
        ],
        i_arg_decl=[
            "{f_type}, intent({f_intent}) :: {c_var}{f_assumed_shape}",
        ],
        i_module_line="iso_c_binding:{f_kind}",
        i_arg_names=["{c_var}"],
#        c_pre_call=[
#            "{c_type} *{cxx_var} = "
#            "{cast_static}{c_type} *{cast1}{c_var_cfi}->base_addr{cast2};",
#        ],
        c_temps=["cfi", "extents", "lower"],
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
        c_local=["cptr", "fptr", "cdesc", "err"],
    ),

    ########################################

    dict(
        # f_in_native_*_cfi
        # f_inout_native_*_cfi
        name="f_in/inout_native_*_cfi",
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
        name="f_in_char_*_cfi",
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
        name="f_out_char_*_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        c_helper="ShroudStrBlankFill",
        c_post_call=[
            "ShroudStrBlankFill({cxx_var}, {c_var_cfi}->elem_len);"
        ],
    ),
    dict(
        name="f_inout_char_*_cfi",
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
        # Copy result into caller's buffer.
        name="f_function_char_*_cfi_copy",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        f_arg_call=["{f_var}"],
        f_need_wrapper=True,

        # Copy result into caller's buffer.
        # c_function_char_*_cfi_copy
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
        c_return_type="void",  # Convert to function.
    ),
    dict(
        # Change function result into an argument
        # F_string_result_arg
        name="f_function_char_*_cfi_arg",
        mixin=[
            "f_function_char_*_cfi_copy",
#            "c_mixin_arg_character_cfi",
            "f_mixin_function_cfi_character-arg",
        ],
    ),
##-    dict(
##-        name="c_function_char_*_cfi_pointer",
##-        mixin=[
##-            "c_mixin_function_character",
##-        ],
##-        c_return_type="void",  # Convert to function.
##-        i_arg_names=["{c_var}"],
##-        i_arg_decl=[        # replace mixin
##-            "character(len=:), intent({f_intent}), pointer :: {c_var}",
##-        ],
##-        cxx_local_var=None,  # replace mixin
##-        c_pre_call=[],         # replace mixin
##-        c_post_call=[
##-# CFI_index_t nbar[1] = {3};
##-#  CFI_CDESC_T(1) c_p;
##-#  CFI_establish((CFI_cdesc_t* )&c_p, bar, CFI_attribute_pointer, CFI_type_int,
##-#                nbar[0]*sizeof(int), 1, nbar);
##-#  CFI_setpointer(f_p, (CFI_cdesc_t *)&c_p, NULL);
##-
##-            # CFI_index_t nbar[1] = {3};
##-            "int {c_local_err};",
##-            "if ({cxx_var} == {nullptr}) {{+",
##-            "{c_local_err} = CFI_setpointer(\t{c_var_cfi},\t {nullptr},\t {nullptr});",
##-            "-}} else {{+",
##-            "CFI_CDESC_T(0) {c_local_fptr};",
##-            "CFI_cdesc_t *{c_local_cdesc} = {cast_reinterpret}CFI_cdesc_t *{cast1}&{c_local_fptr}{cast2};",
##-            "void *{c_local_cptr} = {cxx_nonconst_ptr};",
##-            "size_t {c_local_len} = {stdlib}strlen({cxx_var});",
##-            "{c_local_err} = CFI_establish({c_local_cdesc},\t {c_local_cptr},"
##-            "\t CFI_attribute_pointer,\t CFI_type_char,"
##-            "\t {c_local_len},\t 0,\t {nullptr});",
##-            "if ({c_local_err} == CFI_SUCCESS) {{+",
##-            "{c_var_cfi}->elem_len = {c_local_cdesc}->elem_len;",  # set assumed-length
##-            "{c_local_err} = CFI_setpointer(\t{c_var_cfi},\t {c_local_cdesc},\t {nullptr});",
##-            "-}}",
##-            "-}}",            
##-        ],
##-        c_local=["cptr", "fptr", "cdesc", "len", "err"],
##-    ),
    
    ########################################
    # char **
    dict(
        name="f_in_char_**_cfi",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        i_arg_decl=[
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
        c_temps=["cfi", "len", "size"],

        c_helper="ShroudStrArrayAlloc ShroudStrArrayFree",
        cxx_local_var="pointer",
        c_post_call=[
            "ShroudStrArrayFree({cxx_var}, {c_var_size});",
        ],
    ),

    ########################################
    # std::string
    dict(
        # f_in_string_scalar_cfi
        # f_in_string_*_cfi
        # f_in_string_&_cfi
        name="f_in_string_scalar/*/&_cfi",
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
        c_local=["trim"],
    ),
    dict(
        # f_out_string_*_cfi
        # f_out_string_&_cfi
        name="f_out_string_*/&_cfi",
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
        # f_inout_string_*_cfi
        # f_inout_string_&_cfi
        name="f_inout_string_*/&_cfi",
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
        c_local=["trim"],
    ),
    dict(
        name="f_mixin_function_string_scalar_cfi_copy",
        mixin=[
            "c_mixin_arg_character_cfi",
        ],
        # XXX - avoid calling C directly since the Fortran function
        # is returning an CHARACTER, which CFI can not do.
        # Fortran wrapper passed function result to C which fills it.
        f_need_wrapper=True,
        f_arg_call=["{f_var}"],
        
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
        c_return_type="void",  # Convert to function.
    ),
    # f_function_string_scalar_cfi_copy
    # f_function_string_*_cfi_copy
    # f_function_string_&_cfi_copy
    dict(
        name="f_function_string_scalar/*/&_cfi_copy",
        mixin=[
            "f_mixin_function_string_scalar_cfi_copy",
        ],
    ),
    dict(
        name="f_function_string_scalar_cfi_arg",
        mixin=[
            "f_mixin_function_string_scalar_cfi_copy",
            "f_mixin_function_cfi_character-arg",
        ],
        alias=[
            "f_function_string_&_cfi_arg",
        ],
    ),

    # XXX - consolidate with c_function_*_cfi_pointer?
    # XXX - via a helper to get address and length of string
    dict(
        name="f_shared_function_string_*_cfi_pointer",
        mixin=[
            "c_mixin_function_character",
        ],
        alias=[
            "f_function_string_scalar/*/&_cfi_pointer",
            "f_function_string_scalar/*/&_cfi_pointer_caller/library",
        ],

        # XXX - avoid calling C directly since the Fortran function
        # is returning an pointer, which CFI can not do.
        # Fortran wrapper passed function result to C which fills it.
        f_need_wrapper=True,
        f_arg_decl=[
            "character(len=:), pointer :: {f_var}",
        ],
        f_arg_call=["{f_var}"],
        
        c_return_type="void",  # Convert to function.
        i_arg_names=["{c_var}"],
        i_arg_decl=[        # replace mixin
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
        c_local=["cptr", "fptr", "cdesc", "len", "err"],
    ),

    
    # similar to f_char_scalar_allocatable
    dict(
        name="f_mixin_function_string_scalar_cfi_allocatable",
        # XXX - avoid calling C directly since the Fortran function
        # is returning an allocatable, which CFI can not do.
        # Fortran wrapper passed function result to C which fills it.
        f_need_wrapper=True,
        f_arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        f_arg_call=["{f_var}"],
    ),
    # std::string * function()
    dict(
        # f_function_string_*_cfi_allocatable
        # f_function_string_&_cfi_allocatable
        # f_function_string_scalar_cfi_allocatable_caller
        # f_function_string_*_cfi_allocatable_caller
        # f_function_string_&_cfi_allocatable_caller
        # f_function_string_scalar_cfi_allocatable_library
        # f_function_string_*_cfi_allocatable_library
        # f_function_string_&_cfi_allocatable_library
        name="f_function_string_*/&_cfi_allocatable",
        mixin=[
            "f_mixin_function_string_scalar_cfi_allocatable",
            "c_mixin_function_character",
        ],
        alias=[
            "f_function_string_scalar/*/&_cfi_allocatable_caller/library",
        ],
        i_arg_decl=[
            "character(len=:), intent({f_intent}), allocatable :: {c_var}",
        ],
        c_return_type="void",  # Convert to function.
        i_arg_names=["{c_var}"],
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
    # std::string & function()
    dict(
        name="f_function_string_scalar_cfi_allocatable",
        mixin=[
            "f_mixin_function_string_scalar_cfi_allocatable",
            "c_mixin_function_character",
        ],
        i_arg_names=["{c_var}"],
        i_arg_decl=[        # replace mixin
            "character(len=:), intent({f_intent}), allocatable :: {c_var}",
        ],
        c_return_type="void",  # convert to function
        cxx_local_var=None,  # replace mixin
        c_pre_call=[],         # replace mixin
        c_post_call=[
            "int SH_ret = CFI_allocate({c_var_cfi}, \t(CFI_index_t *) 0, \t(CFI_index_t *) 0, \t{cxx_var}.length());",
            "if (SH_ret == CFI_SUCCESS) {{+",
            "{stdlib}memcpy({c_var_cfi}->base_addr, \t{cxx_var}.data(), \t{c_var_cfi}->elem_len);",
            "-}}",
        ],
    ),

    ##########
    # Pass a cdesc down to describe the memory and a capsule to hold the
    # C++ array. Copy into Fortran argument.
    # [see also f_out_vector_&_cdesc_allocatable_targ_string_scalar]
    dict(
        name="f_out_string_**_cdesc_copy",
        mixin=[
            "f_mixin_str_array",
            "c_mixin_out_array_cdesc",
        ],
        alias=[
            "c_out_string_**_cdesc_copy",
        ],
        c_helper="array_string_out",
        c_pre_call=[
            "std::string *{cxx_var};"
        ],
        c_arg_call=["&{cxx_var}"],
        c_post_call=[
            "{chelper_array_string_out}(\t{c_var_cdesc},\t {cxx_var}, {c_array_size2});",
        ],

    ),

    dict(
        # std::string **arg+intent(out)+dimension(size)
        # Returning a pointer to a string*. However, this needs additional mapping
        # for the C interface.  Fortran calls the +api(cdesc) variant.
        name="f_out_string_**_copy",
        alias=[
            "c_out_string_**_copy",
            "f_out_string_**_cfi_copy",
        ],
        notimplemented=True,
    ),

    ##########
    # Pass a cdesc down to describe the memory and a capsule to hold the
    # C++ array. Allocate in fortran, fill from C.
    # [see also f_out_vector_&_cdesc_allocatable_targ_string_scalar]
    dict(
        name="f_out_string_**_cdesc_allocatable",
        alias=[
            "c_out_string_**_cdesc_allocatable",
        ],
        mixin=["c_mixin_out_array_cdesc"],
        f_arg_decl=[
            "character({f_char_len}), intent(out), allocatable, target :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["C_LOC"]),
        f_declare=[
            "type({F_array_type}) :: {c_var_cdesc}",
            "type({F_array_type}) :: {c_var_out}",
        ],
        f_arg_call=["{c_var_out}"],
        f_post_call=[
            "{c_var_cdesc}%size = {c_var_out}%size;",
            "{c_var_cdesc}%elem_len = {c_var_out}%elem_len;",
            "allocate({f_char_type}{f_var}({c_var_cdesc}%size))",
            "{c_var_cdesc}%cxx%addr = C_LOC({f_var});",
            "{c_var_cdesc}%base_addr = C_LOC({f_var});",
            "call {fhelper_array_string_allocatable}({c_var_cdesc}, {c_var_out})",
        ],
        f_temps=["cdesc", "out"],
        f_helper="array_string_allocatable array_context",
        c_helper="array_string_allocatable array_string_out_len",
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
            "{c_var_cdesc}->elem_len = {chelper_array_string_out_len}({cxx_var}, {c_var_cdesc}->size);",
            "-}}",
            "{c_var_cdesc}->cxx.addr  = {cxx_var};",
            "{c_var_cdesc}->cxx.idtor = 0;",  # XXX - check ownership
        ],
    ),

    dict(
        # std::string **arg+intent(out)+dimension(size)+deref(allocatable)
        # Returning a pointer to a string*. However, this needs additional mapping
        # for the C interface.  Fortran calls the +api(cdesc) variant.
        name="f_out_string_**_allocatable",
        alias=[
            "c_out_string_**_allocatable",
            "f_out_string_**_cfi_allocatable",
        ],
        notimplemented=True,
    ),

    ########################################
    # native
    dict(
        name="f_out_native_*_cfi_allocatable",
    ),
    dict(
        # Set Fortran pointer to point to cxx_var
        name="f_out_native_**_cfi_allocatable",
        mixin=[
            "c_mixin_arg_native_cfi",
            "c_mixin_native_cfi_allocatable",
        ],
        i_arg_decl=[
            "{f_type}, intent({f_intent}), allocatable :: {c_var}{f_assumed_shape}",
        ],
        c_pre_call=[
            "{c_const}{c_type} * {cxx_var};",
        ],
        c_arg_call=["&{cxx_var}"],
    ),
    dict(
        # Set Fortran pointer to point to cxx_var
        name="f_out_native_**_cfi_pointer",
        mixin=[
            "c_mixin_arg_native_cfi",
            "c_mixin_native_cfi_pointer",
        ],
        i_arg_decl=[
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
        # Convert to subroutine and pass result as an argument.
        # Return an allocated copy of data.
        mixin=[
            "c_mixin_arg_native_cfi",
            "c_mixin_native_cfi_allocatable",  # c_post_call
        ],
        f_arg_decl=[
            "{f_type}, allocatable :: {f_var}{f_assumed_shape}",
        ],
        f_arg_call=["{f_var}"],

        i_arg_decl=[
            "{f_type}, intent({f_intent}), allocatable :: {c_var}{f_assumed_shape}",
        ],

        cxx_local_var="result",
        c_return_type="void",  # Convert to function.
    ),

    dict(
        # Pass result as an argument to C wrapper.
        name="f_function_native_*_cfi_pointer",
        # Convert to subroutine and pass result as an argument.
        # Return Fortran pointer to data.
        mixin=[
            "c_mixin_arg_native_cfi",
            "c_mixin_native_cfi_pointer",  # c_post_call
        ],
        f_arg_decl=[
            "{f_type}, pointer :: {f_var}{f_assumed_shape}",
        ],
        f_pre_call=[
            "nullify({f_var})",
        ],
        f_arg_call=["{f_var}"],

        i_arg_decl=[
            "{f_type}, intent({f_intent}), pointer :: {c_var}{f_assumed_shape}",
        ],

        cxx_local_var="result",
        c_return_type="void",  # Convert to function.
    ),

]
