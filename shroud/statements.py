# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
"""
from __future__ import print_function
from __future__ import absolute_import

from . import error
from . import whelpers
from .util import wformat

import collections
import json
import yaml

try:
    # XXX - python 3.7
    import importlib.resources
    def read_json_resource(name):
        fp = importlib.resources.open_binary('shroud', name)
        stmts = json.load(fp)
        return stmts
    def read_yaml_resource(name):
        fp = importlib.resources.open_binary('shroud', name)
        stmts = yaml.safe_load(fp)
        return stmts
except ImportError:
    from pkg_resources import resource_filename
    def read_json_resource(name):
        fp = open(resource_filename('shroud', name), 'rb')
        #stmts = json._load(fp)
        # Use pyYAML to load json to avoid unicode issues.
        stmts = yaml.safe_load(fp)
        return stmts
    def read_yaml_resource(name):
        fp = open(resource_filename('shroud', name), 'rb')
        stmts = yaml.safe_load(fp)
        return stmts

from . import util

from collections import OrderedDict

# The dictionary of c and fortran statements.
fc_dict = OrderedDict() # dictionary of Scope of all expanded fc_statements.


class BindArg(object):
    """
    Information to create wrapper (binding) for a result or argument.

    Use get functions to access instance.
        r_bind = get_func_bind(node, wlang)
        arg_bind = get_arg_bind(node, arg, "f")

    The get_func_find function hides the "+result" implementation
    detail used for function results."
    """
    def __init__(self):
        self.stmt = None
        self.meta = None
        self.fmtdict = None
        self.fstmts = None  # fstatements from YAML file

def fetch_var_bind(node, wlang):
    bindarg = node._bind.setdefault(wlang, BindArg())
    if bindarg.meta is None:
        bindarg.meta = collections.defaultdict(lambda: None)
    return bindarg

def fetch_func_bind(node, wlang):
    bind = node._bind.setdefault(wlang, {})
    bindarg = bind.get("+result")
    if bindarg is None:
        bindarg = BindArg()
        bindarg.meta = collections.defaultdict(lambda: None)
        bind["+result"] = bindarg
    return bindarg

def fetch_arg_bind(node, arg, wlang):
    bind = node._bind.setdefault(wlang, {})
    # XXX - better to turn off wrapping when 'Argument must have name'
    name = arg.declarator.user_name or arg.declarator.arg_name
    bindarg = bind.get(name)
    if bindarg is None:
        bindarg = BindArg()
        bindarg.meta = collections.defaultdict(lambda: None)
        bind[name] = bindarg
    return bindarg

def fetch_typedef_bind(node, wlang):
    # Similar to fetch_arg_bind, except arg is derived direcly from node.
    bind = node._bind.setdefault(wlang, {})
    arg = node.ast
    # XXX - better to turn off wrapping when 'Argument must have name'
    name = arg.declarator.user_name or arg.declarator.arg_name
    bindarg = bind.get(name)
    if bindarg is None:
        bindarg = BindArg()
        bindarg.meta = collections.defaultdict(lambda: None)
        bind[name] = bindarg
    return bindarg

def fetch_name_bind(bind, wlang, name):
    """
    bind - dictionary index by wlang
    """
    bind = bind.setdefault(wlang, {})
    bindarg = bind.setdefault(name, BindArg())
    if bindarg.meta is None:
        bindarg.meta = collections.defaultdict(lambda: None)
    return bindarg

def get_var_bind(node, wlang):
    return node._bind[wlang]

def get_func_bind(node, wlang):
    return node._bind[wlang]["+result"]

def get_arg_bind(node, arg, wlang):
    """
    node - ast.FunctionNode
    arg  - declast.Declaration
    """
    name = arg.declarator.user_name or arg.declarator.arg_name
    return node._bind[wlang][name]

def set_bind_fmtdict(bind, parent):
    """Set the BindArg.fmtdict field."""
    if not bind.fmtdict:
        bind.fmtdict = util.Scope(parent)
    return bind.fmtdict

######################################################################

def collect_arg_typemaps(arg):
    """Return list of typemaps used by argument.
    Templates will provide multiple typemaps.
    """
    if arg.template_arguments:
        typemaps = []
        for targ in arg.template_arguments:
            typemaps.append(targ.typemap)
    else:
        typemaps = [arg.typemap]
    return typemaps

def find_abstract_declarator(arg):
    """Look up the c_statements for an argument.
    If the argument type is a template, look for
    template specialization.

    All function pointers are mapped to "procedure" without
    any reference to the return type.
       - decl: void callback_ptr(int *(*get)(void));
       - decl: void callback_ptr(int  (*get)(void));
    Are the same.
    Funtion pointers from a typedef already have the correct typemap.

    Args:
        arg - declast.Declaration
    """
    declarator = arg.declarator
    if declarator.is_function_pointer():
        decl = ["procedure"]
        abstract = ""
    else:
        decl = [declarator.typemap.sgroup]
        abstract = declarator.get_abstract_declarator()

    if arg.template_arguments:
        decl.append("<")
        for targ in arg.template_arguments:
            decl.append(targ.declarator.typemap.sgroup)
            decl.append(targ.declarator.get_abstract_declarator())
            decl.append(",")
        decl[-1] = ">"
    decl.append(abstract)
    return "".join(decl)

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
        stmt = fc_dict.get("{}_mixin_unknown".format(path[0]))
        error.cursor.warning("Unknown statement: {}".format(name))
    return stmt

def lookup_c_function_stmt(node):
    """Lookup the C statements for a function."""
    ast = node.ast
    declarator = ast.declarator
    subprogram = declarator.get_subprogram()
    r_meta = get_func_bind(node, "c").meta
    sintent = r_meta["intent"]
    if subprogram == "subroutine":
        # intent will be "subroutine", "dtor", "setter"
        stmts = ["c", sintent, r_meta["operator"], r_meta["custom"]]
        result_stmt = lookup_fc_stmts(stmts)
    else:
        # intent will be "function", "ctor", "getter"
        stmts = ["c", sintent, r_meta["abstract"],
                 r_meta["api"], r_meta["deref"], r_meta["owner"],
                 r_meta["operator"], r_meta["custom"]]
    result_stmt = lookup_fc_stmts(stmts)
    return result_stmt

def lookup_f_function_stmt(node):
    """Lookup the Fortran statements for a function."""
    ast = node.ast
    declarator = ast.declarator
    subprogram = declarator.get_subprogram()
    r_meta = get_func_bind(node, "f").meta
    sintent = r_meta["intent"]
    if subprogram == "subroutine":
        # intent will be "subroutine", "dtor", "setter"
        stmts = ["f", sintent, r_meta["operator"], r_meta["custom"]]
        result_stmt = lookup_fc_stmts(stmts)
    else:
        # intent will be "function", "ctor", "getter"
        stmts = ["f", sintent, r_meta["abstract"],
                 r_meta["api"], r_meta["deref"], r_meta["owner"],
                 r_meta["operator"], r_meta["custom"]]
    result_stmt = lookup_fc_stmts(stmts)
    return result_stmt

def lookup_c_arg_stmt(node, arg):
    """Lookup the C statements for an argument."""
    c_meta = get_arg_bind(node, arg, "c").meta
    sapi = c_meta["api"]
    stmts = ["c", c_meta["intent"], c_meta["abstract"],
             sapi, c_meta["deref"], c_meta["owner"]]
    arg_stmt = lookup_fc_stmts(stmts)
    return arg_stmt

def lookup_f_arg_stmt(node, arg):
    """Lookup the Fortran statements for an argument."""
    c_meta = get_arg_bind(node, arg, "f").meta
    sapi = c_meta["api"]
    if c_meta["hidden"]:
        sapi = "hidden"
    stmts = ["f", c_meta["intent"], c_meta["abstract"],
             sapi, c_meta["deref"], c_meta["owner"]]
    arg_stmt = lookup_fc_stmts(stmts)
    return arg_stmt

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
        path   - list of path components ["c"]
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


def apply_fmtdict_from_stmts(bind):
    """Apply fmtdict field from statements.
    Should be done after other defaults are set to
    allow the user to override any value.

    fmtdict:
       f_var: "{F_string_result_as_arg}"
       i_var: "{F_string_result_as_arg}"
       c_var: "{F_string_result_as_arg}"
    """
    stmts = bind.stmt
    fmt = bind.fmtdict
    
    if stmts.fmtdict is not None:
        for key, value in stmts.fmtdict.items():
            setattr(fmt, key, wformat(value, fmt))
    
def compute_return_prefix(arg):
    """Compute how to access variable: dereference, address, as-is"""
    if arg.declarator.is_reference():
        # Convert a return reference into a pointer.
        return "&"
    else:
        return ""


def update_fc_statements_for_language(language, user):
    """Preprocess statements for lookup.

    Update statements for c or c++.
    Fill in cf_tree.

    user:
      file: list of file names
      list: list of dictionaries

    Parameters
    ----------
    language : str
        "c" or "c++"
    user     : dict from YAML file
    """
    stmts = read_json_resource('fc-statements.json')
    fc_statements.extend(stmts)

    if "extend" in user:
        fc_statements.extend(user["extend"])

    update_for_language(fc_statements, language)
    process_mixin(fc_statements, default_stmts, fc_dict)

# Dictionary of statment fields which have changed name.
deprecated_fields = dict(
    c=dict(
        # v0.13 changes
        c_arg_decl="c_prototype",
        c_helper="helper",
        arg_call="c_arg_call",
        pre_call="c_pre_call",
        call="c_call",
        post_call="c_post_call",
        final="c_final",
        ret="c_return",
        temps="c_temps",
        local="c_local",
        f_arg_decl="i_dummy_decl",
        f_result_decl="i_result_decl",
        f_result_var="i_result_var",
        f_module="i_module",
        f_import="i_import",
        # develop changes
        i_arg_decl="i_dummy_decl",
        i_arg_names="i_dummy_arg",
    ),
    f=dict(
        # v0.13 changes
        c_helper="helper",
        f_helper="helper",
        need_wrapper="f_need_wrapper",
        arg_name="f_dummy_arg",
        arg_decl="f_dummy_decl",
        arg_c_call="f_arg_call",
        declare="f_local_decl",
        pre_call="f_pre_call",
        call="f_call",
        post_call="f_post_call",
        result="f_result_var",
        temps="f_temps",
        local="f_local",
        # develop changes
        c_arg_decl="c_prototype",
        f_arg_decl="f_dummy_decl",
        f_arg_name="f_dummy_arg",
        i_arg_decl="i_dummy_decl",
        i_arg_names="i_dummy_arg",
        f_declare="f_local_decl",
    )
)

def check_for_deprecated_names(stmt):
    """
    Report any deprecated keys.
    Update stmt with new key name.

    Used to update from version 0.13.
    The group must have a name field.

    Parameters
    ----------
    stmt : dictionary
    """
    lang = '#'
    if "name" in stmt:
        lang = stmt["name"][0]
    elif "alias" in stmt:
        for alias in stmt["alias"]:
            if alias[0] == "#":
                continue
            lang = alias[0]
            break
    check_stmt_for_deprecated_names(lang, stmt)

def check_stmt_for_deprecated_names(lang, stmt):
    """
    Also used with fstatements in YAML file.

    Parameters
    ----------
    lang : 'c' or 'f'
    stmt : dictionary
    """
    deprecated = deprecated_fields.get(lang)
    if deprecated is None:
        return
    keys = list(stmt.keys()) # dictionary is changing so snapshot keys
    for key in keys:
        if key in deprecated:
            newkey = deprecated[key]
            error.cursor.warning("field {} is deprecated, changed to {}".format(
                key, newkey))
            stmt[newkey] = stmt.pop(key)


def post_mixin_check_statement(name, stmt):
    """check for consistency.
    Called after mixin are applied.
    This makes it easer to a group to change one of
    c_prototype, i_dummy_decl, i_dummy_arg.
    """
    parts = name.split("_")
    lang = parts[0]
    intent = parts[1]

    if lang == "f" and intent not in ["mixin", "setter"]:
        c_prototype = stmt.get("c_prototype", None)
        i_dummy_decl = stmt.get("i_dummy_decl", None)
        i_dummy_arg = stmt.get("i_dummy_arg", None)
        if (c_prototype is not None or
            i_dummy_decl is not None or
            i_dummy_arg is not None):
            err = False
            missing = []
            for field in ["c_prototype", "i_dummy_decl", "i_dummy_arg"]:
                fvalue = stmt.get(field)
                if fvalue is None:
                    err = True
                    missing.append(field)
                elif not isinstance(fvalue, list):
                    err = True
                    error.cursor.warning("{} must be a list.".format(field))
#            if missing:
#                error.cursor.warning("c_prototype, i_dummy_decl and i_dummy_arg must all exist together.\n" +
#                                     "Missing {}.".format(", ".join(missing)))
#                err = True
            if not err:
                length = len(c_prototype)
                if any(len(lst) != length for lst in [i_dummy_decl, i_dummy_arg]):
                    error.cursor.warning(
                        "c_prototype, i_dummy_decl and i_dummy_arg "
                        "must all be same length. Used {}, {}, {}."
                        .format(len(c_prototype), len(i_dummy_decl), len(i_dummy_arg)))

##-    if lang in ["f", "fc"]:
##-        # Default f_dummy_arg is often ok.
##-        f_dummy_arg = stmt.get("f_dummy_arg", None)
##-        f_dummy_decl = stmt.get("f_dummy_decl", None)
##-        if f_dummy_arg is not None or f_dummy_decl is not None:
##-            err = False
##-            for field in ["f_dummy_arg", "f_dummy_decl"]:
##-                fvalue = stmt.get(field)
##-                if fvalue is None:
##-                    err = True
##-                    print("Missing", field, "in", name)
##-                elif not isinstance(fvalue, list):
##-                    err = True
##-                    print(field, "must be a list in", name)
##-            if (f_dummy_arg is None or
##-                f_dummy_decl is None):
##-                print("f_dummy_arg and f_dummy_decl must both exist")
##-                err = True
##-            if err:
##-                raise RuntimeError("Error with fields")
##-            if len(f_dummy_arg) != len(f_dummy_decl):
##-                raise RuntimeError(
##-                    "f_dummy_arg and f_dummy_decl "
##-                    "must all be same length in {}".format(name))

def append_mixin(stmt, mixin):
    """Append each list from mixin to stmt.
    """
    for key, value in mixin.items():
        if key in ["alias", "base", "mixin", "name"]:
            pass
        elif isinstance(value, list):
            if key == "notes":
                # notes do not accumulate like other fields.
                continue
            elif key == "mixin_names":
                # Indent nested mixins
                value = ["  " + val for val in value]
            if key not in stmt:
                stmt[key] = []
            if False:#True:
                # Report the mixin name for debugging
                if "name" in mixin:
                    stmt[key].append("# " + mixin["name"])
                else:
                    stmt[key].append("# append")
            stmt[key].extend(value)
        elif isinstance(value, dict):
            if key not in stmt:
                stmt[key] = {}
            append_mixin(stmt[key], value)
        else:
            stmt[key] = value

valid_intents = [
    "in", "out", "inout",
    "implied",
    "none",    # used with function pointers
    "mixin",
    "function", "subroutine",
    "getter", "setter",
    "ctor", "dtor",
    "descr",
    "helper",
]

def process_mixin(stmts, defaults, stmtdict):
    """Return a dictionary of all statements
    names and aliases will be expanded (ex in/out/inout)
    Each dictionary will have a unique name.

    Add into dictionary.
    Add as aliases
    Add mixin into dictionary

    alias=[
        "c_function_native_*_allocatable",
        "c_function_native_*_raw",
    ],

    Set an index field for each statement.
    variants (ex in/out) and aliases all have the same index.
    """
    # Apply base and mixin
    # This allows mixins to propagate
    # i.e. you can mixin a group which itself has a mixin.
    cursor = error.cursor
    cursor.push_phase("Check statements")
    stmt_cursor = cursor.push_statement()
    mixins = OrderedDict()
    index = 0
    for stmt in stmts:
        stmt_cursor.stmt = stmt
        node = None
        tmp_node = {}
        tmp_name = None
        name = None
        aliases = []
        intent = None
        check_for_deprecated_names(stmt)
        if "alias" in stmt:
            # name is not allowed"
            aliases = [ alias for alias in stmt["alias"] if alias[0] != "#"]
            # XXX - first alias used for lang
            if len(aliases) == 0:
                continue
            tmp_name = aliases[0]
        if "name" in stmt:
            name = stmt["name"]
            tmp_name = name
            if tmp_name[0] == "#":
                continue
        if not tmp_name:
            cursor.warning("Statement must have name or alias")
            continue

        parts = tmp_name.split("_")
        if len(parts) < 2:
            cursor.warning("Statement name is too short, must include language and intent.")
            continue
        lang = parts[0]
        intent = parts[1]
        if intent == "mixin":
            if name is None:
                cursor.warning("Intent mixin only allowed in name, not alias.")
                continue
            if "base" in stmt:
                cursor.warning("Intent mixin group should not have 'base' field.")
                continue
                
            if "alias" in stmt:
                cursor.warning("Intent mixin group should not have 'alias' field.")
                continue
            if "append" in stmt:
                cursor.warning("Intent mixin group should not have 'append' field.")
                continue
            tmp_node["name"] = name
            if name in mixins:
                cursor.warning("Statement name '{}' already exists.".format(name))
            else:
                mixins[name] = tmp_node

        # Apply any mixin groups
        if "mixin" in stmt:
            if "base" in stmt:
                print("XXXX - Groups with mixin cannot have a 'base' field ", name)
            tmp_node["mixin_names"] = []
            for mixin in stmt["mixin"]:
                ### compute mixin permutations
                if mixin[0] == "#":
                    continue
                mparts = mixin.split("_", 2)
                tmp_node["mixin_names"].append("  " + mixin)
                if len(mparts) < 2:
                    cursor.warning("Mixin '{}' must have intent 'mixin'.".format(mixin))
                elif mparts[1] != "mixin":
                    cursor.warning("Mixin '{}' must have intent 'mixin'.".format(mixin))
                elif mixin not in mixins:
                    cursor.warning("Mixin '{}' not found.".format(mixin))
                else:
                    append_mixin(tmp_node, mixins[mixin])

        if intent == "mixin":
            append_mixin(tmp_node, stmt)
        else:
            if "append" in stmt:
                append_mixin(tmp_node, stmt["append"])
            # Replace any mixin values
            tmp_node.update(stmt)

        post_mixin_check_statement(tmp_name, tmp_node)
        tmp_node["index"] = str(index)
        index += 1

        if intent not in valid_intents:
            cursor.warning("Invalid intent '{}'.".format(intent))

        # Create the Scope instance.
        if "base" in stmt:
            if stmt["base"] not in stmtdict:
                cursor.warning("Base '{}' not found.".format(stmt["base"]))
            else:
                node = util.Scope(stmtdict[stmt["base"]])
        elif lang not in defaults:
            cursor.warning("Statement does not start with a known language code: '%s'" % lang)
        else:
            node = util.Scope(defaults[lang])
        if not node:
            continue
        node.update(tmp_node)

        if name:
            stmtdict[name] = node
            node.intent = intent
        if aliases:
            # Install with alias name.
            for alias in aliases:
                apart = alias.split("_", 2)
                intent = apart[1]
                anode = util.Scope(node)
                if intent == "mixin":
                    cursor.warning("Mixin not allowed in alias '{}'."
                                   .format(alias))
                elif intent not in valid_intents:
                    cursor.warning("Invalid intent '{}' in alias '{}'."
                                   .format(intent, alias))
                if alias in stmtdict:
                    cursor.warning("Alias '{}' already exists.".format(alias))
                else:
                    anode.name = alias
                    anode.intent = intent
                    stmtdict[alias] = anode
    cursor.pop_statement()
    cursor.pop_phase("Check statements")
#    cursor.check_for_warnings()
    
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

def lookup_fc_helper(name, scope="helper"):
    """Lookup Fortran/C helper.
    If not found, print error and return h_mixin_unknown.
    """
    helper = fc_dict.get("h_helper_" + name)
    if helper is None:
        helper = fc_dict["h_mixin_unknown"]
        error.cursor.warning("No such {} '{}'".format(scope, name))
    return helper

def add_json_fc_helpers(fmt):
    """Format helper entries in JSON file."""
    for key, stmt in fc_dict.items():
        if key.startswith("h_helper"):
            whelpers.apply_fmtdict_from_helpers(stmt, fmt)

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


def write_cf_tree(fp, options):
    """Write out statements tree.

    Parameters
    ----------
    fp : file
    options : Dict
    """
    print_tree_statements(fp, fc_dict, default_stmts, options)
    tree = update_stmt_tree(fc_dict)
    lines = []
    print_tree_index(tree, lines)
    fp.writelines(lines)


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


def print_tree_statements(fp, statements, defaults, options):
    """Print expanded statements.

    Statements may not have all values directly defined since 'base'
    and 'mixin' brings in other values.  This will dump the values as
    used by Shroud.

    Statements
    ----------
    fp : file
    statements : dict
    defaults : dict
    options : dict

    """
    literalinclude = options["literalinclude"]
    # Convert Scope into a dictionary for YAML.
    # Add all non-null values from the default dict.
    yaml.SafeDumper.ignore_aliases = lambda *args : True
    complete = {}
    for name in sorted(statements.keys()):
        root = name.split("_", 1)[0]
        base = defaults[root]
        value = statements[name]
        all = {}
        if literalinclude:
            all["sphinx-start-after"] = name
        for key in base.__dict__.keys():
            if key[0] == "_":
                continue
            if key == "index":
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
        if literalinclude:
            all["sphinx-end-before"] = name
        complete[name] = all
    yaml.safe_dump(complete, fp, sort_keys=False)

    return
    # DEBUG
    # Dump each group to a file
    # This makes it easier to compare one finalized group to another using diff.
    for name, group in complete.items():
        with open(name, "w") as fp:
            yaml.safe_dump(group, fp, sort_keys=False)


# Listed in the order they are used in the wrapper.
# This order is preserved with option --write-statements.
#  Fortran - interface - C

# C Statements.
#  intent      - Set from name.
#  c_arg_call  - List of arguments passed to C/C++ library function.
#
#  c_prototype  - Add C declaration to C wrapper.
#                Empty list is no arguments, None is default argument.
#  c_call       - code to call the function.
#                 Ex. Will be empty for getter and setter.
#  i_dummy_decl - Add Fortran declaration to Fortran wrapper interface block.
#                Empty list is no arguments, None is default argument.
#  i_dummy_arg - Empty list is no arguments
#  i_result_decl - Declaration for function result.
#                  Can be an empty list to override default.
#  i_module    - Add module info to interface block.
CStmts = util.Scope(
    None,
    name="c_default",
    intent=None,
    comments=[],
    notes=[],      # implementation notes
    usage=[],
    mixin_names=[],
    index="X",

    # code fields
    i_dummy_arg=None,
    i_dummy_decl=None,
    i_result_decl=None,
    i_result_var=None,
    # bookkeeping fields
    i_import=None,
    i_module=None,

    # code fields
    c_return_type=None,
    c_prototype=None,
    c_pre_call=[],
    c_arg_call=[],
    c_call=[],
    c_post_call=[],
    c_final=[],      # tested in strings.yaml, part of ownership
    c_return=[],
    # bookkeeping fields
    c_temps=None,
    c_local=None,
    c_need_wrapper=False,

    fmtdict=None,
    helper=[],
    iface_header=[],
    impl_header=[],
    destructor_header=[],
    destructor_name=None,
    destructor=[],
    owner=None,

    notimplemented=False,
)

# Fortran Statements.
FStmts = util.Scope(
    None,
    name="f_default",
    intent=None,
    comments=[],
    notes=[],      # implementation notes
    usage=[],
    mixin_names=[],
    index="X",

    # code fields
    f_dummy_arg=None,
    f_dummy_decl=None,
    f_local_decl=[],
    f_pre_call=[],
    f_arg_call=None,
    f_call=[],
    f_post_call=[],
    f_result_var=None,
    # bookkeeping fields
    f_module=None,
    f_temps=None,
    f_local=None,
    f_need_wrapper=False,
)

# Fortran/C Statements - both sets of defaults.
FStmts.update(CStmts._to_dict())

HStmts = util.Scope(
    None,
    name="h_default",
    notes=[],
    fmtdict={},
    api="",
    c_fmtname="",
    include=[],
    c_include=[],
    cxx_include=[],
    proto_include=[],
    scope="",
    proto="",
    source=[],
    c_source=[],
    cxx_source=[],
    f_fmtname="",
    derived_type=[],
    interface=[],
    f_source=[],
    modules=None,
    dependent_helpers=[],
)

# Define class for nodes in tree based on their first entry.
# c_native_*_in uses 'c'.
default_stmts = dict(
    c=CStmts,
    f=FStmts,
    h=HStmts,
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

fc_statements = []


######################################################################
# templates

class TemplateFormat(object):
    """Used with Scope and format fields.
    to access type information for a template parameter.

    "{targ[0].f_type} arg"
    """
    def __init__(self, decl):
        """
        decl - declast.Declaration
        """
        self.decl = decl

    def __str__(self):
        return str(self.decl)

    def __getattr__(self, name):
        return getattr(self.decl.typemap, name)
    
    @property
    def cxx_T(self):
        return self.decl.get_first_abstract_declarator()

def set_template_fields(ast, fmt):
    """Set the format fields for template arguments.
    Accessed as "{targs[0].cxx_type}"
    """
    fmt.cxx_T = ast.gen_template_argument()
    fmt.targs = [TemplateFormat(targ) for targ in ast.template_arguments]


######################################################################
# baseclass

class BaseClassFormat(object):
    """Used with Scope and format fields.
    to access type information for the baseclass.

    "{baseclass.cxx_type}"
    """
    def __init__(self, cls):
        """
        cls - ast.ClassNode
        """
        self.cls = cls

    def __str__(self):
        return self.cls.typemap.name

    def __repr__(self):
        return "<BasesClassFormat {}>".format(self.cls.typemap.name)

    def __getattr__(self, name):
        return getattr(self.cls.typemap, name)
