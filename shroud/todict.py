# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Convert some data structures into a dictionary.
Useful for debugging and seralizing instances as json.
"""

import json

from . import visitor


def stringify_baseclass(baseclass):
    """Convert baseclass into printable form by converting typemap object"""
    # Replace typename instance with its name.
    pbase = []
    for basetuple in baseclass:
        (access_specifier, ns_name, baseclass) = basetuple
        pbase.append((access_specifier, ns_name, baseclass.typemap.name))
    return pbase

def add_comment(dct, label, name=None):
    """Add a variable which will sort to the top.
    Helps when trying to locate sections in the written JSON file.
    """
    # "<" sorts towards the top.
    key = "<" + label.upper() + ">"
    if name is None:
        dct[key] = "****************************************"
    else:
        dct[key] = str(name) + " ****************************************"

######################################################################

class Helpers:
    def visit_bool(self, node):
        return node

    def visit_int(self, node):
        return node

    def visit_str(self, node):
        return str(node)

    def visit_list(self, node):
        return [self.visit(n) for n in node]

    def visit_dict(self, node):
        return {key: "" if value is None else self.visit(value) for (key, value) in node.items()}

    ######################################################################

    def visit_Scope(self, node):
        d = {}
        skip = "_" + node.__class__.__name__ + "__"  # __name is skipped
        for key, value in node.__dict__.items():
            if not key.startswith(skip):
                d[key] = value
        return d

    def add_visit_fields(self, node, d, fields):
        """Update dict d with fields which must be visited.

        Parameters
        ----------
        node : 
        d : dict
           Dictionary being filled.
        fields: list of str or tuple
            Attribute or node or tuple of attribute, d key.
        """
        for key in fields:
            dkey = self.rename_fields.get(key, key)
            value = getattr(node, key)
            if value:
                d[dkey] = self.visit(value)

######################################################################
    
class ToDict(visitor.Visitor):
    """Convert to dictionary.
    """
    def __init__(self, labelast=False):
        """
        labelast - If True, add an extra _ast entry with class name.
        """
        super(ToDict, self).__init__()
        self.labelast = labelast

    def visit_NoneType(self, node):
        return node

    def visit_bool(self, node):
        return node

    def visit_int(self, node):
        return node

    def visit_str(self, node):
        return str(node)

    def visit_list(self, node):
        return [self.visit(n) for n in node]

    def visit_dict(self, node):
        return {key: "" if value is None else self.visit(value) for (key, value) in node.items()}

    ######################################################################

    def visit_Block(self, node):
        d = {}
        d["stmts"] = self.visit(node.stmts)
        return d
    
    def visit_Ptr(self, node):
        d = dict(ptr=node.ptr)
        add_true_fields(node, d, ["const", "volatile"])
        return d

    def visit_Declarator(self, node):
        d = {}
        self.add_visit_fields(node, d, ["pointer"])
        if node.name:
            d["name"] = node.name
        if node.func:
            d["func"] = self.visit(node.func)
        if node.params is not None:
            d["params"] = self.visit(node.params)
        if node.array:
            d["array"] = self.visit(node.array)
        if node.init is not None:
            d["init"] = node.init
        add_true_fields(node, d,
                        ["func_const",
                         "is_ctor", "is_dtor",
                         "default_name",
                        ])

        if node.typemap.base != "template":
            # Only print name to avoid too much nesting.
            d["typemap_name"] = node.typemap.name

        attrs = {key: value
                 for (key, value) in node.attrs.items()
                 if value is not None}
        if attrs:
            d["attrs"] = attrs

        return d

    def visit_Declaration(self, node):
        d = dict(
            specifier=node.specifier,
            # #- node.array,
        )
        if self.labelast:
            d["_ast"] = node.__class__.__name__
        if node.tag_body:
            self.add_visit_fields(node, d, ["enum_specifier", "class_specifier"])
        add_non_none_fields(node, d, ["template_argument"])
        if node.typemap.base != "template":
            # Only print name to avoid too much nesting.
            d["typemap_name"] = node.typemap.name
        
        add_true_fields(node, d, [
            "const", "volatile",
            "is_ctor", "is_dtor",
        ])
        if len(node.declarators) > 1:
            lst = []
            d["declarators"] = lst
            for d2 in node.declarators:
                lst.append(self.visit(d2))
        elif node.declarator:
            # ctor and dtor have no declarator
            d["declarator"] = self.visit(node.declarator)
        if node.storage:
            d["storage"] = node.storage
        if node.template_arguments:
            lst = []
            for tp in node.template_arguments:
                lst.append(self.visit(tp))
            d["template_arguments"] = lst
        return d

    def visit_Identifier(self, node):
        d = dict(name=node.name)
        if node.args is not None:
            d["args"] = self.visit(node.args)
        return d

    def visit_AssumedRank(self, node):
        d = dict(assumedrank=True)
        return d

    def visit_BinaryOp(self, node):
        d = dict(
            left=self.visit(node.left), op=node.op, right=self.visit(node.right)
        )
        return d

    def visit_UnaryOp(self, node):
        d = dict(op=node.op, node=self.visit(node.node))
        return d

    def visit_ParenExpr(self, node):
        d = dict(node=self.visit(node.node))
        return d

    def visit_Constant(self, node):
        d = dict(constant=node.value)
        return d

    def visit_CXXClass(self, node):
        d = dict(name=node.name)
        self.add_visit_fields(node, d, ["members"])
        if self.labelast:
            d["_ast"] = node.__class__.__name__
            if node.group:
                d["~group"] = self.visit(node.group)
        if node.baseclass:
            d["baseclass"] = stringify_baseclass(node.baseclass)
        return d

    def visit_Namespace(self, node):
        d = dict(name=node.name)
        if self.labelast:
            d["_ast"] = node.__class__.__name__
            if node.group:
                d["~group"] = self.visit(node.group)
        return d

    def visit_EnumValue(self, node):
        if node.value is None:
            d = dict(name=node.name)
        else:
            d = dict(name=node.name, value=self.visit(node.value))
        return d

    def visit_Enum(self, node):
        if node.scope:
            d = dict(name=node.name, scope=node.scope, 
                     members=self.visit(node.members))
        else:
            d = dict(name=node.name, members=self.visit(node.members))
        if self.labelast:
            d["_ast"] = node.__class__.__name__
        return d

    def visit_Struct(self, node):
        d = dict(name=node.name)
        self.add_visit_fields(node, d, ["members"])
        if node.typemap is not None:
            d['typemap_name'] = node.typemap.name
            if node.children:
                d['scope_prefix'] = node.children[0].prefix
        if self.labelast:
            d["_ast"] = node.__class__.__name__
        return d

    def visit_Template(self, node):
        d = dict(
            parameters=self.visit(node.parameters), decl=self.visit(node.decl)
        )
        if self.labelast:
            d["_ast"] = node.__class__.__name__
        return d

    def visit_TemplateParam(self, node):
        d = dict(name=node.name)
        return d

    ######################################################################

    def visit_Scope(self, node):
        # Do not call visit for most members. It slows things down a lot.
        # Instead, have a list of keys which must be visited.
        d = {}
        skip = "_" + node.__class__.__name__ + "__"  # __name is skipped
        for key, value in node.__dict__.items():
            if key in ["targs"]:
                d[key] = self.visit(value)
            elif not key.startswith(skip):
                d[key] = value
        return d

    def visit_TemplateFormat(self, node):
        # Return the properties of TemplateFormat.
        # Avoid repeating all of the typemap fields.
        return dict(
            cxx_T = node.cxx_T,
            typemap_name = node.decl.typemap.name,
        )

    ######################################################################

    def visit_Typemap(self, node):
        # only export non-default values
        d = dict()
        for key, defvalue in node.defaults.items():
            value = getattr(node, key)
            if key == "cxx_instantiation":
                # Only save Typemap names to avoid too much clutter.
                if value:
                    names = {}
                    for key, ntypemap in value.items():
                        names[key] = ntypemap.name
                    d["cxx_instantiation"] = names
            elif key == "ast":
                if value is not None:
                    d["ast"] = self.visit(value)
            elif key == "typedef":
                if value is not None:
                    d["typedef"] = value.name
            else:
                if value is not defvalue:
                    d[key] = value
        return d

    ######################################################################

    def visit_WrapFlags(self, node):
        d = dict()
        add_true_fields(
            node, d, ["fortran", "c", "lua", "python",
#                      "signature_c", "signature_f",
            ]
        )
        return d

    def visit_LibraryNode(self, node):
        d = dict()
        add_true_fields(node, d, [
            "copyright", "cxx_header", "fortran_header",
            "language", "scope"])
        self.add_visit_fields( # TEMP  deal with wrap_namespace
            node, d, [ "fmtdict", "options", "scope_file", ])
        if node.class_map:
            d["class_map"] = sorted(list(node.class_map.keys()))
        node = node.wrap_namespace   # XXXX TEMP kludge
        self.add_visit_fields(
            node,
            d,
            [
                "classes",
                "enums",
                "functions",
                "namespaces",
                "typedefs",
                "variables",
                "wrap",
                "user_fmt",
#                "fmtdict",
#                "options",
#                "scope_file",
            ],
        )
        return d

    def visit_ClassNode(self, node):
        d = dict(
            cxx_header=node.cxx_header,
            name=node.name,
            typemap_name=node.typemap.name,  # Only print name to avoid too much nesting.
            parse_keyword=node.parse_keyword,
        )
        add_comment(d, "class", node.name)
        add_non_none_fields(node, d, ["linenumber"])
        add_true_fields(
            node, d, [
                "name_api",
                "name_instantiation",
                "python",
                "scope",
                "template_parameters",
            ]
        )
        if node.baseclass:
            d["baseclass"] = stringify_baseclass(node.baseclass)
        if node.parse_keyword != node.wrap_as:
            d["wrap_as"] = node.wrap_as
        if node.typedef_map:
            d["typedef_map"] = [ (tup[0].name, tup[1].name) for tup in node.typedef_map ]
        self.add_visit_fields(
            node,
            d,
            [
                "classes",
                "enums",
                "functions",
                "variables",
                "user_fields",
                "user_fmt",
                "fmtdict",
                "options",
                "template_arguments",
                "wrap",
            ],
        )
        return d

    def visit_FunctionNode(self, node):
        d = dict(ast=self.visit(node.ast), decl=node.decl, name=node.name)
        add_comment(d, "function", "{}  {}".format(node.name, node._function_index))
        self.add_visit_fields(
            node,
            d,
            [
                "_PTR_C_CXX_index",
                "_PTR_F_C_index",
                "_bind",
                "_fmtargs",
                "user_fmt",
                "fmtdict",
                "options",
                "template_arguments",
                "fortran_generic",
                "fstatements",
                "splicer",
                "wrap",
                "C_force_wrapper",
            ],
        )
        add_true_fields(
            node,
            d,
            [
                "cxx_template",
                "default_arg_suffix",
                "declgen",
                "doxygen",
                "linenumber",
                "return_this",
                "have_template_args",
                "template_parameters",
                "C_error_pattern",
                "PY_error_pattern",
                "_default_funcs",
                "_generated",
                "_generated_path",
                "_has_default_arg",
                "_nargs",
                "_overloaded",
                "_gen_fortran_generic",
            ],
        )
        if node._orig_node is not None:
            d["_orig_node_index"] = node._orig_node._function_index
#            d["_orig_node_name"] = node._orig_node.name
        if node.options.debug_index:
            add_non_none_fields(
                node,
                d,
                [
                    "_cxx_overload",
                    "_function_index",
                    "_PTR_C_CXX_index",
                    "_PTR_F_C_index",
                ],
            )
        add_optional_true_fields(
            node,
            d,
            [
                "statements",
                # #- 'CXX_subprogram', 'C_subprogram', 'F_subprogram',
                # #- 'CXX_return_type', 'C_return_type', 'F_return_type',
            ],
        )
        if node.gen_headers_typedef:
            # OrderedDict
            d['gen_headers_typedef'] = list(node.gen_headers_typedef.keys())
        if node.struct_parent:
            d["struct_parent"] = node.struct_parent.typemap.name
        if node.struct_members:
            # struct_members are ast.VariableNode, add name instead
            # to avoid a huge dump.
            d["struct_members"] = list(node.struct_members.keys())

        if node.helpers:
            helpers = {}
            for key, values in node.helpers.items():
                if values:
                    helpers[key] = list(values.keys())
            if helpers:
                d["helpers"] = self.visit(helpers)

        return d

    def visit_EnumNode(self, node):
        d = dict(
            name=node.name,
            typemap_name=node.typemap.name,  # Only print name to avoid too much nesting.
            ast=self.visit(node.ast),
            decl=node.decl,
        )
        add_comment(d, "enum", node.name)
        add_non_none_fields(node, d, ["linenumber"])
        self.add_visit_fields(node, d, [
            "_fmtmembers",
            "user_fmt",
            "fmtdict",
            "options",
            "splicer",
            "wrap",
        ])
        return d

    def visit_NamespaceNode(self, node):
        d = dict(name=node.name)
        add_comment(d, "namespace", node.name)
        self.add_visit_fields(node, d, [
            "classes", "enums", "functions", "namespaces", "typedefs", "variables",
            "user_fmt", "fmtdict", "options", "wrap"])
        add_non_none_fields(node, d, ["linenumber"])
        self.add_visit_fields(node, d, ["scope_file"])
        add_non_none_fields(node, d, ["scope"])
        return d

    def visit_TypedefNode(self, node):
        d = dict(name=node.name)
        add_comment(d, "typedef", node.name)
        self.add_visit_fields(node, d, [
            "_bind",
            "ast",
            "user_fmt",
            "user_fields",
            "fmtdict",
            "options",
            "splicer",
            "wrap",
            "f_kind",
            "f_module",
        ])
        add_non_none_fields(node, d, ["linenumber"])
        return d

    def visit_VariableNode(self, node):
        d = dict(name=node.name, ast=self.visit(node.ast))
        add_comment(d, "variable", node.name)
        self.add_visit_fields(node, d, [
            "user_fmt",
            "fmtdict",
            "options",
            "wrap",
            "_bind",
        ])
        add_non_none_fields(node, d, ["linenumber"])
        return d

    def visit_TemplateArgument(self, node):
        d = dict(instantiation=node.instantiation, asts=self.visit(node.asts))
        #        self.add_visit_fields(node, d, ['fmtdict', 'options'])
        add_non_none_fields(node, d, ["fmtdict", "options"])
        return d

    def visit_FortranGeneric(self, node):
        d = dict(generic=node.generic,
                 function_suffix=node.function_suffix)
        if node.decls:
            d["decls"] = self.visit(node.decls)
        #        self.add_visit_fields(node, d, ['fmtdict', 'options'])
        add_non_none_fields(node, d, ["fmtdict", "options"])
        return d

    def visit_SymbolTable(self, node):
        return {}


    def visit_BindArg(self, node):
        d = {}
        if node.stmt:
            d["stmt"] = node.stmt.name
        if node.fstmts:
            d["fstmts"] = node.fstmts
        if node.meta is not None:
            metaattrs = {key: value
                         for (key, value) in node.meta.items()
                         if value is not None}
            if metaattrs:
                if "fptr" in metaattrs:
                    metaattrs["fptr"] = self.visit(metaattrs["fptr"])
                if "dim_ast" in metaattrs:
                    metaattrs["dim_ast"] = self.visit(metaattrs["dim_ast"])
                d["meta"] = metaattrs
        return d
    
    # Rename some attributes so they sort to the bottom of the JSON dictionary.
    rename_fields = dict(
        _bind="zz_bind",
        _fmtargs="zz_fmtargs",
        fmtdict="zz_fmtdict",
    )
    def add_visit_fields(self, node, d, fields):
        """Update dict d with fields which must be visited.

        Parameters
        ----------
        node : 
        d : dict
           Dictionary being filled.
        fields: list of str or tuple
            Attribute or node or tuple of attribute, d key.
        """
        for key in fields:
            dkey = self.rename_fields.get(key, key)
            value = getattr(node, key)
            if value:
                d[dkey] = self.visit(value)


def add_non_none_fields(node, d, fields):
    """Update dict d  with fields from node which are not None.
    Used to skip empty fields.
    Fields must not be recursive, ex. Bool, Int, Str.
    """
    for key in fields:
        value = getattr(node, key)
        if value is not None:
            d[key] = value


def add_true_fields(node, d, fields):
    """Update dict d with fields from node which are True.
    Used to skip empty list and dictionary fields.
    """
    for key in fields:
        value = getattr(node, key)
        if value:
            d[key] = value


def add_optional_true_fields(node, d, fields):
    """Update dict d  with fields from node which are not false.

    Used to skip empty fields to avoid clutter in JSON file.

    Parameters
    ----------
    node :
        Input instance which contains fields.
    d : dict
        Dictionary being filled.
    fields : list of str
        Fields to add to d.
    """
    for key in fields:
        if hasattr(node, key):
            value = getattr(node, key)
            if value:
                d[key] = value


def to_dict(node, labelast=False):
    """Convert node to a dictionary.
    Useful for debugging.
    """
    visitor = ToDict(labelast)
    return visitor.visit(node)

######################################################################

class PrintNode(visitor.Visitor):
    """Unparse Nodes.
    Create a string from Nodes.
    """

    def param_list(self, node):
        if node.args:
            n = [node.name, "("]
            for arg in node.args:
                n.append(self.visit(arg))
                n.append(",")
            n[-1] = ")"
        else:
            n = [node.name, "()"]
        return "".join(n)

    def comma_list(self, lst):
        if not lst:
            return ""
        n = []
        for item in lst:
            n.append(self.visit(item))
            n.append(", ")
        n.pop()
        return "".join(n)

    def stmt_list(self, lst):
        if not lst:
            return ""
        n = []
        for item in lst:
            n.append(self.visit(item))
            n.append(";\n")
        return "".join(n)

    def visit_Identifier(self, node):
        if node.args is None:
            return node.name
        elif node.args:
            return self.param_list(node)
        else:
            return node.name + "()"

    def visit_AssumedRank(self, node):
        return ".."

    def visit_BinaryOp(self, node):
        return self.visit(node.left) + node.op + self.visit(node.right)

    def visit_UnaryOp(self, node):
        return node.op + self.visit(node.node)

    def visit_ParenExpr(self, node):
        return "(" + self.visit(node.node) + ")"

    def visit_Constant(self, node):
        return node.value

    def visit_Block(self, node):
#        self.add_visit_fields(node, d, ["user_fmt",])
        return self.stmt_list(node.stmts)

    def visit_CXXClass(self, node):
        s = ["class {}".format(node.name)]
        s = []
        if node.baseclass:
            s.append(": ")
            for basetuple in node.baseclass:
                s.append("{} {}".format(basetuple[0], basetuple[1]))
                s.append(", ")
            s.pop()
        if node.members:
            s.append("{\n")
            s.append(self.stmt_list(node.members))
            s.append("}\n")
        return "".join(s)

    def visit_Namespace(self, node):
        return "namespace {}".format(node.name)

    def visit_Declaration(self, node):
        s = str(node)
        if not node.tag_body:
            pass
        elif node.enum_specifier:
            s += self.visit(node.enum_specifier)
        elif node.class_specifier:
            s += self.visit(node.class_specifier)

        if node.is_ctor or node.is_dtor:
            comma = ""
        else:
            comma = " "
        for d2 in node.declarators:
            sdecl = self.visit(d2)
            if sdecl:
                s += comma + sdecl
                comma = ", "
        return s

    def visit_Declarator(self, node):
        return str(node)

    def visit_EnumValue(self, node):
        if node.value is None:
            return node.name
        else:
            return "{} = {}".format(node.name, self.visit(node.value))

    def visit_Enum(self, node):
        return " {{ {} }}".format(
            self.comma_list(node.members)
        )

    def visit_Struct(self, node):
        s = []
        if node.members:
            s.append("{\n")
            s.append(self.stmt_list(node.members))
            s.append("}\n")
        return "".join(s)

    def visit_Template(self, node):
        parms = self.comma_list(node.parameters)
        decl = self.visit(node.decl)
        return "template<{}>  {}".format(parms, decl)

    def visit_TemplateParam(self, node):
        return node.name

    # XXX - Add Declaration nodes, similar to gen_decl


def print_node(node):
    """Convert node to original string.
    """
    visitor = PrintNode()
    return visitor.visit(node)

######################################################################

class PrintFmt(Helpers, visitor.Visitor):
    """Collect fmtdict members.
    Used for development of statements.
    """

    rename_fields = dict()
    
    def visit_ClassNode(self, node):
        d = dict()
        add_comment(d, "class", node.name)
        self.add_visit_fields(
            node,
            d,
            [
                "classes",
                "enums",
                "functions",
                "variables",
#                "user_fields",
#                "user_fmt",
                "fmtdict",
            ],
        )
        return d

    def visit_EnumNode(self, node):
        d = dict()
        add_comment(d, "enum", node.name)
        self.add_visit_fields(node, d, [
#            "_fmtmembers",
#            "user_fmt",
            "fmtdict",
        ])
        return d

    def visit_FunctionNode(self, node):
        d = dict()
        add_comment(d, "function", node.name)
        self.add_visit_fields(
            node,
            d,
            [
                "_fmtargs",
#                "user_fmt",
                "fmtdict",
            ],
        )
        return d

    def visit_LibraryNode(self, node):
        d = dict()
        self.add_visit_fields( # TEMP  deal with wrap_namespace
            node, d, [ "fmtdict"])
        node = node.wrap_namespace   # XXXX TEMP kludge
        self.add_visit_fields(
            node,
            d,
            [
                "classes",
                "enums",
                "functions",
                "namespaces",
                "typedefs",
                "variables",
#                "user_fmt",
            ],
        )
        return d

    def visit_NamespaceNode(self, node):
        d = dict()
        add_comment(d, "namespace", node.name)
        self.add_visit_fields(node, d, [
            "classes", "enums", "functions", "namespaces", "typedefs", "variables",
#            "user_fmt",
            "fmtdict"])
        return d

    def visit_TypedefNode(self, node):
        d = dict()
        add_comment(d, "typedef", node.name)
        self.add_visit_fields(node, d, [
#            "user_fmt",
            "fmtdict",
        ])
        return d

    def visit_VariableNode(self, node):
        d = dict()
        add_comment(d, "variable", node.name)
        self.add_visit_fields(node, d, [
            "user_fmt",
            "fmtdict",
        ])
        return d

    
def print_fmt(node):
    """Dump format strings of nodes.
    """
    visitor = PrintFmt()
    return visitor.visit(node)

######################################################################

class PrintNodeIdentifier(PrintNode):
    """Print a node but convert identifier using symbols.
    symbols[name][key] = replacement symbol.

    Used with enums when converting an enum expression.
    Convert C++ enums in Fortran/C enum identifiers.
    """
    def __init__(self, symbols, key):
        self.symbols = symbols
        self.key = key
        super(PrintNodeIdentifier, self).__init__()

    def visit_Identifier(self, node):
        if node.args is None:
            if node.name in self.symbols:
                return self.symbols[node.name][self.key]
            return node.name
        elif node.args:
            return self.param_list(node)
        else:
            return node.name + "()"

def print_node_identifier(node, symbols, key):
    """Convert node to original string and change identifiers
    """
    visitor = PrintNodeIdentifier(symbols, key)
    return visitor.visit(node)

######################################################################

def print_node_as_json(node):
    """Print a node as json.
    Useful for debugging.
    """
    dd = to_dict(node)
    print(json.dumps(dd, indent=4, sort_keys=True, separators=(',', ': ')))
