# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
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

class ToDict(visitor.Visitor):
    """Convert to dictionary.
    """

    def visit_bool(self, node):
        return str(node)

    def visit_int(self, node):
        return str(node)

    def visit_str(self, node):
        return str(node)

    def visit_list(self, node):
        return [self.visit(n) for n in node]

    def visit_dict(self, node):
        return {key: self.visit(value) for (key, value) in node.items()}

    ######################################################################

    def visit_Ptr(self, node):
        d = dict(ptr=node.ptr)
        add_true_fields(node, d, ["const", "volatile"])
        return d

    def visit_Declarator(self, node):
        d = dict(pointer=self.visit(node.pointer))
        if node.name:
            d["name"] = node.name
        elif node.func:
            d["func"] = self.visit(node.func)
        return d

    def visit_Declaration(self, node):
        d = dict(
            specifier=node.specifier,
            # #- node.array,
            typemap_name=node.typemap.name,  # print name to avoid too much nesting
        )
        attrs = {key: value
                 for (key, value) in node.attrs.items()
                 if value is not None}
        if attrs:
            d["attrs"] = attrs

        metaattrs = {key: value
                 for (key, value) in node.metaattrs.items()
                 if value is not None}
        if metaattrs:
            if "struct_member" in metaattrs:
                # struct_member is a ast.VariableNode, add name instead
                # to avoid huge dump.
                metaattrs["struct_member"] = metaattrs["struct_member"].name
            if "dimension" in metaattrs:
                metaattrs["dimension"] = self.visit(metaattrs["dimension"])
            d["metaattrs"] = metaattrs
        
        add_true_fields(node, d, ["const", "func_const", "volatile"])
        if node.declarator:
            # ctor and dtor have no declarator
            d["declarator"] = self.visit(node.declarator)
        if node.storage:
            d["storage"] = node.storage
        if node.params is not None:
            d["params"] = self.visit(node.params)
        if node.array:
            d["array"] = self.visit(node.array)
        if node.init is not None:
            d["init"] = node.init
        if node.template_arguments:
            lst = []
            for tp in node.template_arguments:
                lst.append(self.visit(tp))
            d["template_arguments"] = lst
        if node.stmts_suffix:
            d["stmts_suffix"] = node.stmts_suffix
        if node.ftrim_char_in:
            d["ftrim_char_in"] = node.ftrim_char_in
        return d

    def visit_Identifier(self, node):
        d = dict(name=node.name)
        if node.args is not None:
            d["args"] = self.visit(node.args)
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
        return dict(name=node.name)

    def visit_Namespace(self, node):
        return dict(name=node.name)

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
        return d

    def visit_Struct(self, node):
        d = dict(name=node.name, members=self.visit(node.members))
        return d

    def visit_Template(self, node):
        d = dict(
            parameters=self.visit(node.parameters), decl=self.visit(node.decl)
        )
        return d

    def visit_TemplateParam(self, node):
        d = dict(name=node.name)
        return d

    ######################################################################

    def visit_Scope(self, node):
        d = {}
        skip = "_" + node.__class__.__name__ + "__"  # __name is skipped
        for key, value in node.__dict__.items():
            if not key.startswith(skip):
                d[key] = value
        return d

    ######################################################################

    def visit_Typemap(self, node):
        # only export non-default values
        a = {}
        for key, defvalue in node.defaults.items():
            value = getattr(node, key)
            if value is not defvalue:
                a[key] = value
        return a

    ######################################################################

    def visit_LibraryNode(self, node):
        d = dict()
        add_true_fields(node, d, ["copyright", "cxx_header", "language", "scope"])
        self.add_visit_fields( # TEMP  deal with wrap_namespace
            node, d, [ "fmtdict", "options", "scope_file", ])
        node = node.wrap_namespace   # XXXX TEMP kludge
        self.add_visit_fields(
            node,
            d,
            [
                "classes",
                "enums",
                "functions",
                "namespaces",
                "variables",
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
            typemap_name=node.typemap.name,  # print name to avoid too much nesting
        )
        add_non_none_fields(node, d, ["linenumber"])
        add_true_fields(
            node, d, ["as_struct", "python", "scope", "template_parameters"]
        )
        self.add_visit_fields(
            node,
            d,
            [
                "enums",
                "functions",
                "variables",
                "fmtdict",
                "options",
                "template_arguments",
            ],
        )
        return d

    def visit_FunctionNode(self, node):
        d = dict(ast=self.visit(node.ast), decl=node.decl)
        self.add_visit_fields(
            node,
            d,
            [
                "_fmtargs",
                "_fmtresult",
                "fmtdict",
                "options",
                "template_arguments",
                "fortran_generic",
                "fstatements",
                "splicer",
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
                "_has_default_arg",
                "_nargs",
                "_overloaded",
                # generated by Preprocess
                # #- 'CXX_subprogram',  'C_subprogram',  'F_subprogram',
                # #- 'CXX_return_type', 'C_return_type', 'F_return_type',
            ],
        )
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
            d['gen_headers_typedef'] = sorted(node.gen_headers_typedef.keys())

        return d

    def visit_EnumNode(self, node):
        d = dict(
            name=node.name,
            typemap_name=node.typemap.name,  # print name to avoid too much nesting
            ast=self.visit(node.ast),
            decl=node.decl,
        )
        add_non_none_fields(node, d, ["linenumber"])
        self.add_visit_fields(node, d, ["_fmtmembers", "fmtdict", "options"])
        return d

    def visit_NamespaceNode(self, node):
        d = dict(name=node.name)
        self.add_visit_fields(node, d, [
            "classes", "enums", "functions", "namespaces", "variables",
            "fmtdict", "options"])
        add_non_none_fields(node, d, ["linenumber"])
        self.add_visit_fields(node, d, ["scope_file"])
        add_non_none_fields(node, d, ["scope"])
        return d

    def visit_VariableNode(self, node):
        d = dict(name=node.name, ast=self.visit(node.ast))
        self.add_visit_fields(node, d, ["fmtdict", "options"])
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

    def add_visit_fields(self, node, d, fields):
        """Update dict d with fields which must be visited."""
        for key in fields:
            value = getattr(node, key)
            if value:
                d[key] = self.visit(value)


def add_non_none_fields(node, d, fields):
    """Update dict d  with fields from node which are not None.
    Used to skip empty fields.
    """
    for key in fields:
        value = getattr(node, key)
        if value is not None:
            d[key] = value


def add_true_fields(node, d, fields):
    """Update dict d  with fields from node which are not None.
    Used to skip empty fields.
    """
    for key in fields:
        value = getattr(node, key)
        if value:
            d[key] = value


def add_optional_true_fields(node, d, fields):
    """Update dict d  with fields from node which are not None.
    Used to skip empty fields.
    """
    for key in fields:
        if hasattr(node, key):
            value = getattr(node, key)
            if value:
                d[key] = value


def to_dict(node):
    """Convert node to a dictionary.
    Useful for debugging.
    """
    visitor = ToDict()
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
            n.append(";")
        return "".join(n)

    def visit_Identifier(self, node):
        if node.args is None:
            return node.name
        elif node.args:
            return self.param_list(node)
        else:
            return node.name + "()"

    def visit_BinaryOp(self, node):
        return self.visit(node.left) + node.op + self.visit(node.right)

    def visit_UnaryOp(self, node):
        return node.op + self.visit(node.node)

    def visit_ParenExpr(self, node):
        return "(" + self.visit(node.node) + ")"

    def visit_Constant(self, node):
        return node.value

    def visit_CXXClass(self, node):
        return "class {};".format(node.name)

    def visit_Namespace(self, node):
        return "namespace {}".format(node.name)

    def visit_Declaration(self, node):
        return str(node)

    def visit_EnumValue(self, node):
        if node.value is None:
            return node.name
        else:
            return "{} = {}".format(node.name, self.visit(node.value))

    def visit_Enum(self, node):
        if node.scope:
            return "enum {} {} {{ {} }};".format(
                node.name, node.scope, self.comma_list(node.members)
            )
        else:
            return "enum {} {{ {} }};".format(
                node.name, self.comma_list(node.members)
            )

    def visit_Struct(self, node):
        return "struct {} {{ {} }};".format(
            node.name, self.stmt_list(node.members)
        )

    # XXX - Add Declaration nodes, similar to gen_decl


def print_node(node):
    """Convert node to original string.
    """
    visitor = PrintNode()
    return visitor.visit(node)



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


def print_node_as_json(node):
    """Print a node as json.
    Useful for debugging.
    """
    dd = to_dict(node)
    print(json.dumps(dd, indent=4, sort_keys=True, separators=(',', ': ')))
