# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################
"""
Generate Lua module for C++ code.
"""
from __future__ import print_function
from __future__ import absolute_import

from . import statements
from . import typemap
from . import util
from .util import wformat, append_format


class Wrapl(util.WrapperMixin):
    """Generate Lua bindings.
    """

    def __init__(self, newlibrary, config, splicers):
        """
        Args:
            newlibrary - ast.LibraryNode.
            config -
            splicers -
        """
        self.newlibrary = newlibrary
        self.language = newlibrary.language
        self.patterns = newlibrary.patterns
        self.config = config
        self.log = config.log
        self._init_splicer(splicers)
        self.comment = "//"
        self.cont = ""
        self.linelen = newlibrary.options.C_line_length
        self.doxygen_begin = "/**"
        self.doxygen_cont = " *"
        self.doxygen_end = " */"
        self.helpers = Helpers(self.language)
        update_statements_for_language(self.language)

    def reset_file(self):
        pass

    def wrap_library(self):
        newlibrary = self.newlibrary
        fmt_library = newlibrary.fmtdict

        # Format variables
        fmt_library.LUA_used_param_state = False
        newlibrary.eval_template("LUA_module_name")
        newlibrary.eval_template("LUA_module_reg")
        newlibrary.eval_template("LUA_module_filename")
        newlibrary.eval_template("LUA_header_filename")

        # Some kludges, need to compute correct value in wrapl.py
        fmt_library.LUA_metadata = "XXLUA_metadata"
        fmt_library.LUA_userdata_type = "XXLUA_userdata_type"
        fmt_library.push_arg = "XXXpush_arg"

        # XXX - Without this c_const is undefined.
        #       Need to sort out where it should be set.
        #       Start with difference bewteen const method and const result.
        #       const double *get() const
        fmt_library.c_const="LUAc_const"

        # Variables to accumulate output lines
        self.luaL_Reg_module = []
        self.body_lines = []
        self.class_lines = []
        self.lua_type_structs = []

        self.wrap_namespace(newlibrary.wrap_namespace)
        self.write_header(newlibrary)
        self.write_module(newlibrary)
    #        self.write_helper()

    def wrap_namespace(self, node):
        """Wrap a library or namespace.

        Args:
            node - ast.LibraryNode, ast.NamespaceNode
        """
        self._push_splicer("class")
        for cls in node.classes:
            if not cls.wrap.lua:
                continue
            name = cls.name
            self.reset_file()
            self._push_splicer(name)
            self.wrap_class(cls)
            #            self.write_extension_type(cls)
            self._pop_splicer(name)
        self._pop_splicer("class")

        self.reset_file()
        if node.functions:
            self._push_splicer("function")
            self.wrap_functions(None, node.functions)
            self._pop_splicer("function")

        for ns in node.namespaces:
            if ns.wrap.lua:
                self.wrap_namespace(ns)

    def wrap_class(self, node):
        """
        Args:
            node -
        """
        fmt_class = node.fmtdict

        fmt_class.LUA_userdata_var = "SH_this"
        node.eval_template("LUA_userdata_type")
        node.eval_template("LUA_userdata_member")
        node.eval_template("LUA_class_reg")
        node.eval_template("LUA_metadata")
        node.eval_template("LUA_ctor_name")
        fmt_class.LUA_this_call = wformat(
            "{LUA_userdata_var}->{LUA_userdata_member}->", fmt_class
        )

        self._create_splicer("C_declaration", self.lua_type_structs)
        self.lua_type_structs.append("")
        self.lua_type_structs.append("typedef struct {+")
        append_format(
            self.lua_type_structs,
            "{namespace_scope}{cxx_class} * {LUA_userdata_member};",
            fmt_class,
        )
        self._create_splicer("C_object", self.lua_type_structs)
        append_format(
            self.lua_type_structs, "-}} {LUA_userdata_type};", fmt_class
        )

        self.luaL_Reg_class = []

        # wrap methods
        self._push_splicer("method")
        self.wrap_functions(node, node.functions)
        self._pop_splicer("method")
        self.append_luaL_Reg(
            self.body_lines, fmt_class.LUA_class_reg, self.luaL_Reg_class
        )

        append_format(
            self.class_lines,
            """
/* Create the metatable and put it on the stack. */
luaL_newmetatable({LUA_state_var}, "{LUA_metadata}");
/* Duplicate the metatable on the stack (We now have 2). */
lua_pushvalue({LUA_state_var}, -1);
/* Pop the first metatable off the stack and assign it to __index
 * of the second one. We set the metatable for the table to itself.
 * This is equivalent to the following in lua:
 * metatable = {{}}
 * metatable.__index = metatable
 */
lua_setfield({LUA_state_var}, -2, "__index");

/* Set the methods to the metatable that should be accessed via object:func */
#if LUA_VERSION_NUM < 502
luaL_register({LUA_state_var}, NULL, {LUA_class_reg});
#else
luaL_setfuncs({LUA_state_var}, {LUA_class_reg}, 0);
#endif
""",
            fmt_class,
        )

    def wrap_functions(self, cls, functions):
        """Wrap functions for a library or class.
        Create one wrapper for overloaded functions and the
        different variations of default-arguments

        Args:
            cls - ast.ClassNode.
            functions - list of ast.FunctionNode.
        """

        # Find overloaded functions.
        # maintain order, but gather overloads together
        overloaded_methods = {}
        overloads = []
        for function in functions:
            if not function.wrap.lua:
                continue
            name = function.ast.name
            if name in overloaded_methods:
                overloaded_methods[name].append(function)
            else:
                first = [function]
                overloads.append(first)
                overloaded_methods[name] = first

        for overload in overloads:
            self.wrap_function(cls, overload)

    #        for function in functions:
    #            self.wrap_function(cls, function)

    def wrap_function(self, cls, overloads):
        """Write a Lua wrapper for a C++ function.

        Args:
            cls  - ast.ClassNode or None for functions
            overloads - a list of ast.FunctionNode to wrap.

        fmt.c_var   - name of variable from lua stack.
        fmt.cxx_var - name of variable in c++ call.
        """

        # First overload defines options
        node = overloads[0]

        ast = node.ast
        fmt_func = node.fmtdict
        fmt = util.Scope(fmt_func)
        node.eval_template("LUA_name")
        node.eval_template("LUA_name_impl")

        CXX_subprogram = ast.get_subprogram()

        # XXX       ast = node.ast
        # XXX       result_type = ast.typename
        # XXX       result_is_ptr = ast.is_pointer()
        # XXX       result_is_ref = ast.is_reference()

        is_ctor = ast.is_ctor()
        is_dtor = ast.is_dtor()
        if is_dtor:
            CXX_subprogram = "subroutine"
            fmt.LUA_name = "__gc"

        # Loop over all overloads and default argument and
        # sort by the number of arguments expected.

        # Create lists of input arguments
        all_calls = []
        maxargs = 0
        for function in overloads:
            nargs = 0
            in_args = []
            out_args = []
            for arg in function.ast.params:
                arg_typemap = arg.typemap
                if arg.init is not None:
                    all_calls.append(
                        LuaFunction(
                            function, CXX_subprogram, in_args[:], out_args
                        )
                    )
                in_args.append(arg)
            # no defaults, use all arguments
            all_calls.append(
                LuaFunction(function, CXX_subprogram, in_args[:], out_args)
            )
            maxargs = max(maxargs, len(in_args))

        # Gather calls by number of arguments
        by_count = [[] for i in range(maxargs + 1)]
        for a_call in all_calls:
            by_count[a_call.nargs].append(a_call)

        self.splicer_lines = []
        lines = self.splicer_lines
        self.stmts_comments = []

        if len(all_calls) == 1:
            call = all_calls[0]
            fmt.nresults = call.nresults
            self.do_function(cls, call, fmt)
            append_format(lines, "return {nresults};", fmt)
        else:
            lines.append("int SH_nresult = 0;")
            fmt.LUA_used_param_state = True
            append_format(
                lines, "int SH_nargs = lua_gettop({LUA_state_var});", fmt
            )

            # Find type of each argument
            itype_vars = []
            for iarg in range(1, maxargs + 1):
                itype_vars.append("SH_itype{}".format(iarg))
                fmt.itype_var = itype_vars[-1]
                fmt.iarg = iarg
                append_format(
                    lines,
                    "int {itype_var} = " "lua_type({LUA_state_var}, {iarg});",
                    fmt,
                )

            lines.append("switch (SH_nargs) {")
            for nargs, calls in enumerate(by_count):
                if len(calls) == 0:
                    continue
                lines.append("case {}:".format(nargs))
                lines.append(1)
                ifelse = "if"

                for call in calls:
                    fmt.nresults = call.nresults
                    checks = []
                    for iarg, arg in enumerate(call.inargs):
                        arg_typemap = arg.typemap
                        fmt.itype_var = itype_vars[iarg]
                        fmt.itype = arg_typemap.LUA_type
                        append_format(checks, "{itype_var} == {itype}", fmt)

                    # Select cases to help with formating of output
                    if nargs == 0:
                        # put within a compound statement to
                        # scope local variables
                        lines.extend(["{", 1])
                        self.do_function(cls, call, fmt)
                        append_format(lines, "SH_nresult = {nresults};", fmt)
                        lines.extend([-1, "}"])
                    elif nargs == 1:
                        lines.append("{} ({}) {{+".format(ifelse, checks[0]))
                        self.do_function(cls, call, fmt)
                        append_format(
                            lines, "SH_nresult = {nresults};\n" "-}}", fmt
                        )
                    elif nargs == 2:
                        lines.append("{} ({} &&+".format(ifelse, checks[0]))
                        lines.append("{}) {{".format(checks[1]))
                        self.do_function(cls, call, fmt)
                        append_format(
                            lines, "SH_nresult = {nresults};\n" "-}}", fmt
                        )
                    else:
                        lines.append("{} ({} &&+".format(ifelse, checks[0]))
                        for check in checks[1:-1]:
                            lines.append("{} &&".format(check))
                        lines.append("{}) {{".format(checks[-1]))
                        self.do_function(cls, call, fmt)
                        append_format(
                            lines, "SH_nresult = {nresults};\n" "-}}", fmt
                        )
                    ifelse = "else if"
                if nargs > 0:
                    # Trap errors when the argument types do not match
                    append_format(
                        lines,
                        "else {{+\n"
                        'luaL_error({LUA_state_var}, "error with arguments");\n'
                        "-}}",
                        fmt,
                    )
                lines.append("break;")
                lines.append(-1)
            append_format(
                lines,
                "default:+\n"
                'luaL_error({LUA_state_var}, "error with arguments");\n'
                "break;\n"
                "-}}\n"
                "return SH_nresult;",
                fmt,
            )

        body = self.body_lines
        body.append("")
        if node.options.debug:
            for node in overloads:
                body.append("// " + node.declgen)
        body.extend(self.stmts_comments)
        if node.options.doxygen:
            for node in overloads:
                if node.doxygen:
                    self.write_doxygen(body, node.doxygen)
        if fmt.LUA_used_param_state:
            append_format(
                body,
                "static int {LUA_name_impl}" "(lua_State *{LUA_state_var})",
                fmt,
            )
        else:
            append_format(body, "static int {LUA_name_impl}(lua_State *)", fmt)
        body.extend(["{", 1])
        self._create_splicer(fmt.LUA_name, body, self.splicer_lines)
        body.extend([-1, "}"])

        # Save pointer to function
        if cls:
            if is_ctor:
                append_format(
                    self.luaL_Reg_module,
                    '{{"{LUA_ctor_name}", ' "{LUA_name_impl}}},",
                    fmt,
                )
            else:
                append_format(
                    self.luaL_Reg_class,
                    '{{"{LUA_name}", {LUA_name_impl}}},',
                    fmt,
                )

        else:
            append_format(
                self.luaL_Reg_module, '{{"{LUA_name}", {LUA_name_impl}}},', fmt
            )

    def do_function(self, cls, luafcn, fmt_func):
        """
        Wrap a single function/overload/default-argument
        variation of a function.

        Args:
            cls - ast.ClassNode
            luafcn - LuaFunction
            fmt_func - local format dictionary
        """
        node = luafcn.function
        #        if not options.wrap_lua:
        #            return

        if cls:
            cls_function = "method"
        else:
            cls_function = "function"
        self.log.write("Lua {0} {1.declgen}\n".format(cls_function, node))

        #        fmt_func = node.fmtdict
        fmtargs = node._fmtargs
        #        fmt = util.Scope(fmt_func)
        #        fmt.doc_string = 'documentation'
        #        node.eval_template('LUA_name')
        #        node.eval_template('LUA_name_impl')

        ast = node.ast
        CXX_subprogram = ast.get_subprogram()
        result_typemap = ast.typemap
        is_ctor = ast.is_ctor()
        is_dtor = ast.is_dtor()
        stmts_comments = self.stmts_comments
        stmts_comments_args = []  # Used to reorder comments

        #        is_const = ast.const
        # XXX        if is_ctor:   # or is_dtor:
        # XXX            # XXX - have explicit delete
        # XXX            # need code in __init__ and __del__
        # XXX            return

        # XXX if a class, then knock off const since the PyObject
        # is not const, otherwise, use const from result.
        # This has been replaced by gen_arg methods, but not sure about const.
        #        if result_typemap.base == 'shadow':
        #            is_const = False
        #        else:
        #            is_const = None
        # return value
        #        fmt.rv_decl = self.std_c_decl(
        #            'cxx_type', ast, name=fmt.LUA_result, const=is_const)

        fmt_result = node._fmtresult.setdefault("fmtl", util.Scope(fmt_func))
        if CXX_subprogram == "function":
            fmt_result.cxx_var = wformat("{CXX_local}{LUA_result}", fmt_result)
            if is_ctor or ast.is_pointer():
                #                fmt_result.c_member = '->'
                fmt_result.cxx_member = "->"
                fmt_result.cxx_addr = ""
            else:
                #                fmt_result.c_member = '.'
                fmt_result.cxx_member = "."
                fmt_result.cxx_addr = "&"
            if result_typemap.cxx_to_c:
                fmt_result.c_var = wformat(
                    result_typemap.cxx_to_c, fmt_result
                )  # if C++
            else:
                fmt_result.c_var = fmt_result.cxx_var

            fmt_func.rv_decl = ast.gen_arg_as_cxx(
                name=fmt_result.cxx_var, params=None
            )
            fmt_func.rv_asgn = fmt_func.rv_decl + " =\t "

        node_stmt = util.Scope(LuaStmts)
        declare_code = []  # Declare variables and pop values.
        node_stmt.pre_call = []  # Extract arguments.
        node_stmt.call = []  # Call C++ function.
        node_stmt.post_call = []  # Push results.

        # post_parse = []
        cxx_call_list = []

        # find class object
        if cls:
            cls_typedef = cls.typemap
            if not is_ctor:
                fmt_func.LUA_used_param_state = True
                fmt_func.c_var = wformat(cls_typedef.LUA_pop, fmt_func)
                append_format(
                    node_stmt.call,
                    "{LUA_userdata_type} * {LUA_userdata_var} =\t {c_var};",
                    fmt_func,
                )

        # parse arguments
        # call function based on number of default arguments provided
        # XXX default_calls = []   # each possible default call
        # XXX if '_has_default_arg' in node:
        # XXX     append_format(declare_code, 'int SH_nargs =
        # XXX          lua_gettop({LUA_state_var});', fmt_func)

        # Only process nargs.
        # Each variation of default-arguments produces a new call.
        LUA_index = 1
        for iarg in range(luafcn.nargs):
            arg = ast.params[iarg]
            arg_name = arg.name
            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg = fmt_arg0.setdefault("fmtl", util.Scope(fmt_func))
            fmt_arg.LUA_index = LUA_index
            fmt_arg.c_var = arg_name
            fmt_arg.cxx_var = arg_name
            fmt_arg.lua_var = "SH_Lua_" + arg_name
            fmt_arg.c_var_len = "L" + arg_name
            if arg.is_pointer():
                fmt_arg.c_deref = " *"
                fmt_arg.c_member = "->"
                fmt_arg.cxx_member = "->"
            else:
                fmt_arg.c_deref = ""
                fmt_arg.c_member = "."
                fmt_arg.cxx_member = "."
            attrs = arg.attrs
            meta = arg.metaattrs

            arg_typemap = arg.typemap
            fmt_arg.cxx_type = arg_typemap.cxx_type

            intent_blk = None
            intent = meta["intent"]
            sgroup = arg_typemap.sgroup
            spointer = arg.get_indirect_stmt()
            stmts = None
            stmts = ["lua", intent, sgroup, spointer]
            if intent_blk is None:
                intent_blk = lookup_stmts(stmts)
            # Useful for debugging.  Requested and found path.
            fmt_arg.stmt0 = statements.compute_name(stmts)
            fmt_arg.stmt1 = intent_blk.name
            # Add some debug comments to function.
            if node.options.debug:
                stmts_comments_args.append(
                    "// ----------------------------------------")
                stmts_comments_args.append("// Argument:  " + arg.gen_decl())
                self.document_stmts(
                    stmts_comments_args, arg, fmt_arg.stmt0, fmt_arg.stmt1)
            
            if intent in ["inout", "in"]:
                # XXX lua_pop = wformat(arg_typemap.LUA_pop, fmt_arg)
                # lua_pop is a C++ expression
                fmt_arg.pop_expr = wformat(arg_typemap.LUA_pop, fmt_arg)
                if self.language == "c":
                    pass
                elif arg_typemap.c_to_cxx:
                    fmt_arg.c_var = fmt_arg.pop_expr
                    fmt_arg.pop_expr = wformat(arg_typemap.c_to_cxx, fmt_arg)
                LUA_index += 1

            if intent in ["inout", "out"]:
                # output variable must be a pointer
                # XXX - fix up for strings
                # XXX  format, vargs = self.intent_out(
                # XXX      arg_typemap, fmt_arg, post_call)
                # XXX  build_format.append(format)
                # XXX  build_vargs.append('*' + vargs)

                # append_format(post_call_code, arg_typemap.LUA_push, fmt_arg)
                tmp = wformat(arg_typemap.LUA_push, fmt_arg)
                node_stmt.post_call.append(tmp + ";")
                # XXX - needs work with pointers: int *out+intent(out)

            self.append_code(intent_blk, node_stmt, fmt_arg, fmt_func)

            cxx_call_list.append(fmt_arg.cxx_var)
        # --- End loop over function parameters

        # call with arguments
        fmt_func.cxx_call_list = ",\t ".join(cxx_call_list)
        #        call_code.extend(post_parse)

        sgroup = None
        spointer = ast.get_indirect_stmt()
#        print("DDDDDDDDDDDDDD", ast.name)
        sintent = ast.metaattrs["intent"]
        if is_ctor:
            sgroup ="shadow"
            fmt_func.LUA_used_param_state = True
#            self.helpers.add_helper("maker", fmt_func)
        elif is_dtor:
            sgroup ="shadow"
            fmt_func.LUA_used_param_state = True
        elif CXX_subprogram == "subroutine":
            sgroup = "subroutine"
            spointer = None
            sintent = None
        else:
            sgroup = result_typemap.sgroup
        stmts = ["lua", sintent, sgroup, spointer]
#        print("XXXXXX", stmts)
        result_blk = lookup_stmts(stmts)
        fmt_result.stmt0 = statements.compute_name(stmts)
        fmt_result.stmt1 = result_blk.name
        if node.options.debug:
            stmts_comments.append(
                "// ----------------------------------------")
            stmts_comments.append(
                "// Function:  " + ast.gen_decl(params=None))
            self.document_stmts(
                stmts_comments, ast, fmt_result.stmt0, fmt_result.stmt1)
            stmts_comments.extend(stmts_comments_args)
            

        #        if 'LUA_error_pattern' in node:
        #            lfmt = util.Scope(fmt)
        #            lfmt.c_var = fmt.LUA_result
        #            lfmt.cxx_var = fmt.LUA_result
        #            append_format(call_code,
        #                 self.patterns[node['LUA_error_pattern']], lfmt)

        # Compute return value
        if CXX_subprogram == "function" and not is_ctor:
            fmt_result.push_arg = fmt_result.c_var
            fmt_result.push_expr = wformat(result_typemap.LUA_push, fmt_result)
        self.append_code(result_blk, node_stmt, fmt_result, fmt_func)

        lines = self.splicer_lines
        lines.extend(declare_code)
        lines.extend(node_stmt.pre_call)
        lines.extend(node_stmt.call)
        # int lua_checkstack (lua_State *L, int extra)
        lines.extend(node_stmt.post_call)  # return values


    def append_code(self, blk, node_stmt, fmt, fmt_func):
        """Append code from blk

        Args:
            blk - util.Scope
            node_stmt - LuaStmts
            fmt - util.Scope
            fmt_func - util.Scope
        """
        if blk.pre_call:
            fmt_func.LUA_used_param_state = True
            for line in blk.pre_call:
                append_format(node_stmt.pre_call, line, fmt)
        if blk.call:
            for line in blk.call:
                append_format(node_stmt.call, line, fmt)
        if blk.post_call:
            fmt_func.LUA_used_param_state = True
            for line in blk.post_call:
                append_format(node_stmt.post_call, line, fmt)
        
    def write_header(self, node):
        """
        Args:
            node -
        """
        fmt = node.fmtdict
        fname = fmt.LUA_header_filename

        header_impl = util.Header(self.newlibrary)
        header_impl.add_cxx_header(node)

        output = []

        # add guard
        guard = fname.replace(".", "_").upper()
        output.extend(["#ifndef %s" % guard, "#define %s" % guard])
        util.extern_C(output, "begin")

        header_impl.write_headers(output)

        output.append('#include "lua.h"')
        output.extend(self.lua_type_structs)
        append_format(
            output,
            "\n" "int luaopen_{LUA_module_name}(lua_State *{LUA_state_var});\n",
            fmt,
        )
        util.extern_C(output, "end")
        output.append("#endif  /* %s */" % guard)
        self.write_output_file(fname, self.config.python_dir, output)

    def append_luaL_Reg(self, output, name, lines):
        """Create luaL_Reg struct

        Args:
            output -
            name -
            lines -
        """
        output.append("")
        self._create_splicer("additional_functions", output)
        output.extend(
            ["", "static const struct luaL_Reg {} [] = {{".format(name), 1]
        )
        output.extend(lines)
        self._create_splicer("register", output)
        output.extend(["{NULL, NULL}   /*sentinel */", -1, "};"])

    def write_module(self, node):
        """
        Args:
            node - ast.LibraryNode.
        """
        fmt = node.fmtdict
        fname = fmt.LUA_module_filename

        hinclude, hsource = self.helpers.find_file_helper_code()

        header_impl = util.Header(self.newlibrary)
        header_impl.add_cxx_header(node)
        header_impl.add_shroud_file(fmt.LUA_header_filename)
        header_impl.add_shroud_dict(hinclude)
        
        output = []

        header_impl.write_headers(output)

        util.extern_C(output, "begin")
        output.append('#include "lauxlib.h"')
        util.extern_C(output, "end")

        output.append("")
        self._create_splicer("include", output)

        self._create_splicer("C_definition", output)

        output.extend(hsource)
        output.extend(self.body_lines)

        self.append_luaL_Reg(output, fmt.LUA_module_reg, self.luaL_Reg_module)
        output.append("")
        util.extern_C(output, "begin")
        append_format(
            output,
            "int luaopen_{LUA_module_name}" "(lua_State *{LUA_state_var}) {{+",
            fmt,
        )
        output.extend(self.class_lines)
        append_format(
            output,
            "\n"
            "#if LUA_VERSION_NUM < 502\n"
            'luaL_register({LUA_state_var}, "{LUA_module_name}", '
            "{LUA_module_reg});\n"
            "#else\n"
            "luaL_newlib({LUA_state_var}, {LUA_module_reg});\n"
            "#endif\n"
            "return 1;\n"
            "-}}",
            fmt,
        )
        util.extern_C(output, "end")

        self.write_output_file(fname, self.config.lua_dir, output)


class LuaFunction(object):
    """Gather information used to write a wrapper for
    an overloaded/default-argument function
    """

    def __init__(self, function, subprogram, inargs, outargs):
        """
        Args:
            function -
            subprogram -
            inargs -
            outargs -
        """
        self.function = function
        self.subprogram = subprogram  # 'function' or 'subroutine'
        self.nargs = len(inargs)
        self.nresults = len(outargs)
        self.inargs = inargs
        self.outargs = outargs

        if subprogram == "function":
            self.nresults += 1


class Helpers(object):
    def __init__(self, language):
        self.c_helpers = {}  # c and c++
        self.language = language

    def add_helper(self, helpers, fmt):
        """Add a list of C helpers."""
        c_helper = wformat(helpers, fmt)
        for i, helper in enumerate(c_helper.split()):
            self.c_helpers[helper] = True
            setattr(fmt, "hnamefunc" + str(i),
                    LuaHelpers[helper].get("name", helper))
        
#        self.c_helper[name] = True
#        # Adjust for alias like with type char.
#        return whelpers.CHelpers[name]["name"]

    def _gather_helper_code(self, name, done):
        """Add code from helpers.

        First recursively process dependent_helpers
        to add code in order.

        Args:
            name - Name of helper.
            done - Dictionary of previously processed helpers.
        """
        if name in done:
            return  # avoid recursion
        done[name] = True

        helper_info = LuaHelpers[name]
        if "dependent_helpers" in helper_info:
            for dep in helper_info["dependent_helpers"]:
                # check for recursion
                self._gather_helper_code(dep, done)

        scope = helper_info.get("scope", "file")
        # assert scope in ["file", "utility"]

        lang_key = self.language + "_include"
        if lang_key in helper_info:
            for include in helper_info[lang_key]:
                self.helper_summary["include"][scope][include] = True
        elif "include" in helper_info:
            for include in helper_info["include"]:
                self.helper_summary["include"][scope][include] = True

        for key in ["proto", "source"]:
            lang_key = self.language + "_" + key 
            if lang_key in helper_info:
                self.helper_summary[key][scope].append(helper_info[lang_key])
            elif key in helper_info:
                self.helper_summary[key][scope].append(helper_info[key])

    def gather_helper_code(self, helpers):
        """Gather up all helpers requested and insert code into output.

        helpers should be self.c_helper or self.shared_helper

        Args:
            helpers - dictionary of helper names.
        """
        self.helper_summary = dict(
            include=dict(file={}, pwrap_impl={}),
            proto=dict(file=[], pwrap_impl=[]),
            source=dict(file=[], pwrap_impl=[]),
        )
        self.helper_need_numpy = False

        done = {}  # Avoid duplicates by keeping track of what's been written.
        for name in sorted(helpers.keys()):
            self._gather_helper_code(name, done)

#        print("XXXXXXXX", self.helper_summary)

    def find_file_helper_code(self):
        """Get "file" helper code.
        Add to shared_helper, then reset.

        Return dictionary of headers and list of source files.
        """
#        if self.newlibrary.options.PY_write_helper_in_util:
#            self.shared_helper.update(self.c_helper)
#        if True:
#            self.c_helpers = {}
#            return {}, []
        self.gather_helper_code(self.c_helpers)
#        self.shared_helper.update(self.c_helpers)
        self.c_helper = {}
        return (
            self.helper_summary["include"]["file"],
            self.helper_summary["source"]["file"],
#            self.helper_need_numpy
        )

        


LuaHelpers = dict(
    maker=dict(
        source="""
// Test adding helper
""",
    ),
)

######################################################################

# The tree of Python Scope statements.
lua_tree = {}
lua_dict = {} # dictionary of Scope of all expanded lua_statements,
default_scope = None  # for statements

def update_statements_for_language(language):
    """Preprocess statements for lookup.

    Update statements for c or c++.
    Fill in py_tree.

    Parameters
    ----------
    language : str
        "c" or "c++"
    """
    statements.update_for_language(lua_statements, language)
    statements.update_stmt_tree(lua_statements, lua_dict, lua_tree, default_stmts)
    global default_scope
    default_scope = statements.default_scopes["lua"]


def write_stmts_tree(fp):
    """Write out statements tree.

    Parameters
    ----------
    fp : file
    """
    lines = []
    statements.print_tree_index(lua_tree, lines)
    fp.writelines(lines)
    statements.print_tree_statements(fp, lua_dict, default_stmts)
    

def lookup_stmts(path):
    return statements.lookup_stmts_tree(lua_tree, path)

LuaStmts = util.Scope(None,
    name="lua_default",
    pre_call=[],
    call=[],
    post_call=[],
)

default_stmts = dict(
    lua=LuaStmts,
)
        
lua_statements = [
    # Factor out some common code patterns to use as mixins.
    dict(
        # Used to capture return value.
        # Used with intent(result).
        name="lua_mixin_callfunction",
        call=[
            "{rv_asgn}{LUA_this_call}{function_name}({cxx_call_list});",
        ],
    ),
#    dict(
#        # Pop an argument off of the stack
#        name="lua_mixin_pop",
#        pre_call=[
#            "// pre_call",
#        ],
#    ),
    dict(
        # Used to capture return value.
        # Used with intent(result).
        name="lua_mixin_push",
        post_call=[
            "{push_expr};",
        ],
    ),
    #####
    # subroutine
    dict(
        name="lua_subroutine",
        call=[
            "{LUA_this_call}{function_name}({cxx_call_list});",
        ],
    ),
    #####
    # void
    dict(
        name="lua_function_void_*",
        mixin=[
            "lua_mixin_callfunction",
        ],
    ),
    #####
    # bool
    dict(
        name="lua_in_bool_scalar",
        pre_call=[
            "bool {c_var} = {pop_expr};",
        ],
    ),
    dict(
        name="lua_function_bool_scalar",
        mixin=[
            "lua_mixin_callfunction",
            "lua_mixin_push"
        ],
    ),
    #####
    # native
    dict(
        name="lua_in_native_scalar",
        pre_call=[
            "{cxx_type} {cxx_var} =\t {pop_expr};",
        ],
    ),
    dict(
        name="lua_inout_native_*",
        pre_call=[
            "// lua_native_*_inout;",
        ],
    ),
    dict(
        name="lua_function_native_scalar",
        mixin=[
            "lua_mixin_callfunction",
            "lua_mixin_push"
        ],
    ),
    #####
    # string
    dict(
        name="lua_in_string_*",
        pre_call=[
            "const char * {c_var} = \t{pop_expr};",
        ],
    ),
    dict(
        name="lua_in_string_&",
        base="lua_in_string_*",
    ),
    dict(
        name="lua_function_string_scalar",
        mixin=[
            "lua_mixin_callfunction",
            "lua_mixin_push"
        ],
    ),
    dict(
        name="lua_function_string_&",
        mixin=[
            "lua_mixin_callfunction",
            "lua_mixin_push"
        ],
    ),
    #####
    # shadow
    dict(
        name="lua_ctor_shadow",
        call=[
            "{LUA_userdata_type} * {LUA_userdata_var} ="
                "\t ({LUA_userdata_type} *) lua_newuserdata"
                "({LUA_state_var}, sizeof(*{LUA_userdata_var}));",
            "{LUA_userdata_var}->{LUA_userdata_member} ="
                "\t new {namespace_scope}{cxx_class}({cxx_call_list});",
            "/* Add the metatable to the stack. */",
            'luaL_getmetatable(L, "{LUA_metadata}");',
            "/* Set the metatable on the userdata. */",
            "lua_setmetatable(L, -2);",
        ],
    ),
    dict(
        name="lua_dtor_shadow",
        call=[
            "delete {LUA_userdata_var}->{LUA_userdata_member};",
            "{LUA_userdata_var}->{LUA_userdata_member} = NULL;",
        ],
    ),
    dict(
        name="lua_in_shadow_*",
        pre_call=[
            "{cxx_type} * {cxx_var} =\t {pop_expr};",
        ],
    ),
    dict(
        name="lua_function_shadow_*",
        mixin=[
            "lua_mixin_callfunction",
            "lua_mixin_push",
        ],
    ),
]
