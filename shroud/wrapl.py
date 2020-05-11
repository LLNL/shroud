# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
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
            if not cls.options.wrap_lua:
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
            if ns.options.wrap_lua:
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
            if not function.options.wrap_lua:
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

        if node.return_this:
            # XXX           result_type = 'void'
            # XXX           result_is_ptr = False
            CXX_subprogram = "subroutine"

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

    def do_function(self, cls, luafcn, fmt):
        """
        Wrap a single function/overload/default-argument
        variation of a function.

        Args:
            cls -
            luafcn -
            fmt - local format dictionary
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

        CXX_subprogram = node.CXX_subprogram
        result_typemap = node.CXX_result_typemap
        ast = node.ast
        is_ctor = ast.is_ctor()
        is_dtor = ast.is_dtor()

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

        if CXX_subprogram == "function":
            fmt_result0 = node._fmtresult
            fmt_result = fmt_result0.setdefault("fmtl", util.Scope(fmt))
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

            fmt.rv_decl = ast.gen_arg_as_cxx(
                name=fmt_result.cxx_var, params=None
            )
            fmt.rv_asgn = fmt.rv_decl + " =\t "

        LUA_decl = []  # declare variables and pop values
        LUA_code = []  # call C++ function
        LUA_push = []  # push results

        # post_parse = []
        cxx_call_list = []

        # find class object
        if cls:
            cls_typedef = cls.typemap
            if not is_ctor:
                fmt.LUA_used_param_state = True
                fmt.c_var = wformat(cls_typedef.LUA_pop, fmt)
                append_format(
                    LUA_code,
                    "{LUA_userdata_type} * {LUA_userdata_var} =\t {c_var};",
                    fmt,
                )

        # parse arguments
        # call function based on number of default arguments provided
        # XXX default_calls = []   # each possible default call
        # XXX if '_has_default_arg' in node:
        # XXX     append_format(LUA_decl, 'int SH_nargs =
        # XXX          lua_gettop({LUA_state_var});', fmt)

        # Only process nargs.
        # Each variation of default-arguments produces a new call.
        fmt_arg = util.Scope(fmt)
        LUA_index = 1
        for iarg in range(luafcn.nargs):
            arg = ast.params[iarg]
            arg_name = arg.name
            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg = fmt_arg0.setdefault("fmtl", util.Scope(fmt))
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

            lua_pop = None

            arg_typemap = arg.typemap
            fmt_arg.cxx_type = arg_typemap.cxx_type
            if attrs["intent"] in ["inout", "in"]:
                # XXX lua_pop = wformat(arg_typemap.LUA_pop, fmt_arg)
                # lua_pop is a C++ expression
                fmt_arg.c_var = wformat(arg_typemap.LUA_pop, fmt_arg)
                if arg_typemap.c_to_cxx is None:
                    lua_pop = fmt_arg.c_var
                else:
                    lua_pop = wformat(arg_typemap.c_to_cxx, fmt_arg)
                LUA_index += 1

            if attrs["intent"] in ["inout", "out"]:
                # output variable must be a pointer
                # XXX - fix up for strings
                # XXX  format, vargs = self.intent_out(
                # XXX      arg_typemap, fmt_arg, post_call)
                # XXX  build_format.append(format)
                # XXX  build_vargs.append('*' + vargs)

                # append_format(LUA_push, arg_typemap.LUA_push, fmt_arg)
                fmt.LUA_used_param_state = True
                tmp = wformat(arg_typemap.LUA_push, fmt_arg)
                LUA_push.append(tmp + ";")

            # argument for C++ function
            # This has been replaced by gen_arg methods, but not sure about const.
            #            lang = 'cxx_type'
            #            arg_const = False
            #            if arg_typemap.base == 'string':
            #                # C++ will coerce char * to std::string
            #                lang = 'c_type'
            #                arg_const = True  # lua_tostring is const
            #            if arg.is_reference():
            #                # convert a reference to a pointer
            #                ptr = True
            #            else:
            #                ptr = False

            if lua_pop:
                fmt.LUA_used_param_state = True
                decl_suffix = " =\t {};".format(lua_pop)
            else:
                decl_suffix = ";"
            if arg_typemap.base == "string":
                LUA_decl.append(
                    arg.gen_arg_as_c(continuation=True) + decl_suffix
                )
            else:
                LUA_decl.append(
                    arg.gen_arg_as_cxx(as_ptr=True, continuation=True)
                    + decl_suffix
                )

            cxx_call_list.append(fmt_arg.cxx_var)

        # call with arguments
        fmt.cxx_call_list = ",\t ".join(cxx_call_list)
        #        LUA_code.extend(post_parse)

        if is_ctor:
            fmt.LUA_used_param_state = True
            append_format(
                LUA_code,
                "{LUA_userdata_type} * {LUA_userdata_var} ="
                "\t ({LUA_userdata_type} *) lua_newuserdata"
                "({LUA_state_var}, sizeof(*{LUA_userdata_var}));\n"
                "{LUA_userdata_var}->{LUA_userdata_member} ="
                "\t new {namespace_scope}{cxx_class}({cxx_call_list});\n"
                "/* Add the metatable to the stack. */\n"
                'luaL_getmetatable(L, "{LUA_metadata}");\n'
                "/* Set the metatable on the userdata. */\n"
                "lua_setmetatable(L, -2);",
                fmt,
            )
        elif is_dtor:
            fmt.LUA_used_param_state = True
            append_format(
                LUA_code,
                "delete {LUA_userdata_var}->{LUA_userdata_member};\n"
                "{LUA_userdata_var}->{LUA_userdata_member} = NULL;",
                fmt,
            )
        elif CXX_subprogram == "subroutine":
            append_format(
                LUA_code,
                "{LUA_this_call}{function_name}({cxx_call_list});",
                fmt,
            )
        else:
            append_format(
                LUA_code,
                "{rv_asgn}{LUA_this_call}{function_name}({cxx_call_list});",
                fmt,
            )

        #        if 'LUA_error_pattern' in node:
        #            lfmt = util.Scope(fmt)
        #            lfmt.c_var = fmt.LUA_result
        #            lfmt.cxx_var = fmt.LUA_result
        #            append_format(LUA_code,
        #                 self.patterns[node['LUA_error_pattern']], lfmt)

        # Compute return value
        if CXX_subprogram == "function" and not is_ctor:
            fmt.LUA_used_param_state = True
            tmp = wformat(result_typemap.LUA_push, fmt_result)
            LUA_push.append(tmp + ";")

        lines = self.splicer_lines
        lines.extend(LUA_decl)
        lines.extend(LUA_code)
        # int lua_checkstack (lua_State *L, int extra)
        lines.extend(LUA_push)  # return values

    def write_header(self, node):
        """
        Args:
            node -
        """
        fmt = node.fmtdict
        fname = fmt.LUA_header_filename

        output = []

        # add guard
        guard = fname.replace(".", "_").upper()
        output.extend(["#ifndef %s" % guard, "#define %s" % guard])
        util.extern_C(output, "begin")

        for include in node.cxx_header:
            output.append('#include "%s"' % include)

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

        output = []

        for include in node.cxx_header:
            output.append('#include "{}"'.format(include))
        append_format(output, '#include "{LUA_header_filename}"', fmt)

        util.extern_C(output, "begin")
        output.append('#include "lauxlib.h"')
        util.extern_C(output, "end")

        self._create_splicer("include", output)

        self._create_splicer("C_definition", output)

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
    and overloaded/default-argument function
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
