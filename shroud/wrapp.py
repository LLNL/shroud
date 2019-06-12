# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-738041.
#
# All rights reserved.
#
# This file is part of Shroud.
#
# For details about use and distribution, please read LICENSE.
#
########################################################################
"""
Generate Python module for C++ code.

Entire library in a single header.
One Extension module per class


Variables prefixes used by generated code:
SH_     C or C++ version of argument
SHPy_   Python object which corresponds to the argument
SHTPy_  A temporary object, usually from PyArg_Parse
        to be converted to SHPy_ object.
SHDPy_  PyArray_Descr object
SHD_    npy_intp array for shape
SHC_    PyCapsule owner of memory of NumPy array.  Used to deallocate memory.

"""
from __future__ import print_function
from __future__ import absolute_import

import collections
import os
import re

from . import declast
from . import todict
from . import util
from . import whelpers
from .util import wformat, append_format

# If multiple values are returned, save up into to build a tuple to return.
# else build value from ctor, then return ctorvar.
# The value may be built earlier (bool, array), if so ctor will be None.
# format   - Format arg to PyBuild_Tuple
# vargs    - Variable for PyBuild_Tuple
# ctor     - Code to construct a Python object
# ctorvar  - Variable created by ctor
BuildTuple = collections.namedtuple("BuildTuple", "format vargs ctor ctorvar")


class Wrapp(util.WrapperMixin):
    """Generate Python bindings.
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
        self.patterns = self.newlibrary.patterns
        self.config = config
        self.log = config.log
        self._init_splicer(splicers)
        self.comment = "//"
        self.cont = ""
        self.linelen = newlibrary.options.C_line_length
        self.doxygen_begin = "/**"
        self.doxygen_cont = " *"
        self.doxygen_end = " */"
        self.need_numpy = False
        self.enum_impl = []
        self.arraydescr = []  # Create PyArray_Descr for struct
        self.decl_arraydescr = []
        self.define_arraydescr = []
        self.call_arraydescr = []

    def XXX_begin_output_file(self):
        """Start a new class for output"""
        pass

    def XXX_end_output_file(self):
        pass

    def XXX_begin_class(self):
        pass

    def reset_file(self):
        """Start a new output file"""
        self.PyMethodBody = []
        self.PyMethodDef = []
        self.PyGetSetBody = []
        self.PyGetSetDef = []
        self.c_helper = {}
#        self.c_helper_include = {}  # include files in generated C header

    def wrap_library(self):
        newlibrary = self.newlibrary
        fmt_library = newlibrary.fmtdict

        if self.language == "c":
            fmt_library.PY_header_filename_suffix = "h"
            fmt_library.PY_impl_filename_suffix = "c"
            fmt_library.PY_extern_C_begin = ""
        else:
            fmt_library.PY_header_filename_suffix = "hpp"
            fmt_library.PY_impl_filename_suffix = "cpp"
            fmt_library.PY_extern_C_begin = 'extern "C" '

        # Format variables
        newlibrary.eval_template("PY_module_filename")
        newlibrary.eval_template("PY_header_filename")
        newlibrary.eval_template("PY_utility_filename")
        fmt_library.PY_obj = "obj"  # name of cpp class pointer in PyObject
        fmt_library.PY_PyObject = "PyObject"
        fmt_library.PY_param_self = "self"
        fmt_library.PY_param_args = "args"
        fmt_library.PY_param_kwds = "kwds"
        fmt_library.PY_used_param_self = False
        fmt_library.PY_used_param_args = False
        fmt_library.PY_used_param_kwds = False

        # Variables to accumulate output lines
        self.py_type_object_creation = []
        self.py_class_decl = []
        self.py_utility_definition = []
        self.py_utility_declaration = []
        self.py_utility_functions = []
        # reserved the 0 slot of capsule_order
        # self.add_capsule_code('--none--', ['// not yet implemented'])

        # preprocess all classes first to allow them to reference each other
        for node in newlibrary.classes:
            if not node.options.wrap_python:
                continue
            ntypemap = node.typemap
            fmt = node.fmtdict
            ntypemap.PY_format = "O"

            # PyTypeObject for class
            node.eval_template("PY_PyTypeObject")

            # PyObject for class
            node.eval_template("PY_PyObject")

            fmt.PY_to_object_func = wformat("PP_{cxx_class}_to_Object", fmt)
            fmt.PY_from_object_func = wformat("PP_{cxx_class}_from_Object", fmt)

            ntypemap.PY_PyTypeObject = fmt.PY_PyTypeObject
            ntypemap.PY_PyObject = fmt.PY_PyObject
            ntypemap.PY_to_object = fmt.PY_to_object_func
            ntypemap.PY_from_object = fmt.PY_from_object_func

        self._push_splicer("class")
        for node in newlibrary.classes:
            if not node.options.wrap_python:
                continue
            name = node.name
            self.reset_file()
            self._push_splicer(name)
            if node.as_struct:
                self.create_arraydescr(node)
            else:
                self.wrap_class(node)
                self.write_extension_type(newlibrary, node)
            self._pop_splicer(name)
        self._pop_splicer("class")

        self.reset_file()
        self.wrap_enums(None)

        if newlibrary.functions:
            self._push_splicer("function")
            #            self._begin_class()
            self.wrap_functions(None, newlibrary.functions)
            self._pop_splicer("function")

        self.write_utility()
        self.write_header(newlibrary)
        self.write_module(newlibrary)

    def wrap_enums(self, cls):
        """Wrap enums for library or cls

        Args:
            cls - ast.ClassNode.
        """
        if cls is None:
            enums = self.newlibrary.enums
        else:
            enums = cls.enums
        if not enums:
            return
        self._push_splicer("enums")
        for enum in enums:
            self.wrap_enum(enum, cls)
        self._pop_splicer("enums")

    def wrap_enum(self, node, cls):
        """Wrap an enumeration.
        If module, use PyModule_AddIntConstant.
        If class, create a descriptor.
        Without a setter, it will be read-only.

        Args:
            node -
            cls -
        """
        fmtmembers = node._fmtmembers

        ast = node.ast
        output = self.enum_impl
        if cls is None:
            # library enumerations
            # m is module pointer from module_middle
            output.append("")
            append_format(output, "// enum {namespace_scope}{enum_name}",
                          node.fmtdict)
            for member in ast.members:
                fmt_id = fmtmembers[member.name]
                append_format(
                    output,
                    'PyModule_AddIntConstant(m, "{enum_member_name}",'
                    " {namespace_scope}{enum_member_name});",
                    fmt_id,
                )
        else:
            output.append("\n{+")
            append_format(output, "// enumeration {enum_name}", node.fmtdict)
            output.append("PyObject *tmp_value;")
            for member in ast.members:
                fmt_id = fmtmembers[member.name]
                append_format(
                    output,
                    "tmp_value = PyLong_FromLong("
                    "{namespace_scope}{enum_member_name});\n"
                    "PyDict_SetItemString("
                    "(PyObject*) {PY_PyTypeObject}.tp_dict,"
                    ' "{enum_member_name}", tmp_value);\n'
                    "Py_DECREF(tmp_value);",
                    fmt_id,
                )
            output.append("-}")

    def wrap_class(self, node):
        """
        Args:
            node - ast.ClassNode.
        """
        self.log.write("class {1.name}\n".format(self, node))
        fmt_class = node.fmtdict

        node.eval_template("PY_type_filename")
        fmt_class.PY_this_call = wformat("self->{PY_obj}->", fmt_class)

        output = self.py_type_object_creation
        output.append("")
        if node.cpp_if:
            output.append("#" + node.cpp_if)
        output.append(
            wformat(
                """// {cxx_class}
{PY_PyTypeObject}.tp_new   = PyType_GenericNew;
{PY_PyTypeObject}.tp_alloc = PyType_GenericAlloc;
if (PyType_Ready(&{PY_PyTypeObject}) < 0)
+return RETVAL;-
Py_INCREF(&{PY_PyTypeObject});
PyModule_AddObject(m, "{cxx_class}", (PyObject *)&{PY_PyTypeObject});""",
                fmt_class,
            )
        )
        if node.cpp_if:
            output.append("#endif // " + node.cpp_if)

        # header declarations
        output = self.py_class_decl
        output.append("")
        output.append("// ------------------------------")
        if node.cpp_if:
            output.append("#" + node.cpp_if)
        self.write_namespace(node, "begin", output)
        output.append("class {};  // forward declare".format(node.name))
        self.write_namespace(node, "end", output, comment=False)

        output.append(wformat("extern PyTypeObject {PY_PyTypeObject};", fmt_class))

        self._create_splicer("C_declaration", output)
        append_format(
            output,
            "\n"
            "typedef struct {{\n"
            "PyObject_HEAD\n"
            "+{namespace_scope}{cxx_class} * {PY_obj};",
            fmt_class,
        )
        self._create_splicer("C_object", output)
        append_format(output, "-}} {PY_PyObject};", fmt_class)
        output.append("")

        self.create_class_utility_functions(node)
        if node.cpp_if:
            output.append("#endif // " + node.cpp_if)

        self.wrap_enums(node)

        for var in node.variables:
            self.wrap_class_variable(var)

        # wrap methods
        self.tp_init_default = "0"
        self._push_splicer("method")
        self.wrap_functions(node, node.functions)
        self._pop_splicer("method")

    def create_class_utility_functions(self, node):
        """Create some utility functions to convert to and from a PyObject.
        These functions are used by PyArg_ParseTupleAndKeywords
        and Py_BuildValue node is a C++ class.
        """
        fmt = node.fmtdict

        fmt.PY_capsule_name = wformat("PY_{cxx_class}_capsule_name", fmt)

        if node.cpp_if:
            cpp_if = "#" + node.cpp_if + "\n"
            cpp_endif = "\n#endif  // " + node.cpp_if
        else:
            cpp_if = ""
            cpp_endif = ""

        self._push_splicer("utility")
        append_format(
            self.py_utility_definition,
            cpp_if + 'const char *{PY_capsule_name} = "{cxx_class}";' + cpp_endif,
            fmt,
        )
        append_format(
            self.py_class_decl,
            "extern const char *{PY_capsule_name};",
            fmt,
        )

        # To
        to_object = wformat(
            """PyObject *voidobj;
PyObject *args;
PyObject *rv;

voidobj = PyCapsule_New(addr, {PY_capsule_name}, NULL);
args = PyTuple_New(1);
PyTuple_SET_ITEM(args, 0, voidobj);
rv = PyObject_Call((PyObject *) &{PY_PyTypeObject}, args, NULL);
Py_DECREF(args);
return rv;""",
            fmt,
        )
        to_object = to_object.split("\n")

        proto = wformat(
            "PyObject *{PY_to_object_func}({namespace_scope}{cxx_class} *addr)",
            fmt,
        )
        self.py_class_decl.append(proto + ";")

        self.py_utility_functions.append("")
        if node.cpp_if:
            self.py_utility_functions.append("#" + node.cpp_if)
        self.py_utility_functions.append(proto)
        self.py_utility_functions.append("{+")
        self._create_splicer("to_object", self.py_utility_functions, to_object)
        self.py_utility_functions.append("-}")

        # From
        from_object = wformat(
            """if (obj->ob_type != &{PY_PyTypeObject}) {{
    // raise exception
    return 0;
}}
{PY_PyObject} * self = ({PY_PyObject} *) obj;
*addr = self->{PY_obj};
return 1;""",
            fmt,
        )
        from_object = from_object.split("\n")

        proto = wformat(
            "int {PY_from_object_func}(PyObject *obj, void **addr)", fmt
        )
        self.py_class_decl.append(proto + ";")

        self.py_utility_functions.append("")
        self.py_utility_functions.append(proto)
        self.py_utility_functions.append("{")
        self.py_utility_functions.append(1)
        self._create_splicer(
            "from_object", self.py_utility_functions, from_object
        )
        self.py_utility_functions.append(-1)
        self.py_utility_functions.append("}")
        if node.cpp_if:
            self.py_utility_functions.append("#endif  // " + node.cpp_if)

        self._pop_splicer("utility")

    def create_arraydescr(self, node):
        """Create a NumPy PyArray_Descr for a struct.
        Install into module.

        struct {
          int ifield;
          double dfield;
        };

        numpy.dtype(
          {'names': ['ifield', 'dfield'],
           'formats': [np.int32, np.float64]},
           'offsets':[0,8],
           'itemsize':12},
          align=True)

        Args:
            node -
        """
        fmt = node.fmtdict

        self.need_numpy = True

        self.decl_arraydescr.append(
            wformat(
                "extern PyArray_Descr *{PY_struct_array_descr_variable};", fmt
            )
        )
        self.define_arraydescr.append(
            wformat("PyArray_Descr *{PY_struct_array_descr_variable};", fmt)
        )
        append_format(
            self.call_arraydescr,
            "{PY_struct_array_descr_variable} = {PY_struct_array_descr_create}();\n"
            'PyModule_AddObject(m, "{PY_struct_array_descr_name}",'
            " \t(PyObject *) {PY_struct_array_descr_variable});",
            fmt,
        )
        output = self.arraydescr
        output.append("")
        append_format(
            output,
            "// Create PyArray_Descr for {cxx_class}\n"
            "PyArray_Descr *{PY_struct_array_descr_create}()",
            fmt,
        )
        output.append("{")
        output.append(1)

        nvars = len(node.variables)
        output.extend(
            [
                "int ierr;",
                "PyObject *obj = NULL;",
                "PyObject * lnames = NULL;",
                "PyObject * ldescr = NULL;",
                "PyObject * dict = NULL;",
                "PyArray_Descr *dtype = NULL;",
                "",
                "lnames = PyList_New({});".format(nvars),
                "if (lnames == NULL) goto fail;",
                "ldescr = PyList_New({});".format(nvars),
                "if (ldescr == NULL) goto fail;",
            ]
        )

        for i, var in enumerate(node.variables):
            ast = var.ast
            output.extend(
                [
                    "",
                    "// " + var.ast.name,
                    'obj = PyString_FromString("{}");'.format(ast.name),
                    "if (obj == NULL) goto fail;",
                    "PyList_SET_ITEM(lnames, {}, obj);".format(i),
                ]
            )

            arg_typemap = ast.typemap
            output.extend(
                [
                    "obj = (PyObject *) PyArray_DescrFromType({});".format(
                        arg_typemap.PYN_typenum
                    ),
                    "if (obj == NULL) goto fail;",
                    "PyList_SET_ITEM(ldescr, {}, obj);".format(i),
                ]
            )

            # XXX - add offset and itemsize to be explicit?

        output.extend(
            [
                "obj = NULL;",
                "",
                "dict = PyDict_New();",
                "if (dict == NULL) goto fail;",
                'ierr = PyDict_SetItemString(dict, "names", lnames);',
                "if (ierr == -1) goto fail;",
                "lnames = NULL;",
                'ierr = PyDict_SetItemString(dict, "formats", ldescr);',
                "if (ierr == -1) goto fail;",
                "ldescr = NULL;",
                # 'Py_INCREF(Py_True);',
                # 'ierr = PyDict_SetItemString(descr, "aligned", Py_True);',
                # 'if (ierr == -1) goto fail;',
                "ierr = PyArray_DescrAlignConverter(dict, &dtype);",
                "if (ierr == 0) goto fail;",
                "return dtype;",
            ]
        )
        output.extend(
            [
                "^fail:",
                "Py_XDECREF(obj);",
                "if (lnames != NULL) {+",
                "for (int i=0; i < {}; i++) {{+".format(nvars),
                "Py_XDECREF(PyList_GET_ITEM(lnames, i));",
                "-}",
                "Py_DECREF(lnames);",
                "-}",
                "if (ldescr != NULL) {+",
                "for (int i=0; i < {}; i++) {{+".format(nvars),
                "Py_XDECREF(PyList_GET_ITEM(ldescr, i));",
                "-}",
                "Py_DECREF(ldescr);",
                "-}",
                "Py_XDECREF(dict);",
                "Py_XDECREF(dtype);",
                "return NULL;",
            ]
        )
        #    int PyArray_RegisterDataType(descr)

        output.append(-1)
        output.append("}")

    def wrap_class_variable(self, node):
        """Wrap a VariableNode in a class with descriptors.

        Args:
            node - ast.VariableNode.
        """
        options = node.options
        fmt_var = node.fmtdict
        fmt_var.PY_getter = wformat(options.PY_member_getter_template, fmt_var)
        fmt_var.PY_setter = "NULL"  # readonly

        fmt = util.Scope(fmt_var)
        fmt.c_var = wformat("{PY_param_self}->{PY_obj}->{field_name}", fmt_var)
        fmt.c_deref = ""  # XXX needed for PY_ctor
        fmt.py_var = "value"  # Used with PY_get

        ast = node.ast
        arg_typemap = ast.typemap

        if arg_typemap.PY_ctor:
            fmt.ctor = wformat(arg_typemap.PY_ctor, fmt)
        else:
            fmt.ctor = "UUUctor"
        fmt.cxx_decl = ast.gen_arg_as_cxx(name="rv")

        output = self.PyGetSetBody
        append_format(
            output,
            "\nstatic PyObject *{PY_getter}("
            "{PY_PyObject} *{PY_param_self},"
            "\t void *SHROUD_UNUSED(closure))\n"
            "{{+\nPyObject * rv = {ctor};\nreturn rv;"
            "\n-}}",
            fmt,
        )

        # setter
        if not ast.attrs.get("readonly", False):
            fmt_var.PY_setter = wformat(
                options.PY_member_setter_template, fmt_var
            )
            if arg_typemap.PY_get:
                fmt.get = wformat(arg_typemap.PY_get, fmt)
            else:
                fmt.get = "UUUget"

            append_format(
                output,
                "\nstatic int {PY_setter}("
                "{PY_PyObject} *{PY_param_self}, PyObject *{py_var},"
                "\t void *SHROUD_UNUSED(closure))\n{{+\n"
                "{cxx_decl} = {get};",
                fmt,
            )
            output.append("if (PyErr_Occurred()) {\n+return -1;-\n}")
            # XXX - allow user to add error checks on value
            output.append(fmt.c_var + " = rv;")
            output.append("return 0;\n-}")

        # Set pointers to functions
        self.PyGetSetDef.append(
            # XXX - the (char *) only needed for C++
            wformat(
                '{{(char *)"{variable_name}",\t '
                "(getter){PY_getter},\t "
                "(setter){PY_setter},\t "
                "NULL, "  # doc
                "NULL}},",
                fmt_var,
            )
        )  # closure

    def allocatable_blk(self, allocatable, node, arg, fmt_arg):
        """Allocate NumPy Array.
        Assumes intent(out)

        Args:
            allocatable -
            node -
            arg -
            fmt_arg -

        Examples:
        (int arg1, int arg2 +intent(out)+allocatable(mold=arg1))
        """
        fmt_arg.py_type = "PyObject"

        attr_allocatable(self.language, allocatable, node, arg, fmt_arg)
        index = "intent_out_{}_allocatable_numpy".format(self.language)
        blk = py_statements_local[index]
        self.need_numpy = self.need_numpy or blk.get("need_numpy", False)
        return blk

    def dimension_blk(self, arg, fmt_arg, options):
        """Create code needed for a dimensioned array argument.
        Convert it to use Numpy.

        Args:
            arg - argument node.
            fmt_arg -
            options - Scope

        Return a dictionary which defines fields
        of code to insert into the wrapper.

        Examples:
        ----------------------------------------
        (int * arg1 +intent(in) +dimension(:))
        """
        intent = arg.attrs["intent"]
        whelpers.add_to_PyList_helper(arg)
        if intent == "out":
            dimension = arg.attrs.get("dimension", None)
            if dimension is None:
                raise RuntimeError(
                    "Argument must have non-default dimension attribute")
            if dimension == "*":
                raise RuntimeError(
                    "Argument dimension must not be assumed-length")
            fmt_arg.npy_ndims = "1"
            fmt_arg.npy_dims = "SHD_" +  fmt_arg.cxx_var # ast.name
            fmt_arg.pointer_shape = dimension
        else:
            fmt_arg.py_type = "PyObject"
            fmt_arg.pytmp_var = "SHTPy_" + fmt_arg.c_var

        index = "intent_{}_{}_dimension_{}".format(intent, self.language, options.PY_array_arg)
        blk = py_statements_local[index]
        self.need_numpy = self.need_numpy or blk.get("need_numpy", False)
        return blk

    def array_result(self, capsule_order, ast, typemap, fmt):
        """
        Deal with function result which is a NumPy array.

        A pointer or allocatable result, which is not a string,
        creates a NumPy array.
        Return an intent_blk with post_call set which contains
        code to create NumPy array.

        Args:
            capsule_order - index into capsule_order of code to free memory.
                            None = do not release memory.
            ast - Abstract Syntax Tree of argument or result
            typemap - typemap of C++ variable.
            fmt - format dictionary
        """
        post_call = []

        fmt.PyObject = typemap.PY_PyObject or "PyObject"
        fmt.PyTypeObject = typemap.PY_PyTypeObject

        # Create a 1-d array from pointer.
        # A string is not really an array, so do not deal with it here.
        dim = ast.attrs.get("dimension", None)
        # Dimensions must be in npy_intp type array.
        self.need_numpy = True
        if dim:
            fmt.npy_ndims = "1"
            fmt.npy_dims = "SHD_" + ast.name
            fmt.pointer_shape = dim
            append_format(
                post_call,
                "npy_intp {npy_dims}[1] = {{{{ {pointer_shape} }}}};",
                fmt,
            )
        else:
            fmt.npy_ndims = "0"
            fmt.npy_dims = "NULL"
        if typemap.PYN_descr:
            fmt.PYN_descr = typemap.PYN_descr
            append_format(
                post_call,
                "Py_INCREF({PYN_descr});\n"
                "PyObject * {py_var} = "
                "PyArray_NewFromDescr(&PyArray_Type, \t{PYN_descr},\t"
                " {npy_ndims}, {npy_dims}, \tNULL, {cxx_var}, 0, NULL);",
                fmt,
            )
        else:
            append_format(
                post_call,
                "PyObject * {py_var} = "
                "PyArray_SimpleNewFromData({npy_ndims},\t {npy_dims},"
                "\t {numpy_type},\t {cxx_var});",
                fmt,
            )

        if capsule_order is not None:
            # If NumPy owns the memory, add a way to delete it
            # by creating a capsule base object.
            fmt.py_capsule = "SHC_" + fmt.c_var
            context = do_cast(
                self.language,
                "const",
                "char *",
                "{}[{}]".format(
                    fmt.PY_numpy_array_dtor_context, capsule_order
                ),
            )
            append_format(
                post_call,
                "PyObject * {py_capsule} = "
                'PyCapsule_New({cxx_var}, "{PY_numpy_array_capsule_name}", '
                "\t{PY_numpy_array_dtor_function});\n"
                "PyCapsule_SetContext({py_capsule}, " + context + ");\n"
                "PyArray_SetBaseObject((PyArrayObject *) {py_var}, {py_capsule});",  # 0=ok, -1=error
                fmt,
            )

        # Return a dictionary which is used as an intent_blk.
        return dict(
            post_call=post_call,
        )

    def implied_blk(self, node, arg, pre_call):
        """Add the implied attribute to the pre_call block.

        Called after all input arguments have their fmtpy dictionary
        updated.
        Added into wrapper after post_parse code is inserted --
        i.e. all intent in,inout arguments have been evaluated
        and PyArrayObjects created.

        Args:
            node -
            arg -
            pre_call -
        """
        implied = arg.attrs.get("implied", None)
        if implied:
            fmt = node._fmtargs[arg.name]["fmtpy"]
            fmt.pre_call_intent = py_implied(implied, node)
            append_format(pre_call, "{cxx_decl} = {pre_call_intent};", fmt)

    def intent_out(self, typemap, intent_blk, fmt, post_call):
        """Add code for post-call.
        Create PyObject from C++ value to return.
        Used with function results and intent(OUT) arguments.

        Args:
            typemap - typemap of C++ variable.
            intent_blk -
            fmt - format dictionary
            post_call   - list of post_call code for function.

        NumPy intent(OUT) arguments will create a Python object as part of pre-call.
        Return a BuildTuple instance.
        """
        fmt.PyObject = typemap.PY_PyObject or "PyObject"
        fmt.PyTypeObject = typemap.PY_PyTypeObject

        if "post_call" in intent_blk:
            # Explicit code exists to create object.
            # If post_call is None, the Object has already been created
            util.append_format_cmds(post_call, intent_blk, "post_call", fmt)
            build_format = "O"
            vargs = fmt.py_var
            ctor = None
            ctorvar = fmt.py_var
        else:
            # Decide values for Py_BuildValue
            build_format = typemap.PY_build_format or typemap.PY_format
            vargs = typemap.PY_build_arg
            if not vargs:
                vargs = "{cxx_var}"
            vargs = wformat(vargs, fmt)

            if typemap.PY_ctor:
                ctor = wformat(
                    "{PyObject} * {py_var} = " + typemap.PY_ctor + ";", fmt
                )
                ctorvar = fmt.py_var
            else:
                fmt.PY_build_format = build_format
                fmt.vargs = vargs
                ctor = wformat(
                    "{PyObject} * {py_var} = "
                    'Py_BuildValue("{PY_build_format}", {vargs});',
                    fmt,
                )
                ctorvar = fmt.py_var

        return BuildTuple(build_format, vargs, ctor, ctorvar)

    def wrap_functions(self, cls, functions):
        """Wrap functions for a library or class.
        Compute overloading map.

        Args:
            cls - ast.ClassNode
            functions -
        """
        overloaded_methods = {}
        for function in functions:
            flist = overloaded_methods.setdefault(function.ast.name, [])
            if not function.options.wrap_python:
                continue
            flist.append(function)
        self.overloaded_methods = overloaded_methods

        for function in functions:
            self.wrap_function(cls, function)

        self.multi_dispatch(functions)

    def wrap_function(self, cls, node):
        """Write a Python wrapper for a C or C++ function.

        Args:
            cls  - ast.ClassNode or None for functions
            node - ast.FunctionNode.

        fmt.c_var   - name of variable in PyArg_ParseTupleAndKeywords
        fmt.cxx_var - name of variable in c++ call.
        fmt.py_var  - name of PyObject variable

        # Used to prevent compiler warnings about unused variables.
        fmt.PY_used_param_args - True/False if parameter args is used
        fmt.PY_used_param_kwds - True/False if parameter kwds is used

        fmt.PY_error_return - 'NULL' for all wrappers except constructors
                              which are called via tp_init and require -1.
        """

        # need_rv - need Return Value declaration.
        #           The simplest case is to assign to rv as part of calling function.
        #           When default arguments are present, a switch statement is create
        #           so set need_rv = True to declare variable once, then call several time
        # need_malloc - Result is a scalar but we need to put it into a NumPy array,
        #           so allocate memory.

        options = node.options
        if not options.wrap_python:
            return

        if cls:
            cls_function = "method"
        else:
            cls_function = "function"
        self.log.write("Python {0} {1.declgen}\n".format(cls_function, node))

        fmt_func = node.fmtdict
        fmtargs = node._fmtargs
        fmt = util.Scope(fmt_func)
        fmt.PY_doc_string = "documentation"

        CXX_subprogram = node.CXX_subprogram
        result_typemap = node.CXX_result_typemap
        ast = node.ast
        is_ctor = ast.is_ctor()
        is_dtor = ast.is_dtor()
        #        is_const = ast.const
        ml_flags = []
        is_struct_scalar = False
        need_malloc = False
        result_return_pointer_as = ast.return_pointer_as

        if cls:
            if "static" in ast.storage:
                ml_flags.append("METH_STATIC")
                fmt_func.PY_this_call = (
                    fmt_func.namespace_scope + fmt_func.class_scope
                )
            else:
                fmt.PY_used_param_self = True

        if is_dtor:
            # Added in tp_del from write_tp_func.
            return
        elif is_ctor:
            fmt_func.PY_type_method = "tp_init"
            node.eval_template("PY_type_impl")
            fmt_func.PY_name_impl = fmt_func.PY_type_impl
            self.tp_init_default = fmt_func.PY_type_impl
            fmt.PY_error_return = "-1"
        else:
            node.eval_template("PY_name_impl")
            fmt.PY_error_return = "NULL"

        # XXX if a class, then knock off const since the PyObject
        # is not const, otherwise, use const from result.
        # This has been replaced by gen_arg methods, but not sure about const.
        #        if result_typemap.base == 'shadow':
        #            is_const = False
        #        else:
        #            is_const = None
        if CXX_subprogram == "function":
            fmt_result0 = node._fmtresult
            fmt_result = fmt_result0.setdefault(
                "fmtpy", util.Scope(fmt)
            )  # fmt_func

            CXX_result = ast
            if result_typemap.base == "struct" and not CXX_result.is_pointer():
                # Allocate variable to the type returned by the function.
                # No need to convert to C.
                is_struct_scalar = True
                result_return_pointer_as = "pointer"
                fmt_result.cxx_var = wformat("{C_local}{C_result}", fmt_result)
            elif result_typemap.cxx_to_c is None:
                fmt_result.cxx_var = wformat("{C_local}{C_result}", fmt_result)
            else:
                fmt_result.cxx_var = wformat(
                    "{CXX_local}{C_result}", fmt_result
                )

            if is_struct_scalar:
                # Force result to be a pointer to a struct
                need_malloc = True
                fmt.C_rv_decl = CXX_result.gen_arg_as_cxx(
                    name=fmt_result.cxx_var,
                    force_ptr=True,
                    params=None,
                    continuation=True,
                )
            else:
                fmt.C_rv_decl = CXX_result.gen_arg_as_cxx(
                    name=fmt_result.cxx_var, params=None, continuation=True
                )

            if CXX_result.is_pointer():
                fmt_result.c_deref = "*"
                fmt_result.cxx_addr = ""
                fmt_result.cxx_member = "->"
            else:
                fmt_result.c_deref = ""
                fmt_result.cxx_addr = "&"
                fmt_result.cxx_member = "."
            fmt_result.c_var = fmt_result.cxx_var
            fmt_result.py_var = fmt.PY_result
            fmt_result.numpy_type = result_typemap.PYN_typenum
        #            fmt_pattern = fmt_result

        PY_code = []

        # arguments to PyArg_ParseTupleAndKeywords
        parse_format = []
        parse_vargs = []

        # arguments to Py_BuildValue
        build_tuples = []

        # Code blocks
        PY_decl = []  # variables for function
        post_parse = []
        pre_call = []
        post_call = []  # Create objects passed to PyBuildValue
        cleanup_code = []
        fail_code = []

        cxx_call_list = []

        # parse arguments
        # call function based on number of default arguments provided
        default_calls = []  # each possible default call
        found_default = False
        if node._has_default_arg:
            PY_decl.append("Py_ssize_t SH_nargs = 0;")
            PY_code.extend(
                [
                    "if (args != NULL) SH_nargs += PyTuple_Size(args);",
                    "if (kwds != NULL) SH_nargs += PyDict_Size(args);",
                ]
            )

        goto_fail = False
        args = ast.params
        arg_names = []
        arg_offsets = []
        arg_implied = []  # Collect implied arguments
        offset = 0
        for arg in args:
            arg_name = arg.name
            fmt_arg0 = fmtargs.setdefault(arg_name, {})
            fmt_arg = fmt_arg0.setdefault("fmtpy", util.Scope(fmt))
            fmt_arg.c_var = arg_name
            fmt_arg.cxx_var = arg_name
            fmt_arg.py_var = "SHPy_" + arg_name

            arg_typemap = arg.typemap
            fmt_arg.numpy_type = arg_typemap.PYN_typenum
            # Add formats used by py_statements
            fmt_arg.c_type = arg_typemap.c_type
            fmt_arg.cxx_type = arg_typemap.cxx_type
            if arg.const:
                fmt_arg.c_const = "const "
            else:
                fmt_arg.c_const = ""
            if arg.is_pointer():
                fmt_arg.c_deref = "*"
                fmt_arg.cxx_addr = ""
                fmt_arg.cxx_member = "->"
            else:
                fmt_arg.c_deref = ""
                fmt_arg.cxx_addr = "&"
                fmt_arg.cxx_member = "."
            attrs = arg.attrs

            dimension = arg.attrs.get("dimension", False)
            pass_var = fmt_arg.c_var  # The variable to pass to the function
            # local_var - 'funcptr', 'pointer', or 'scalar'
            if arg.is_function_pointer():
                fmt_arg.c_decl = arg.gen_arg_as_c(continuation=True)
                fmt_arg.cxx_decl = arg.gen_arg_as_cxx(continuation=True)
                # not sure how function pointers work with Python.
                local_var = "funcptr"
            elif arg_typemap.base == "string":
                charlen = arg.attrs.get("charlen", False)
                if charlen:
                    fmt_arg.charlen = charlen
                    fmt_arg.c_decl = wformat("{c_const}char {c_var}[{charlen}]", fmt_arg)
                    fmt_arg.cxx_decl = fmt_arg.c_decl
                else:
                    fmt_arg.c_decl = wformat("{c_const}char * {c_var}", fmt_arg)
                    #                fmt_arg.cxx_decl = wformat('{c_const}char * {cxx_var}', fmt_arg)
                    fmt_arg.cxx_decl = arg.gen_arg_as_cxx()
                local_var = "pointer"
            elif arg.attrs.get("allocatable", False):
                fmt_arg.c_decl = wformat("{c_type} * {c_var}", fmt_arg)
                fmt_arg.cxx_decl = wformat("{cxx_type} * {cxx_var}", fmt_arg)
                local_var = "pointer"
            elif arg.attrs.get("dimension", False):
                fmt_arg.c_decl = wformat("{c_type} * {c_var}", fmt_arg)
                fmt_arg.cxx_decl = wformat("{cxx_type} * {cxx_var}", fmt_arg)
                local_var = "pointer"
            else:
                # non-strings should be scalars
                fmt_arg.c_deref = ""
                #                fmt_arg.cxx_addr = '&'
                #                fmt_arg.cxx_member = '.'
                fmt_arg.c_decl = wformat("{c_type} {c_var}", fmt_arg)
                fmt_arg.cxx_decl = wformat("{cxx_type} {cxx_var}", fmt_arg)
                local_var = "scalar"

            allocatable = attrs.get("allocatable", False)
            hidden = attrs.get("hidden", False)
            implied = attrs.get("implied", False)
            intent = attrs["intent"]
            if implied:
                arg_implied.append(arg)
                intent_blk = {}
            elif allocatable:
                intent_blk = self.allocatable_blk(
                    allocatable, node, arg, fmt_arg
                )
            elif dimension:
                intent_blk = self.dimension_blk(arg, fmt_arg, options)
            else:
                py_statements = arg_typemap.py_statements
                stmts = "intent_" + intent
                intent_blk = py_statements.get(stmts, {})

            goto_fail = goto_fail or intent_blk.get("goto_fail", False)
            cxx_local_var = intent_blk.get("cxx_local_var", "")
            if cxx_local_var:
                # With PY_PyTypeObject, there is no c_var, only cxx_var
                if not arg_typemap.PY_PyTypeObject:
                    fmt_arg.cxx_var = "SH_" + fmt_arg.c_var
                local_var = cxx_local_var
                pass_var = fmt_arg.cxx_var
                # cxx_member used with typemap fields like PY_ctor.
                if cxx_local_var == "scalar":
                    fmt_arg.cxx_member = "."
                elif cxx_local_var == "pointer":
                    fmt_arg.cxx_member = "->"

            if implied:
                pass
            elif intent in ["inout", "in"]:
                # names to PyArg_ParseTupleAndKeywords
                arg_names.append(arg_name)
                arg_offsets.append("(char *) SH_kwcpp+%d" % offset)
                offset += len(arg_name) + 1

                # XXX default should be handled differently
                if arg.init is not None:
                    if not found_default:
                        parse_format.append("|")  # add once
                        found_default = True
                    # call for default arguments  (num args, arg string)
                    default_calls.append(
                        (
                            len(cxx_call_list),
                            len(post_parse),
                            len(pre_call),
                            ",\t ".join(cxx_call_list),
                        )
                    )

                # Declare C variable - may be PyObject.
                # add argument to call to PyArg_ParseTypleAndKeywords
                if dimension:
                    # Use NumPy with dimensioned arguments
                    pass_var = fmt_arg.cxx_var
                    parse_format.append("O")
                    parse_vargs.append("&" + fmt_arg.pytmp_var)
                elif arg_typemap.PY_PyTypeObject:
                    # Expect object of given type
                    # cxx_var is declared by py_statements.intent_out.post_parse.
                    fmt_arg.py_type = arg_typemap.PY_PyObject or "PyObject"
                    append_format(PY_decl, "{py_type} * {py_var};", fmt_arg)
                    pass_var = fmt_arg.cxx_var
                    parse_format.append(arg_typemap.PY_format)
                    parse_format.append("!")
                    parse_vargs.append("&" + arg_typemap.PY_PyTypeObject)
                    parse_vargs.append("&" + fmt_arg.py_var)
                elif arg_typemap.PY_from_object:
                    # Use function to convert object
                    # cxx_var created directly (no c_var)
                    append_format(PY_decl, "{cxx_decl};", fmt_arg)
                    pass_var = fmt_arg.cxx_var
                    parse_format.append(arg_typemap.PY_format)
                    parse_format.append("&")
                    parse_vargs.append(arg_typemap.PY_from_object)
                    parse_vargs.append("&" + fmt_arg.cxx_var)
                else:
                    append_format(PY_decl, "{c_decl};", fmt_arg)
                    parse_format.append(arg_typemap.PY_format)
                    parse_vargs.append("&" + fmt_arg.c_var)

            if intent in ["inout", "out"]:
                if intent == "out":
                    if allocatable or dimension:
                        # If an array, a local NumPy array has already been defined.
                        pass
                    elif not cxx_local_var:
                        pass_var = fmt_arg.cxx_var
                        append_format(
                            pre_call, "{cxx_decl};  // intent(out)", fmt_arg
                        )

                if not hidden:
                    # output variable must be a pointer
                    build_tuples.append(
                        self.intent_out(arg_typemap, intent_blk, fmt_arg, post_call)
                    )

            # Code to convert parsed values (C or Python) to C++.
            util.append_format_cmds(PY_decl, intent_blk, "decl", fmt_arg)
            util.append_format_cmds(
                post_parse, intent_blk, "post_parse", fmt_arg
            )
            util.append_format_cmds(pre_call, intent_blk, "pre_call", fmt_arg)
            util.append_format_cmds(
                cleanup_code, intent_blk, "cleanup", fmt_arg
            )
            util.append_format_cmds(fail_code, intent_blk, "fail", fmt_arg)
            if "c_helper" in intent_blk:
                c_helper = wformat(intent_blk["c_helper"], fmt_arg)
                for helper in c_helper.split():
                    self.c_helper[helper] = True

            if intent != "out" and not cxx_local_var and arg_typemap.c_to_cxx:
                # Make intermediate C++ variable
                # Needed to pass address of variable
                # Helpful with debugging.
                fmt_arg.cxx_var = "SH_" + fmt_arg.c_var
                fmt_arg.cxx_decl = arg.gen_arg_as_cxx(
                    name=fmt_arg.cxx_var, params=None, continuation=True
                )
                fmt_arg.cxx_val = wformat(arg_typemap.c_to_cxx, fmt_arg)
                append_format(post_parse, "{cxx_decl} =\t {cxx_val};", fmt_arg)
                pass_var = fmt_arg.cxx_var

            # Pass correct value to wrapped function.
            if local_var == "scalar":
                if arg.is_pointer():
                    cxx_call_list.append("&" + pass_var)
                else:
                    cxx_call_list.append(pass_var)
            elif local_var == "pointer":
                if arg.is_pointer():
                    cxx_call_list.append(pass_var)
                else:
                    cxx_call_list.append("*" + pass_var)
            elif local_var == "funcptr":
                cxx_call_list.append(pass_var)
            else:
                raise RuntimeError("unexpected value of local_var")
        # end for arg in args:

        # Add implied argument initialization to pre_call code
        for arg in arg_implied:
            intent_blk = self.implied_blk(node, arg, pre_call)

        need_blank = False  # needed before next debug header
        if not arg_names:
            # no input arguments
            ml_flags.append("METH_NOARGS")
        else:
            ml_flags.append("METH_VARARGS")
            ml_flags.append("METH_KEYWORDS")
            fmt.PY_used_param_args = True
            fmt.PY_used_param_kwds = True
            need_blank = True

            if self.language == "cxx":
                kw_const = "const "
                fmt.PyArg_kwlist = "const_cast<char **>(SHT_kwlist)"
            else:
                kw_const = ""
                fmt.PyArg_kwlist = "SHT_kwlist"
            PY_decl.append(
                kw_const
                + 'char *SHT_kwlist[] = {\f"'
                + '",\f"'.join(arg_names)
                + '",\fNULL };'
            )
            parse_format.extend([":", fmt.function_name])
            fmt.PyArg_format = "".join(parse_format)
            fmt.PyArg_vargs = ",\t ".join(parse_vargs)
            append_format(
                PY_code,
                "if (!PyArg_ParseTupleAndKeywords"
                "({PY_param_args}, {PY_param_kwds},\t "
                '"{PyArg_format}",\t {PyArg_kwlist}, '
                "\t{PyArg_vargs}))\n"
                "+return {PY_error_return};-",
                fmt,
            )

        # call with all arguments
        default_calls.append(
            (
                len(cxx_call_list),
                len(post_parse),
                len(pre_call),
                ",\t ".join(cxx_call_list),
            )
        )

        # If multiple calls (because of default argument values),
        # declare return value once; else delare on call line.
        if found_default:
            if CXX_subprogram == "function":
                fmt.PY_rv_asgn = fmt_result.cxx_var + " =\t "
            PY_code.append("switch (SH_nargs) {")
        else:
            if CXX_subprogram == "function":
                fmt.PY_rv_asgn = fmt.C_rv_decl + " =\t "
        need_rv = False

        # build up code for a function
        for nargs, len_post_parse, len_pre_call, call_list in default_calls:
            if found_default:
                PY_code.append("case %d:" % nargs)
                PY_code.append(1)
                need_blank = False
                if len_post_parse or len_pre_call:
                    # Only add scope if necessary
                    PY_code.append("{")
                    PY_code.append(1)
                    extra_scope = True
                else:
                    extra_scope = False

            if len_post_parse:
                if options.debug:
                    if need_blank:
                        PY_code.append("")
                    PY_code.append("// post_parse")
                PY_code.extend(post_parse[:len_post_parse])
                need_blank = True

            if self.language == "cxx" and goto_fail:
                # Need an extra scope to deal with C++ error
                # error: jump to label 'fail' crosses initialization of ...
                PY_code.append("{")
                PY_code.append(1)
                fail_scope = True
                need_blank = False
            else:
                fail_scope = False

            if len_pre_call:
                if options.debug:
                    if need_blank:
                        PY_code.append("")
                    PY_code.append("// pre_call")
                PY_code.extend(pre_call[:len_pre_call])
                need_blank = True
            fmt.PY_call_list = call_list

            if options.debug and need_blank:
                PY_code.append("")

            capsule_order = None
            if is_ctor:
                append_format(
                    PY_code,
                    "self->{PY_obj} = new {namespace_scope}"
                    "{cxx_class}({PY_call_list});",
                    fmt,
                )
            elif CXX_subprogram == "subroutine":
                append_format(
                    PY_code,
                    "{PY_this_call}{function_name}({PY_call_list});",
                    fmt,
                )
            elif need_malloc:
                # Allocate space for scalar returned by function.
                # This allows NumPy to pointer to the memory.
                need_rv = True
                fmt.cxx_type = result_typemap.cxx_type
                capsule_type = CXX_result.gen_arg_as_cxx(
                    name=None, force_ptr=True, params=None, continuation=True
                )
                if self.language == "c":
                    append_format(
                        PY_code,
                        "{C_rv_decl} = malloc(sizeof({cxx_type}));",
                        fmt,
                    )
                    del_lines = ["free(ptr);"]
                else:
                    append_format(PY_code, "{C_rv_decl} = new {cxx_type};", fmt)
                    del_lines = [
                        "{} cxx_ptr =\t static_cast<{}>(ptr);".format(
                            capsule_type, capsule_type
                        ),
                        "delete cxx_ptr;",
                    ]
                capsule_order = self.add_capsule_code(capsule_type, del_lines)
                append_format(
                    PY_code,
                    "*{cxx_var} = {PY_this_call}{function_name}({PY_call_list});",
                    fmt_result,
                )
            else:
                need_rv = True
                append_format(
                    PY_code,
                    "{PY_rv_asgn}{PY_this_call}{function_name}({PY_call_list});",
                    fmt,
                )

            if node.PY_error_pattern:
                lfmt = util.Scope(fmt)
                lfmt.c_var = fmt.C_result
                lfmt.cxx_var = fmt.C_result
                append_format(
                    PY_code, self.patterns[node.PY_error_pattern], lfmt
                )

            if found_default:
                PY_code.append("break;")
                PY_code.append(-1)
                if extra_scope:
                    PY_code.append("}")
                    PY_code.append(-1)
                    need_blank = False
        # End of loop over default arguments.
        if found_default:
            PY_code.append(
                "default:+\n"
                "PyErr_SetString(PyExc_ValueError,"
                "\t \"Wrong number of arguments\");\n"
                "return NULL;\n"
#                "goto fail;\n"
                "-}")
# XXX - need to add a extra scope to deal with goto in C++
#            goto_fail = True;
        else:
            need_rv = False

        if need_rv:
            PY_decl.append(fmt.C_rv_decl + ";")
        if len(PY_decl):
            # Add blank line after declarations.
            PY_decl.append("")

        # Compute return value
        if CXX_subprogram == "function":
            if (
                    result_return_pointer_as in ["pointer", "allocatable"]
                    and result_typemap.base != "string"
            ):
                # Returning a NumPy array.
                result_blk = self.array_result(
                    capsule_order, ast, result_typemap, fmt_result)
            else:
                # XXX - wrapc uses result instead of intent_out
                result_blk = result_typemap.py_statements.get("intent_out", {})
                if build_tuples and result_typemap.name == 'bool':
                    # This kludges around a very specific problem.
                    # bool creates an object since Py_BuildValue does not know bool until Python 3.3
                    # If there are additional return arguments, a tuple will be created
                    # which is also named py_var. So create a temporary name.
                    fmt_result.py_var += "_tmp"

            ttt0 = self.intent_out(
                result_typemap, result_blk, fmt_result, post_call)
            # Add result to front of result tuple.
            build_tuples.insert(0, ttt0)

        # If only one return value, return the ctor
        # else create a tuple with Py_BuildValue.
        if is_ctor:
            return_code = "return 0;"
        elif not build_tuples:
            return_code = "Py_RETURN_NONE;"
        elif len(build_tuples) == 1:
            # return a single object already created in build_stmts
            ctor = build_tuples[0].ctor
            if ctor:
                post_call.append(ctor)
            fmt.py_var = build_tuples[0].ctorvar
            return_code = wformat("return (PyObject *) {py_var};", fmt)
        else:
            # create tuple object
            fmt.PyBuild_format = "".join([ttt.format for ttt in build_tuples])
            fmt.PyBuild_vargs = ",\t ".join([ttt.vargs for ttt in build_tuples])
            append_format(
                post_call,
                "PyObject * {PY_result} = "
                'Py_BuildValue("{PyBuild_format}",\t {PyBuild_vargs});',
                fmt,
            )
            return_code = wformat("return {PY_result};", fmt)

        need_blank = False  # put return right after call
        if post_call and not is_ctor:
            # ctor does not need to build return values
            if options.debug:
                PY_code.append("")
                PY_code.append("// post_call")
            PY_code.extend(post_call)
            need_blank = True

        if cleanup_code:
            if options.debug:
                PY_code.append("")
                PY_code.append("// cleanup")
            PY_code.extend(cleanup_code)
            need_blank = True

        if options.debug and need_blank:
            PY_code.append("")
        PY_code.append(return_code)

        if fail_scope:
            PY_code.append(-1)
            PY_code.append("}")
        if goto_fail:
            PY_code.extend(["", "^fail:"])
            PY_code.extend(fail_code)
            append_format(PY_code, "return {PY_error_return};", fmt)

        PY_impl = [1] + PY_decl + PY_code + [-1]

        expose = True
        if is_ctor:
            expose = False
        if len(self.overloaded_methods[ast.name]) > 1:
            # Only expose a multi-dispatch name, not each overload
            expose = False
        elif found_default:
            # Only one wrapper to deal with default arugments.
            # [C creates a wrapper per default argument]
            fmt = util.Scope(fmt)
            fmt.function_suffix = ""

        fmt.PY_ml_flags = "|".join(ml_flags)
        self.create_method(node, expose, is_ctor, fmt, PY_impl)

    def create_method(self, node, expose, is_ctor, fmt, PY_impl):
        """Format the function.

        Args:
            node    - function node to wrap
                      or None when called from multi_dispatch.
            expose  - True if exposed to user.
            is_ctor - True if this is a constructor.
            fmt     - dictionary of format values.
            PY_impl - list of implementation lines.
        """
        if node:
            cpp_if = node.cpp_if
        else:
            cpp_if = False

        body = self.PyMethodBody
        body.append("")
        if cpp_if:
            body.append("#" + node.cpp_if)
        if expose:
            append_format(
                body,
                "static char {PY_name_impl}__doc__[] =\n"
                '"{PY_doc_string}"\n'
                ";\n",
                fmt,
            )

        if node and node.options.doxygen and node.doxygen:
            self.write_doxygen(body, node.doxygen)
        if is_ctor:
            body.append("static int")
        else:
            body.append("static PyObject *")
        append_format(body, "{PY_name_impl}(", fmt)

        if fmt.PY_used_param_self:
            append_format(body, "  {PY_PyObject} *{PY_param_self},", fmt)
        else:
            append_format(
                body, "  PyObject *SHROUD_UNUSED({PY_param_self}),", fmt
            )
        if fmt.PY_used_param_args:
            append_format(body, "  PyObject *{PY_param_args},", fmt)
        else:
            append_format(
                body, "  PyObject *SHROUD_UNUSED({PY_param_args}),", fmt
            )
        if fmt.PY_used_param_args:
            append_format(body, "  PyObject *{PY_param_kwds})", fmt)
        else:
            append_format(
                body, "  PyObject *SHROUD_UNUSED({PY_param_kwds}))", fmt
            )

        body.append("{")
        # use function_suffix in splicer name since a single C++ function may
        # produce several methods.
        # XXX - make splicer name customizable?
        #        self._create_splicer(fmt.function_name, self.PyMethodBody, PY_impl)
        if node and node.options.debug:
            self.PyMethodBody.append("// " + node.declgen)
        self._create_splicer(
            fmt.underscore_name +
            fmt.function_suffix +
            fmt.template_suffix,
            self.PyMethodBody,
            PY_impl,
        )
        self.PyMethodBody.append("}")
        if cpp_if:
            body.append("#endif // " + cpp_if)

        if expose is True:
            if cpp_if:
                self.PyMethodDef.append("#" + cpp_if)
            # default name
            append_format(
                self.PyMethodDef,
                '{{"{function_name}{function_suffix}{template_suffix}",\t '
                "(PyCFunction){PY_name_impl},\t "
                "{PY_ml_flags},\t "
                "{PY_name_impl}__doc__}},",
                fmt,
            )
            if cpp_if:
                self.PyMethodDef.append("#endif // " + cpp_if)

    #        elif expose is not False:
    #            # override name
    #            fmt = util.Scope(fmt)
    #            fmt.expose = expose
    #            self.PyMethodDef.append( wformat('{{"{expose}", (PyCFunction){PY_name_impl}, {PY_ml_flags}, {PY_name_impl}__doc__}},', fmt))

    def write_tp_func(self, node, fmt_type, output):
        """Create functions for tp_init et.al.

        Args:
            node -
            fmt_type - dictionary used with PyTypeObject_template
                       to fill in type function names.
            output - list for generated functions.

        python:
          type: [ repr, str ]

        """
        # Type bodies must be filled in by user, no real way to guess
        # how to implement.
        # Except tp_init (constructor) and tp_del (destructor).
        fmt_func = node.fmtdict
        fmt = util.Scope(fmt_func)
        template = node.options.PY_type_impl_template
        PyObj = fmt_func.PY_PyObject
        if "type" in node.python:
            selected = node.python["type"][:]
            for auto in ["del"]:
                # Make some methods are there
                if auto not in selected:
                    selected.append(auto)
        else:
            selected = ["del"]

        # Dictionary of methods for bodies
        default_body = dict(richcompare=self.not_implemented)
        default_body["del"] = self.tp_del

        self._push_splicer("type")
        for typename in typenames:
            tp_name = "tp_" + typename
            if typename == "init":
                # The constructor method is used for tp_init
                fmt_type[tp_name] = self.tp_init_default
                continue
            if typename not in selected:
                fmt_type[tp_name] = fmt_type["nullptr"]
                continue
            fmt.PY_type_method = tp_name
            func_name = wformat(template, fmt)
            fmt_type[tp_name] = func_name
            tup = typefuncs[typename]
            output.append("static " + tup[0])
            output.append(
                ("{name} " + tup[1]).format(  # object used by tup[1]
                    name=func_name, object=PyObj
                )
            )
            output.append("{")
            default = default_body.get(typename, self.not_implemented_error)
            default = default(typename, tup[2])

            # format and indent default bodies
            fmted = [1]
            for line in default:
                append_format(fmted, line, fmt_func)
            fmted.append(-1)

            self._create_splicer(typename, output, fmted)
            output.append("}")
        self._pop_splicer("type")


######
    def _gather_helper_code(self, name, done):
        """Add code from helpers.

        First recursively process dependent_helpers
        to add code in order.

        Args:
            name -
            done -
        """
        if name in done:
            return  # avoid recursion
        done[name] = True

        helper_info = whelpers.CHelpers[name]
        if "dependent_helpers" in helper_info:
            for dep in helper_info["dependent_helpers"]:
                # check for recursion
                self._gather_helper_code(dep, done)

        if self.language == "c":
            lang_header = "c_header"
            lang_source = "c_source"
        else:
            lang_header = "cxx_header"
            lang_source = "cxx_source"
        scope = helper_info.get("scope", "file")

        if lang_header in helper_info:
            for include in helper_info[lang_header].split():
                self.helper_header[scope][include] = True
        elif "header" in helper_info:
            for include in helper_info["header"].split():
                self.helper_header[scope][include] = True

        if lang_source in helper_info:
            self.helper_source[scope].append(helper_info[lang_source])
        elif "source" in helper_info:
            self.helper_source[scope].append(helper_info["source"])

    def gather_helper_code(self, helpers):
        """Gather up all helpers requested and insert code into output.

        helpers should be self.c_helper or self.shared_helper

        Args:
            helpers -
        """
        # per class
        self.helper_source = dict(file=[], utility=[])
        self.helper_header = dict(file={}, utility={})

        done = {}  # avoid duplicates and recursion
        for name in sorted(helpers.keys()):
            self._gather_helper_code(name, done)
######

    def write_extension_type(self, library, node):
        """
        Args:
            library - ast.LibraryNode.
            node - ast.ClassNode
        """
        fmt = node.fmtdict
        fname = fmt.PY_type_filename

        self.header_impl_include = {}
        self.gather_helper_code(self.c_helper)
        # always include helper header
#        self.c_helper_include[library.fmtdict.C_header_utility] = True
#        self.shared_helper.update(self.c_helper)  # accumulate all helpers

        output = []
        if node.cpp_if:
            output.append("#" + node.cpp_if)

        append_format(output, '#include "{PY_header_filename}"', fmt)
        #        if self.need_numpy:
        #            output.append('#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION')
        #            output.append('#include "numpy/arrayobject.h"')
        self._push_splicer("impl")

        # Use headers from class if they exist or else library
        header_impl_include = {}
        if node and node.cxx_header:
            for include in node.cxx_header.split():
                header_impl_include[include] = True
        else:
            for include in library.cxx_header.split():
                header_impl_include[include] = True
        header_impl_include.update(self.helper_header["file"])
        self.write_headers(header_impl_include, output)

        self._create_splicer("include", output)
        output.append(cpp_boilerplate)
        if self.helper_source["file"]:
            output.extend(self.helper_source["file"])
        self._create_splicer("C_definition", output)
        self._create_splicer("additional_methods", output)
        self._pop_splicer("impl")

        fmt_type = dict(
            PY_module_name=fmt.PY_module_name,
            PY_PyObject=fmt.PY_PyObject,
            PY_PyTypeObject=fmt.PY_PyTypeObject,
            cxx_class=fmt.cxx_class,
            nullptr="0",  # 'NULL',
        )
        self.write_tp_func(node, fmt_type, output)

        output.extend(self.PyMethodBody)

        self._push_splicer("impl")
        self._create_splicer("after_methods", output)
        self._pop_splicer("impl")

        output.extend(self.PyGetSetBody)
        if self.PyGetSetDef:
            fmt_type["tp_getset"] = wformat(
                "{PY_prefix}{cxx_class}_getset", fmt
            )
            append_format(
                output, "\nstatic PyGetSetDef {tp_getset}[] = {{+", fmt_type
            )
            output.extend(self.PyGetSetDef)
            self._create_splicer("PyGetSetDef", output)
            output.append("{NULL}            /* sentinel */")
            output.append("-};")
        else:
            fmt_type["tp_getset"] = fmt_type["nullptr"]

        fmt_type["tp_methods"] = wformat("{PY_prefix}{cxx_class}_methods", fmt)
        append_format(
            output, "static PyMethodDef {tp_methods}[] = {{+", fmt_type
        )
        output.extend(self.PyMethodDef)
        self._create_splicer("PyMethodDef", output)
        output.append(
            "{NULL,   (PyCFunction)NULL, 0, NULL}" "            /* sentinel */"
        )
        output.append("-};")

        append_format(output, PyTypeObject_template, fmt_type)
        if node.cpp_if:
            output.append("#endif // " + node.cpp_if)

        self.config.pyfiles.append(os.path.join(self.config.python_dir, fname))
        self.write_output_file(fname, self.config.python_dir, output)

    def multi_dispatch(self, functions):
        """Look for overloaded methods.
        When found, create a method which will call each of the
        overloaded methods looking for the one which will accept
        the given arguments.

        Args:
            functions -
        """
        mdone = {}
        for function in functions:
            # preserve order of multi-dispatch functions
            mname = function.ast.name
            if mname in mdone:
                continue
            mdone[mname] = True
            methods = self.overloaded_methods[mname]
            if len(methods) < 2:
                continue  # not overloaded

            node = methods[0]
            fmt_func = node.fmtdict
            fmt = util.Scope(fmt_func)
            fmt.function_suffix = ""
            fmt.template_suffix = ""
            fmt.PY_doc_string = "documentation"
            fmt.PY_ml_flags = "METH_VARARGS|METH_KEYWORDS"
            fmt.PY_used_param_self = True
            fmt.PY_used_param_args = True
            fmt.PY_used_param_kwds = True

            is_ctor = node.ast.is_ctor()

            body = []
            body.append(1)
            body.append("Py_ssize_t SHT_nargs = 0;")
            body.extend(
                [
                    "if (args != NULL) SHT_nargs += PyTuple_Size(args);",
                    "if (kwds != NULL) SHT_nargs += PyDict_Size(args);",
                ]
            )
            if is_ctor:
                fmt.PY_type_method = "tp_init"
                fmt.PY_name_impl = wformat(
                    node.options.PY_type_impl_template, fmt
                )
                fmt.PY_type_impl = fmt.PY_name_impl
                self.tp_init_default = fmt.PY_type_impl
                return_code = "return rv;"
                return_arg = "rv"
                fmt.PY_error_return = "-1"
                self.tp_init_default = fmt.PY_name_impl
                body.append("int rv;")
                expose = False
            else:
                fmt.PY_name_impl = wformat(
                    node.options.PY_name_impl_template, fmt
                )
                return_code = "return rvobj;"
                return_arg = "rvobj"
                fmt.PY_error_return = "NULL"
                body.append("PyObject *rvobj;")
                expose = True

            for overload in methods:
                if overload.cpp_if:
                    body.append("#" + overload.cpp_if)
                if overload._nargs:
                    body.append(
                        "if (SHT_nargs >= %d && SHT_nargs <= %d) {+"
                        % overload._nargs
                    )
                else:
                    body.append(
                        "if (SHT_nargs == %d) {+" % len(overload.ast.params)
                    )
                append_format(
                    body,
                    return_arg + " = {PY_name_impl}(self, args, kwds);",
                    overload.fmtdict,
                )
                body.append("if (!PyErr_Occurred()) {+")
                body.append(return_code)
                body.append(
                    "-} else if (! PyErr_ExceptionMatches"
                    "(PyExc_TypeError)) {+"
                )
                body.append(return_code)
                body.append("-}\nPyErr_Clear();\n-}")
                if overload.cpp_if:
                    body.append("#endif // " + overload.cpp_if)

            body.append(
                "PyErr_SetString(PyExc_TypeError, "
                '"wrong arguments multi-dispatch");'
            )
            append_format(body, "return {PY_error_return};", fmt)
            body.append(-1)

            self.create_method(None, expose, is_ctor, fmt, body)

    def write_header(self, node):
        """
        Args:
            node - ast.LibraryNode.
        """
        fmt = node.fmtdict
        fname = fmt.PY_header_filename

        output = []

        # add guard
        guard = fname.replace(".", "_").upper()
        output.extend(["#ifndef %s" % guard, "#define %s" % guard])

        output.append("#include <Python.h>")

        self._push_splicer("header")
        self._create_splicer("include", output)

        if self.py_class_decl:
            output.append("")
            output.extend(self.py_class_decl)
            output.append("// ------------------------------")

        #        output.extend(self.define_arraydescr)

        output.append("")
        self._create_splicer("C_declaration", output)
        self._pop_splicer("header")

        if self.py_utility_declaration:
            output.append("")
            output.append("// utility functions")
            output.extend(self.py_utility_declaration)

        append_format(
            output,
            """
extern PyObject *{PY_prefix}error_obj;

#if PY_MAJOR_VERSION >= 3
{PY_extern_C_begin}PyMODINIT_FUNC PyInit_{PY_module_name}(void);
#else
{PY_extern_C_begin}PyMODINIT_FUNC init{PY_module_name}(void);
#endif
""",
            fmt,
        )
        output.append("#endif  /* %s */" % guard)
        #        self.config.pyfiles.append(
        #            os.path.join(self.config.python_dir, fname))
        self.write_output_file(fname, self.config.python_dir, output)

    def write_module(self, node):
        """
        Write the Python extension module.

        Args:
            node - ast.LibraryNode.
        """
        fmt = node.fmtdict
        fname = fmt.PY_module_filename

        fmt.PY_library_doc = "library documentation"

        self.header_impl_include = {}
        self.gather_helper_code(self.c_helper)
        # always include helper header
#        self.c_helper_include[library.fmtdict.C_header_utility] = True
#        self.shared_helper.update(self.c_helper)  # accumulate all helpers

        output = []

        append_format(output, '#include "{PY_header_filename}"', fmt)
        if self.need_numpy:
            output.append("#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION")
            output.append('#include "numpy/arrayobject.h"')
        for include in node.cxx_header.split():
            output.append('#include "%s"' % include)
        self.write_headers(self.helper_header["file"], output)
        output.append("")
        self._create_splicer("include", output)
        output.append(cpp_boilerplate)
        if self.helper_source["file"]:
            output.extend(self.helper_source["file"])
        output.append("")
        self._create_splicer("C_definition", output)

        append_format(output, "PyObject *{PY_prefix}error_obj;", fmt)
        output.extend(self.define_arraydescr)

        self._create_splicer("additional_functions", output)
        output.extend(self.PyMethodBody)

        append_format(
            output, "static PyMethodDef {PY_prefix}methods[] = {{", fmt
        )
        output.extend(self.PyMethodDef)
        output.append(
            "{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */"
        )
        output.append("};")

        output.extend(self.arraydescr)

        append_format(output, module_begin, fmt)
        self._create_splicer("C_init_locals", output)
        append_format(output, module_middle, fmt)
        if self.need_numpy:
            output.append("import_array();")
        output.extend(self.py_type_object_creation)
        output.extend(self.enum_impl)
        if self.call_arraydescr:
            output.append("")
            output.append("// Define PyArray_Descr for structs")
            output.extend(self.call_arraydescr)
        append_format(output, module_middle2, fmt)
        self._create_splicer("C_init_body", output)
        append_format(output, module_end, fmt)

        self.config.pyfiles.append(os.path.join(self.config.python_dir, fname))
        self.write_output_file(fname, self.config.python_dir, output)

    def write_utility(self):
        node = self.newlibrary
        fmt = node.fmtdict
        output = []
        append_format(output, '#include "{PY_header_filename}"', fmt)
        if self.capsule_order:
            # header file may be needed to fully qualify types capsule destructors
            for include in node.cxx_header.split():
                output.append('#include "%s"' % include)
            output.append("")
        output.extend(self.py_utility_definition)
        output.append("")
        output.extend(self.py_utility_functions)
        if self.capsule_order:
            self.write_capsule_code(output, fmt)
        self.config.pyfiles.append(
            os.path.join(self.config.python_dir, fmt.PY_utility_filename)
        )
        self.write_output_file(
            fmt.PY_utility_filename, self.config.python_dir, output
        )

    def write_capsule_code(self, output, fmt):
        """Write a function used to delete memory when a
        NumPy array is deleted.

        Create a global variable of of context pointer used
        to switch to case used to release memory.

        Args:
            output -
            fmt -
        """

        append_format(
            self.py_utility_declaration,
            "extern const char * {PY_numpy_array_dtor_context}[];",
            fmt,
        )
        append_format(
            self.py_utility_declaration,
            "extern void {PY_numpy_array_dtor_function}(PyObject *cap);",
            fmt,
        )

        output.append(
            "\n"
            "// Code used to release arrays for NumPy objects\n"
            "// via a Capsule base object with a destructor.\n"
            "// Context strings"
        )
        append_format(
            output, "const char * {PY_numpy_array_dtor_context}[] = {{+", fmt
        )
        for name in self.capsule_order:
            output.append('"{}",'.format(name))
        output.append("NULL")
        output.append("-};")

        append_format(
            output,
            "\n// destructor function for PyCapsule\n"
            "void {PY_numpy_array_dtor_function}(PyObject *cap)\n"
            "{{+\n"
            # 'const char* name = PyCapsule_GetName(cap);\n'
            'void *ptr = PyCapsule_GetPointer(cap, "{PY_numpy_array_capsule_name}");',
            fmt,
        )

        output.append(
            "const char * context = "
            + do_cast(
                self.language,
                "static",
                "const char *",
                "PyCapsule_GetContext(cap)",
            )
            + ";"
        )

        start = "if"
        for i, name in enumerate(self.capsule_order):
            output.append(
                start
                + " (context == {}[{}]) {{".format(
                    fmt.PY_numpy_array_dtor_context, i
                )
            )
            output.append(1)
            for line in self.capsule_code[name][1]:
                output.append(line)
            output.append(-1)
            start = "} else if "
        output.append("} else {+")
        output.append("// no such destructor")
        output.append("-}")

        output.append("-}")

    capsule_code = {}
    capsule_order = []

    def add_capsule_code(self, name, lines):
        """Add unique names to capsule_code.
        Return index of name.

        Args:
            name -
            lines -
        """
        if name not in self.capsule_code:
            self.capsule_code[name] = (str(len(self.capsule_code)), lines)
            self.capsule_order.append(name)
        return self.capsule_code[name][0]

    def not_implemented_error(self, msg, ret):
        """A standard splicer for unimplemented code
        ret is the return value (NULL or -1 or '')

        Args:
            msg -
            ret -
        """
        lines = ['PyErr_SetString(PyExc_NotImplementedError, "%s");' % msg]
        if ret:
            lines.append("return %s;" % ret)
        else:
            lines.append("return;")
        return lines

    def not_implemented(self, msg, ret):
        """A standard splicer for rich comparison

        Args:
            msg -
            ret -
        """
        return ["Py_INCREF(Py_NotImplemented);", "return Py_NotImplemented;"]

    def tp_del(self, msg, ret):
        """default method for tp_del.

        Args:
            msg = 'del'
            ret = ''
        """
        return ["delete self->{PY_obj};", "self->{PY_obj} = NULL;"]


# --- Python boiler plate

# Avoid warning errors about unused parameters
# Include in each source file and not the header file because
# we don't want to pollute the user's files.
cpp_boilerplate = """
#ifdef __cplusplus
#define SHROUD_UNUSED(param)
#else
#define SHROUD_UNUSED(param) param
#endif

#if PY_MAJOR_VERSION >= 3
#define PyInt_AsLong PyLong_AsLong
#define PyInt_FromLong PyLong_FromLong
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#endif"""

typenames = [
    "dealloc",
    "print",
    "compare",
    "getattr",
    "setattr",  # deprecated
    "getattro",
    "setattro",
    "repr",
    "hash",
    "call",
    "str",
    "init",
    "alloc",
    "new",
    "free",
    "del",
    "richcompare",
]


# return type, prototype, default return value
typefuncs = {
    "dealloc": ("void", "({object} *self)", ""),
    "print": ("int", "({object} *self, FILE *fp, int flags)", "-1"),
    "getattr": ("PyObject *", "({object} *self, char *name)", "NULL"),
    "setattr": ("int", "({object} *self, char *name, PyObject *value)", "-1"),
    "compare": ("int", "({object} *self, PyObject *)", "-1"),
    "repr": ("PyObject *", "({object} *self)", "NULL"),
    "hash": ("long", "({object} *self)", "-1"),
    "call": (
        "PyObject *",
        "({object} *self, PyObject *args, PyObject *kwds)",
        "NULL",
    ),
    "str": ("PyObject *", "({object} *self)", "NULL"),
    "getattro": ("PyObject *", "({object} *self, PyObject *name)", "NULL"),
    "setattro": (
        "int",
        "({object} *self, PyObject *name, PyObject *value)",
        "-1",
    ),
    "init": ("int", "({object} *self, PyObject *args, PyObject *kwds)", "-1"),
    "alloc": ("PyObject *", "(PyTypeObject *type, Py_ssize_t nitems)", "NULL"),
    "new": (
        "PyObject *",
        "(PyTypeObject *type, PyObject *args, PyObject *kwds)",
        "NULL",
    ),
    "free": ("void", "(void *op)", ""),
    "del": ("void", "({object} *self)", ""),
    "richcompare": (
        "PyObject *",
        "({object} *self, PyObject *other, int opid)",
        "",
    ),
}

# Note: that these strings have some format character to control indenting
#  + indent
#  - deindent
#  0 noindention

PyTypeObject_template = """
static char {cxx_class}__doc__[] =
"virtual class"
;

/* static */
PyTypeObject {PY_PyTypeObject} = {{+
PyVarObject_HEAD_INIT(NULL, 0)
"{PY_module_name}.{cxx_class}",                       /* tp_name */
sizeof({PY_PyObject}),         /* tp_basicsize */
{nullptr},                              /* tp_itemsize */
/* Methods to implement standard operations */
(destructor){tp_dealloc},                 /* tp_dealloc */
(printfunc){tp_print},                   /* tp_print */
(getattrfunc){tp_getattr},                 /* tp_getattr */
(setattrfunc){tp_setattr},                 /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
{nullptr},                               /* tp_reserved */
#else
(cmpfunc){tp_compare},                     /* tp_compare */
#endif
(reprfunc){tp_repr},                    /* tp_repr */
/* Method suites for standard classes */
{nullptr},                              /* tp_as_number */
{nullptr},                              /* tp_as_sequence */
{nullptr},                              /* tp_as_mapping */
/* More standard operations (here for binary compatibility) */
(hashfunc){tp_hash},                    /* tp_hash */
(ternaryfunc){tp_call},                 /* tp_call */
(reprfunc){tp_str},                    /* tp_str */
(getattrofunc){tp_getattro},                /* tp_getattro */
(setattrofunc){tp_setattro},                /* tp_setattro */
/* Functions to access object as input/output buffer */
{nullptr},                              /* tp_as_buffer */
/* Flags to define presence of optional/expanded features */
Py_TPFLAGS_DEFAULT,             /* tp_flags */
{cxx_class}__doc__,         /* tp_doc */
/* Assigned meaning in release 2.0 */
/* call function for all accessible objects */
(traverseproc){nullptr},                /* tp_traverse */
/* delete references to contained objects */
(inquiry){nullptr},                     /* tp_clear */
/* Assigned meaning in release 2.1 */
/* rich comparisons */
(richcmpfunc){tp_richcompare},                 /* tp_richcompare */
/* weak reference enabler */
{nullptr},                              /* tp_weaklistoffset */
/* Added in release 2.2 */
/* Iterators */
(getiterfunc){nullptr},                 /* tp_iter */
(iternextfunc){nullptr},                /* tp_iternext */
/* Attribute descriptor and subclassing stuff */
{tp_methods},                             /* tp_methods */
{nullptr},                              /* tp_members */
{tp_getset},                             /* tp_getset */
{nullptr},                              /* tp_base */
{nullptr},                              /* tp_dict */
(descrgetfunc){nullptr},                /* tp_descr_get */
(descrsetfunc){nullptr},                /* tp_descr_set */
{nullptr},                              /* tp_dictoffset */
(initproc){tp_init},                   /* tp_init */
(allocfunc){tp_alloc},                  /* tp_alloc */
(newfunc){tp_new},                    /* tp_new */
(freefunc){tp_free},                   /* tp_free */
(inquiry){nullptr},                     /* tp_is_gc */
{nullptr},                              /* tp_bases */
{nullptr},                              /* tp_mro */
{nullptr},                              /* tp_cache */
{nullptr},                              /* tp_subclasses */
{nullptr},                              /* tp_weaklist */
(destructor){tp_del},                 /* tp_del */
{nullptr},                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
(destructor){nullptr},                  /* tp_finalize */
#endif
-}};"""


module_begin = """
/*
 * init{library_lower} - Initialization function for the module
 * *must* be called init{library_lower}
 */
static char {PY_prefix}_doc__[] =
"{PY_library_doc}"
;

struct module_state {{
    PyObject *error;
}};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#if PY_MAJOR_VERSION >= 3
static int {library_lower}_traverse(PyObject *m, visitproc visit, void *arg) {{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}}

static int {library_lower}_clear(PyObject *m) {{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}}

static struct PyModuleDef moduledef = {{
    PyModuleDef_HEAD_INIT,
    "{library_lower}", /* m_name */
    {PY_prefix}_doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    {PY_prefix}methods, /* m_methods */
    NULL, /* m_reload */
    {library_lower}_traverse, /* m_traverse */
    {library_lower}_clear, /* m_clear */
    NULL  /* m_free */
}};

#define RETVAL m
#define INITERROR return NULL
#else
#define RETVAL
#define INITERROR return
#endif

{PY_extern_C_begin}PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_{PY_module_name}(void)
#else
init{PY_module_name}(void)
#endif
{{+
PyObject *m = NULL;
const char * error_name = "{library_lower}.Error";
"""

module_middle = """

/* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
m = PyModule_Create(&moduledef);
#else
m = Py_InitModule4("{PY_module_name}", {PY_prefix}methods,\t
+{PY_prefix}_doc__,
(PyObject*)NULL,PYTHON_API_VERSION);
#endif
-if (m == NULL)
+return RETVAL;-
struct module_state *st = GETSTATE(m);
"""

module_middle2 = """
{PY_prefix}error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
if ({PY_prefix}error_obj == NULL)
+return RETVAL;-
st->error = {PY_prefix}error_obj;
PyModule_AddObject(m, "Error", st->error);
"""

module_end = """
/* Check for errors */
if (PyErr_Occurred())
+Py_FatalError("can't initialize module {PY_module_name}");-
return RETVAL;
-}}
"""


class ToImplied(todict.PrintNode):
    """Convert implied expression to Python wrapper code.

    expression has already been checked for errors by generate.check_implied.
    Convert functions:
      size     -  PyArray_SIZE
      len      -  strlen
      len_trim -
    """

    def __init__(self, expr, func):
        super(ToImplied, self).__init__()
        self.expr = expr
        self.func = func

    def visit_Identifier(self, node):
        # Look for functions
        if node.args is None:
            return node.name
        ### functions
        elif node.name == "size":
            # size(arg)
            argname = node.args[0].name
            #            arg = self.func.ast.find_arg_by_name(argname)
            fmt = self.func._fmtargs[argname]["fmtpy"]
            return wformat("PyArray_SIZE({py_var})", fmt)
        elif node.name == "len":
            # len(arg)
            argname = node.args[0].name
            #            arg = self.func.ast.find_arg_by_name(argname)
            fmt = self.func._fmtargs[argname]["fmtpy"]

            # find argname in function parameters
            arg = self.func.ast.find_arg_by_name(argname)
            if arg.attrs["intent"] == "out":
                #   char *text+intent(out)+charlen(XXX), 
                #   int ltext+implied(len(text)))
                # len(text) in this case is the value of "charlen"
                # since no variable is actually passed in as an argument.
                return arg.attrs["charlen"]
            # XXX - need code for len_trim?
            return wformat("strlen({cxx_var})", fmt)
        elif node.name == "len_trim":
            # len_trim(arg)
            argname = node.args[0].name
            #            arg = self.func.ast.find_arg_by_name(argname)
            fmt = self.func._fmtargs[argname]["fmtpy"]
            return wformat("strlen({cxx_var})", fmt)
            # XXX - need code for len_trim
            return wformat("ShroudLenTrim({cxx_var}, strlen({cxx_var}))", fmt)
            #c_helper="ShroudLenTrim"
        else:
            return self.param_list(node)


def py_implied(expr, func):
    """Convert string to Python code.

    Args:
        expr -
        func -
    """
    node = declast.ExprParser(expr).expression()
    visitor = ToImplied(expr, func)
    return visitor.visit(node)


def attr_allocatable(language, allocatable, node, arg, fmt_arg):
    """parse allocatable and add values to fmt_arg.

    arguments to PyArray_NewLikeArray
      (prototype, order, descr, subok)
    descr_args - code to create PyArray_Descr.

    Valid values of allocatable:
       mold=name

    Args:
        language -
        allocatable -
        node -
        arg -
        fmt_arg - format dictionary for arg. 
    """
    fmtargs = node._fmtargs

    prototype = "--NONE--"
    order = "NPY_CORDER"
    descr = "NULL"
    subok = "0"
    descr_code = ""

    p = re.compile("mold\s*=\s*(\w+)")
    m = p.match(allocatable)
    if m is not None:
        moldvar = m.group(1)
        moldarg = node.ast.find_arg_by_name(moldvar)
        if moldarg is None:
            raise RuntimeError(
                "Mold argument '{}' does not exist: {}".format(
                    moldvar, allocatable
                )
            )
        if "dimension" not in moldarg.attrs:
            raise RuntimeError(
                "Mold argument '{}' must have dimension attribute: {}".format(
                    moldvar, allocatable
                )
            )
        fmt = fmtargs[moldvar]["fmtpy"]
        # copy from the numpy array for the argument
        prototype = fmt.py_var

        # Create Descr if types are different
        if arg.typemap.name != moldarg.typemap.name:
            arg_typemap = arg.typemap
            descr = "SHDPy_" + arg.name
            descr_code = (
                "PyArray_Descr * {} = "
                "PyArray_DescrFromType({});\n".format(
                    descr, arg_typemap.PYN_typenum
                )
            )

    fmt_arg.npy_prototype = prototype
    fmt_arg.npy_order = order
    fmt_arg.npy_descr = descr
    fmt_arg.npy_subok = subok
    fmt_arg.npy_descr_code = descr_code


def do_cast(lang, kind, typ, var):
    """Do cast based on language.

    Args:
        lang - c, c++
        kind - reinterpret, static
        typ -
        var -
    """
    if lang == "c":
        return "(%s) %s" % (typ, var)
    else:
        return "%s_cast<%s>\t(%s)" % (kind, typ, var)


# put into list to avoid duplicating text below
array_error = [
    "if ({py_var} == NULL) {{+",
    "PyErr_SetString(PyExc_ValueError,"
    '\t "{c_var} must be a 1-D array of {c_type}");',
    "goto fail;",
    "-}}",
]


py_statements_local = dict(
## numpy
# language=c
    intent_in_c_dimension_numpy=dict(
        need_numpy=True,
        decl=[
            "{py_type} * {pytmp_var};",
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = (PyArrayObject *) PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_IN_ARRAY);",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = PyArray_DATA({py_var});",
        ],
        cleanup=[
            "Py_DECREF({py_var});"
        ],
        fail=[
            "Py_XDECREF({py_var});"
        ],
        goto_fail=True,
    ),

    intent_inout_c_dimension_numpy=dict(
        need_numpy=True,
        decl=[
            "{py_type} * {pytmp_var};",
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = (PyArrayObject *) PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_INOUT_ARRAY);",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = PyArray_DATA({py_var});",
        ],
        post_call=None,  # Object already created in post_parse
        goto_fail=True,
    ),

    intent_out_c_dimension_numpy=dict(
        need_numpy=True,
        decl=[
            "PyArrayObject * {py_var} = NULL;",
            "npy_intp {npy_dims}[1] = {{ {pointer_shape} }};"
        ],
        post_parse=[
            "{py_var} = (PyArrayObject *) PyArray_SimpleNew("
            "{npy_ndims}, {npy_dims}, {numpy_type});",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = PyArray_DATA({py_var});",
        ],
        post_call=None,  # Object already created in post_parse
        fail=[
            "Py_XDECREF({py_var});"
        ],
        goto_fail=True,
    ),

# language=c++
# use C++ casts
    intent_in_cxx_dimension_numpy=dict(
        need_numpy=True,
        decl=[
            "{py_type} * {pytmp_var};",
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = reinterpret_cast<PyArrayObject *>\t(PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_IN_ARRAY));",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        cleanup=[
            "Py_DECREF({py_var});"
        ],
        fail=[
            "Py_XDECREF({py_var});"
        ],
        goto_fail=True,
    ),

    intent_inout_cxx_dimension_numpy=dict(
        need_numpy=True,
        decl=[
            "{py_type} * {pytmp_var};",
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = reinterpret_cast<PyArrayObject *>\t(PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_INOUT_ARRAY));",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        post_call=None,  # Object already created in post_parse
        goto_fail=True,
    ),

    intent_out_cxx_dimension_numpy=dict(
        need_numpy=True,
        decl=[
            "PyArrayObject * {py_var} = NULL;",
            "npy_intp {npy_dims}[1] = {{ {pointer_shape} }};"
        ],
        post_parse=[
            "{py_var} = reinterpret_cast<PyArrayObject *>\t(PyArray_SimpleNew("
            "{npy_ndims}, {npy_dims}, {numpy_type}));",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        post_call=None,  # Object already created in post_parse
        fail=[
            "Py_XDECREF({py_var});"
        ],
        goto_fail=True,
    ),

## allocatable
    intent_out_c_allocatable_numpy=dict(
        need_numpy=True,
        decl=["PyArrayObject * {py_var} = NULL;"],
        pre_call=[
            "{npy_descr_code}"
            "{py_var} = PyArray_NewLikeArray("
            "\t{npy_prototype},\t {npy_order},\t {npy_descr},\t {npy_subok});",
            "if ({py_var} == NULL)",
            "+goto fail;-",
            "{cxx_decl} = PyArray_DATA({py_var});",
            ],
        post_call=None,  # Object already created in pre_call
        fail=["Py_XDECREF({py_var});"],
        goto_fail=True,
    ),

# language=c++
# use C++ casts
    intent_out_cxx_allocatable_numpy=dict(
        need_numpy=True,
        decl=["PyArrayObject * {py_var} = NULL;"],
        pre_call=[
            "{npy_descr_code}"
            "{py_var} = reinterpret_cast<PyArrayObject *>\t(PyArray_NewLikeArray"
            "(\t{npy_prototype},\t {npy_order},\t {npy_descr},\t {npy_subok}));",
            "if ({py_var} == NULL)",
            "+goto fail;-",
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
            ],
        post_call=None,  # Object already created in pre_call
        fail=["Py_XDECREF({py_var});"],
        goto_fail=True,
    ),

########################################
## list
# language=c
    intent_in_c_dimension_list=dict(
        decl=[
            "{py_type} * {pytmp_var};",
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = (PyArrayObject *) PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_IN_ARRAY);",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = PyArray_DATA({py_var});",
        ],
        cleanup=[
            "Py_DECREF({py_var});"
        ],
        fail=[
            "Py_XDECREF({py_var});"
        ],
        goto_fail=True,
    ),

    intent_inout_c_dimension_list=dict(
        decl=[
            "{py_type} * {pytmp_var};",
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = (PyArrayObject *) PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_INOUT_ARRAY);",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = PyArray_DATA({py_var});",
        ],
        post_call=None,  # Object already created in post_parse
        goto_fail=True,
    ),

    intent_out_c_dimension_list=dict(
        c_helper="to_PyList_{cxx_type}",
        c_header="<stdlib.h>",  # malloc/free
        decl=[
            "PyObject *{py_var} = NULL;",
            "{cxx_decl} = NULL;",
        ],
        pre_call=[
#            "{cxx_decl}[{pointer_shape}];",
            "{cxx_var} = malloc(sizeof({cxx_type}) * {pointer_shape});",
            "if ({cxx_var} == NULL) goto fail;",
        ],
        post_call=[
            "{py_var} = SHROUD_to_PyList_{cxx_type}({cxx_var}, {pointer_shape});",
            "if ({py_var} == NULL) goto fail;",
            "free({cxx_var});",
            "{cxx_var} = NULL;",
        ],
        fail=[
            "Py_XDECREF({py_var});",
            "if({cxx_var} != NULL) free({cxx_var});",
        ],
        goto_fail=True,
    ),

# language=c++
# use C++ casts
    intent_in_cxx_dimension_list=dict(
        decl=[
            "{py_type} * {pytmp_var};",
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = reinterpret_cast<PyArrayObject *>\t(PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_IN_ARRAY));",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        cleanup=[
            "Py_DECREF({py_var});"
        ],
        fail=[
            "Py_XDECREF({py_var});"
        ],
        goto_fail=True,
    ),

    intent_inout_cxx_dimension_list=dict(
        decl=[
            "{py_type} * {pytmp_var};",
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = reinterpret_cast<PyArrayObject *>\t(PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_INOUT_ARRAY));",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        post_call=None,  # Object already created in post_parse
        goto_fail=True,
    ),

    intent_out_cxx_dimension_list=dict(
        decl=[
            "PyArrayObject * {py_var} = NULL;",
            "npy_intp {npy_dims}[1] = {{ {pointer_shape} }};"
        ],
        post_parse=[
            "{py_var} = reinterpret_cast<PyArrayObject *>\t(PyArray_SimpleNew("
            "{npy_ndims}, {npy_dims}, {numpy_type}));",
        ] + array_error,
        pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        post_call=None,  # Object already created in post_parse
        fail=[
            "Py_XDECREF({py_var});"
        ],
        goto_fail=True,
    ),

## allocatable
    intent_out_c_allocatable_list=dict(
        decl=["PyArrayObject * {py_var} = NULL;"],
        pre_call=[
            "{npy_descr_code}"
            "{py_var} = PyArray_NewLikeArray("
            "\t{npy_prototype},\t {npy_order},\t {npy_descr},\t {npy_subok});",
            "if ({py_var} == NULL)",
            "+goto fail;-",
            "{cxx_decl} = PyArray_DATA({py_var});",
            ],
        post_call=None,  # Object already created in pre_call
        fail=["Py_XDECREF({py_var});"],
        goto_fail=True,
    ),

# language=c++
# use C++ casts
    intent_out_cxx_allocatable_list=dict(
        decl=["PyArrayObject * {py_var} = NULL;"],
        pre_call=[
            "{npy_descr_code}"
            "{py_var} = reinterpret_cast<PyArrayObject *>\t(PyArray_NewLikeArray"
            "(\t{npy_prototype},\t {npy_order},\t {npy_descr},\t {npy_subok}));",
            "if ({py_var} == NULL)",
            "+goto fail;-",
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
            ],
        post_call=None,  # Object already created in pre_call
        fail=["Py_XDECREF({py_var});"],
        goto_fail=True,
    ),

)
