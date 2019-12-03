# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Generate Python module for C or C++ code.

Entire library in a single header.
One Extension module per class


Variables prefixes used by generated code:
SH_     C or C++ version of argument
SHPy_   Python object which corresponds to the argument {py_var}
SHTPy_  A temporary object, usually from PyArg_Parse
        to be converted to SHPy_ object. {pytmp_var}
SHData_ Data of NumPy object (fmt.data_var} - intermediate variable
        of PyArray_DATA cast to correct type.
SHDPy_  PyArray_Descr object  {pydescr_var}
SHD_    npy_intp array for shape, {npy_dims}
SHC_    PyCapsule owner of memory of NumPy array. {py_capsule}
        Used to deallocate memory.
SHSize_ Size of dimension argument (fmt.size_var}
SHPyResult Return Python object.
        Necessary when a return object is combined with others by Py_BuildValue.
"""

from __future__ import print_function
from __future__ import absolute_import

import collections
import os
import re

from . import declast
from . import todict
from . import typemap
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
BuildTuple = collections.namedtuple("BuildTuple", "format vargs blk ctorvar")

# type_object_creation - code to add variables to module.
ModuleTuple = collections.namedtuple(
    "ModuleTuple",
    "type_object_creation"
)

# Info used per file.  Each library, namespace and class create a file.
FileTuple = collections.namedtuple(
    "FileTuple", "MethodBody MethodDef GetSetBody GetSetDef")


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
        self.module_init_decls = []
        self.arraydescr = []  # Create PyArray_Descr for struct
        self.decl_arraydescr = []
        self.define_arraydescr = []
        self.call_arraydescr = []
        self.need_blah = False
        update_for_language(self.language)

    def XXX_begin_output_file(self):
        """Start a new class for output"""
        pass

    def XXX_end_output_file(self):
        pass

    def XXX_begin_class(self):
        pass

    def reset_file(self):
        """Start a new output file"""
        self.header_impl_include = {}  # header files in implementation file
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
        newlibrary.eval_template("PY_header_filename")
        newlibrary.eval_template("PY_utility_filename")
        fmt_library.PY_PyObject = "PyObject"
        fmt_library.PyObject = "PyObject"
        fmt_library.PY_param_self = "self"
        fmt_library.PY_param_args = "args"
        fmt_library.PY_param_kwds = "kwds"
        fmt_library.PY_used_param_self = False
        fmt_library.PY_used_param_args = False
        fmt_library.PY_used_param_kwds = False

        fmt_library.npy_ndims = "0"   # number of dimensions
        fmt_library.npy_dims = "NULL" # shape variable
        fmt_library.npy_intp = ""     # shape array definition

        # Variables to accumulate output lines
        self.py_class_decl = []  # code for header file
        self.py_utility_definition = []
        self.py_utility_declaration = []
        self.py_utility_functions = []
        append_format(self.module_init_decls,
                      "PyObject *{PY_prefix}error_obj;", fmt_library)

        # reserved the 0 slot of capsule_order
        self.add_capsule_code('--none--', ['// Do not release'])
        fmt_library.capsule_order = "0"
        self.need_blah = False  # Not needed if no there gc routines are added.

        self.wrap_namespace(newlibrary.wrap_namespace, top=True)
        self.write_utility()
        self.write_header(newlibrary)

    def wrap_namespace(self, node, top=False):
        """Wrap a library or namespace.

        Each class is written into its own file.

        Args:
            node - ast.LibraryNode, ast.NamespaceNode
            top  - True if top level module, else submodule.
        """
        node.eval_template("PY_module_filename")
        modinfo = ModuleTuple([])
        fileinfo = FileTuple([], [], [], [])

        if top:
            # have one namespace level, then replace name each time
            self._push_splicer("namespace")
            self._push_splicer("XXX") # placer holder
        for ns in node.namespaces:
            if ns.options.wrap_python:
                self.wrap_namespace(ns)
                self.register_submodule(ns, modinfo)
        if top:
            self._pop_splicer("XXX")  # This name will not match since it is replaced.
            self._pop_splicer("namespace")
        else:
            # Skip file component in scope_file for splicer name.
            self._update_splicer_top("::".join(node.scope_file[1:]))

        # preprocess all classes first to allow them to reference each other
        for cls in node.classes:
            if not cls.options.wrap_python:
                continue

            # XXX - classes and structs as classes
            ntypemap = cls.typemap
            fmt = cls.fmtdict
            ntypemap.PY_format = "O"

            # PyTypeObject for class
            cls.eval_template("PY_PyTypeObject")

            # PyObject for class
            cls.eval_template("PY_PyObject")

            fmt.PY_to_object_func = wformat("PP_{cxx_class}_to_Object", fmt)
            fmt.PY_from_object_func = wformat("PP_{cxx_class}_from_Object", fmt)

            ntypemap.PY_PyTypeObject = fmt.PY_PyTypeObject
            ntypemap.PY_PyObject = fmt.PY_PyObject
            ntypemap.PY_to_object = fmt.PY_to_object_func
            ntypemap.PY_from_object = fmt.PY_from_object_func

        self._push_splicer("class")
        for cls in node.classes:
            if not cls.options.wrap_python:
                continue
            name = cls.name
            self.reset_file()
            self._push_splicer(name)
            if cls.as_struct and cls.options.PY_struct_arg != "class":
                self.create_arraydescr(cls)
            else:
                self.need_blah = True
                self.wrap_class(cls, modinfo)
            self._pop_splicer(name)
        self._pop_splicer("class")

        self.reset_file()
        self.wrap_enums(node)

        if node.functions:
            self._push_splicer("function")
            #            self._begin_class()
            self.wrap_functions(None, node.functions, fileinfo)
            self._pop_splicer("function")

        self.write_module(node, modinfo, fileinfo, top)

    def register_submodule(self, ns, modinfo):
        """Create code to add submodule to a module.

        Args:
            ns - ast.NamespaceNode
            modinfo - ModuleTuple
        """
        fmt_ns = ns.fmtdict

        self.module_init_decls.append(
            wformat("PyObject *{PY_prefix}init_{PY_module_init}(void);", fmt_ns))

        output = modinfo.type_object_creation
        output.append(
            wformat("""
{{+
PyObject *submodule = {PY_prefix}init_{PY_module_init}();
if (submodule == NULL)
+INITERROR;-
Py_INCREF(submodule);
PyModule_AddObject(m, (char *) "{PY_module_name}", submodule);
-}}""",
                    fmt_ns,
                )
        )

    def wrap_enums(self, node):
        """Wrap enums for library, namespace or class.

        Args:
            node - ast.LibraryNode, ast.NamespaceNode, ast.ClassNode
        """
        enums = node.enums
        if not enums:
            return
        self._push_splicer("enums")
        for enum in enums:
            self.wrap_enum(enum)
        self._pop_splicer("enums")

    def wrap_enum(self, node):
        """Wrap an enumeration.
        If module, use PyModule_AddIntConstant.
        If class, create a descriptor.
        Without a setter, it will be read-only.

        Args:
            node -
        """
        fmtmembers = node._fmtmembers

        ast = node.ast
        output = self.enum_impl
        if node.parent.nodename != "class":
            # library/namespace enumerations
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

    def wrap_class(self, node, modinfo):
        """Create an extension type for a C++ class.

        Wrapper code added to py_type_object_creation.

        Args:
            node - ast.ClassNode.
            modinfo - ModuleTuple
        """
        self.log.write("class {1.name}\n".format(self, node))
        fileinfo = FileTuple([], [], [], [])
        fmt_class = node.fmtdict

        node.eval_template("PY_type_filename")
        fmt_class.PY_this_call = wformat("self->{PY_type_obj}->", fmt_class)

        # Create code for module to add type to module
        output = modinfo.type_object_creation
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
            "+{namespace_scope}{cxx_class} * {PY_type_obj};\n"
            "int {PY_type_dtor};",
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
            self.wrap_class_variable(var, fileinfo)

        # wrap methods
        self.tp_init_default = "0"
        self._push_splicer("method")
        self.wrap_functions(node, node.functions, fileinfo)
        self._pop_splicer("method")

        self.write_extension_type(node, fileinfo)

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
*addr = self->{PY_type_obj};
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
#           'offsets':[0,8],
#           'itemsize':12},
          align=True)

        Args:
            node - ast.ClassNode
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
            "static PyArray_Descr *{PY_struct_array_descr_create}({void_proto})",
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

    def wrap_class_variable(self, node, fileinfo):
        """Wrap a VariableNode in a class with descriptors.

        Args:
            node - ast.VariableNode.
            fileinfo - FileTuple
        """
        options = node.options
        fmt_var = node.fmtdict
        fmt_var.PY_getter = wformat(options.PY_member_getter_template, fmt_var)
        fmt_var.PY_setter = "NULL"  # readonly

        fmt = util.Scope(fmt_var)
        fmt.c_var = wformat("{PY_param_self}->{PY_type_obj}->{field_name}", fmt_var)
        fmt.c_deref = ""  # XXX needed for PY_ctor
        fmt.py_var = "value"  # Used with PY_get

        ast = node.ast
        arg_typemap = ast.typemap

        if arg_typemap.PY_ctor:
            fmt.ctor = wformat(arg_typemap.PY_ctor, fmt)
        else:
            fmt.ctor = "UUUctor"
        fmt.cxx_decl = ast.gen_arg_as_cxx(name="rv")

        output = fileinfo.GetSetBody
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
        fileinfo.GetSetDef.append(
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

    def check_dimension_blk(self, arg):
        """Check dimension attribute.
        Convert it to use Numpy.

        Args:
            arg - argument node.
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

    def set_fmt_fields(self, ast, fmt, is_result=False):
        """
        Set format fields for ast.
        Used with arguments and results.

        Args:
            ast - declast.Declaration
                  Abstract Syntax Tree of argument or result
            fmt - format dictionary
        """
        typemap = ast.typemap
        if typemap.PY_PyObject:
            fmt.PyObject = typemap.PY_PyObject
        if typemap.PY_PyTypeObject:
            fmt.PyTypeObject = typemap.PY_PyTypeObject
        if typemap.PYN_descr:
            fmt.PYN_descr = typemap.PYN_descr

        if typemap.base == "vector":
            vtypemap = ast.template_arguments[0].typemap
            fmt.numpy_type = vtypemap.PYN_typenum
            fmt.cxx_T = ast.template_arguments[0].typemap.name
            fmt.npy_ndims = "1"
            if is_result:
                fmt.npy_dims = "SHD_" + fmt.C_result
            else:
                fmt.npy_dims = "SHD_" + ast.name
#            fmt.pointer_shape = dimension
            # Dimensions must be in npy_intp type array.
            # XXX - assumes 1-d
            fmt.npy_intp = "npy_intp {}[1];\n".format(fmt.npy_dims)
#            fmt.npy_intp = "npy_intp {}[1] = {{{}->size()}};\n".format(fmt.npy_dims, fmt.cxx_var)

        dimension = ast.attrs.get("dimension", None)
        if dimension:
            # (*), (:), (:,:)
            if dimension[0] not in ["*", ":"]:
                fmt.npy_ndims = "1"
                if is_result:
                    fmt.npy_dims = "SHD_" + fmt.C_result
                else:
                    fmt.npy_dims = "SHD_" + ast.name
                fmt.pointer_shape = dimension
                # Dimensions must be in npy_intp type array.
                # XXX - assumes 1-d
                fmt.npy_intp = "npy_intp {}[1] = {{{}}};\n".format(
                    fmt.npy_dims, dimension)

#        fmt.c_type = typemap.c_type
        fmt.cxx_type = wformat(typemap.cxx_type, fmt) # expand cxx_T

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

    def intent_out(self, typemap, intent_blk, fmt):
        """Add code for post-call.
        Create PyObject from C++ value to return.
        Used with function results, intent(OUT) and intent(INOUT) arguments.

        Args:
            typemap - typemap of C++ variable.
            intent_blk -
            fmt - format dictionary

        NumPy intent(OUT) arguments will create a Python object as part of pre-call.
        Return a BuildTuple instance.
        """
        if "post_call" in intent_blk:
            # Explicit code exists to create object.
            # If post_call is None, the Object has already been created
            build_format = "O"
            vargs = fmt.py_var
            blk = None
            ctorvar = fmt.py_var
        else:
            # Decide values for Py_BuildValue
            build_format = typemap.PY_build_format or typemap.PY_format
            vargs = typemap.PY_build_arg
            if not vargs:
                vargs = "{cxx_var}"
            vargs = wformat(vargs, fmt)

            if typemap.PY_ctor:
                decl = "{PyObject} * {py_var} = NULL;"
                post_call = "{py_var} = " + typemap.PY_ctor + ";"
                ctorvar = fmt.py_var
            else:
                # ex. long long does not define PY_ctor.
                fmt.PY_build_format = build_format
                fmt.vargs = vargs
                decl = "{PyObject} * {py_var} = NULL;"
                post_call = '{py_var} = Py_BuildValue("{PY_build_format}", {vargs});'
                ctorvar = fmt.py_var
            blk = dict(
                decl=[wformat(decl, fmt)],
                post_call=[wformat(post_call, fmt)],
            )

        return BuildTuple(build_format, vargs, blk, ctorvar)

    def wrap_functions(self, cls, functions, fileinfo):
        """Wrap functions for a library or class.
        Compute overloading map.

        Args:
            cls - ast.ClassNode
            functions -
            fileinfo - FileTuple
        """
        overloaded_methods = {}
        for function in functions:
            flist = overloaded_methods.setdefault(function.ast.name, [])
            if not function.options.wrap_python:
                continue
            flist.append(function)
        self.overloaded_methods = overloaded_methods

        for function in functions:
            self.wrap_function(cls, function, fileinfo)

        self.multi_dispatch(functions, fileinfo)

    def wrap_function(self, cls, node, fileinfo):
        """Write a Python wrapper for a C or C++ function.

        Args:
            cls  - ast.ClassNode or None for functions
            node - ast.FunctionNode.
            fileinfo - FileTuple

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
            fmt_result, result_typeflag, need_malloc, result_blk = self.process_result(node, fmt)
        else:
            result_blk = {}

        PY_code = []

        # arguments to PyArg_ParseTupleAndKeywords
        parse_format = []
        parse_vargs = []

        # arguments to Py_BuildValue
        build_tuples = []

        # Code blocks
        # Accumulate code from statements.
        decl_code = []  # variables for function
        post_parse_code = []
        pre_call_code = []
        post_call_code = []  # Create objects passed to PyBuildValue
        cleanup_code = []
        fail_code = []

        cxx_call_list = []

        # parse arguments
        # call function based on number of default arguments provided
        default_calls = []  # each possible default call
        found_default = False
        if node._has_default_arg:
            decl_code.append("Py_ssize_t SH_nargs = 0;")
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
            fmt_arg.data_var = "SHData_" + arg_name
            fmt_arg.size_var = "SHSize_" + arg_name

            arg_typemap = arg.typemap
            fmt_arg.numpy_type = arg_typemap.PYN_typenum
            # Add formats used by py_statements
            fmt_arg.c_type = arg_typemap.c_type
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

            self.set_fmt_fields(arg, fmt_arg)
            as_object = False
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
            elif arg_typemap.base == "vectorXXX":
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
            elif dimension:
                fmt_arg.c_decl = wformat("{c_type} * {c_var}", fmt_arg)
                fmt_arg.cxx_decl = wformat("{cxx_type} * {cxx_var}", fmt_arg)
                local_var = "pointer"
                as_object = True
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
                # Allocate NumPy Array.
                # Assumes intent(out)
                # ex. (int arg1, int arg2 +intent(out)+allocatable(mold=arg1))
                attr_allocatable(self.language, allocatable, node, arg, fmt_arg)
                if intent != "out":
                    raise RuntimeError(
                        "Argument must have intent(out)")
                intent_blk = typemap.lookup_stmts(
                    py_statements_local,
                    ["intent_out", "allocatable", node.options.PY_array_arg])
            elif arg_typemap.base == "struct":
                intent_blk = typemap.lookup_stmts(
                    py_statements_local,
                    ["struct", "intent_" + intent, options.PY_struct_arg])
            elif arg_typemap.base == "vector":
                intent_blk = typemap.lookup_stmts(
                    py_statements_local,
                    ["intent_" + intent, "vector", options.PY_array_arg])
                whelpers.add_to_PyList_helper_vector(arg)
            elif dimension:
                # ex. (int * arg1 +intent(in) +dimension(:))
                self.check_dimension_blk(arg)
                intent_blk = typemap.lookup_stmts(
                    py_statements_local,
                    ["intent_" + intent, "dimension", options.PY_array_arg])
            else:
                py_statements = arg_typemap.py_statements
                intent_blk = typemap.lookup_stmts(py_statements, ["intent_" + intent])

            if "parse_as_object" in intent_blk:
                as_object = True
            cxx_local_var = intent_blk.get("cxx_local_var", "")
            create_out_decl = intent_blk.get("create_out_decl", False)
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
                # Argument is implied from other arguments.
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
                            len(post_parse_code),
                            len(pre_call_code),
                            ",\t ".join(cxx_call_list),
                        )
                    )

                # Declare C variable - may be PyObject.
                # add argument to call to PyArg_ParseTypleAndKeywords
                if as_object:
                    # Use NumPy/list with dimensioned or struct arguments.
                    fmt_arg.pytmp_var = "SHTPy_" + fmt_arg.c_var
#                    fmt_arg.pydescr_var = "SHDPy_" + arg.name
                    pass_var = fmt_arg.cxx_var
                    parse_format.append("O")
                    parse_vargs.append("&" + fmt_arg.pytmp_var)
                elif arg_typemap.PY_PyTypeObject:
                    # Expect object of given type
                    # cxx_var is declared by py_statements.intent_out.post_parse.
                    fmt_arg.py_type = arg_typemap.PY_PyObject or "PyObject"
                    append_format(decl_code, "{py_type} * {py_var};", fmt_arg)
                    pass_var = fmt_arg.cxx_var
                    parse_format.append(arg_typemap.PY_format)
                    parse_format.append("!")
                    parse_vargs.append("&" + arg_typemap.PY_PyTypeObject)
                    parse_vargs.append("&" + fmt_arg.py_var)
                elif arg_typemap.PY_from_object:
                    # Use function to convert object
                    # cxx_var created directly (no c_var)
                    append_format(decl_code, "{cxx_decl};", fmt_arg)
                    pass_var = fmt_arg.cxx_var
                    parse_format.append(arg_typemap.PY_format)
                    parse_format.append("&")
                    parse_vargs.append(arg_typemap.PY_from_object)
                    parse_vargs.append("&" + fmt_arg.cxx_var)
                else:
                    append_format(decl_code, "{c_decl};", fmt_arg)
                    parse_format.append(arg_typemap.PY_format)
                    parse_vargs.append("&" + fmt_arg.c_var)

            if intent in ["inout", "out"]:
                if intent == "out":
                    if allocatable or dimension or create_out_decl:
                        # If an array, a local NumPy array has already been defined.
                        pass
                    elif not cxx_local_var:
                        pass_var = fmt_arg.cxx_var
                        append_format(
                            pre_call_code, "{cxx_decl};  // intent(out)", fmt_arg
                        )

                if not hidden:
                    # output variable must be a pointer
                    build_tuples.append(
                        self.intent_out(arg_typemap, intent_blk, fmt_arg)
                    )

            # Code to convert parsed values (C or Python) to C++.
            allocate_local_blk = self.add_stmt_capsule(arg, intent_blk, fmt_arg)
            if allocate_local_blk:
                update_code_blocks(locals(), allocate_local_blk, fmt_arg)
            update_code_blocks(locals(), intent_blk, fmt_arg)
            goto_fail = goto_fail or intent_blk.get("goto_fail", False)
            self.need_numpy = self.need_numpy or intent_blk.get("need_numpy", False)
            if "c_helper" in intent_blk:
                c_helper = wformat(intent_blk["c_helper"], fmt_arg)
                for helper in c_helper.split():
                    self.c_helper[helper] = True
            self.add_statements_headers(intent_blk)

            if intent != "out" and not cxx_local_var and arg_typemap.c_to_cxx:
                # Make intermediate C++ variable
                # Needed to pass address of variable
                # Helpful with debugging.
                fmt_arg.cxx_var = "SH_" + fmt_arg.c_var
                fmt_arg.cxx_decl = arg.gen_arg_as_cxx(
                    name=fmt_arg.cxx_var, params=None, continuation=True
                )
                fmt_arg.cxx_val = wformat(arg_typemap.c_to_cxx, fmt_arg)
                append_format(post_parse_code, "{cxx_decl} =\t {cxx_val};", fmt_arg)
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

        # Add implied argument initialization to pre_call_code
        for arg in arg_implied:
            intent_blk = self.implied_blk(node, arg, pre_call_code)

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
            decl_code.append(
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
                len(post_parse_code),
                len(pre_call_code),
                ",\t ".join(cxx_call_list),
            )
        )

        # Result pre_call is added once before all default argument cases.
        if "pre_call" in result_blk:
            PY_code.extend(["", "// result pre_call"])
            PY_code.extend(result_blk["pre_call"])

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
        for nargs, post_parse_len, pre_call_len, call_list in default_calls:
            if found_default:
                PY_code.append("case %d:" % nargs)
                PY_code.append(1)
                need_blank = False
                if post_parse_len or pre_call_len:
                    # Only add scope if necessary
                    PY_code.append("{")
                    PY_code.append(1)
                    extra_scope = True
                else:
                    extra_scope = False

            if post_parse_len:
                if options.debug:
                    if need_blank:
                        PY_code.append("")
                    PY_code.append("// post_parse")
                PY_code.extend(post_parse_code[:post_parse_len])
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

            if pre_call_len:
                if options.debug:
                    if need_blank:
                        PY_code.append("")
                    PY_code.append("// pre_call")
                PY_code.extend(pre_call_code[:pre_call_len])
                need_blank = True
            fmt.PY_call_list = call_list

            if options.debug and need_blank:
                PY_code.append("")

            capsule_order = None
            if is_ctor:
                self.create_ctor_function(cls, node, PY_code, fmt)
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
                CXX_result = node.ast
                capsule_type = CXX_result.gen_arg_as_cxx(
                    name=None, force_ptr=True, params=None, continuation=True,
                    with_template_args=True,
                )
                append_format(decl_code, "{C_rv_decl} = NULL;", fmt_result)
                PY_code.extend(self.allocate_memory(
                    self.language, fmt_result.cxx_var, capsule_type, fmt_result,
                    "goto fail", result_typeflag))
                append_format(fail_code, "if ({cxx_var} != NULL) {{+\n"
                              "{PY_release_memory_function}({capsule_order}, {cxx_var});\n"
                              "-}}",
                              fmt_result)
                goto_fail = True
                fmt_result.py_capsule = "SHC_" + fmt_result.c_var
                fmt_result.cxx_addr = ""
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
            decl_code.append(fmt.C_rv_decl + ";")

        # Compute return value
        if CXX_subprogram == "function":
            ttt0 = self.intent_out(result_typemap, result_blk, fmt_result)
            # Add result to front of return tuple.
            build_tuples.insert(0, ttt0)
            if ttt0.format == "O":
                # If an object has already been created,
                # use another variable for the result.
                fmt.PY_result = "SHPyResult"
            self.add_stmt_capsule(ast, result_blk, fmt_result)
            update_code_blocks(locals(), result_blk, fmt_result)
            goto_fail = goto_fail or result_blk.get("goto_fail", False)
            self.need_numpy = self.need_numpy or result_blk.get("need_numpy", False)

        # If only one return value, return the ctor
        # else create a tuple with Py_BuildValue.
        if is_ctor:
            return_code = "return 0;"
        elif not build_tuples:
            return_code = "Py_RETURN_NONE;"
        elif len(build_tuples) == 1:
            # return a single object already created in build_stmts
            blk = build_tuples[0].blk
            if blk:
                decl_code.extend(blk["decl"])
                post_call_code.extend(blk["post_call"])
            fmt.py_var = build_tuples[0].ctorvar
            return_code = wformat("return (PyObject *) {py_var};", fmt)
        else:
            # fmt=format for function. Do not use fmt_result here.
            # There may be no return value, only intent(OUT) arguments.
            # create tuple object
            fmt.PyBuild_format = "".join([ttt.format for ttt in build_tuples])
            fmt.PyBuild_vargs = ",\t ".join([ttt.vargs for ttt in build_tuples])
            rv_blk = dict(
                decl=["PyObject *{PY_result} = NULL;  // return value object"],
                post_call=["{PY_result} = "
                           'Py_BuildValue("{PyBuild_format}",\t {PyBuild_vargs});'],
                # Since this is the last statement before the Return,
                # no need to check for error. Just return NULL.
                # fail=["Py_XDECREF(SHPyResult);"],
            )
            update_code_blocks(locals(), rv_blk, fmt)
            return_code = wformat("return {PY_result};", fmt)

        need_blank = False  # put return right after call
        if post_call_code and not is_ctor:
            # ctor does not need to build return values
            if options.debug:
                PY_code.append("")
                PY_code.append("// post_call")
            PY_code.extend(post_call_code)
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

        if len(decl_code):
            # Add blank line after declarations.
            decl_code.append("")
        PY_impl = [1] + decl_code + PY_code + [-1]

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
        self.create_method(node, expose, is_ctor, fmt, PY_impl, fileinfo)

    def create_method(self, node, expose, is_ctor, fmt, PY_impl, fileinfo):
        """Format the function.

        Args:
            node    - function node to wrap
                      or None when called from multi_dispatch.
            expose  - True if exposed to user.
            is_ctor - True if this is a constructor.
            fmt     - dictionary of format values.
            PY_impl - list of implementation lines.
            fileinfo - FileTuple
        """
        if node:
            cpp_if = node.cpp_if
        else:
            cpp_if = False

        body = fileinfo.MethodBody
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
        #        self._create_splicer(fmt.function_name, body, PY_impl)
        if node and node.options.debug:
            body.append("// " + node.declgen)
        self._create_splicer(
            fmt.underscore_name +
            fmt.function_suffix +
            fmt.template_suffix,
            body,
            PY_impl,
        )
        body.append("}")
        if cpp_if:
            body.append("#endif // " + cpp_if)

        if expose is True:
            if cpp_if:
                fileinfo.MethodDef.append("#" + cpp_if)
            # default name
            append_format(
                fileinfo.MethodDef,
                '{{"{function_name}{function_suffix}{template_suffix}",\t '
                "(PyCFunction){PY_name_impl},\t "
                "{PY_ml_flags},\t "
                "{PY_name_impl}__doc__}},",
                fmt,
            )
            if cpp_if:
                fileinfo.MethodDef.append("#endif // " + cpp_if)

    #        elif expose is not False:
    #            # override name
    #            fmt = util.Scope(fmt)
    #            fmt.expose = expose
    #            fileinfo.MethodDef.append( wformat('{{"{expose}", (PyCFunction){PY_name_impl}, {PY_ml_flags}, {PY_name_impl}__doc__}},', fmt))

    def create_ctor_function(self, cls, node, code, fmt):
        """
        Wrap a function which is a constructor.
        Typical c++ constructors are created.
        But also used for structs which are treated as constructors.
        Explicitly assign to fields since C does not have constructors.

        Allocate an instance.
        XXX - do memory reference stuff
        """
        assert cls is not None
        capsule_type = fmt.namespace_scope + fmt.cxx_type + " *"
        var = "self->" + fmt.PY_type_obj
        if cls.as_struct:
            typeflag = "struct"
        else:
            typeflag = None
        code.extend(self.allocate_memory(
            self.language, var, capsule_type, fmt, "return -1", typeflag))
        append_format(code,
                      "self->{PY_type_dtor} = {capsule_order};",
                      fmt)

        if cls.as_struct and cls.options.PY_struct_arg == "class":
            code.append("// initialize fields")
            append_format(code, "{namespace_scope}{cxx_type} *SH_obj = self->{PY_type_obj};", fmt)
            for var in node.ast.params:
                code.append("SH_obj->{} = {};".format(var.name, var.name))

    def process_result(self, node, fmt):
        """Work on formatting for result values.

        Return fmt_result
        Args:
            node    - FunctionNode to wrap.
            fmt     - dictionary of format values.
        """
        options = node.options
        ast = node.ast
        is_ctor = ast.is_ctor()
        CXX_subprogram = node.CXX_subprogram
        result_typemap = node.CXX_result_typemap

        result_typeflag = None
        need_malloc = False
        result_return_pointer_as = node.ast.return_pointer_as
        result_blk = {}

        if CXX_subprogram == "function":
            fmt_result0 = node._fmtresult
            fmt_result = fmt_result0.setdefault(
                "fmtpy", util.Scope(fmt)
            )  # fmt_func

            CXX_result = node.ast
            if result_typemap.base == "vector" and not CXX_result.is_pointer():
                if CXX_result.is_pointer():
                    pass
                else:
                    # Allocate variable to the type returned by the function.
                    result_typeflag = "vector"
                    result_return_pointer_as = "pointer"
                    fmt_result.cxx_var = wformat("{C_result}", fmt_result)
            elif result_typemap.base == "struct" and not CXX_result.is_pointer():
                # Allocate variable to the type returned by the function.
                # No need to convert to C.
                result_typeflag = "struct"
                result_return_pointer_as = "pointer"
                fmt_result.cxx_var = wformat("{C_result}", fmt_result)
            elif result_typemap.cxx_to_c is None:
                fmt_result.cxx_var = wformat("{C_result}", fmt_result)
            else:
                fmt_result.cxx_var = wformat(
                    "{CXX_local}{C_result}", fmt_result
                )

            if result_typeflag:
                # Force result to be a pointer to a struct/vector
                need_malloc = True
                fmt.C_rv_decl = CXX_result.gen_arg_as_cxx(
                    name=fmt_result.cxx_var,
                    force_ptr=True,
                    params=None,
                    continuation=True,
                    with_template_args=True,
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
            fmt_result.data_var = "SHData_" + fmt_result.C_result
            fmt_result.size_var = "SHSize_" + fmt_result.C_result
            fmt_result.numpy_type = result_typemap.PYN_typenum
            #            fmt_pattern = fmt_result

            self.set_fmt_fields(ast, fmt_result, True)
            if is_ctor:
                # Code added by create_ctor_function.
                result_blk = {}
            elif result_typemap.base == "struct":
                result_blk = typemap.lookup_stmts(
                    py_statements_local,
                    ["struct", "result", options.PY_struct_arg])
            elif result_typemap.base == "vector":
                result_blk = typemap.lookup_stmts(
                    py_statements_local,
                    ["result", "vector", options.PY_struct_arg]) # XXX PY_array_arg])
                whelpers.add_to_PyList_helper_vector(ast)
            elif (
                    result_return_pointer_as in ["pointer", "allocatable"]
                    and result_typemap.base != "string"
            ):
                result_blk = typemap.lookup_stmts(
                    py_statements_local,
                    ["result", "dimension", options.PY_array_arg])
            else:
                result_blk = typemap.lookup_stmts(
                    result_typemap.py_statements, ["result"])
            
        return fmt_result, result_typeflag, need_malloc, result_blk

    def XXXadd_stmt_capsule(self, stmts, fmt):
        """Create code to release memory.
        Processes "capsule_type" and "del_lines".

        For example, std::vector intent(out) must eventually release
        the vector via a capsule owned by the NumPy array.

        XXX - Move update_code_blocks here....
        """
        # Create capsule destructor
        capsule_type = stmts.get("capsule_type", None)
        if capsule_type:
            capsule_type = wformat(capsule_type, fmt)
            fmt.capsule_type = capsule_type

            del_lines = stmts.get("del_lines", [])
            if del_lines:
                # Expand things like {cxx_T}
                del_work = []
                for line in del_lines:
                    append_format(del_work, line, fmt)
                del_lines = del_work
            
            capsule_order = self.add_capsule_code(
                self.language + " " + capsule_type, del_lines)
            fmt.capsule_order = capsule_order
            fmt.py_capsule = "SHC_" + fmt.c_var
                
    def add_stmt_capsule(self, ast, stmts, fmt):
        """Create code to allocate/release memory.

        For example, std::vector intent(out) must allocate an vector
        instance and eventually release it via a capsule owned by the
        NumPy array.

        XXX - Move update_code_blocks here....

        The results will be processed by format so literal curly must be protected.

        """
        if ast.is_pointer():
            return None
        allocate_local_var = stmts.get("allocate_local_var", False)
        if allocate_local_var:
            fmt.cxx_alloc_decl = ast.gen_arg_as_cxx(
                name=fmt.cxx_var, force_ptr=True, params=None,
                with_template_args=True, continuation=True,
            )
            capsule_type = ast.gen_arg_as_cxx(
                name=None, force_ptr=True, params=None,
                with_template_args=True,
            )
            fmt.py_capsule = "SHC_" + fmt.c_var
            typemap = ast.typemap
#            result_typeflag = ast.typemap.base
#        result_typemap = node.CXX_result_typemap
            
            return dict(
                decl = ["{cxx_alloc_decl} = NULL;"],
                pre_call = self.allocate_memory_new(
                    fmt.cxx_var, capsule_type, fmt,
                    "goto fail", ast.typemap.base),
                fail = [
                    "if ({cxx_var} != NULL) {{+\n"
                    "{PY_release_memory_function}({capsule_order}, {cxx_var});\n"
                    "-}}"],
                goto_fail=True,
                )
        return None
        
    def allocate_memory(self, lang, var, capsule_type, fmt,
                       error, as_type):
        """Return code to allocate an item.
        Call PyErr_NoMemory if necessary.
        Set fmt.capsule_order which is used to release it.

        Args:
            lang   - c or c++
            var    - Name of variable for assignment.
            capsule_type
            fmt
            error   - error code ex. "goto fail" or "return -1"
            as_type - "struct", "vector", None
        """
        lines = []
        if lang == "c":
            alloc = var + " = malloc(sizeof({cxx_type}));"
            del_lines = [wformat("{stdlib}free(ptr);",fmt)]
        else:
            if as_type == "vector":
                # Expand cxx_T.
                alloc = var + " = new " + wformat(fmt.cxx_type, fmt) + ";"
            elif as_type == "struct":
                alloc = var + " = new {namespace_scope}{cxx_type};"
            else:
                alloc = var + " = new {namespace_scope}{cxx_type}({PY_call_list});"
            del_lines = [
                "{} cxx_ptr =\t static_cast<{}>(ptr);".format(
                    capsule_type, capsule_type
                ),
                "delete cxx_ptr;",
            ]
        append_format(lines, alloc, fmt)
        lines.append("if ({} == NULL) {{+\n"
                     "PyErr_NoMemory();\n{};\n-}}".format(var, error))
        capsule_order = self.add_capsule_code(lang + " " + capsule_type, del_lines)
        fmt.capsule_order = capsule_order
        return lines

    def allocate_memory_new(self, var, capsule_type, fmt,
                       error, as_type):
        """Return code to allocate an item.
        Call PyErr_NoMemory if necessary.
        Set fmt.capsule_order which is used to release it.

        Args:
            var    - Name of variable for assignment.
            capsule_type
            fmt
            error   - error code ex. "goto fail" or "return -1"
            as_type - "struct", "vector", None
        """
        lines = []
        if self.language == "c":
            alloc = "{cxx_var} = malloc(sizeof({cxx_type}));"
            del_lines = ["{stdlib}free(ptr);"]
        else:
            if as_type == "vector":
                # Expand cxx_T.
                alloc = "{cxx_var} = new " + wformat(fmt.cxx_type, fmt) + ";"
            elif as_type == "struct":
                alloc = "{cxx_var} = new {namespace_scope}{cxx_type};"
            else:
                alloc = "{cxx_var} = new {namespace_scope}{cxx_type}({PY_call_list});"
            del_lines = [
                "{} cxx_ptr =\t static_cast<{}>(ptr);".format(
                    capsule_type, capsule_type
                ),
                "delete cxx_ptr;",
            ]
        lines.append(alloc)
        lines.append("if ({cxx_var} == NULL) {{+\n"
                     "PyErr_NoMemory();\ngoto fail;\n-}}")
        capsule_order = self.add_capsule_code(self.language + " " + capsule_type, del_lines)
        fmt.capsule_order = capsule_order
        return lines

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

    def write_extension_type(self, node, fileinfo):
        """
        Args:
            node - ast.ClassNode
            fileinfo - FileTuple
        """
        fmt = node.fmtdict
        fname = fmt.PY_type_filename

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

        # Use headers from implementation
        header_impl_include = self.header_impl_include
        self.find_header(node)
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
            PY_module_scope=fmt.PY_module_scope,
            PY_PyObject=fmt.PY_PyObject,
            PY_PyTypeObject=fmt.PY_PyTypeObject,
            cxx_class=fmt.cxx_class,
            nullptr="0",  # 'NULL',
        )
        self.write_tp_func(node, fmt_type, output)

        output.extend(fileinfo.MethodBody)

        self._push_splicer("impl")
        self._create_splicer("after_methods", output)
        self._pop_splicer("impl")

        output.extend(fileinfo.GetSetBody)
        if fileinfo.GetSetDef:
            fmt_type["tp_getset"] = wformat(
                "{PY_prefix}{cxx_class}_getset", fmt
            )
            append_format(
                output, "\nstatic PyGetSetDef {tp_getset}[] = {{+", fmt_type
            )
            output.extend(fileinfo.GetSetDef)
            self._create_splicer("PyGetSetDef", output)
            output.append("{NULL}            /* sentinel */")
            output.append("-};")
        else:
            fmt_type["tp_getset"] = fmt_type["nullptr"]

        fmt_type["tp_methods"] = wformat("{PY_prefix}{cxx_class}_methods", fmt)
        append_format(
            output, "static PyMethodDef {tp_methods}[] = {{+", fmt_type
        )
        output.extend(fileinfo.MethodDef)
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

    def multi_dispatch(self, functions, fileinfo):
        """Look for overloaded methods.
        When found, create a method which will call each of the
        overloaded methods looking for the one which will accept
        the given arguments.

        Args:
            functions - list of ast.FunctionNode
            fileinfo - FileTuple
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

            self.create_method(None, expose, is_ctor, fmt, body, fileinfo)

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

        if self.py_utility_declaration:
            output.append("")
            output.append("// utility functions")
            output.extend(self.py_utility_declaration)

        if self.py_class_decl:
            output.extend(self.py_class_decl)
            output.append("// ------------------------------")

        #        output.extend(self.define_arraydescr)

        output.append("")
        self._create_splicer("C_declaration", output)
        self._pop_splicer("header")

        append_format(
            output,
            """
extern PyObject *{PY_prefix}error_obj;

#if PY_MAJOR_VERSION >= 3
{PY_extern_C_begin}PyMODINIT_FUNC PyInit_{PY_module_init}(void);
#else
{PY_extern_C_begin}PyMODINIT_FUNC init{PY_module_init}(void);
#endif
""",
            fmt,
        )
        output.append("#endif  /* %s */" % guard)
        #        self.config.pyfiles.append(
        #            os.path.join(self.config.python_dir, fname))
        self.write_output_file(fname, self.config.python_dir, output)

    def write_module(self, node, modinfo, fileinfo, top):
        """
        Write the Python extension module.
        Used with a Library or Namespace node

        Args:
            node - ast.LibraryNode
            modinfo - ModuleTuple
            fileinfo - FileTuple
            top - True = top module, else submodule.
        """
        fmt = node.fmtdict
        fname = fmt.PY_module_filename

        fmt.PY_library_doc = "library documentation"

        self.gather_helper_code(self.c_helper)
        # always include helper header
#        self.c_helper_include[library.fmtdict.C_header_utility] = True
#        self.shared_helper.update(self.c_helper)  # accumulate all helpers

        output = []

        append_format(output, '#include "{PY_header_filename}"', fmt)
        if top and self.need_numpy:
            output.append("#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION")
            output.append('#include "numpy/arrayobject.h"')

        if node.cxx_header:
            for include in node.cxx_header.split():
                output.append('#include "%s"' % include)
        else:
            for include in self.newlibrary.cxx_header.split():
                output.append('#include "%s"' % include)

        self.header_impl_include.update(self.helper_header["file"])
        self.write_headers(self.header_impl_include, output)
        output.append("")
        self._create_splicer("include", output)
        output.append(cpp_boilerplate)
        if self.helper_source["file"]:
            output.extend(self.helper_source["file"])
        output.append("")
        self._create_splicer("C_definition", output)

        if top:
            output.extend(self.module_init_decls)
        output.extend(self.define_arraydescr)

        self._create_splicer("additional_functions", output)
        output.extend(fileinfo.MethodBody)

        append_format(
            output, "static PyMethodDef {PY_prefix}methods[] = {{", fmt
        )
        output.extend(fileinfo.MethodDef)
        output.append(
            "{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */"
        )
        output.append("};")

        output.extend(self.arraydescr)

        if top:
            self.write_init_module(fmt, output, modinfo)
        else:
            self.write_init_submodule(fmt, output, modinfo)

        self.config.pyfiles.append(os.path.join(self.config.python_dir, fname))
        self.write_output_file(fname, self.config.python_dir, output)

    def write_init_module(self, fmt, output, modinfo):
        """Initialize the top level module.

        Uses Python's API for importing a module from a shared library.
        Deal with numpy initialization.
        """
        append_format(output, module_begin, fmt)
        self._create_splicer("C_init_locals", output)
        append_format(output, module_middle, fmt)
        if self.need_numpy:
            output.append("")
            output.append("import_array();")
        output.extend(modinfo.type_object_creation)
        output.extend(self.enum_impl)
        if self.call_arraydescr:
            output.append("")
            output.append("// Define PyArray_Descr for structs")
            output.extend(self.call_arraydescr)
        append_format(output, module_middle2, fmt)
        self._create_splicer("C_init_body", output)
        append_format(output, module_end, fmt)

    def write_init_submodule(self, fmt, output, modinfo):
        """Initialize namespace module.

        Always return a PyObject.
        """
        append_format(output, submodule_begin, fmt)
        output.extend(modinfo.type_object_creation)
#        output.extend(self.enum_impl)
        append_format(output, submodule_end, fmt)

    def write_utility(self):
        node = self.newlibrary
        fmt = node.fmtdict
        output = []
        append_format(output, '#include "{PY_header_filename}"', fmt)
        if len(self.capsule_order) > 1:
            # header file may be needed to fully qualify types capsule destructors
            for include in node.cxx_header.split():
                output.append('#include "%s"' % include)
            output.append("")
        output.extend(self.py_utility_definition)
        output.append("")
        output.extend(self.py_utility_functions)

        if self.need_blah:
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
            output,
            "\n// ----------------------------------------\n"
            "typedef struct {{+\n"
            "const char *name;\n"
            "void (*dtor)(void *ptr);\n"
            "-}} {PY_dtor_context_typedef};",
            fmt
        )

        # Create variable with as array of {PY_dtor_context_typedef}
        # to contain function pointers to routines to release memory.
        fcnnames = []
        for i, name in enumerate(self.capsule_order):
            fcnname = fmt.PY_capsule_destructor_function + "_" + str(i)
            fcnnames.append((name, fcnname))
            output.append("\n// {} - {}".format(i, name))
            output.append("static void {}(void *ptr)".format(fcnname))
            output.append("{+")
            for line in self.capsule_code[name][1]:
                output.append(line)
            output.append("-}")

        output.append(
            "\n"
            "// Code used to release arrays for NumPy objects\n"
            "// via a Capsule base object with a destructor.\n"
            "// Context strings"
        )
        append_format(
            output, "static {PY_dtor_context_typedef} {PY_dtor_context_array}[] = {{+", fmt
        )
        for name in fcnnames:
            output.append('{{"{}", {}}},'.format(name[0], name[1]))
        output.append("{NULL, NULL}")
        output.append("-};")

        # Write function to release from extension type.
        proto = wformat("void {PY_release_memory_function}(int icontext, void *ptr)", fmt)
        self.py_utility_declaration.append("extern " + proto + ";")
        output.append("\n// Release memory based on icontext.")
        output.append(proto)
        append_format(
            output,
            "{{+\n"
            "{PY_dtor_context_array}[icontext].dtor(ptr);\n"
            "-}}", fmt)

        # Write function to release NumPy capsule base object.
        proto = wformat("void *{PY_fetch_context_function}(int icontext)", fmt)
        self.py_utility_declaration.append("extern " + proto + ";")
        output.append("\n//Fetch garbage collection context.")
        output.append(proto)
        append_format(
            output,
            "{{+\n"
            "return {PY_dtor_context_array} + icontext;\n"
#            "return {cast_static}void *{cast1}({PY_dtor_context_array} + icontext){cast2};\n"
            "-}}",
            fmt)

        proto = wformat("void {PY_capsule_destructor_function}(PyObject *cap)", fmt)
        self.py_utility_declaration.append("extern " + proto + ";")
        output.append("\n// destructor function for PyCapsule")
        output.append(proto)
        append_format(
            output,
            "{{+\n"
            # 'const char* name = PyCapsule_GetName(cap);\n'
            'void *ptr = PyCapsule_GetPointer(cap, "{PY_numpy_array_capsule_name}");',
            fmt,
        )
        if self.language == "c":
            append_format(
                output,
                "{PY_dtor_context_typedef} * context = PyCapsule_GetContext(cap);", fmt)
        else:
            append_format(
                output,
                "{PY_dtor_context_typedef} * context = "
                "static_cast<{PY_dtor_context_typedef} *>\t("
                "PyCapsule_GetContext(cap));", fmt)
        output.append("context->dtor(ptr);")
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
        self.need_blah = True
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
        return [
            "Py_INCREF(Py_NotImplemented);",
            "return Py_NotImplemented;"
        ]

    def tp_del(self, msg, ret):
        """default method for tp_del.

        Args:
            msg = 'del'
            ret = ''
        """
        return [
            "{PY_release_memory_function}(self->{PY_type_dtor}, self->{PY_type_obj});",
            "self->{PY_type_obj} = NULL;"
        ]


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
"{PY_module_scope}.{cxx_class}",                       /* tp_name */
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
PyInit_{PY_module_init}(void)
#else
init{PY_module_init}(void)
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
struct module_state *st = GETSTATE(m);"""

# XXX - +INITERROR;-
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

# A submodule always returns a PyObject.
submodule_begin = """
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {{
    PyModuleDef_HEAD_INIT,
    "{PY_module_scope}", /* m_name */
    {PY_prefix}_doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    {PY_prefix}methods, /* m_methods */
    NULL, /* m_reload */
//    {library_lower}_traverse, /* m_traverse */
//    {library_lower}_clear, /* m_clear */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL  /* m_free */
}};
#endif
#define RETVAL NULL

PyObject *{PY_prefix}init_{PY_module_init}(void)
{{+
PyObject *m;
#if PY_MAJOR_VERSION >= 3
m = PyModule_Create(&moduledef);
#else
m = Py_InitModule3((char *) "{PY_module_scope}", {PY_prefix}methods, NULL);
#endif
if (m == NULL)
+return NULL;-
"""
submodule_end = """
return m;
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
        """
        Args:
            node - declast.Identifier
        """
        # Look for functions
        if node.args is None:
            return node.name
        ### functions
        elif node.name == "size":
            # size(arg)
            argname = node.args[0].name
            #            arg = self.func.ast.find_arg_by_name(argname)
            fmt = self.func._fmtargs[argname]["fmtpy"]
            if self.func.options.PY_array_arg == "numpy":
                return wformat("PyArray_SIZE({py_var})", fmt)
            else:
                return fmt.size_var
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
        expr - string expression
        func - declast.FunctionNode
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
        fmt_arg.size_var = fmt.size_var

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


def update_code_blocks(symtab, stmts, fmt):
    """ Accumulate info from statements.
    Append to lists in symtab.

    Args:
        symtab - result of locals() of caller
        stmts  - dictionary
        fmt    - format dictionary (Scope)
    """
    for clause in ["decl", "post_parse", "pre_call",
                   "post_call", "cleanup", "fail"]:
        if clause in stmts:
            util.append_format_cmds(symtab[clause + "_code"], stmts, clause, fmt)

    # If capsule_order is defined, then add some additional code to 
    # do reference counting.
    if fmt.inlocal("capsule_order"):
        suffix = "_capsule"
    else:
        suffix = "_keep"
    for clause in ["decl", "post_call", "fail"]:
        name = clause + suffix
        if "post_call_capsule" in stmts:
            util.append_format_cmds(symtab[clause + "_code"], stmts, name, fmt)


def XXXdo_cast(lang, kind, typ, var):
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


def update_for_language(lang):
    """
    Move language specific entries to current language.

    foo_bar=dict(
      c_decl=[],
      cxx_decl=[],
    )

    For lang==c,
      foo_bar["decl"] = foo_bar["c_decl"]
    """
    for item in py_statements_local.values():
        for clause in ["decl", "post_parse", "pre_call", "post_call",
                       "cleanup", "fail"]:
            specific = lang + "_" + clause
            if specific in item:
                item[clause] = item[specific]

# put into list to avoid duplicating text below
array_error = [
    "if ({py_var} == NULL) {{+",
    "PyErr_SetString(PyExc_ValueError,"
    '\t "{c_var} must be a 1-D array of {c_type}");',
    "goto fail;",
    "-}}",
]
# Use cxx_T instead of c_type for vector.
template_array_error = [
    "if ({py_var} == NULL) {{+",
    "PyErr_SetString(PyExc_ValueError,"
    '\t "{c_var} must be a 1-D array of {cxx_T}");',
    "goto fail;",
    "-}}",
]

malloc_error = [
    "if ({cxx_var} == NULL) {{+",
    "PyErr_NoMemory();",
    "goto fail;",
    "-}}",
]

decl_capsule=[
    "PyObject *{py_capsule} = NULL;",
]
post_call_capsule=[
    "{py_capsule} = "
    'PyCapsule_New({cxx_var}, "{PY_numpy_array_capsule_name}", '
    "\t{PY_capsule_destructor_function});",
    "if ({py_capsule} == NULL) goto fail;",
    "PyCapsule_SetContext({py_capsule},"
    "\t {PY_fetch_context_function}({capsule_order}));",
    "if (PyArray_SetBaseObject(\t"
    "{cast_reinterpret}PyArrayObject *{cast1}{py_var}{cast2},"
    "\t {py_capsule}) < 0)\t goto fail;",
]
fail_capsule=[
    "Py_XDECREF({py_capsule});",
]

# Code clauses are used for C and C++.
# Differences are dealt with by format entries stdlib and cast.
# Language specific clauses are used in update_for_language.
# Function calls which return 'void *', do not require casts in C.
# It doesn't hurt to add them, but I dislike the clutter.
py_statements_local = dict(
####################
## numpy
    intent_in_dimension_numpy=dict(
        need_numpy=True,
        decl=[
            "PyObject * {pytmp_var};",
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_IN_ARRAY){cast2};",
        ] + array_error,
        c_pre_call=[
            "{cxx_decl} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        cleanup=[
            "Py_DECREF({py_var});",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),

    intent_inout_dimension_numpy=dict(
        need_numpy=True,
        decl=[
            "PyObject * {pytmp_var};",
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_INOUT_ARRAY){cast2};",
        ] + array_error,
        c_pre_call=[
            "{cxx_decl} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        post_call=None,  # Object already created in post_parse
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),

    intent_out_dimension_numpy=dict(
        need_numpy=True,
        decl=[
            "{npy_intp}"
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}PyArray_SimpleNew("
            "{npy_ndims}, {npy_dims}, {numpy_type}){cast2};",
        ] + array_error,
        c_pre_call=[
            "{cxx_decl} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        post_call=None,  # Object already created in post_parse
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),

    result_dimension_numpy=dict(
        need_numpy=True,
        decl=[
            "PyObject * {py_var} = NULL;",
        ],
        post_call=[
            "{npy_intp}"
            "{py_var} = "
            "PyArray_SimpleNewFromData({npy_ndims},\t {npy_dims},"
            "\t {numpy_type},\t {cxx_var});",
            "if ({py_var} == NULL) goto fail;",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
        decl_capsule=decl_capsule,
        post_call_capsule=post_call_capsule,
        fail_capsule=fail_capsule,
    ),

########################################
## allocatable
    intent_out_allocatable_numpy=dict(
        need_numpy=True,
        decl=["PyArrayObject * {py_var} = NULL;"],
        pre_call=[
            "{npy_descr_code}"
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}PyArray_NewLikeArray"
            "(\t{npy_prototype},\t {npy_order},\t {npy_descr},\t {npy_subok}){cast2};",
            "if ({py_var} == NULL)",
            "+goto fail;-",
            "{cxx_decl} = {cast_static}{cxx_type} *{cast1}PyArray_DATA({py_var}){cast2};",
            ],
        post_call=None,  # Object already created in pre_call
        fail=["Py_XDECREF({py_var});"],
        goto_fail=True,
    ),

########################################
## list
    intent_in_dimension_list=dict(
        c_helper="from_PyObject_{cxx_type}",
        decl=[
            "PyObject *{pytmp_var} = NULL;",
            "{cxx_decl} = NULL;",
        ],
        post_parse=[
            "Py_ssize_t {size_var};",
            "if (SHROUD_from_PyObject_{c_type}\t({pytmp_var}"
            ",\t \"{c_var}\",\t &{cxx_var}, \t &{size_var}) == -1)",
            "+goto fail;-",
        ],
        cleanup=[
            "{stdlib}free({cxx_var});",
        ],
        fail=[
            "if ({cxx_var} != NULL) {stdlib}free({cxx_var});",
        ],
        goto_fail=True,
    ),

    intent_inout_dimension_list=dict(
#        c_helper="update_PyList_{cxx_type}",
        c_helper="to_PyList_{cxx_type}",
        decl=[
            "PyObject *{pytmp_var} = NULL;",
            "{cxx_decl} = NULL;",
        ],
        post_parse=[
            "Py_ssize_t {size_var};",
            "if (SHROUD_from_PyObject_{c_type}\t({pytmp_var}"
            ",\t \"{c_var}\",\t &{cxx_var}, \t &{size_var}) == -1)",
            "+goto fail;-",
        ],
        post_call=[
#            "SHROUD_update_PyList_{cxx_type}({pytmp_var}, {cxx_var}, {size_var});",
            "PyObject *{py_var} = SHROUD_to_PyList_{cxx_type}\t({cxx_var},\t {size_var});",
            "if ({py_var} == NULL) goto fail;",
        ],
        cleanup=[
            "{stdlib}free({cxx_var});",
        ],
        fail=[
            "if ({cxx_var} != NULL)\t {stdlib}free({cxx_var});",
        ],
        goto_fail=True,
    ),

    intent_out_dimension_list=dict(
        c_helper="to_PyList_{cxx_type}",
        c_header="<stdlib.h>",  # malloc/free
        cxx_header="<cstdlib>",  # malloc/free
        decl=[
            "PyObject *{py_var} = NULL;",
            "{cxx_decl} = NULL;",
        ],
        c_pre_call=[
#            "{cxx_decl}[{pointer_shape}];",
            "{cxx_var} = malloc(\tsizeof({cxx_type}) * {pointer_shape});",
        ] + malloc_error,
        cxx_pre_call=[
#            "{cxx_decl}[{pointer_shape}];",
            "{cxx_var} = static_cast<{cxx_type} *>\t(std::malloc(\tsizeof({cxx_type}) * {pointer_shape}));",
        ] + malloc_error,
        post_call=[
            "{py_var} = SHROUD_to_PyList_{cxx_type}\t({cxx_var},\t {pointer_shape});",
            "if ({py_var} == NULL) goto fail;",
        ],
        cleanup=[
            "{stdlib}free({cxx_var});",
            "{cxx_var} = NULL;",
        ],
        fail=[
            "Py_XDECREF({py_var});",
            "if ({cxx_var} != NULL)\t {stdlib}free({cxx_var});",
        ],
        goto_fail=True,
    ),

########################################
## allocatable
    intent_out_allocatable_list=dict(
        c_helper="to_PyList_{cxx_type}",
        c_header="<stdlib.h>",  # malloc/free
        cxx_header="<cstdlib>",  # malloc/free
        decl=[
            "{cxx_decl} = NULL;",
        ],
        c_pre_call=[
            "{cxx_var} = malloc(sizeof({cxx_type}) * {size_var});",
        ] + malloc_error,
        cxx_pre_call=[
            "{cxx_var} = static_cast<{cxx_type} *>\t(std::malloc(sizeof({cxx_type}) * {size_var}));",
        ] + malloc_error,
        post_call=[
            "PyObject *{py_var} = SHROUD_to_PyList_{cxx_type}\t({cxx_var},\t {size_var});",
            "if ({py_var} == NULL) goto fail;",
        ],
        cleanup=[
            "{stdlib}free({cxx_var});",
        ],
        fail=[
            "if ({cxx_var} != NULL)\t {stdlib}free({cxx_var});",
        ],
        goto_fail=True,
    ),

########################################
# struct
# numpy
    struct_intent_in_numpy=dict(
        need_numpy=True,
        parse_as_object=True,
        cxx_local_var="pointer",
        decl=[
            "PyObject * {pytmp_var} = NULL;",
            "PyArrayObject * {py_var} = NULL;",
#            "PyArray_Descr * {pydescr_var} = {PYN_descr};",
        ],
        post_parse=[
            # PyArray_FromAny steals a reference from PYN_descr
            # and will decref it if an error occurs.
            "Py_INCREF({PYN_descr});",
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}"
            "PyArray_FromAny(\t{pytmp_var},\t {PYN_descr},"
            "\t 0,\t 1,\t NPY_ARRAY_IN_ARRAY,\t NULL){cast2};",
        ] + array_error,
        c_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        cleanup=[
            "Py_DECREF({py_var});",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    struct_intent_inout_numpy=dict(
        need_numpy=True,
        parse_as_object=True,
        cxx_local_var="pointer",
        decl=[
            "PyObject * {pytmp_var} = NULL;",
            "PyArrayObject * {py_var} = NULL;",
#            "PyArray_Descr * {pydescr_var} = {PYN_descr};",
        ],
        post_parse=[
            # PyArray_FromAny steals a reference from PYN_descr
            # and will decref it if an error occurs.
            "Py_INCREF({PYN_descr});",
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}"
            "PyArray_FromAny(\t{pytmp_var},\t {PYN_descr},"
            "\t 0,\t 1,\t NPY_ARRAY_IN_ARRAY,\t NULL){cast2};",
        ] + array_error,
        c_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        post_call=None,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    struct_intent_out_numpy=dict(
        # XXX - expand to array of struct
        need_numpy=True,
        create_out_decl=True,
        cxx_local_var="pointer",
        decl=[
#            "{npy_intp}"
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "Py_INCREF({PYN_descr});",
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}"
            "PyArray_NewFromDescr(\t&PyArray_Type,\t {PYN_descr},"
            "\t 0,\t NULL,\t NULL,\t NULL,\t 0,\t NULL){cast2};",
        ] + array_error,
        c_pre_call=[
#            "{cxx_decl} = PyArray_DATA({py_var});",
            "{cxx_type} *{cxx_var} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
#            "{cxx_decl} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
            "{cxx_type} *{cxx_var} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        post_call=None,  # Object already created in post_parse
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    struct_result_numpy=dict(
        # XXX - expand to array of struct
        need_numpy=True,
        decl=[
            "PyObject * {py_var} = NULL;",
        ],
        post_call=[
            "{npy_intp}"
            "Py_INCREF({PYN_descr});",
            "{py_var} = "
            "PyArray_NewFromDescr(&PyArray_Type, \t{PYN_descr},\t"
            " {npy_ndims}, {npy_dims}, \tNULL, {cxx_var}, 0, NULL);",
            "if ({py_var} == NULL) goto fail;",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
        decl_capsule=decl_capsule,
        post_call_capsule=post_call_capsule,
        fail_capsule=fail_capsule,
    ),

##########
    struct_intent_in_class=dict(
        cxx_local_var="pointer",
        post_parse=[
            "{c_const}{cxx_type} * {cxx_var} ="
            "\t {py_var} ? {py_var}->{PY_type_obj} : NULL;",
        ],
    ),
    struct_intent_inout_class=dict(
        cxx_local_var="pointer",
        post_parse=[
            "{c_const}{cxx_type} * {cxx_var} ="
            "\t {py_var} ? {py_var}->{PY_type_obj} : NULL;",
        ],
        post_call=None,  # Object was passed in
    ),
    struct_intent_out_class=dict(
        create_out_decl=True,
        cxx_local_var="pointer",
        decl=[
            "{PyObject} * {py_var} = NULL;",
        ],
        c_pre_call=[
            "{cxx_type} * {cxx_var} = malloc(sizeof({cxx_type}));",
        ],
        c_dealloc_capsule=[
            "free(ptr);",
        ],
        cxx_pre_call=[
            "{cxx_type} * {cxx_var} = new {cxx_type};",
        ],
        cxx_dealloc_capsule=[
            "delete cxx_ptr;",
        ],
        post_call=[
            "{py_var} ="
            "\t PyObject_New({PyObject}, &{PyTypeObject});",
            "if ({py_var} == NULL) goto fail;",
            "{py_var}->{PY_type_obj} = {cxx_addr}{cxx_var};",
            "{py_var}->{PY_type_dtor} = {capsule_order};",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    struct_result_class=dict(
        cxx_local_var="pointer",
        decl=[
            "{PyObject} *{py_var} = NULL;  // struct_result_class",
        ],
        post_call=[
            "{py_var} ="
            "\t PyObject_New({PyObject}, &{PyTypeObject});",
            "if ({py_var} == NULL) goto fail;",
            "{py_var}->{PY_type_obj} = {cxx_addr}{cxx_var};",
            "{py_var}->{PY_type_dtor} = {capsule_order};",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),


########################################
# std::vector  only used with C++
# list
    intent_in_vector_list=dict(
        # Convert input list argument into a C++ std::vector.
        # Pass to C++ function.
        # cxx_var is released by the compiler.
        c_helper="from_PyObject_vector_{cxx_T}",
        cxx_local_var="scalar",
        decl=[
            "PyObject * {pytmp_var};",  # Object set by ParseTupleAndKeywords.
        ],
        pre_call=[
            "std::vector<{cxx_T}> {cxx_var};",
            "if (SHROUD_from_PyObject_vector_{cxx_T}\t({pytmp_var}"
            ",\t \"{c_var}\",\t {cxx_var}) == -1)",
            "+goto fail;-",
        ],
        goto_fail=True,
    ),
    intent_out_vector_list=dict(
        # Create a pointer a std::vector and pass to C++ function.
        # Create a Python list with the std::vector.
        # cxx_var is released by the compiler.
        c_helper="to_PyList_vector_{cxx_T}",
        cxx_local_var="scalar",
        decl=[
            "PyObject * {py_var} = NULL;",
        ],
        pre_call=[
            "std::vector<{cxx_T}> {cxx_var};",
        ],
        post_call=[
            "{py_var} = SHROUD_to_PyList_vector_{cxx_T}\t({cxx_var});",
            "if ({py_var} == NULL) goto fail;",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    # XXX - must release after copying result.
    result_vector_list=dict(
        decl=[
            "PyObject * {py_var} = NULL;",
        ],
        post_call=[
            "{py_var} = SHROUD_to_PyList_vector_{cxx_T}\t({cxx_var});",
            "if ({py_var} == NULL) goto fail;",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),

##########
# numpy
# cxx_var will always be a pointer since we must save it in a capsule.
    intent_in_vector_numpy=dict(
        # Convert input argument into a NumPy array to make sure it is contiguous,
        # create a local std::vector which will copy the values.
        # Pass to C++ function.
        need_numpy=True,
        cxx_local_var="scalar",
        decl=[
            "PyObject * {pytmp_var};",  # Object set by ParseTupleAndKeywords.
            "PyArrayObject * {py_var} = NULL;",
        ],
        post_parse=[
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_IN_ARRAY){cast2};",
        ] + template_array_error,
        pre_call=[
            "{cxx_T} * {data_var} = static_cast<{cxx_T} *>(PyArray_DATA({py_var}));",
            "std::vector<{cxx_T}> {cxx_var}\t(\t{data_var},\t "
            "{data_var}+PyArray_SIZE({py_var}));",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    intent_out_vector_numpy=dict(
        # Create a pointer a std::vector and pass to C++ function.
        # Create a NumPy array with the std::vector as the capsule object.
        need_numpy=True,
        cxx_local_var="pointer",
        allocate_local_var=True,
        decl=[
            "PyObject * {py_var} = NULL;",
        ],
        post_call=[
            "{npy_intp}"
            "{npy_dims}[0] = {cxx_var}->size();",
            "{py_var} = "
            "PyArray_SimpleNewFromData({npy_ndims},\t {npy_dims},"
            "\t {numpy_type},\t {cxx_var}->data());",
            "if ({py_var} == NULL) goto fail;",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
        decl_capsule=decl_capsule,
        post_call_capsule=post_call_capsule,
        fail_capsule=fail_capsule,
    ),
    result_vector_numpy=dict(
        need_numpy=True,
        decl=[
            "PyObject * {py_var} = NULL;",
        ],
        post_call=[
            "{npy_intp}"
            "{npy_dims}[0] = {cxx_var}->size();",
            "{py_var} = "
            "PyArray_SimpleNewFromData({npy_ndims},\t {npy_dims},"
            "\t {numpy_type},\t {cxx_var}->data());",
            "if ({py_var} == NULL) goto fail;",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
        decl_capsule=decl_capsule,
        post_call_capsule=post_call_capsule,
        fail_capsule=fail_capsule,
    ),

)
