# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
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
SHD_    npy_intp array for shape, {npy_dims_var}
SHC_    PyCapsule owner of memory of NumPy array. {py_capsule}
        Used to deallocate memory.
SHSize_ Size of dimension argument {fmt.size_var}
SHValue PY_typedef_converter variable {fmt.value_var}
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
from .util import wformat, append_format, append_format_lst

# The tree of Python Scope statements.
py_tree = {}
default_scope = None  # for statements

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
        self.header_type_include = {}  # header files in module header
        self.shared_helper = {} # All accumulated helpers
        update_typemap_for_language(self.language)

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
        fmt_library.PY_cleanup_decref = "Py_DECREF"

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
        fmt_library.PY_member_object = "XXXPY_member_object"
        fmt_library.PY_member_data = "XXXPY_member_data"

        fmt_library.npy_rank = "0"   # number of dimensions
        fmt_library.npy_dims_var = fmt_library.nullptr # shape variable
        fmt_library.npy_intp_decl = ""     # shape array definition
        fmt_library.npy_intp_asgn = ""     # shape array assignment

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
        self.write_utility_file()
        self.write_module_header(newlibrary)
        self.write_setup()

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

            fmt.PY_to_object_idtor_func = wformat("PP_{cxx_class}_to_Object_idtor", fmt)
            fmt.PY_to_object_func = wformat("PP_{cxx_class}_to_Object", fmt)
            fmt.PY_from_object_func = wformat("PP_{cxx_class}_from_Object", fmt)

            ntypemap.PY_PyTypeObject = fmt.PY_PyTypeObject
            ntypemap.PY_PyObject = fmt.PY_PyObject
            ntypemap.PY_to_object_idtor = fmt.PY_to_object_idtor_func
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
if (submodule == {nullptr})
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
        options = node.options
        fmt_class = node.fmtdict
        node.create_node_map()

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

        self.find_header(node, self.header_type_include)
            
        # header declarations
        output = self.py_class_decl
        output.append("")
        output.append("// ------------------------------")
        if node.cpp_if:
            output.append("#" + node.cpp_if)

        output.append(wformat("extern PyTypeObject {PY_PyTypeObject};", fmt_class))

        self._create_splicer("C_declaration", output)
        output.append("")
        if options.literalinclude:
            output.append("// start object " + fmt_class.PY_PyObject)
        append_format(
            output,
            "typedef struct {{\n"
            "PyObject_HEAD\n"
            "+{namespace_scope}{cxx_type} * {PY_type_obj};\n"
            "int {PY_type_dtor};",
            fmt_class,
        )

        # Create a PyObject pointer for each pointer member
        # to contain the actual data.
        self.init_member_obj(node)
        # object which holds data - NumPy array
        # Returned by getter
        self.process_member_obj(
            node, "PyObject *{PY_member_object};", output)
        # object which hold data - PyCapsule
        # Used to release memory
        self.process_member_obj(
            node, "PyObject *{PY_member_data};", output)

        self._create_splicer("C_object", output)
        append_format(output, "-}} {PY_PyObject};", fmt_class)
        if options.literalinclude:
            output.append("// end object " + fmt_class.PY_PyObject)
        output.append("")

        self.create_class_utility_functions(node)
        if node.cpp_if:
            output.append("#endif // " + node.cpp_if)

        self.wrap_enums(node)

        for var in node.variables:
            self.wrap_class_variable(node, var, fileinfo)

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
        output = self.py_utility_functions

        ########################################
        # To object helper with idtor argument.
        to_object = wformat(
            """{PY_PyObject} *obj =\t PyObject_New({PY_PyObject}, &{PY_PyTypeObject});
if (obj == {nullptr})+
return {nullptr};
-obj->{PY_type_obj} = addr;
obj->{PY_type_dtor} = idtor;""",
            fmt,
        )
        to_object = to_object.split("\n")
        self.process_member_obj(
            node, "obj->{PY_member_object} = {nullptr};", to_object)
        self.process_member_obj(
            node, "obj->{PY_member_data} = {nullptr};", to_object)
        append_format(
            to_object,
            "return {cast_reinterpret}PyObject *{cast1}obj{cast2};",
            fmt)

        proto = wformat(
            "PyObject *{PY_to_object_idtor_func}({namespace_scope}{cxx_type} *addr,\t int idtor)",
            fmt,
        )
        self.py_class_decl.append(proto + ";")

        output.append("")
        if node.cpp_if:
            output.append("#" + node.cpp_if)
        output.append("// Wrap pointer to struct/class.")
        output.append(proto)
        output.append("{+")
        self._create_splicer("to_object", output, to_object)
        output.append("-}")

        ########################################
        # To object.
        to_object = wformat(
            """PyObject *voidobj;
PyObject *args;
PyObject *rv;

voidobj = PyCapsule_New(addr, {PY_capsule_name}, {nullptr});
args = PyTuple_New(1);
PyTuple_SET_ITEM(args, 0, voidobj);
rv = PyObject_Call((PyObject *) &{PY_PyTypeObject}, args, {nullptr});
Py_DECREF(args);
return rv;""",
            fmt,
        )
        to_object = to_object.split("\n")

        proto = wformat(
            "PyObject *{PY_to_object_func}({namespace_scope}{cxx_type} *addr)",
            fmt,
        )
        self.py_class_decl.append(proto + ";")

        output.append("")
        if node.cpp_if:
            output.append("#" + node.cpp_if)
        output.append("// converter which may be used with PyBuild.")
        output.append(proto)
        output.append("{+")
        self._create_splicer("to_object", output, to_object)
        output.append("-}")

        ########################################
        # From object.
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

        output.append("")
        output.append("// converter which may be used with PyArg_Parse.")
        output.append(proto)
        output.append("{+")
        self._create_splicer(
            "from_object", self.py_utility_functions, from_object
        )
        output.append("-}")
        if node.cpp_if:
            output.append("#endif  // " + node.cpp_if)

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
        options = node.options
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
        if options.literalinclude:
            output.append("// start " + fmt.PY_struct_array_descr_create)
        append_format(
            output,
            "// Create PyArray_Descr for {cxx_class}\n"
            "static PyArray_Descr *{PY_struct_array_descr_create}({void_proto})",
            fmt,
        )
        output.append("{")
        output.append(1)

        tmpfmt = util.Scope(fmt)
        tmpfmt.nvars = len(node.variables)
        append_format(
            output,
            "int ierr;\n"
            "PyObject *obj = {nullptr};\n"
            "PyObject * lnames = {nullptr};\n"
            "PyObject * ldescr = {nullptr};\n"
            "PyObject * dict = {nullptr};\n"
            "PyArray_Descr *dtype = {nullptr};\n"
            "\n"
            "lnames = PyList_New({nvars});\n"
            "if (lnames == {nullptr}) goto fail;\n"
            "ldescr = PyList_New({nvars});\n"
            "if (ldescr == {nullptr}) goto fail;",
            tmpfmt
        )

        for i, var in enumerate(node.variables):
            ast = var.ast
            output.extend(
                [
                    "",
                    "// " + var.ast.name,
                    'obj = PyString_FromString("{}");'.format(ast.name),
                    "if (obj == {}) goto fail;".format(fmt.nullptr),
                    "PyList_SET_ITEM(lnames, {}, obj);".format(i),
                ]
            )

            arg_typemap = ast.typemap
            if ast.is_pointer():
                PYN_typenum = "NPY_INTP"
            else:
                PYN_typenum = arg_typemap.PYN_typenum
            output.extend(
                [
                    "obj = (PyObject *) PyArray_DescrFromType({});".format(
                        PYN_typenum
                    ),
                    "if (obj == {}) goto fail;".format(fmt.nullptr),
                    "PyList_SET_ITEM(ldescr, {}, obj);".format(i),
                ]
            )

            # XXX - add offset and itemsize to be explicit?

        append_format(
            output,
            "obj = {nullptr};\n"
            "\n"
            "dict = PyDict_New();\n"
            "if (dict == {nullptr}) goto fail;\n"
            'ierr = PyDict_SetItemString(dict, "names", lnames);\n'
            "if (ierr == -1) goto fail;\n"
            "lnames = {nullptr};\n"
            'ierr = PyDict_SetItemString(dict, "formats", ldescr);\n'
            "if (ierr == -1) goto fail;\n"
            "ldescr = {nullptr};\n"
            # 'Py_INCREF(Py_True);\n'
            # 'ierr = PyDict_SetItemString(descr, "aligned", Py_True);\n'
            # 'if (ierr == -1) goto fail;\n'
            "ierr = PyArray_DescrAlignConverter(dict, &dtype);\n"
            "if (ierr == 0) goto fail;\n"
            "return dtype;",
            fmt
        )
        append_format(
            output,
            "^fail:\n"
            "Py_XDECREF(obj);\n"
            "if (lnames != {nullptr}) {{+\n"
            "for (int i=0; i < {nvars}; i++) {{+\n"
            "Py_XDECREF(PyList_GET_ITEM(lnames, i));\n"
            "-}}\n"
            "Py_DECREF(lnames);\n"
            "-}}\n"
            "if (ldescr != {nullptr}) {{+\n"
            "for (int i=0; i < {nvars}; i++) {{+\n"
            "Py_XDECREF(PyList_GET_ITEM(ldescr, i));\n"
            "-}}\n"
            "Py_DECREF(ldescr);\n"
            "-}}\n"
            "Py_XDECREF(dict);\n"
            "Py_XDECREF(dtype);\n"
            "return {nullptr};",
            tmpfmt
        )
        #    int PyArray_RegisterDataType(descr)

        output.append(-1)
        output.append("}")
        if options.literalinclude:
            output.append("// end " + fmt.PY_struct_array_descr_create)

    def wrap_class_variable(self, parent, node, fileinfo):
        """Wrap a VariableNode in a class/struct with descriptors.

        Args:
            node - ast.VariableNode.
            fileinfo - FileTuple
        """
        options = node.options
        ast = node.ast
        arg_typemap = ast.typemap
        
        fmt_var = node.fmtdict
        fmt_var.PY_getter = wformat(options.PY_member_getter_template, fmt_var)
        fmt_var.PY_setter = fmt_var.nullptr  # readonly
        # How to find other fields in struct.
        fmt_var.PY_struct_context = wformat("{PY_param_self}->{PY_type_obj}->", fmt_var)

        fmt = util.Scope(fmt_var)
        # c_var is used with PY_ctor
        fmt.c_var_raw = wformat("{PY_struct_context}{field_name}", fmt_var)
        fmt.c_var = fmt.c_var_raw
        fmt.ctor_expr = fmt.c_var_raw
        fmt.cxx_var = fmt.c_var
        fmt.c_var_non_const = fmt.c_var
        fmt.c_var_obj = wformat("{PY_param_self}->{PY_member_object}", fmt)
        fmt.c_var_data = wformat("{PY_param_self}->{PY_member_data}", fmt)
        fmt.cxx_var_obj = fmt.c_var_obj
        fmt.cxx_var_data = fmt.c_var_data
        fmt.c_deref = ""  # XXX needed for PY_ctor
        fmt.py_var = "value"  # Used with PY_get
        fmt.PY_array_arg = options.PY_array_arg
        fmt.c_type = arg_typemap.c_type

        py_struct_dimension(parent, node, fmt)
        indirect_stmt = ast.get_indirect_stmt()

        if arg_typemap.PY_get:
            fmt.PY_get = wformat(arg_typemap.PY_get, fmt)
        
        if arg_typemap.PYN_descr:
            # class
            fmt.PYN_descr = arg_typemap.PYN_descr
        else:
            fmt.PYN_typenum = arg_typemap.PYN_typenum

        stmts = ['py', 'descr',
                 arg_typemap.sgroup,
                 indirect_stmt,
        ]
        if indirect_stmt != "scalar":
            # Pointers and static arrays.
            stmts.append(options.PY_array_arg)
            if ast.const:
                # get a non-const pointer for NumPy
                fmt.c_var_non_const = wformat(
                    "{cast_const}{c_type} *{cast1}{c_var}{cast2}", fmt)

        stmt0 = typemap.compute_name(stmts)
        intent_blk = lookup_stmts(stmts)
        output = fileinfo.GetSetBody
        ########################################
        # getter
        output.append("")
        if options.debug:
            self.document_stmts(output, stmt0, intent_blk.name)
        append_format(
            output,
            "static PyObject *{PY_getter}("
            "{PY_PyObject} *{PY_param_self},"
            "\t void *SHROUD_UNUSED(closure))\n"
            "{{+",
            fmt,
        )
        fmt.cxx_decl = ast.gen_arg_as_cxx(name="rv")
        if arg_typemap.PY_ctor:
            fmt.ctor = wformat(arg_typemap.PY_ctor, fmt)
        else:
            fmt.ctor = "UUUctor"

        if intent_blk.name == 'py_default':
            intent_blk = None
        if intent_blk:
            self.update_descr_code_blocks(
                "getter", intent_blk, fmt, output)
        else:
#            linenumber = options.get("__line__", "?")
            output.append("#error no py_statements getter for {}"
                          .format(stmts0))
        output.append("-}")

        ########################################
        # setter
        if not ast.attrs["readonly"]:
            fmt_var.PY_setter = wformat(
                options.PY_member_setter_template, fmt_var
            )

            output.append("")
            if options.debug:
                self.document_stmts(output, stmt0, intent_blk.name)
            append_format(
                output,
                "static int {PY_setter}("
                "{PY_PyObject} *{PY_param_self}, PyObject *{py_var},"
                "\t void *SHROUD_UNUSED(closure))\n{{+",
                fmt
            )

            if intent_blk:
                fmt.cast_type = ast.as_cast(
                    language=self.language)
                self.update_descr_code_blocks(
                    "setter", intent_blk, fmt, output)
            else:
                output.append("#error no py_statements setter for {}"
                              .format(stmts0))
            # XXX - allow user to add error checks on value
            output.append("return 0;\n-}")

        # Set pointers to functions
        fileinfo.GetSetDef.append(
            # XXX - the (char *) only needed for C++
            wformat(
                '{{(char *)"{variable_name}",\t '
                "(getter){PY_getter},\t "
                "(setter){PY_setter},\t "
                "{nullptr}, "  # doc
                "{nullptr}}},",
                fmt_var,
            )
        )  # closure

    def update_descr_code_blocks(self, name, stmts, fmt, output):
        """Format descr code.

        Args:
            name   - "getter" "setter"
            stmts  - PyStmts
            fmt    - Scope
            output - descr code/
        """
        if stmts.need_numpy:
            self.need_numpy = True
        helpers = getattr(stmts, name + "_helper", None)
        if helpers:
            helpers = wformat(helpers, fmt)
            for i, helper in enumerate(helpers.split()):
                setattr(fmt, "hnamefunc" + str(i),
                        self.add_helper(helper))
        # update_code_blocks
        for cmd in getattr(stmts, name):
            output.append(wformat(cmd, fmt))

    def set_fmt_fields(self, cls, fcn, ast, fmt, is_result=False):
        """
        Set format fields for ast.
        Used with arguments and results.

        Args:
            cls - ast.ClassNode or None
            fcn   - ast.FunctionNode of calling function.
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
            fmt.npy_rank = "1"
            if is_result:
                fmt.npy_dims_var = "SHD_" + fmt.C_result
            else:
                fmt.npy_dims_var = "SHD_" + ast.name
            # Dimensions must be in npy_intp type array.
            # XXX - assumes 1-d
            fmt.npy_intp_decl = wformat("npy_intp {npy_dims_var}[1];\n", fmt)
            # XXX - cxx_var may not have prefix yet.
            fmt.npy_intp_asgn = wformat("{npy_dims_var}[0] = {cxx_var}->size();\n", fmt)

        dimension = ast.attrs["dimension"]
        if dimension:
            class_context = "self->{}->".format(fmt.PY_type_obj)
            visitor = ToDimension(cls, fcn, fmt, class_context)
            visitor.visit(ast.metaattrs["dimension"])
            fmt.rank = str(visitor.rank)

            fmt.npy_rank = str(visitor.rank)
            if is_result:
                fmt.npy_dims_var = "SHD_" + fmt.C_result
            else:
                fmt.npy_dims_var = "SHD_" + ast.name
            # Dimensions must be in npy_intp type array.
            fmt.npy_intp_decl = wformat(
                "npy_intp {npy_dims_var}[{npy_rank}];\n", fmt)

            # Assign each rank of dimension.
            fmtdim = []
            fmtsize = []
            for i, dim in enumerate(visitor.shape):
                fmtdim.append("{}[{}] = {};\n".format(
                    fmt.npy_dims_var, i, dim))
                fmtsize.append("({})".format(dim))
            fmt.npy_intp_asgn = "\n".join(fmtdim)
            if len(fmtsize) > 1:
                fmt.array_size = "*\t".join(fmtsize)
            else:
                fmt.array_size = visitor.shape[0]
        elif ast.is_indirect():
            fmt.array_size = "1"  # assume scalar

#        fmt.c_type = typemap.c_type
        fmt.cxx_type = wformat(typemap.cxx_type, fmt) # expand cxx_T

    def set_cxx_nonconst_ptr(self, ast, fmt):
        """Set fmt.cxx_nonconst_ptr.
        A non-const pointer to cxx_var (which may be same as c_var).
        cxx_addr is used with references.
        """
        if self.language == "c":
            if ast.const:
                fmt.cxx_nonconst_ptr = wformat(
                    "({cxx_type} *) {cxx_addr}{cxx_var}", fmt)
            else:
                fmt.cxx_nonconst_ptr = wformat(
                    "{cxx_addr}{cxx_var}", fmt)
        elif ast.const:
            # cast away constness
            fmt.cxx_nonconst_ptr = wformat(
                "const_cast<{cxx_type} *>\t({cxx_addr}{cxx_var})",
                fmt
            )
        else:
            fmt.cxx_nonconst_ptr = wformat("{cxx_addr}{cxx_var}", fmt)

    def set_fmt_hnamefunc(self, blk, fmt):
        """process helper functions from py_statements.c_helper"""
        if blk.c_helper:
            c_helper = wformat(blk.c_helper, fmt)
            for i, helper in enumerate(c_helper.split()):
                setattr(fmt, "hnamefunc" + str(i),
                    self.add_helper(helper))

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
        implied = arg.attrs["implied"]
        if implied:
            fmt = node._fmtargs[arg.name]["fmtpy"]
            fmt.pre_call_intent = py_implied(implied, node)
            append_format(pre_call, "{cxx_var} = {pre_call_intent};", fmt)

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
        if intent_blk.object_created:
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
                declare = "{PyObject} * {py_var} = {nullptr};"
                post_call = "{py_var} = " + typemap.PY_ctor + ";"
                ctorvar = fmt.py_var
            else:
                # ex. long long does not define PY_ctor.
                fmt.PY_build_format = build_format
                fmt.vargs = vargs
                declare = "{PyObject} * {py_var} = {nullptr};"
                post_call = '{py_var} = Py_BuildValue("{PY_build_format}", {vargs});'
                ctorvar = fmt.py_var
            blk = PyStmts(
                declare=[wformat(declare, fmt)],
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
            if not function.options.PY_create_generic:
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

        A cxx local variable exists when cxx_local_var is defined.
        """

        # need_rv_decl - need Return Value declaration.
        #  The simplest case is to assign to rv as part of calling function.
        #  When default arguments are present, a switch statement is create
        #  so set need_rv_decl = True to declare variable once,
        #  then call wrapped function several times.
        #  If goto_fail, set to True to avoid "crosses initialization" error.
        options = node.options
        if not options.wrap_python:
            return
        if options.PY_array_arg not in ["numpy", "list"]:
            linenumber = options.get("__line__", "?")
            raise RuntimeError(
                "Illegal value for PY_array_arg around line {}: {}".
                format(linenumber, options.PY_array_arg))
        if options.PY_struct_arg not in ["numpy", "list", "class"]:
            linenumber = options.get("__line__", "?")
            raise RuntimeError(
                "Illegal value for PY_struct_arg around line {}: {}".
                format(linenumber, options.PY_struct_arg))

        if cls:
            cls_function = "method"
        else:
            cls_function = "function"
        self.log.write("Python {0} {1.declgen}\n".format(cls_function, node))

        fmt_func = node.fmtdict
        fmtargs = node._fmtargs
        fmt = util.Scope(fmt_func)
        fmt.PY_doc_string = "documentation"
        fmt.PY_array_arg = options.PY_array_arg

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
            fmt.PY_error_return = fmt_func.nullptr

        # XXX if a class, then knock off const since the PyObject
        # is not const, otherwise, use const from result.
        # This has been replaced by gen_arg methods, but not sure about const.
        #        if result_typemap.base == 'shadow':
        #            is_const = False
        #        else:
        #            is_const = None
        if CXX_subprogram == "function":
            fmt_result, result_blk = self.process_function_result(cls, node, fmt)
        else:
            fmt_result = fmt
            result_blk = default_scope
            fmt_result.stmt0 = result_blk.name
            fmt_result.stmt1 = result_blk.name
        stmts_comments = []
        if options.debug:
            stmts_comments.append(
                "// ----------------------------------------")
            stmts_comments.append(
                "// Function:  " + ast.gen_decl(params=None))
            self.document_stmts(
                stmts_comments, fmt_result.stmt0, fmt_result.stmt1)
        self.set_fmt_hnamefunc(result_blk, fmt_result)
        if result_blk.fmtdict is not None:
            for key, value in result_blk.fmtdict.items():
                setattr(fmt_result, key, wformat(value, fmt_result))

        PY_code = []

        # arguments to PyArg_ParseTupleAndKeywords
        parse_format = []
        parse_vargs = []

        # arguments to Py_BuildValue
        build_tuples = []

        # Code blocks
        # Accumulate code from statements.
        declare_code = []  # variables for function
        post_declare_code = []
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
        found_optional = False  # Optional added to parse_format
        set_optional = []
        if node._has_default_arg:
            declare_code.append("Py_ssize_t SH_nargs = 0;")
            append_format(
                PY_code,
                "if (args != {nullptr}) SH_nargs += PyTuple_Size(args);\n"
                "if (kwds != {nullptr}) SH_nargs += PyDict_Size(args);",
                fmt
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
            fmt_arg.value_var = "SHValue_" + arg_name

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
                fmt_arg.ctor_expr = "*" + fmt_arg.c_var
            else:
                fmt_arg.c_deref = ""
                fmt_arg.cxx_addr = "&"
                fmt_arg.cxx_member = "."
                fmt_arg.ctor_expr = fmt_arg.c_var
            update_fmt_from_typemap(fmt_arg, arg_typemap)
            attrs = arg.attrs

            self.set_fmt_fields(cls, node, arg, fmt_arg)
            self.set_cxx_nonconst_ptr(arg, fmt_arg)
            pass_var = fmt_arg.c_var  # The variable to pass to the function
            as_object = False
            rank = arg.attrs["rank"]
            dimension = arg.attrs["dimension"]
            hidden = attrs["hidden"]
            implied = attrs["implied"]
            intent = attrs["intent"]
            sgroup = arg_typemap.sgroup
            spointer = arg.get_indirect_stmt()
            stmts = None

            intent_blk = None
            if node._generated == "struct_as_class_ctor":
                stmts = ["py", "ctor", sgroup, spointer, options.PY_array_arg]
                intent_blk = lookup_stmts(stmts)
                if intent_blk.name == "py_default":
                    intent_blk = None
                struct_member = arg.metaattrs["struct_member"]
                struct_fmt = struct_member.fmtdict
                fmt_arg.field_name = struct_fmt.field_name
                fmt_arg.PY_member_object = struct_fmt.PY_member_object
                field_size = struct_member.ast.get_array_size()
                if field_size is not None:
                    fmt_arg.field_size = field_size
                if not found_optional:
                    parse_format.append("|")  # add once
                    found_optional = True
            deref = attrs["deref"] or "pointer"
            if intent_blk is not None:
                pass
            elif arg.is_function_pointer():
                intent_blk = default_scope
            elif implied:
                arg_implied.append(arg)
                intent_blk = default_scope
            elif sgroup == "char":
                stmts = ["py", sgroup, spointer, intent]
                charlen = arg.attrs["charlen"]
                if charlen:
                    fmt_arg.charlen = charlen
                    stmts.append("charlen")
            elif arg_typemap.base == "struct":
                stmts = ["py", sgroup, spointer, intent, arg_typemap.PY_struct_as]
            elif arg_typemap.base == "vector":
                stmts = ["py", sgroup, intent, options.PY_array_arg]
            elif rank or dimension:
                # ex. (int * arg1 +intent(in) +rank(1))
                stmts = ["py", sgroup, spointer, intent,
                         deref, options.PY_array_arg]
            elif deref == "raw":
                # A single pointer.
                stmts = ["py", sgroup, spointer, intent, deref]
            else:
                # Scalar argument
                # ex. (int * arg1 +intent(in))
                stmts = ["py", sgroup, spointer, intent]
            if options.debug:
                stmts_comments.append(
                    "// ----------------------------------------")
                stmts_comments.append("// Argument:  " + arg.gen_decl())
            if stmts is not None:
                if intent_blk is None:
                    intent_blk = lookup_stmts(stmts)
                # Useful for debugging.  Requested and found path.
                fmt_arg.stmt0 = typemap.compute_name(stmts)
                fmt_arg.stmt1 = intent_blk.name
                # Add some debug comments to function.
                if options.debug:
                    self.document_stmts(
                        stmts_comments, fmt_arg.stmt0, fmt_arg.stmt1)
            elif options.debug:
                stmts_comments.append(
                    self.comment + " Exact:     " + intent_blk.name)

            self.set_fmt_hnamefunc(intent_blk, fmt_arg)
            
            cxx_local_var = intent_blk.cxx_local_var
            if cxx_local_var:
                # cxx_local_var is used when explicitly converting
                # to a C++ var in post_declare code.
                # For example, char * to std::string or
                # extracting a class/struct pointer out of a PyObject.
                # With PY_PyTypeObject, there is no c_var, only cxx_var
                if not arg_typemap.PY_PyTypeObject:
                    fmt_arg.cxx_var = "SH_" + fmt_arg.c_var
                pass_var = fmt_arg.cxx_var
                # cxx_member used with typemap fields like PY_ctor.
                if cxx_local_var == "scalar":
                    fmt_arg.cxx_member = "."
                elif cxx_local_var == "pointer":
                    fmt_arg.cxx_member = "->"
            elif intent != "out" and arg_typemap.c_to_cxx:
                # Make intermediate C++ variable
                # Needed to pass address of variable.
                # Convert type like with enums or MPI_Comm.
                # Helpful with debugging.
                fmt_arg.cxx_var = "SH_" + fmt_arg.c_var
                fmt_arg.cxx_decl = arg.gen_arg_as_cxx(
                    name=fmt_arg.cxx_var, params=None, continuation=True
                )
                fmt_arg.cxx_val = wformat(arg_typemap.c_to_cxx, fmt_arg)
                append_format(post_declare_code,
                              "{cxx_decl} =\t {cxx_val};", fmt_arg)
                pass_var = fmt_arg.cxx_var

            if intent_blk.fmtdict is not None:
                for key, value in intent_blk.fmtdict.items():
                    setattr(fmt_arg, key, wformat(value, fmt_arg))

            # Declare argument variable.
            if intent_blk.arg_declare is not None:
                # Explicit declarations from py_statements.
                for line in intent_blk.arg_declare:
                    append_format(declare_code, line, fmt_arg)
            else:
                # Since all declarations are at the top, remove const
                # since it will be assigned later.
                junk = arg.gen_arg_as_c(remove_const=True, continuation=True)
                declare_code.append(junk + ";")
            
            if implied or hidden:
                # Argument is implied from other arguments.
                pass
            elif intent in ["inout", "in"]:
                # names to PyArg_ParseTupleAndKeywords
                arg_names.append(arg_name)
                arg_offsets.append("(char *) SH_kwcpp+%d" % offset)
                offset += len(arg_name) + 1

                # XXX default should be handled differently
                if arg.init is not None:
                    # Default value argument.
                    if not found_optional:
                        parse_format.append("|")  # add once
                        found_optional = True
                    found_default = True
                    # Cleanup should always do Py_XDECREF instead of
                    # Py_DECREF since PyObject pointers may be NULL due
                    # to different paths of execution in switch statement.
                    fmt_func.PY_cleanup_decref = "Py_XDECREF"
                    # call for default arguments  (num args, arg string)
                    default_calls.append(
                        (
                            len(cxx_call_list),
                            len(post_declare_code),
                            len(post_parse_code),
                            len(pre_call_code),
                            ",\t ".join(cxx_call_list),
                        )
                    )

                # Declare C variable - may be PyObject.
                # add argument to call to PyArg_ParseTypleAndKeywords
                if intent_blk.parse_format:
                    # Explicitly specified parse_format
                    # Must also define parse_args.
                    fmt_arg.pytmp_var = "SHTPy_" + fmt_arg.c_var
                    parse_format.append(intent_blk.parse_format)
                    for varg in intent_blk.parse_args:
                        append_format(parse_vargs, varg, fmt_arg)
                elif arg_typemap.PY_PyTypeObject:
                    # Expect object of given type
                    # cxx_var is declared by py_statements.intent_out.post_parse.
                    fmt_arg.py_type = arg_typemap.PY_PyObject or "PyObject"
                    append_format(declare_code, "{py_type} * {py_var};", fmt_arg)
                    pass_var = fmt_arg.cxx_var
                    parse_format.append(arg_typemap.PY_format)
                    parse_format.append("!")
                    parse_vargs.append("&" + arg_typemap.PY_PyTypeObject)
                    parse_vargs.append("&" + fmt_arg.py_var)
                elif arg_typemap.PY_from_object:
                    # Use function to convert object
                    # cxx_var created directly (no c_var)
#                    append_format(declare_code, "{cxx_decl};", fmt_arg)
                    print("XXXX unused")
                    # XXX this code is not hit by the testsuite.
                    # XXX not sure how if cxx_decl is still needed.
                    pass_var = fmt_arg.cxx_var
                    parse_format.append(arg_typemap.PY_format)
                    parse_format.append("&")
                    parse_vargs.append(arg_typemap.PY_from_object)
                    parse_vargs.append("&" + fmt_arg.cxx_var)
                else:
                    parse_format.append(arg_typemap.PY_format)
                    parse_vargs.append("&" + fmt_arg.c_var)

            if intent in ["inout", "out"]:
                if not hidden:
                    # output variable must be a pointer
                    build_tuples.append(
                        self.intent_out(arg_typemap, intent_blk, fmt_arg)
                    )

            # Code to convert parsed values (C or Python) to C++.
            allocate_local_blk = self.add_stmt_capsule(arg, intent_blk, fmt_arg)
            if allocate_local_blk:
                update_code_blocks(locals(), allocate_local_blk, fmt_arg)
            goto_fail = goto_fail or intent_blk.goto_fail
            self.need_numpy = self.need_numpy or intent_blk.need_numpy
            update_code_blocks(locals(), intent_blk, fmt_arg)
            self.add_statements_headers(intent_blk)

            # Pass correct value to wrapped function.
            if intent_blk.arg_call:
                for arg in intent_blk.arg_call:
                    append_format(cxx_call_list, arg, fmt_arg)
            else:
                cxx_call_list.append(pass_var)
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
            declare_code.append(
                kw_const
                + 'char *SHT_kwlist[] = {\f"'
                + '",\f"'.join(arg_names)
                + '",\f'
                + fmt.nullptr
                + ' };'
            )
            parse_format.extend([":", fmt.function_name])
            fmt.PyArg_format = "".join(parse_format)
            fmt.PyArg_vargs = ",\t ".join(parse_vargs)
            PY_code.extend(set_optional)
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
                len(post_declare_code),
                len(post_parse_code),
                len(pre_call_code),
                ",\t ".join(cxx_call_list),
            )
        )

        # Add function pre_call code.
        need_blank0 = True
        pre_call_deref = ""
        if CXX_subprogram == "function":
            allocate_result_blk = self.add_stmt_capsule(ast, result_blk, fmt_result)
            # Result pre_call is added once before all default argument cases.
            if allocate_result_blk and allocate_result_blk.pre_call:
                PY_code.extend(["", "// result pre_call"])
                util.append_format_cmds(PY_code, allocate_result_blk, "pre_call", fmt_result)
                need_blank0 = False
                pre_call_deref = "*"
        if result_blk.pre_call:
            if need_blank0:
                PY_code.extend(["", "// result pre_call"])
            PY_code.extend(result_blk.pre_call)

        # If multiple calls (because of default argument values),
        # declare return value once; else delare on call line.
        need_rv_decl = False
        if CXX_subprogram == "function":
            if is_ctor:
                pass
            elif found_default or goto_fail:
                fmt.PY_rv_asgn = pre_call_deref + fmt_result.cxx_var + " =\t "
                need_rv_decl = True
            elif allocate_result_blk:
                fmt.PY_rv_asgn = "*" + fmt_result.cxx_var + " =\t "
            else:
                fmt.PY_rv_asgn = fmt.C_rv_decl + " =\t "
            if result_typemap.sgroup == "struct":
                # Avoid unused variable.
                # XXX - major kludge.  struct only access declaration via self->obj.
                need_rv_decl = False
        if found_default:
            PY_code.append("switch (SH_nargs) {")

        # build up code for a function
        for nargs, post_declare_len,  post_parse_len, pre_call_len, call_list in default_calls:
            if found_default:
                PY_code.append("case %d:" % nargs)
                PY_code.append(1)
                need_blank = False
                if post_declare_len or post_parse_len or pre_call_len:
                    # Only add scope if necessary.
                    # There may be declarations in these code blocks.
                    # Need to avoid error:
                    # jump to label 'fail' crosses initialization of ...
                    PY_code.append("{")
                    PY_code.append(1)
                    extra_scope = True
                else:
                    extra_scope = False

            if post_declare_len:
                if options.debug:
                    if need_blank:
                        PY_code.append("")
                    PY_code.append("// post_declare")
                PY_code.extend(post_declare_code[:post_declare_len])
                need_blank = True

            if post_parse_len:
                if options.debug:
                    if need_blank:
                        PY_code.append("")
                    PY_code.append("// post_parse")
                PY_code.extend(post_parse_code[:post_parse_len])
                need_blank = True

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
                    "{PY_this_call}{function_name}"
                    "{CXX_template}({PY_call_list});",
                    fmt,
                )
            else:
                append_format(
                    PY_code,
                    "{PY_rv_asgn}{PY_this_call}{function_name}"
                    "{CXX_template}({PY_call_list});",
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
            append_format(
                PY_code,
                "default:+\n"
                "PyErr_SetString(PyExc_ValueError,"
                "\t \"Wrong number of arguments\");\n"
                "return {nullptr};\n"
#                "goto fail;\n"
                "-}}",
                fmt)
# XXX - need to add a extra scope to deal with goto in C++
#            goto_fail = True;

        if need_rv_decl:
            declare_code.append(fmt.C_rv_decl + ";")

        # Compute return value
        if CXX_subprogram == "function":
            ttt0 = self.intent_out(result_typemap, result_blk, fmt_result)
            # Add result to front of return tuple.
            build_tuples.insert(0, ttt0)
            if ttt0.format == "O":
                # If an object has already been created,
                # use another variable for the result.
                fmt.PY_result = "SHPyResult"
            if allocate_result_blk:
                update_code_blocks(locals(), allocate_result_blk, fmt_result)
            update_code_blocks(locals(), result_blk, fmt_result)
            goto_fail = goto_fail or result_blk.goto_fail
            self.need_numpy = self.need_numpy or result_blk.need_numpy

        # If only one return value, return the ctor
        # else create a tuple with Py_BuildValue.
        if is_ctor:
            return_code = "return 0;"
        elif not build_tuples:
            return_code = "Py_RETURN_NONE;"
        elif len(build_tuples) == 1:
            # return a single object already created in build_stmts
            blk = build_tuples[0].blk
            if blk is not None:
                declare_code.extend(blk.declare)
                post_call_code.extend(blk.post_call)
            fmt.py_var = build_tuples[0].ctorvar
            return_code = wformat("return (PyObject *) {py_var};", fmt)
        else:
            # fmt=format for function. Do not use fmt_result here.
            # There may be no return value, only intent(OUT) arguments.
            # create tuple object
            fmt.PyBuild_format = "".join([ttt.format for ttt in build_tuples])
            fmt.PyBuild_vargs = ",\t ".join([ttt.vargs for ttt in build_tuples])
            rv_blk = PyStmts(
                declare=["PyObject *{PY_result} = {nullptr};  // return value object"],
                post_call=["{PY_result} = "
                           'Py_BuildValue("{PyBuild_format}",\t {PyBuild_vargs});'],
                # Since this is the last statement before the Return,
                # no need to check for error. Just return NULL.
                # fail=["Py_XDECREF(SHPyResult);"],
            )
            update_code_blocks(locals(), rv_blk, fmt)
            return_code = wformat("return {PY_result};", fmt)

        need_blank = False  # put return right after call
        if node._generated == "struct_as_class_ctor":
            if options.debug:
                PY_code.append("")
                PY_code.append("// post_call - initialize fields")
            # Create a convience variable to access struct.
            append_format(
                PY_code,
                "{namespace_scope}{cxx_type} *SH_obj = self->{PY_type_obj};",
                fmt)
            PY_code.extend(post_call_code)
            need_blank = True
        elif post_call_code and not is_ctor:
            # ctor does not need to build return values.
            # Called as the __init__ method.
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

        if goto_fail:
            PY_code.extend(["", "^fail:"])
            PY_code.extend(fail_code)
            append_format(PY_code, "return {PY_error_return};", fmt)

        if len(declare_code):
            # Add blank line after declarations.
            declare_code.append("")
        if "py" in node.splicer:
            PY_force = node.splicer["py"]
            PY_impl = None
        else:
            PY_force = None
            PY_impl = [1] + declare_code + PY_code + [-1]

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
        self.create_method(node, expose, is_ctor, fmt,
                           PY_force, PY_impl, stmts_comments,
                           fileinfo)

    def create_method(self, node, expose, is_ctor, fmt,
                      PY_force, PY_impl, stmts_comments,
                      fileinfo):
        """Format the function.

        Args:
            node    - function node to wrap
                      or None when called from multi_dispatch.
            expose  - True if exposed to user.
            is_ctor - True if this is a constructor.
            fmt     - dictionary of format values.
            PY_force - list of inline splicer code.
            PY_impl - list of implementation code.
            stmts_comments -
            fileinfo - FileTuple
        """
        if node:
            cpp_if = node.cpp_if
        else:
            cpp_if = False

        body = fileinfo.MethodBody
        body.append("")
        if node and node.options.debug:
            body.extend(stmts_comments)
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
        self._create_splicer(
            fmt.underscore_name +
            fmt.function_suffix +
            fmt.template_suffix,
            body,
            PY_impl,
            PY_force,
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

        Args:
            cls  - ast.ClassNode
            node - ast.FunctionNode
            code - list of generated wrapper code.
            fmt  -
        """
        assert cls is not None
        capsule_type = fmt.namespace_scope + fmt.cxx_type + " *"
        var = "self->" + fmt.PY_type_obj
        if cls.as_struct:
            typeflag = "struct"
        else:
            typeflag = None
        util.append_format_lst(
            code,
            self.allocate_memory(
                var, capsule_type, fmt, "return -1", typeflag),
            fmt
        )
        append_format(code,
                      "self->{PY_type_dtor} = {capsule_order};",
                      fmt)

    def process_function_result(self, cls, node, fmt):
        """Work on formatting for function result values.

        Return fmt_result
        Args:
            node    - FunctionNode to wrap.
            fmt     - dictionary of format values.
        """
        options = node.options
        ast = node.ast
        attrs = ast.attrs
        is_ctor = ast.is_ctor()
        result_typemap = node.CXX_result_typemap

        result_blk = default_scope

        fmt_result0 = node._fmtresult
        fmt_result = fmt_result0.setdefault(
            "fmtpy", util.Scope(fmt)
        )  # fmt_func
        CXX_result = node.ast

        # Mangle result variable name to avoid possible conflict with arguments.
        fmt_result.cxx_var = wformat(
            "{CXX_local}{C_result}", fmt_result
        )

        fmt.C_rv_decl = CXX_result.gen_arg_as_cxx(
            name=fmt_result.cxx_var, params=None,
            with_template_args=True, continuation=True
        )

        if CXX_result.is_pointer():
            fmt_result.c_deref = "*"
            fmt_result.cxx_addr = ""
            fmt_result.cxx_member = "->"
            fmt_result.ctor_expr = "*" + fmt_result.cxx_var
        else:
            fmt_result.c_deref = ""
            fmt_result.cxx_addr = "&"
            fmt_result.cxx_member = "."
            fmt_result.ctor_expr = fmt_result.cxx_var
        fmt_result.c_var = fmt_result.cxx_var
        fmt_result.py_var = fmt.PY_result
        fmt_result.data_var = "SHData_" + fmt_result.C_result
        fmt_result.size_var = "SHSize_" + fmt_result.C_result
        fmt_result.value_var = "SHValue_" + fmt_result.C_result
        fmt_result.numpy_type = result_typemap.PYN_typenum
        #            fmt_pattern = fmt_result
        update_fmt_from_typemap(fmt_result, result_typemap)

        self.set_fmt_fields(cls, node, ast, fmt_result, True)
        self.set_cxx_nonconst_ptr(ast, fmt_result)
        sgroup = result_typemap.sgroup
        stmts = None
        if is_ctor:
            # Code added by create_ctor_function.
            result_blk = default_scope
            fmt_result.stmt0 = result_blk.name
            fmt_result.stmt1 = result_blk.name
        elif result_typemap.base == "struct":
            stmts = ["py", sgroup, "result", options.PY_struct_arg]
        elif result_typemap.base == "vector":
            stmts = ["py", sgroup, "result", options.PY_array_arg]
        elif sgroup == "native":
            spointer = ast.get_indirect_stmt()
            stmts = ["py", sgroup, spointer, "result"]
            if spointer != "scalar":
                deref = attrs["deref"] or "pointer"
                stmts.append(deref)
                if deref != "scalar":
                    stmts.append(options.PY_array_arg)
        else:
            spointer = ast.get_indirect_stmt()
            stmts = ["py", sgroup, spointer, "result"]
        if stmts is not None:
            result_blk = lookup_stmts(stmts)
            # Useful for debugging.  Requested and found path.
            fmt_result.stmt0 = typemap.compute_name(stmts)
            fmt_result.stmt1 = result_blk.name
                
        return fmt_result, result_blk

    def XXXadd_stmt_capsule(self, stmts, fmt):
        """Create code to release memory.
        Processes "capsule_type" and "del_lines".

        For example, std::vector intent(out) must eventually release
        the vector via a capsule owned by the NumPy array.

        XXX - Remove ability to set capsule_type in statements.
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

        For example, std::vector intent(out) must allocate a vector
        instance and eventually release it via a capsule owned by the
        NumPy array.

        XXX - Move update_code_blocks here....

        The results will be processed by format so literal curly must be protected.

        """
        if ast.is_pointer():
            return None
        allocate_local_var = stmts.allocate_local_var
        if allocate_local_var:
            # We're creating a pointer to a struct which will later then be assigned to.
            # Have to discard constness or the assignment will produce a compile error.
            #  *result = returnConstStructByValue()
            fmt.cxx_alloc_decl = ast.gen_arg_as_cxx(
                name=fmt.cxx_var, force_ptr=True, params=None,
                remove_const=True,
                with_template_args=True, continuation=True,
            )
            capsule_type = ast.gen_arg_as_cxx(
                name=None, force_ptr=True, params=None,
                with_template_args=True,
            )
            fmt.py_capsule = "SHC_" + fmt.c_var
            # A pointer is always created by allocate_result_blk.
            fmt.c_deref = "*"
            fmt.cxx_addr = ""
            fmt.cxx_member = "->"
            fmt.ctor_expr = "*" + fmt.c_var
            typemap = ast.typemap
#            result_typeflag = ast.typemap.base
#        result_typemap = node.CXX_result_typemap
            
            return PyStmts(
                declare=["{cxx_alloc_decl} = {nullptr};"],
                pre_call=self.allocate_memory(
                    fmt.cxx_var, capsule_type, fmt,
                    "goto fail", ast.typemap.base),
                fail=[
                    "if ({cxx_var} != {nullptr}) {{+\n"
                    "{PY_release_memory_function}({capsule_order}, {cxx_var});\n"
                    "-}}"],
                goto_fail=True,
            )
        return None
        
    def allocate_memory(self, var, capsule_type, fmt,
                        error, as_type):
        """Return code to allocate an item.
        Call PyErr_NoMemory if necessary.
        Set fmt.capsule_order which is used to release it.

        When called from create_ctor_function var and error
        will be different than when called for arguments.

        Args:
            var    - Name of variable for assignment.
            capsule_type
            fmt
            error   - error code ex. "goto fail" or "return -1"
            as_type - "struct", "vector", None
        """
        lines = []
        if self.language == "c":
            alloc = var + " = malloc(sizeof({cxx_type}));"
            del_lines = ["free(ptr);"]
        else:
            if as_type == "vector":
                alloc = var + " = new {cxx_type};"
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
        lines.append(alloc)
        # This line is formatted later, thus {{{{ for a single {.
        lines.append("if ({} == {{nullptr}}) {{{{+\n"
                     "PyErr_NoMemory();\n{};\n-}}}}".format(var, error))
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
                fmt_type[tp_name] = fmt_func.nullptr
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
            ret = fmt_func.nullptr if tup[2] == "NULL" else tup[2]
            default = default(node, typename, ret)

            # format and indent default bodies
            fmted = [1]
            for line in default:
                append_format(fmted, line, fmt_func)
            fmted.append(-1)

            self._create_splicer(typename, output, fmted)
            output.append("}")
        self._pop_splicer("type")


######
    def add_helper(self, name):
        """Use a helper function.
        Return the name of the function associated with helper.
        """
        self.c_helper[name] = True
        # Adjust for alias like with type char.
        return whelpers.CHelpers[name]["name"]
        
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

        scope = helper_info.get("scope", "file")
        # assert scope in ["file", "utility"]

        self.helper_need_numpy = (
            helper_info.get("need_numpy", False) or self.helper_need_numpy)

        lang_key = self.language + "_include"
        if lang_key in helper_info:
            for include in helper_info[lang_key].split():
                self.helper_summary["include"][scope][include] = True
        elif "include" in helper_info:
            for include in helper_info["include"].split():
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

        done = {}  # avoid duplicates and recursion
        for name in sorted(helpers.keys()):
            self._gather_helper_code(name, done)

    def find_file_helper_code(self):
        """Get "file" helper code.
        Add to shared_helper, then reset.

        Return dictionary of headers and list of source files.
        """
        if self.newlibrary.options.PY_write_helper_in_util:
            self.shared_helper.update(self.c_helper)
            self.c_helper = {}
            return {}, [], False
        self.gather_helper_code(self.c_helper)
        self.shared_helper.update(self.c_helper)
        self.c_helper = {}
        return (
            self.helper_summary["include"]["file"],
            self.helper_summary["source"]["file"],
            self.helper_need_numpy
        )

    def find_utility_helper_code(self):
        """Get "pwrap_impl" helper code.
        Added to PY_utility_filename and shared among files.

        Return list of code with typedefs.
        """
        self.gather_helper_code(self.shared_helper)
        return (
            self.helper_summary["include"]["pwrap_impl"],
            self.helper_summary["source"]["pwrap_impl"]
        )

    def find_shared_file_helper_code(self):
        """Get "file" helper code when added to utility file.
        """
        if self.newlibrary.options.PY_write_helper_in_util:
            self.gather_helper_code(self.shared_helper)
            return (
                self.helper_summary["include"]["file"],
                self.helper_summary["source"]["file"],
                self.helper_need_numpy,
            )
        return {}, [], False

######

    def write_extension_type(self, node, fileinfo):
        """
        Args:
            node - ast.ClassNode
            fileinfo - FileTuple
        """
        fmt = node.fmtdict
        fname = fmt.PY_type_filename

        hinclude, hsource, helper_need_numpy = self.find_file_helper_code()
        if helper_need_numpy:
            self.need_numpy = True
        # always include helper header
#        self.c_helper_include[library.fmtdict.C_header_utility] = True
#        self.shared_helper.update(self.c_helper)  # accumulate all helpers

        output = []
        if node.cpp_if:
            output.append("#" + node.cpp_if)

        append_format(output, '#include "{PY_header_filename}"', fmt)
        if self.need_numpy:
            self.add_numpy_includes(output)
        self._push_splicer("impl")

        # Use headers from implementation
        header_impl_include = self.header_impl_include
        header_impl_include.update(hinclude)
        self.write_headers(header_impl_include, output)

        self._create_splicer("include", output)
        output.append(cpp_boilerplate)
        output.extend(hsource)
        self._create_splicer("C_definition", output)
        self._create_splicer("additional_methods", output)
        self._pop_splicer("impl")

        fmt_type = dict(
            PY_module_scope=fmt.PY_module_scope,
            PY_PyObject=fmt.PY_PyObject,
            PY_PyTypeObject=fmt.PY_PyTypeObject,
            cxx_class=fmt.cxx_class,
            nullptr=fmt.nullptr,
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
            append_format(output, "{{{nullptr}}}            /* sentinel */", fmt)
            output.append("-};")
        else:
            fmt_type["tp_getset"] = fmt.nullptr

        fmt_type["tp_methods"] = wformat("{PY_prefix}{cxx_class}_methods", fmt)
        append_format(
            output, "static PyMethodDef {tp_methods}[] = {{+", fmt_type
        )
        output.extend(fileinfo.MethodDef)
        self._create_splicer("PyMethodDef", output)
        append_format(
            output,
            "{{{nullptr},   (PyCFunction){nullptr}, 0, {nullptr}}}"
            "            /* sentinel */",
            fmt
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
            append_format(
                body,
                "if (args != {nullptr}) SHT_nargs += PyTuple_Size(args);\n"
                "if (kwds != {nullptr}) SHT_nargs += PyDict_Size(args);",
                fmt
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
                fmt.PY_error_return = fmt.nullptr
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

            self.create_method(None, expose, is_ctor, fmt,
                               None, body, [], fileinfo)

    def write_module_header(self, node):
        """Write the header for the module.
        Args:
            node - ast.LibraryNode.
        """
        fmt = node.fmtdict
        fname = fmt.PY_header_filename
        self.find_header(node, self.header_type_include)
        hinclude, hsource = self.find_utility_helper_code()
        output = []

        # add guard
        guard = fname.replace(".", "_").upper()
        output.extend(["#ifndef %s" % guard, "#define %s" % guard])

        output.append("#include <Python.h>")
        self.write_headers(self.header_type_include, output)

        self._push_splicer("header")
        self._create_splicer("include", output)

        output.extend(hsource)
        if self.newlibrary.options.PY_write_helper_in_util:
            if self.helper_summary["proto"]["file"]:
                output.append("")
                output.append("// Helper functions.")
                output.extend(self.helper_summary["proto"]["file"])

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

        hinclude, hsource, helper_need_numpy = self.find_file_helper_code()
        if helper_need_numpy:
            self.need_numpy = True
        # always include helper header
#        self.c_helper_include[library.fmtdict.C_header_utility] = True
#        self.shared_helper.update(self.c_helper)  # accumulate all helpers

        output = []

        append_format(output, '#include "{PY_header_filename}"', fmt)
        if self.need_numpy:
            self.add_numpy_includes(output, top)

        self.header_impl_include.update(hinclude)
        self.write_headers(self.header_impl_include, output)
        output.append("")
        self._create_splicer("include", output)
        output.append(cpp_boilerplate)
        output.extend(hsource)
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
        append_format(
            output,
            "{{{nullptr},   (PyCFunction){nullptr}, 0, {nullptr}}}"
            "            /* sentinel */",
            fmt
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

    def add_numpy_includes(self, output, top=False):
        """Import numpy
        If top is True, then import_array will be called
        in the file being written.
        """
        if not top:
            output.append('#define NO_IMPORT_ARRAY')
        append_format(output,
                      '#define PY_ARRAY_UNIQUE_SYMBOL {PY_ARRAY_UNIQUE_SYMBOL}',
                      self.newlibrary.fmtdict)
        output.append("#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION")
        output.append('#include "numpy/arrayobject.h"')
        
    def write_utility_file(self):
        """
        Do not write the file unless it has contents.
        """
        node = self.newlibrary
        fmt = node.fmtdict
        need_file = False
        hinclude, hsource, need_numpy = self.find_shared_file_helper_code()
        
        output = []
        append_format(output, '#include "{PY_header_filename}"', fmt)
        if need_numpy:
            self.add_numpy_includes(output)
        self.write_headers(hinclude, output)
        output.append(cpp_boilerplate)

        if hsource:
            output.extend(hsource)
            need_file = True
        if self.py_utility_definition:
            output.append("")
            output.extend(self.py_utility_definition)
            need_file = True
        if self.py_utility_functions:
            output.append("")
            output.extend(self.py_utility_functions)
            need_file = True
        if self.need_blah:
            self.write_capsule_code(output, fmt)
            need_file = True
        if need_file:
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
        output.append('{{{}, {}}},'.format(fmt.nullptr, fmt.nullptr))
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

    def write_setup(self):
        """Write a setup.py file for the module"""
        library = self.newlibrary
        options = library.options
        fmt = library.fmtdict
        fname = "setup.py"

        if options.debug_testsuite:
            srcs = [ "'" + os.path.basename(name) + "'"
                     for name in self.config.pyfiles]
        else:
            srcs = [ "'" + name + "'" for name in self.config.pyfiles]
        fmt = dict(
            language="c++" if self.language == "cxx" else self.language,
            name=fmt.library_lower,
            source=",\n         ".join(srcs),
            include_dirs="None",
        )

        output = ["from setuptools import setup, Extension"]
        if self.need_numpy:
            output.append("import numpy")
            fmt["include_dirs"] = "[numpy.get_include()]"

        append_format(
            output, """
module = Extension(
    '{name}',
    sources=[
         {source}
    ],
    language='{language}',
    include_dirs = {include_dirs},
#    libraries = ['tcl83'],
#    library_dirs = ['/usr/local/lib'],      
#    extra_compile_args = [ '-O0', '-g' ],
#    extra_link_args =
)

setup(
    name='{name}',
    ext_modules = [module],""", fmt)
        setup = library.setup
        for key in [
                "author",
                "author_email",
                "description",
#                "long_description",
                "license",
                "url",
                "test_suite",
        ]:
            if key in setup:
                output.append("    {} = '{}',".format(key, setup[key]))
        output.append(")")
        self.comment = '#'
        self.write_output_file(fname, self.config.out_dir, output)

    def not_implemented_error(self, node, msg, ret):
        """A standard splicer for unimplemented code
        ret is the return value (NULL or -1 or '')

        Args:
            node - ast.ClassNode
            msg -
            ret -
        """
        lines = ['PyErr_SetString(PyExc_NotImplementedError, "%s");' % msg]
        if ret:
            lines.append("return %s;" % ret)
        else:
            lines.append("return;")
        return lines

    def not_implemented(self, node, msg, ret):
        """A standard splicer for rich comparison

        Args:
            node - ast.ClassNode
            msg -
            ret -
        """
        return [
            "Py_INCREF(Py_NotImplemented);",
            "return Py_NotImplemented;"
        ]

    def tp_del(self, node, msg, ret):
        """default method for tp_del.

        Args:
            node - ast.ClassNode
            msg  - 'del'
            ret  - ''
        """
        output = [
            "{PY_release_memory_function}(self->{PY_type_dtor}, self->{PY_type_obj});",
            "self->{PY_type_obj} = {nullptr};",
        ]
        self.process_member_obj(
            node, "Py_XDECREF(self->{PY_member_object});", output)
        self.process_member_obj(
            node, "Py_XDECREF(self->{PY_member_data});", output)
        return output

    def init_member_obj(self, node):
        """Update fmt for members of struct-as-class.
        """
        for var in node.variables:
            fmt = var.fmtdict
            if var.ast.is_array():
                fmt.py_var = "SHPy_" + fmt.variable_name
                var.eval_template("PY_member_object")
                var.eval_template("PY_member_data")

    def process_member_obj(self, node, text, output):
        """Loop over variables in the struct-as-class and add
        a line of formatted text for each pointer variable.
        """
        if not node.as_struct:
            return
        print_header = True
        for var in node.variables:
            # var is VariableNode
            if not var.ast.is_array():
                continue
            if print_header:
                output.append("// Python objects for members.")
                print_header = False
            append_format(output, text, var.fmtdict)

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
#define PyInt_FromSize_t PyLong_FromSize_t
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
# [2] will be converted to fmt.nullptr if 'NULL'.
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
PyVarObject_HEAD_INIT({nullptr}, 0)
"{PY_module_scope}.{cxx_class}",                       /* tp_name */
sizeof({PY_PyObject}),         /* tp_basicsize */
0,                              /* tp_itemsize */
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
0,                              /* tp_weaklistoffset */
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
0,                              /* tp_dictoffset */
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
0,                              /* tp_version_tag */
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
    {nullptr}, /* m_reload */
    {library_lower}_traverse, /* m_traverse */
    {library_lower}_clear, /* m_clear */
    NULL  /* m_free */
}};

#define RETVAL m
#define INITERROR return {nullptr}
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
PyObject *m = {nullptr};
const char * error_name = "{library_lower}.Error";
"""

module_middle = """

/* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
m = PyModule_Create(&moduledef);
#else
m = Py_InitModule4("{PY_module_name}", {PY_prefix}methods,\t
+{PY_prefix}_doc__,
(PyObject*){nullptr},PYTHON_API_VERSION);
#endif
-if (m == {nullptr})
+return RETVAL;-
struct module_state *st = GETSTATE(m);"""

# XXX - +INITERROR;-
module_middle2 = """
{PY_prefix}error_obj = PyErr_NewException((char *) error_name, {nullptr}, {nullptr});
if ({PY_prefix}error_obj == {nullptr})
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
static char {PY_prefix}_doc__[] =
"XXX submodule doc"  //"{PY_library_doc}"
;

struct module_state {{
    PyObject *error;
}};

static struct PyModuleDef moduledef = {{
    PyModuleDef_HEAD_INIT,
    "{PY_module_scope}", /* m_name */
    {PY_prefix}_doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    {PY_prefix}methods, /* m_methods */
    {nullptr}, /* m_reload */
//    {library_lower}_traverse, /* m_traverse */
//    {library_lower}_clear, /* m_clear */
    {nullptr}, /* m_traverse */
    {nullptr}, /* m_clear */
    {nullptr}  /* m_free */
}};
#endif
#define RETVAL {nullptr}

PyObject *{PY_prefix}init_{PY_module_init}(void)
{{+
PyObject *m;
#if PY_MAJOR_VERSION >= 3
m = PyModule_Create(&moduledef);
#else
m = Py_InitModule3((char *) "{PY_module_name}", {PY_prefix}methods, {nullptr});
#endif
if (m == {nullptr})
+return {nullptr};-
"""
submodule_end = """
return m;
-}}
"""

def update_fmt_from_typemap(fmt, ntypemap):
    """Copy fields from typemap to use with creating output"""
    # XXX maybe use in wrap_namespace
    if ntypemap.PY_to_object_idtor:
        fmt.PY_to_object_idtor_func = ntypemap.PY_to_object_idtor
        # XXX - not sure if needed, avoid clutter for now.
#        fmt.PY_to_object_func = ntypemap.PY_to_object
#        fmt.PY_from_object_func = ntypemap.PY_from_object

######################################################################

def py_struct_dimension(parent, var, fmt):
    """
    Process ast.array or the dimension attribute.

    Set format fields.
    npy_rank        = rank of NumPy array.  Scalars are 1.
    npy_intp_values = comma separated list of dimensions
    npy_intp_size   = size of array, multiplied ranks.

    ex. npy_intp {npy_dims_var}[{npy_rank}] = {{ {npy_intp_values} }};

    Args:
        parent - ast.ClassNode.
        var    - ast.VariableNode.
        fmt    - util.Scope.
    """
    fmt.npy_dims_var = "dims"  # Name of local variables
    # fmt.npy_intp_asgn     # assign to
    #    fmt.npy_intp_values     # comma separated list of values
    ast = var.ast
    if ast.array: # Fixed size array.
        metadim = ast.array
    elif ast.attrs["dimension"] is not None:
        metadim = ast.metaattrs["dimension"]
    else:
        metadim = None
    if metadim:
        visitor = ToDimension(parent, None, var.fmtdict,
                              var.fmtdict.PY_struct_context)
        visitor.visit(metadim)
        fmt.rank = str(visitor.rank)
        fmt.npy_rank = fmt.rank
        fmt.npy_intp_values = ", ".join(visitor.shape)
        if visitor.rank == 1:
            fmt.npy_intp_size = visitor.shape[0]
        else:
            fmt.npy_intp_size = "*".join(
                ["(" + dim + ")" for dim in visitor.shape])
    else:
        # Scalar
        fmt.rank = "0"
        fmt.npy_rank = "1"
        fmt.npy_intp_values = "1"     # comma separated list of values
        fmt.npy_intp_size   = "1"

######################################################################

class ToDimension(todict.PrintNode):
    """Visit dimension expression.
    Convert cls references to correct scope.  obj->{argname}

    If a name is a member of the struct, prefix with PY_struct_context.

    struct Cstruct_list {
        int nitems;
        int *ivalue     +dimension(nitems+nitems);  # case 1
        double *dvalue  +dimension(nitems*TWO);     # case 2
    }

    case 1:  {context}nitems+{context}nitems
    case 2:  {context}nitems*TWO

    """

    def __init__(self, cls, fcn, fmt, context):
        """
        Args:
            cls  - ast.ClassNode or None
            fcn   - ast.FunctionNode of calling function
                    or None for struct.
            fmt  - util.Scope
            context - how to access Identifiers in cls.
                      Different for function arguments and
                      class/struct members.
        """
        super(ToDimension, self).__init__()
        self.cls = cls
        self.fcn = fcn
        self.fmt = fmt
        self.context = context

        self.rank = 0
        self.shape = []

    def visit_list(self, node):
        # list of dimension expressions
        self.rank = len(node)
        for dim in node:
            sh = self.visit(dim)
            self.shape.append(sh)

    def visit_Identifier(self, node):
        argname = node.name
        if self.fcn and argname == "size" and node.args:
            # size(in)
            argname = node.args[0].name
            #            arg = self.func.ast.find_arg_by_name(argname)
            fmt = self.fcn._fmtargs[argname]["fmtpy"]
            if self.fcn.options.PY_array_arg == "numpy":
                return wformat("PyArray_SIZE({py_var})", fmt)
            else:
                return fmt.size_var
        # Look for members of class/struct.
        elif self.cls is not None and argname in self.cls.map_name_to_node:
            # This name is in the same class as the dimension.
            # Make name relative to the class.
            member = self.cls.map_name_to_node[argname]
            obj = self.fmt.PY_type_obj
            if member.may_have_args():
                if node.args is None:
                    print("{} must have arguments".format(argname))
                else:
                    return "{}{}({})".format(
                        self.context, argname, self.comma_list(node.args))
            else:
                if node.args is not None:
                    print("{} must not have arguments".format(argname))
                else:
                    return "{}{}".format(self.context, argname)
        elif node.args is None:
            return argname  # variable
        else:
            return self.param_list(node) # function
        return "--??--"

######################################################################

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
        argname = node.name
        # Look for functions
        if node.args is None:
            return argname
        ### functions
        elif argname == "size":
            # size(arg)
            argname = node.args[0].name
            #            arg = self.func.ast.find_arg_by_name(argname)
            fmt = self.func._fmtargs[argname]["fmtpy"]
            if self.func.options.PY_array_arg == "numpy":
                return wformat("PyArray_SIZE({py_var})", fmt)
            else:
                return fmt.size_var
        elif argname == "len":
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
        elif argname == "len_trim":
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

######################################################################

def update_code_blocks(symtab, stmts, fmt):
    """ Accumulate info from statements.
    Append to lists in symtab.

    Args:
        symtab - result of locals() of caller
        stmts  - PyStmts
        fmt    - format dictionary (Scope)
    """
    for clause in ["declare", "post_declare", "post_parse", "pre_call",
                   "post_call", "cleanup", "fail"]:
        lstout = symtab[clause + "_code"]
        for cmd in getattr(stmts, clause):
            lstout.append(wformat(cmd, fmt))

    # If capsule_order is defined, then add some additional code to 
    # do reference counting.
    if fmt.inlocal("capsule_order"):
        suffix = "_capsule"
    else:
        suffix = "_keep"
    for clause in ["declare", "post_call", "fail"]:
        name = clause + suffix
        if stmts.post_call_capsule:
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

def update_typemap_for_language(language):
    """Preprocess statements for lookup.

    Update statements for c or c++.
    Fill in py_tree.
    """
    typemap.update_for_language(py_statements, language)
    typemap.update_stmt_tree(py_statements, py_tree, default_stmts)
    global default_scope
    default_scope = typemap.default_scopes["py"]

def lookup_stmts(path):
    return typemap.lookup_stmts_tree(py_tree, path)

class PyStmts(object):
    def __init__(
        self,
        name="py_default",
        arg_declare=None,   # Empty list indicates no declaration.
        post_declare=[],
        fmtdict=None,
            
        allocate_local_var=False,
        arg_call=None,
        c_header=[], c_helper=[],
        cxx_header=[], cxx_local_var=None,
        need_numpy=False,
        object_created=False,
        parse_format=None, parse_args=[],
        declare=[], post_parse=[], pre_call=[],
        post_call=[],
        declare_capsule=[], post_call_capsule=[], fail_capsule=[],
        declare_keep=[], post_call_keep=[], fail_keep=[],
        cleanup=[], fail=[],
        goto_fail=False,
        getter=[], getter_helper=[],
        setter=[], setter_helper=[],
    ):
        self.name = name
        self.arg_declare = arg_declare
        self.post_declare = post_declare
        self.fmtdict = fmtdict

        self.allocate_local_var = allocate_local_var
        self.arg_call = arg_call
        self.c_header = c_header
        self.c_helper = c_helper
        self.cxx_header = cxx_header
        self.cxx_local_var = cxx_local_var
        self.need_numpy = need_numpy
        self.object_created = object_created
        self.parse_format = parse_format
        self.parse_args = parse_args

        self.declare = declare
        self.post_parse = post_parse
        self.pre_call = pre_call
        self.post_call = post_call

        self.declare_capsule = declare_capsule
        self.post_call_capsule = post_call_capsule
        self.fail_capsule = fail_capsule
        self.declare_keep = declare_keep
        self.post_call_keep = post_call_keep
        self.fail_keep = fail_keep

        self.cleanup = cleanup
        self.fail = fail
        self.goto_fail = goto_fail
        # descr
        self.getter = getter
        self.getter_helper = getter_helper
        self.setter = setter
        self.setter_helper = setter_helper

    def to_dict(self):
        d = {}
        for key in [
                "name",
                "allocate_local_var",
                "arg_call",
                "arg_declare",
                "c_header",
                "c_helper",
                "cxx_header",
                "cxx_local_var",
                "fmtdict",
                "need_numpy",
                "object_created",
                "parse_format",
                "parse_args",
                "declare",
                "post_declare",
                "post_parse",
                "pre_call",
                "post_call",
                "declare_capsule",
                "post_call_capsule",
                "fail_capsule",
                "declare_keep",
                "post_call_keep",
                "fail_keep",
                "cleanup",
                "fail",
                "goto_fail",
                "getter",
                "getter_helper",
                "setter",
                "setter_helper",
                ]:
            value = getattr(self, key)
            if value:
                d[key] = value
        return d
                          
default_stmts = dict(
    py=PyStmts,
    base=PyStmts,
)

# put into list to avoid duplicating text below
array_error = [
    "if ({py_var} == {nullptr}) {{+",
    "PyErr_SetString(PyExc_ValueError,"
    '\t "{c_var} must be a 1-D array of {c_type}");',
    "goto fail;",
    "-}}",
]
# Use cxx_T instead of c_type for vector.
template_array_error = [
    "if ({py_var} == {nullptr}) {{+",
    "PyErr_SetString(PyExc_ValueError,"
    '\t "{c_var} must be a 1-D array of {cxx_T}");',
    "goto fail;",
    "-}}",
]

malloc_error = [
    "if ({cxx_var} == {nullptr}) {{+",
    "PyErr_NoMemory();",
    "goto fail;",
    "-}}",
]

declare_capsule=[
    "PyObject *{py_capsule} = {nullptr};",
]
post_call_capsule=[
    "{py_capsule} = "
    'PyCapsule_New({cxx_var}, "{PY_numpy_array_capsule_name}", '
    "\t{PY_capsule_destructor_function});",
    "if ({py_capsule} == {nullptr}) goto fail;",
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
py_statements = [

########################################
# void
    dict(
        # Accept a capsule and extract address.
        name="py_void_*_in",
        declare=[
            "PyObject *{py_var};",
        ],
        parse_format="O",
        parse_args=["&{py_var}"],
        post_parse=[
            "{c_var} = PyCapsule_GetPointer({py_var}, NULL);",
            "if (PyErr_Occurred())",
            "+goto fail;-",
        ],
        arg_call=[
            "{c_var}",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_void_**_out",
        arg_declare=[
            "void *{c_var};",
        ],
        arg_call=[
            "&{c_var}",
        ],
        fmtdict=dict(
            ctor_expr="{cxx_var}",
        ),
    ),
    dict(
        name="py_void_*&_out",
        base="py_void_**_out",
        arg_call=[
            "{c_var}",
        ]
    ),
    dict(
        name="py_void_*_result",
        fmtdict=dict(
            ctor_expr="{cxx_var}",
        ),
    ),

########################################
# bool
    dict(
        name="py_bool_in",
        pre_call=["{cxx_var} = PyObject_IsTrue({py_var});"]
    ),
    dict(
        name="py_bool_inout",
        arg_declare=[
            "bool {cxx_var};",
        ],
        pre_call=["{cxx_var} = PyObject_IsTrue({py_var});"],
        # py_var is already declared for inout
        post_call=[
            "{py_var} = PyBool_FromLong({c_var});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_bool_out",
        arg_declare=[
            "bool {cxx_var};",
        ],
        declare=[
            "{PyObject} * {py_var} = {nullptr};",
        ],
        post_call=[
            "{py_var} = PyBool_FromLong({c_var});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_bool_result",
        declare=[
            "{PyObject} * {py_var} = {nullptr};",
        ],
        post_call=[
            "{py_var} = PyBool_FromLong({c_var});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_bool_*_out",
        base="py_bool_out",
        arg_call=["&{cxx_var}"],
    ),
    dict(
        name="py_bool_*_inout",
        base="py_bool_inout",
        arg_call=["&{cxx_var}"],
    ),
    
####################
    dict(
        name="py_native_*_in",
        arg_declare=["{c_type} {c_var};"],
        arg_call=["&{c_var}"],
    ),
    dict(
        name="py_native_*_inout",
        arg_declare=["{c_type} {c_var};"],
        arg_call=["&{c_var}"],
        fmtdict=dict(
            ctor_expr="{c_var}",
        ),
    ),
    dict(
        name="py_native_*_out",
        arg_declare=["{c_type} {c_var};"],
        arg_call=["&{c_var}"],
        fmtdict=dict(
            ctor_expr="{c_var}",
        ),
    ),
    dict(
        name="py_native_&_in",
        arg_declare=["{c_type} {c_var};"],
    ),
    dict(
        name="py_native_&_inout",
        arg_declare=["{c_type} {c_var};"],
    ),
    dict(
        name="py_native_&_out",
        arg_declare=["{c_const}{c_type} {c_var};"],
    ),

####################
## numpy
    dict(
        name="py_native_*_in_pointer_numpy",
        need_numpy=True,
        declare=[
            "PyObject * {pytmp_var};",
            "PyArrayObject * {py_var} = {nullptr};",
        ],
        parse_format="O",
        parse_args=["&{pytmp_var}"],
        post_parse=[
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_IN_ARRAY){cast2};",
        ] + array_error,
        c_pre_call=[
            "{c_var} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_var} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        arg_call=["{c_var}"],
        cleanup=[
            "{PY_cleanup_decref}({py_var});",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),

    dict(
        name="py_native_*_inout_pointer_numpy",
        need_numpy=True,
        parse_format="O",
        parse_args=["&{pytmp_var}"],
        declare=[
            "PyObject * {pytmp_var};",
            "PyArrayObject * {py_var} = {nullptr};",
        ],
        post_parse=[
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_INOUT_ARRAY){cast2};",
        ] + array_error,
        c_pre_call=[
            "{c_var} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_var} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        arg_call=["{c_var}"],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),

    dict(
        name="py_native_*_out_pointer_numpy",
        need_numpy=True,
        declare=[
            "{npy_intp_decl}"  # Must contain a newline if non-blank.
            "PyArrayObject * {py_var} = {nullptr};",
        ],
        post_parse=[
            "{npy_intp_asgn}"  # Must contain a newline if non-blank.
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}PyArray_SimpleNew("
            "{npy_rank}, {npy_dims_var}, {numpy_type}){cast2};",
        ] + array_error,
        c_pre_call=[
            "{c_var} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_var} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        arg_call=["{c_var}"],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),

    dict(
        name="py_native_*_result_pointer_list",
        c_helper="to_PyList_{cxx_type}",
        declare=[
            "PyObject *{py_var} = {nullptr};",
        ],
        post_call=[
            "{py_var} = {hnamefunc0}\t({cxx_var},\t {array_size});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        # XXX - library owns memory, test on +owner attribute
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_native_*_result_pointer_numpy",
        need_numpy=True,
        declare=[
            "{npy_intp_decl}"
            "PyObject * {py_var} = {nullptr};",
        ],
        post_call=[
            "{npy_intp_asgn}"
            "{py_var} = "
            "PyArray_SimpleNewFromData({npy_rank},\t {npy_dims_var},"
            "\t {numpy_type},\t {cxx_nonconst_ptr});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
        declare_capsule=declare_capsule,
        post_call_capsule=post_call_capsule,
        fail_capsule=fail_capsule,
    ),
    dict(
        name="py_native_&_result_pointer_numpy",
        base="py_native_*_result_pointer_numpy",
    ),
    dict(
        name="py_native_*_result_allocatable_numpy",
        base="py_native_*_result_pointer_numpy",
    ),

    dict(
        name="py_native_**_out_pointer_numpy",
        base="py_native_*_result_pointer_numpy",
        # Declare a local variable for the argument.
        arg_declare=[
            "{c_const}{c_type} *{c_var};",
        ],
        declare=[
            "{npy_intp_decl}"
            "PyObject *{py_var} = {nullptr};"
        ],
        arg_call=["&{cxx_var}"],
    ),
    dict(
        name="py_native_*&_out_pointer_numpy",
        base="py_native_**_out_pointer_numpy",
        arg_call=["{cxx_var}"],
    ),

########################################
## list
    dict(
        name="py_native_*_in_pointer_list",
        c_helper="get_from_object_{cxx_type}_list",
        parse_format="O",
        parse_args=["&{pytmp_var}"],
        arg_declare=[ # initialize
            "{cxx_type} * {cxx_var} = {nullptr};",
        ],
        declare=[
            "PyObject *{pytmp_var} = {nullptr};",
            "{PY_typedef_converter} {value_var} = {PY_value_init};",
            "{value_var}.name = \"{c_var}\";",
            "Py_ssize_t {size_var};",
        ],
        post_parse=[
            "if ({hnamefunc0}\t({pytmp_var}, &{value_var}) == 0)",
            "+goto fail;-",
            "{cxx_var} = {cast_static}{cxx_type} *{cast1}{value_var}.data{cast2};",
            "{size_var} = {value_var}.size;",
        ],
        arg_call=["{cxx_var}"],
        cleanup=[
            "Py_XDECREF({value_var}.dataobj);",
        ],
        fail=[
            "Py_XDECREF({value_var}.dataobj);",
        ],
        goto_fail=True,
    ),

    dict(
        name="py_native_*_inout_pointer_list",
#        c_helper="update_PyList_{cxx_type}",
        c_helper="get_from_object_{cxx_type}_list to_PyList_{cxx_type}",
        parse_format="O",
        parse_args=["&{pytmp_var}"],
        arg_declare=[
            "{cxx_type} * {cxx_var} = {nullptr};",
        ],
        declare=[
            "PyObject *{py_var};",
            "PyObject *{pytmp_var} = {nullptr};",
            "{PY_typedef_converter} {value_var} = {PY_value_init};",
            "{value_var}.name = \"{c_var}\";",
            "Py_ssize_t {size_var};",
        ],
        post_parse=[
            "if ({hnamefunc0}\t({pytmp_var}, &{value_var}) == 0)",
            "+goto fail;-",
            "{cxx_var} = {cast_static}{cxx_type} *{cast1}{value_var}.data{cast2};",
            "{size_var} = {value_var}.size;",
        ],
        arg_call=["{cxx_var}"],
        post_call=[
#            "SHROUD_update_PyList_{cxx_type}({pytmp_var}, {cxx_var}, {size_var});",
            "{py_var} = {hnamefunc1}\t({cxx_var},\t {size_var});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        cleanup=[
            "Py_XDECREF({value_var}.dataobj);",
        ],
        fail=[
            "Py_XDECREF({value_var}.dataobj);",
        ],
        goto_fail=True,
    ),

    dict(
        name="py_native_*_out_pointer_list",
        c_helper="to_PyList_{cxx_type}",
        c_header=["<stdlib.h>"],  # malloc/free
        cxx_header=["<cstdlib>"],  # malloc/free
        arg_declare=[
            "{cxx_type} * {cxx_var} = {nullptr};",
        ],
        declare=[
            "PyObject *{py_var} = {nullptr};",
        ],
        c_pre_call=[
            "{c_var} = malloc(\tsizeof({c_type}) * ({array_size}));",
        ] + malloc_error,
        cxx_pre_call=[
            "{cxx_var} = static_cast<{cxx_type} *>\t(std::malloc(\tsizeof({cxx_type}) * ({array_size})));",
        ] + malloc_error,
        arg_call=["{c_var}"],
        post_call=[
            "{py_var} = {hnamefunc0}\t({cxx_var},\t {array_size});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        cleanup=[
            "{stdlib}free({cxx_var});",
            "{cxx_var} = {nullptr};",
        ],
        fail=[
            "Py_XDECREF({py_var});",
            "if ({cxx_var} != {nullptr})\t {stdlib}free({cxx_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_native_**_out_pointer_list",
        base="py_native_*_result_pointer_list",
        # Declare a local variable for the argument.
        arg_declare=[
            "{c_const}{c_type} *{c_var};",
        ],
        declare=[
            "PyObject *{py_var} = {nullptr};"
        ],
        arg_call=["&{cxx_var}"],
    ),
    
########################################
## allocatable
    dict(
        name="py_native_*_out_allocatable_list",
        base="py_native_*_out_pointer_list",
    ),
    dict(
        name="py_native_*_out_allocatable_numpy",
        base="py_native_*_out_pointer_numpy",
    ),

########################################
## raw
    dict(
        # Declare a local pointer, pass address to library, convert to capsule.
        name="py_native_**_out_raw",
        arg_declare=[
            "{c_type} *{c_var};",
        ],
        declare=[
            "PyObject *{py_var} = {nullptr};"
        ],
        arg_call=["&{cxx_var}"],
        post_call=[
            "{py_var} = PyCapsule_New({cxx_var}, NULL, NULL);",
        ],
        object_created=True,
    ),
########################################
# char
    dict(
        # Get a string from argument but only pass first character to C++.
        # Since char is a scalar, need to actually get a char * - parse_format, parse_args.
        # XXX - make sure c_var is only 1 long?
        name="py_char_scalar_in",
        arg_declare=[
            "char *{c_var};",
        ],
        parse_format="s",
        parse_args=["&{c_var}"],
        arg_call=["{c_var}[0]"],
    ),
    dict(
        name="py_char_scalar_result",
        declare=[
            "{PyObject} * {py_var} = {nullptr};",
        ],
        post_call=[
            "{py_var} = PyString_FromStringAndSize(&{cxx_var}, 1);",
        ],
        object_created=True,
    ),
    dict(
        name="py_char_*_result",
        fmtdict=dict(
            ctor_expr="{c_var}",
        ),
    ),
    dict(
        name="py_char_*_in",
        arg_call=["{c_var}"],
    ),
    dict(
        name="py_char_*_out_charlen",
        arg_declare=[
            "{c_const}char {c_var}[{charlen}];  // intent(out)",
        ],
        arg_call=["{c_var}"],
        fmtdict=dict(
            ctor_expr="{c_var}",
        ),
    ),
    dict(
        name="py_char_*_inout",
        arg_call=["{c_var}"],
        fmtdict=dict(
            ctor_expr="{c_var}",
        ),
    ),

    dict(
        name="py_char_**_in",
        c_helper="get_from_object_charptr",
        parse_format="O",
        parse_args=["&{pytmp_var}"],
        arg_declare=[
            "{c_const}char ** {cxx_var} = {nullptr};",
        ],
        declare=[
            "PyObject * {pytmp_var};", # set by PyArg_Parse
            "{PY_typedef_converter} {value_var} = {PY_value_init};",
            "{value_var}.name = \"{c_var}\";",
            "Py_ssize_t {size_var};",
        ],
        pre_call=[
            "if ({hnamefunc0}\t({pytmp_var}, &{value_var}) == 0)",
            "+goto fail;-",
            "{cxx_var} = {cast_static}char **{cast1}{value_var}.data{cast2};",
        ],
        arg_call=["{cxx_var}"],
        post_call=[
            "Py_XDECREF({value_var}.dataobj);",
        ],
        fail=[
            "Py_XDECREF({value_var}.dataobj);",
        ],
        goto_fail=True,
    ),
    
########################################
# string
# ctor_expr is arguments to PyString_FromStringAndSize.
    dict(
        name="py_string_scalar_in",
        cxx_local_var="scalar",
        post_declare=["{c_const}std::string {cxx_var}({c_var});"],
        fmtdict=dict(
            ctor_expr="{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size()",
        ),
    ),
    dict(
        name="py_string_&_in",
        base="py_string_scalar_in",
    ),
    dict(
        name="py_string_scalar_inout",
        cxx_local_var="scalar",
        post_declare=["{c_const}std::string {cxx_var}({c_var});"],
        fmtdict=dict(
            ctor_expr="{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size()",
        ),
    ),
    dict(
        name="py_string_&_inout",
        base="py_string_scalar_inout",
    ),
    dict(
        name="py_string_scalar_out",
        arg_declare=[],
        cxx_local_var="scalar",
        post_declare=["{c_const}std::string {cxx_var};"],
        fmtdict=dict(
            ctor_expr="{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size()",
        ),
    ),
    dict(
        name="py_string_&_out",
        base="py_string_scalar_out",
    ),
    dict(
        name="py_string_scalar_result",
        fmtdict=dict(
            ctor_expr="{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size()",
        ),
    ),
    dict(
        name="py_string_*_result",
        fmtdict=dict(
            ctor_expr="{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size()",
        ),
    ),
    dict(
        name="py_string_&_result",
        base="py_string_*_result",
    ),
    dict(
        name="py_string_*_in",
        base="py_string_scalar_in",
        arg_call=["&{cxx_var}"],
    ),
    dict(
        name="py_string_*_inout",
        base="py_string_scalar_inout",
        arg_call=["&{cxx_var}"],
    ),
    dict(
        name="py_string_*_out",
        base="py_string_scalar_out",
        arg_call=["&{cxx_var}"],
    ),

########################################
# struct
# "struct", intent, PY_struct_arg
# numpy
# Note that Typemap.c_type is a C wrapper over a C++ struct
# created in wrapc.py. Do not use here.
    
# and does not apply in Python.
    dict(
        name="py_struct_in_list",
        arg_declare=[],
    ),
    dict(
        name="py_struct_inout_list",
        arg_declare=[],
    ),
    dict(
        name="py_struct_out_list",
        arg_declare=[],
        post_declare=[
            "{cxx_type} {cxx_var};",
        ],
    ),

    dict(
        name="py_struct_scalar_in_list",
        base="py_struct_in_list",
    ),

    # struct-list-cxx   (XXX - is not compiled)
    dict(
        name="py_struct_*_in_list",
        base="py_struct_in_list",
        arg_call=["&{cxx_var}"],
    ),
    dict(
        name="py_struct_*_inout_list",
        base="py_struct_inout_list",
        arg_call=["&{cxx_var}"],
    ),
    dict(
        name="py_struct_*_out_list",
        base="py_struct_out_list",
        arg_call=["&{cxx_var}"],
    ),

    dict(
        name="py_struct_*_in_numpy",
        need_numpy=True,
        parse_format="O",
        parse_args=["&{pytmp_var}"],
        cxx_local_var="pointer",
        arg_declare=[ # Must be a pointer of cxx_type.
            "{cxx_type} *{cxx_var};",
        ],
        declare=[
            "PyObject * {pytmp_var} = {nullptr};",
            "PyArrayObject * {py_var} = {nullptr};",
#            "PyArray_Descr * {pydescr_var} = {PYN_descr};",
        ],
        post_parse=[
            # PyArray_FromAny steals a reference from PYN_descr
            # and will decref it if an error occurs.
            "Py_INCREF({PYN_descr});",
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}"
            "PyArray_FromAny(\t{pytmp_var},\t {PYN_descr},"
            "\t 0,\t 1,\t NPY_ARRAY_IN_ARRAY,\t {nullptr}){cast2};",
        ] + array_error,
        c_pre_call=[
            "{c_var} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_var} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        cleanup=[
            "{PY_cleanup_decref}({py_var});",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_struct_*_inout_numpy",
        need_numpy=True,
        parse_format="O",
        parse_args=["&{pytmp_var}"],
        arg_declare=[ # Must be a pointer.
            "{cxx_type} *{cxx_var};",
        ],
        declare=[
            "PyObject * {pytmp_var} = {nullptr};",
            "PyArrayObject * {py_var} = {nullptr};",
#            "PyArray_Descr * {pydescr_var} = {PYN_descr};",
        ],
        post_parse=[
            # PyArray_FromAny steals a reference from PYN_descr
            # and will decref it if an error occurs.
            "Py_INCREF({PYN_descr});",
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}"
            "PyArray_FromAny(\t{pytmp_var},\t {PYN_descr},"
            "\t 0,\t 1,\t NPY_ARRAY_IN_ARRAY,\t {nullptr}){cast2};",
        ] + array_error,
        c_pre_call=[
            "{c_var} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_var} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_struct_*_out_numpy",
        # XXX - expand to array of struct
        need_numpy=True,
#        allocate_local_var=True,  # needed to release memory
        arg_declare=[
            "{cxx_type} *{cxx_var};", 
        ],
        declare=[
#            "{npy_intp_decl}"
            "PyArrayObject * {py_var} = {nullptr};",
        ],
        post_parse=[
#            "{npy_intp_asgn}"
            "Py_INCREF({PYN_descr});",
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}"
            "PyArray_NewFromDescr(\t&PyArray_Type,\t {PYN_descr},"
            "\t 0,\t {nullptr},\t {nullptr},\t {nullptr},\t 0,\t {nullptr}){cast2};",
        ] + array_error,
        c_pre_call=[
            "{c_var} = PyArray_DATA({py_var});",
        ],
        cxx_pre_call=[
            "{cxx_var} = static_cast<{cxx_type} *>\t(PyArray_DATA({py_var}));",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_struct_result_numpy",
        # XXX - expand to array of struct
        need_numpy=True,
        allocate_local_var=True,
        declare=[
            "{npy_intp_decl}"
            "PyObject * {py_var} = {nullptr};",
        ],
        post_call=[
            "{npy_intp_asgn}"
            "Py_INCREF({PYN_descr});",
            "{py_var} = "
            "PyArray_NewFromDescr(&PyArray_Type, \t{PYN_descr},\t"
            " {npy_rank}, {npy_dims_var}, \t{nullptr}, {cxx_var}, 0, {nullptr});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
        declare_capsule=declare_capsule,
        post_call_capsule=post_call_capsule,
        fail_capsule=fail_capsule,
    ),

    dict(
        name="py_struct_&_in_numpy",
        base="py_struct_*_in_numpy",
        arg_call=["*{cxx_var}"],
    ),
    dict(
        name="py_struct_&_inout_numpy",
        base="py_struct_*_inout_numpy",
        arg_call=["*{cxx_var}"],
    ),
    dict(
        name="py_struct_&_out_numpy",
        base="py_struct_*_out_numpy",
        arg_call=["*{cxx_var}"],
    ),
    dict(
        name="py_struct_scalar_in_numpy",
        base="py_struct_*_in_numpy",
        arg_call=["*{cxx_var}"],
    ),
# cannot support inout/out with call-by-value
#        name="py_struct_*_inout_numpy",
#        name="py_struct_*_out_numpy",

##########
# Since cxx_var is always a pointer, use that case as the base for
# pass by value.
    dict(
        name="py_struct_*_in_class",
        arg_declare=[], # No C variable, the pointer is extracted from PyObject.
        cxx_local_var="pointer",
        post_declare=[
            "{c_const}{cxx_type} * {cxx_var} ="
            "\t {py_var} ? {py_var}->{PY_type_obj} : {nullptr};",
        ],
    ),
    dict(
        name="py_struct_*_inout_class",
        arg_declare=[], # No C variable, the pointer is extracted from PyObject.
        cxx_local_var="pointer",
        post_declare=[
            "{c_const}{cxx_type} * {cxx_var} ="
            "\t {py_var} ? {py_var}->{PY_type_obj} : {nullptr};",
        ],
        object_created=True,
    ),
    dict(
        name="py_struct_*_out_class",
#        allocate_local_var=True,  # needed to release memory
        cxx_local_var="pointer",
        arg_declare=[
            "{cxx_type} *{cxx_var} = {nullptr};", 
        ],
        declare=[
            "PyObject *{py_var} = {nullptr};",
        ],
        c_pre_call=[
            "{c_var} = malloc(sizeof({c_type}));",
        ],
        c_dealloc_capsule=[
            "free(ptr);",
        ],
        cxx_pre_call=[
            "{cxx_var} = new {cxx_type};",
        ],
        cxx_dealloc_capsule=[
            "delete cxx_ptr;",
        ],
        post_call=[
            "{py_var} = {PY_to_object_idtor_func}({cxx_addr}{cxx_var},\t {capsule_order});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_struct_result_class",
        cxx_local_var="pointer",
        allocate_local_var=True,
        declare=[
            "PyObject *{py_var} = {nullptr};  // struct_result_class",
        ],
        post_call=[
            "{py_var} = {PY_to_object_idtor_func}({cxx_addr}{cxx_var},\t {capsule_order});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),


    dict(
        name="py_struct_scalar_in_class",
        base="py_struct_*_in_class",
        arg_call=["*{cxx_var}"],
    ),
# cannot support inout/out with call-by-value
#        name="py_struct_scalar_inout_class",
#        name="py_struct_scalar_out_class",
    dict(
        name="py_struct_&_in_class",
        base="py_struct_*_in_class",
        arg_call=["*{cxx_var}"],
    ),
    dict(
        name="py_struct_&_inout_class",
        base="py_struct_*_inout_class",
        arg_call=["*{cxx_var}"],
    ),
    dict(
        # XXX - this memory will leak
        name="py_struct_&_out_class",
        base="py_struct_*_out_class",
        arg_call=["*{cxx_var}"],
        post_call=[
            "{py_var} = {PY_to_object_idtor_func}({cxx_var},\t {capsule_order});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
    ),

########################################
# shadow a.k.a class
    dict(
        name="py_shadow_*_in",
        arg_declare=[], # No C variable, the pointer is extracted from PyObject.
        cxx_local_var="pointer",
        post_declare=[
            "{c_const}{cxx_type} * {cxx_var} ="
            "\t {py_var} ? {py_var}->{PY_type_obj} : {nullptr};"
        ],
    ),
    dict(
        name="py_shadow_*_inout",
        arg_declare=[], # No C variable, the pointer is extracted from PyObject.
        cxx_local_var="pointer",
        post_declare=[
            "{c_const}{cxx_type} * {cxx_var} ="
            "\t {py_var} ? {py_var}->{PY_type_obj} : {nullptr};"
        ],
    ),
    dict(
        name="py_shadow_*_out",
        declare=[
            "{PyObject} *{py_var} = {nullptr};"
        ],
        post_call=[
            "{py_var} ="
            "\t PyObject_New({PyObject}, &{PyTypeObject});",
            "if ({py_var} == {nullptr}) goto fail;",
            "{py_var}->{PY_type_obj} = {cxx_addr}{cxx_var};",
        ],
        object_created=True,
#            post_call_capsule=[
#                "{py_var}->{PY_type_dtor} = {PY_numpy_array_dtor_context} + {capsule_order};",
#            ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_shadow_*_result",
#            declare=[
#                "{PyObject} *{py_var} = {nullptr};"
#            ],
        post_call=[
            "{PyObject} * {py_var} ="
            "\t PyObject_New({PyObject}, &{PyTypeObject});",
#                "if ({py_var} == {nullptr}) goto fail;",
            "{py_var}->{PY_type_obj} = {cxx_addr}{cxx_var};",
        ],
        object_created=True,
#            post_call_capsule=[
#                "{py_var}->{PY_type_dtor} = {PY_numpy_array_dtor_context} + {capsule_order};",
#            ],
#            fail=[
#                "Py_XDECREF({py_var});",
#            ],
#            goto_fail=True,
    ),
    dict(
        name="py_shadow_&_result",
        base="py_shadow_*_result",
    ),
    dict(
        name="py_shadow_scalar_in",
        base="py_shadow_*_in",
        arg_call=["*{cxx_var}"],
    ),
    dict(
        name="py_shadow_&_in",
        base="py_shadow_*_in",
        arg_call=["*{cxx_var}"],
    ),
    
########################################
# std::vector  only used with C++
# list
    dict(
        name="py_vector_in_list",
        # Convert input list argument into a C++ std::vector.
        # Pass to C++ function.
        # cxx_var is released by the compiler.
        c_helper="create_from_PyObject_vector_{cxx_T}",
        parse_format="O",
        parse_args=["&{pytmp_var}"],
        arg_declare=[],
        declare=[
            "PyObject * {pytmp_var};",  # Object set by ParseTupleAndKeywords.
        ],
        cxx_local_var="scalar",
        post_declare=[
            "std::vector<{cxx_T}> {cxx_var};",
        ],
        pre_call=[
            "if ({hnamefunc0}\t({pytmp_var}"
            ",\t \"{c_var}\",\t {cxx_var}) == -1)",
            "+goto fail;-",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_vector_out_list",
        # Create a pointer a std::vector and pass to C++ function.
        # Create a Python list with the std::vector.
        # cxx_var is released by the compiler.
        c_helper="to_PyList_vector_{cxx_T}",
        arg_declare=[],
        declare=[
            "PyObject * {py_var} = {nullptr};",
        ],
        cxx_local_var="scalar",
        post_declare=[
            "std::vector<{cxx_T}> {cxx_var};",
        ],
        post_call=[
            "{py_var} = {hnamefunc0}\t({cxx_var});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    # XXX - must release after copying result.
    dict(
        name="py_vector_result_list",
        declare=[
            "PyObject * {py_var} = {nullptr};",
        ],
        post_call=[
            "{py_var} = SHROUD_to_PyList_vector_{cxx_T}\t({cxx_var});",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),

##########
# numpy
# cxx_var will always be a pointer since we must save it in a capsule.
# vectors have the dimension attribute added by generate.py
    dict(
        name="py_vector_in_numpy",
        # Convert input argument into a NumPy array to make sure it is contiguous,
        # create a local std::vector which will copy the values.
        # Pass to C++ function.
        need_numpy=True,
        parse_format="O",
        parse_args=["&{pytmp_var}"],
        cxx_local_var="scalar",
        arg_declare=[],
        declare=[
            "PyObject * {pytmp_var};",  # Object set by ParseTupleAndKeywords.
            "PyArrayObject * {py_var} = {nullptr};",
        ],
        post_declare=[
            "std::vector<{cxx_T}> {cxx_var};",
            "{cxx_T} * {data_var};",
        ],
        post_parse=[
            "{py_var} = {cast_reinterpret}PyArrayObject *{cast1}PyArray_FROM_OTF("
            "\t{pytmp_var},\t {numpy_type},\t NPY_ARRAY_IN_ARRAY){cast2};",
        ] + template_array_error,
        pre_call=[
            "{data_var} = static_cast<{cxx_T} *>(PyArray_DATA({py_var}));",
            "{cxx_var}.assign(\t{data_var},\t {data_var}+PyArray_SIZE({py_var}));",
        ],
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
    ),
    dict(
        name="py_vector_out_numpy",
        # Create a pointer a std::vector and pass to C++ function.
        # Create a NumPy array with the std::vector as the capsule object.
        need_numpy=True,
        cxx_local_var="pointer",
        allocate_local_var=True,
        arg_declare=[],
        declare=[
            "{npy_intp_decl}"
            "PyObject * {py_var} = {nullptr};",
        ],
        arg_call=["*{cxx_var}"],
        post_call=[
#            "{npy_intp_asgn}"
            "{npy_dims_var}[0] = {cxx_var}->size();",
            "{py_var} = "
            "PyArray_SimpleNewFromData({npy_rank},\t {npy_dims_var},"
            "\t {numpy_type},\t {cxx_var}->data());",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
        declare_capsule=declare_capsule,
        post_call_capsule=post_call_capsule,
        fail_capsule=fail_capsule,
    ),
    dict(
        name="py_vector_result_numpy",
        need_numpy=True,
        allocate_local_var=True,
        declare=[
            "{npy_intp_decl}"
            "PyObject * {py_var} = {nullptr};",
        ],
        post_call=[
#            "{npy_intp_asgn}"
            "{npy_dims_var}[0] = {cxx_var}->size();",
            "{py_var} = "
            "PyArray_SimpleNewFromData({npy_rank},\t {npy_dims_var},"
            "\t {numpy_type},\t {cxx_var}->data());",
            "if ({py_var} == {nullptr}) goto fail;",
        ],
        object_created=True,
        fail=[
            "Py_XDECREF({py_var});",
        ],
        goto_fail=True,
        declare_capsule=declare_capsule,
        post_call_capsule=post_call_capsule,
        fail_capsule=fail_capsule,
    ),


    ########################################
    # ctor
    dict(
        name="base_py_ctor_array",
        arg_declare=[],  # No local variable, filled into struct directly.
        declare=[
            "{PY_typedef_converter} {value_var} = {PY_value_init};",
            "{value_var}.name = \"{field_name}\";",
            ],
        parse_format="O&",
        parse_args=["{hnamefunc0}", "&{value_var}"],
        post_call=[
            "SH_obj->{field_name} = "
            "{cast_static}{c_type} *{cast1}{value_var}.data{cast2};",
            "self->{PY_member_object} = {value_var}.obj;"
            "  // steal reference",
        ],
    ),
    dict(
        # Fill an array struct member.
        # helper is set by groups which use this as base.
        name="base_py_ctor_array_fill",
        arg_declare=[],  # No local variable, filled into struct directly.
        declare=[
            # Initialize to NULL since it is optional.
            "PyObject *{py_var} = {nullptr};",
        ],
        parse_format="O",
        parse_args=["&{py_var}"],
        post_call=[
            "if ({py_var} != {nullptr}) {{+",
            "if ({hnamefunc0}(\t{py_var},\t \"{c_var}\","
            "\t SH_obj->{field_name},\t {field_size}) == -1)",
            "+goto fail;-",
            "self->{PY_member_object} = {nullptr};",
            "-}}",
        ],
        goto_fail=True,
    ),
    
    dict(
        name="py_ctor_native",
        arg_declare=[],  # No local variable, assign to struct in post_call.
        declare=[
            "{c_type} {c_var} = 0;",
        ],
        post_call=[
            "SH_obj->{field_name} = {field_name};",
        ],
    ),
    dict(
        name="py_ctor_native_[]",
        base="base_py_ctor_array_fill",
        c_helper="fill_from_PyObject_{c_type}_{PY_array_arg}",
    ),
    dict(
        name="py_ctor_native_*",
        base="base_py_ctor_array",
        c_helper="get_from_object_{c_type}_{PY_array_arg}",
    ),
    
    dict(
        name="py_ctor_char_[]",
        base="base_py_ctor_array_fill",
        c_helper="fill_from_PyObject_char",
    ),
    dict(
        name="py_ctor_char_*",
        base="base_py_ctor_array",
        c_helper="get_from_object_char",
    ),
    dict(
        name="py_ctor_char_**",
        base="base_py_ctor_array",
        c_helper="get_from_object_charptr",
        # Need explicit post_call to change cast to char **.
        post_call=[
            "SH_obj->{field_name} = "
            "{cast_static}char **{cast1}{value_var}.data{cast2};",
            "self->{PY_member_object} = {value_var}.obj;"
            "  // steal reference",
        ],
    ),
    
    ########################################
    # descriptors
    dict(
        name="py_descr_native",
        setter=[
            "{cxx_decl} = {PY_get};",
            "if (PyErr_Occurred()) {{+",
            "return -1;",
            "-}}",
            "{c_var} = rv;",
        ],
        getter=[
            "PyObject * rv = {ctor};",
            "return rv;",
        ],
    ),

    dict(
        name="py_descr_native_*_list",
        setter_helper="get_from_object_{c_type}_list",
        setter=[
            "{PY_typedef_converter} cvalue;",
            "Py_XDECREF({c_var_obj});",
            "if ({hnamefunc0}({py_var}, &cvalue) == 0) {{+",
            "{c_var} = {nullptr};",
            "{c_var_obj} = {nullptr};",
            # Exception is set by hnamefunc0
            "return -1;",
            "-}}",
            "{c_var} = {cast_static}{cast_type}{cast1}cvalue.data{cast2};",
            "{c_var_obj} = cvalue.obj;  // steal reference",
        ],
        getter_helper="to_PyList_{c_type}",
        getter=[
            "if ({c_var} == {nullptr}) {{+",
            "Py_RETURN_NONE;",
            "-}}",
            "if ({c_var_obj} != {nullptr}) {{+",
            "Py_INCREF({c_var_obj});",
            "return {c_var_obj};",
            "-}}",
            "PyObject *rv = {hnamefunc0}({c_var}, {npy_intp_size});",
            "return rv;",
        ],
    ),
    dict(
        name="py_descr_char_*",
        setter_helper="get_from_object_{c_type}_list",
        setter=[
            "{PY_typedef_converter} cvalue;",
            "Py_XDECREF({c_var_data});",
            "if ({hnamefunc0}({py_var}, &cvalue) == 0) {{+",
            "{c_var} = {nullptr};",
            "{c_var_data} = {nullptr};",
            # Exception is set by hnamefunc0
            "return -1;",
            "-}}",
            "{c_var} = {cast_static}{cast_type}{cast1}cvalue.data{cast2};",
            "{c_var_data} = cvalue.dataobj;  // steal reference",
        ],
#        getter_helper="to_PyList_{c_type}",
        getter=[
            "if ({c_var} == {nullptr}) {{+",
            "Py_RETURN_NONE;",
            "-}}",
            # Always create a new object since the struct may change value.
#            "if ({c_var_obj} != {nullptr}) {{+",
#            "Py_INCREF({c_var_obj});",
#            "return {c_var_obj};",
#            "-}}",
            "PyObject * rv = {ctor};", # difference from py_descr_native_*_list
            "return rv;",
        ],
    ),
    # XXX - only helper is different from py_descr_native_*_list
    dict(
        name="py_descr_char_**_list",
        setter_helper="get_from_object_charptr",
        setter=[
            "{PY_typedef_converter} cvalue;",
            "Py_XDECREF({c_var_data});",
            "if ({hnamefunc0}({py_var}, &cvalue) == 0) {{+",
            "{c_var} = {nullptr};",
            "{c_var_data} = {nullptr};",
            "// XXXX set error",
            "return -1;",
            "-}}",
            "{c_var} = {cast_static}{cast_type}{cast1}cvalue.data{cast2};",
            "{c_var_data} = cvalue.dataobj;  // steal reference",
        ],
        getter_helper="to_PyList_char",
        getter=[
            "if ({c_var} == {nullptr}) {{+",
            "Py_RETURN_NONE;",
            "-}}",
            # Always create a new object since the struct may change value.
#            "if ({c_var_obj} != {nullptr}) {{+",
#            "Py_INCREF({c_var_obj});",
#            "return {c_var_obj};",
#            "-}}",
            "PyObject *rv = {hnamefunc0}({c_var}, {npy_intp_size});",
            "return rv;",
        ],
    ),

    dict(
        name="py_descr_native_[]_list",
        need_numpy = True,
        setter_helper="fill_from_PyObject_{c_type}_{PY_array_arg}",
        setter=[
            "Py_XDECREF({c_var_obj});",
            "{c_var_obj} = {nullptr};",
            "if ({hnamefunc0}(\t{py_var},\t \"{field_name}\","
            "\t {c_var},\t {npy_intp_size}) == -1) {{+",
            "return -1;",
            "-}}",
        ],
        getter_helper="to_PyList_{c_type}",
        getter=[
            "PyObject *rv = {hnamefunc0}({c_var}, {npy_intp_size});",
            "return rv;",
        ]
    ),
    dict(
        name="py_descr_native_[]_numpy",
        need_numpy = True,
        setter_helper="fill_from_PyObject_{c_type}_{PY_array_arg}",
        setter=[
            "Py_XDECREF({c_var_obj});",
            "{c_var_obj} = {nullptr};",
            "if ({hnamefunc0}(\t{py_var},\t \"{field_name}\","
            "\t {c_var},\t {npy_intp_size}) == -1) {{+",
            "return -1;",
            "-}}",
        ],
        getter=[
            "if ({c_var_obj} == {nullptr}) {{+",
            "// Create Numpy object which points to struct member.",
            "npy_intp {npy_dims_var}[{rank}] = {{ {npy_intp_values} }};",
            "{c_var_obj} = PyArray_SimpleNewFromData("
            "\t{npy_rank},\t {npy_dims_var},\t {PYN_typenum},\t {c_var});",
            "-}}",
            "Py_INCREF({c_var_obj});",
            "return {c_var_obj};",
        ]
    ),
    dict(
        name="py_descr_native_*_numpy",
        need_numpy = True,
        setter_helper="get_from_object_{c_type}_numpy",
        setter=[
            "{PY_typedef_converter} cvalue;",
            "Py_XDECREF({c_var_obj});",
            "if ({hnamefunc0}({py_var}, &cvalue) == 0) {{+",
            "{c_var} = {nullptr};",
            "{c_var_obj} = {nullptr};",
            "// XXXX set error",
            "return -1;",
            "-}}",
            "{c_var} = {cast_static}{cast_type}{cast1}cvalue.data{cast2};",
            "{c_var_obj} = cvalue.obj;  // steal reference",
        ],
        getter=[
            "if ({c_var} == {nullptr}) {{+",
            "Py_RETURN_NONE;",
            "-}}",
            "if ({c_var_obj} != {nullptr}) {{+",
            "Py_INCREF({c_var_obj});",
            "return {c_var_obj};",
            "-}}",
            "npy_intp {npy_dims_var}[{npy_rank}] = {{ {npy_intp_values} }};",
            "PyObject *rv = PyArray_SimpleNewFromData"
            "(\t{npy_rank},\t {npy_dims_var},\t {PYN_typenum},\t {c_var_non_const});",
            "if (rv != {nullptr}) {{+",
            "Py_INCREF(rv);",
            "{c_var_obj} = rv;",
            "-}}",
            "return rv;",
        ]
    ),

    dict(
        name="py_descr_char_[]",
        setter_helper="fill_from_PyObject_char", #_{PY_array_arg}",
        setter=[
            "Py_XDECREF({c_var_obj});",
            "{c_var_obj} = {nullptr};",
            "if ({hnamefunc0}(\t{py_var},\t \"{field_name}\","
            "\t {c_var},\t {npy_intp_size}) == -1) {{+",
            "return -1;",
            "-}}",
        ],
        # XXX - PyString_FromStringAndSize({c_var}, sizeof({c_var});
        # c_var_obj is not cached since if the struct changes the
        # object should be remade.
        getter=["""if ({c_var_obj} != {nullptr}) {{+
Py_INCREF({c_var_obj});
return {c_var_obj};
-}}
PyObject * rv = PyString_FromString({c_var});
// XXX assumes is null terminated
return rv;"""],
    ),
    
]
