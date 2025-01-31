# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
"""Helper functions for Python wrappers.

C helper functions which may be added to a implementation file.

 c_fmtname     = name of function created by the helper.
               ex. SHROUD_get_from_object_char_{numpy,list}
 need_numpy  = If True, NumPy headers will be added.
 scope
     ``pwrap_impl`` - Added to PY_utility_filename and shared among files.


# Helper in wrapper classes

Methods in wrappers to deal with helpers.
  add_helper - Build up a list of helpers from statements.
    - wrapf.ModuleInfo.add_f_helper and add_c_helper
    - wrapc.Wrapc.add_c_helper
    - wrapp.Wrapp.add_helper
  gather_helper_code - Write helpers in a sorted order (so the generated
   files will compare). Write dependent helpers so their declaration is before
   their use.

Most C API functions also return an error indicator, usually NULL if
they are supposed to return a pointer, or -1 if they return an integer.

O& converter - status = converter(PyObject *object, void *address);
The returned status should be 1 for a successful conversion and 0 if
the conversion has failed.

"""

# Note about PRIVATE Fortran helpers
# If a single subroutine uses multiple modules created by Shroud
# some compilers will rightly complain that they each define this function.
#  "Procedure shroud_copy_string_and_free has more than one interface accessible
#  by use association. The interfaces are assumed to be the same."
# It should be marked PRIVATE to prevent users from calling it directly.
# However, gfortran does not like that.
#  "Symbol 'shroud_copy_string_and_free' at (1) is marked PRIVATE but has been given
#  the binding label 'ShroudCopyStringAndFree'"
# See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=49111
# Instead, mangle the name with C_prefix.
# See FHelpers copy_string
#
# This also applies to derived types which are bind(C).


from . import statements
from . import typemap
from . import util

import json

wformat = util.wformat

# Used with literalinclude
# format fields {lstart} and {lend}
cstart = "// start "
cend   = "// end "
fstart = "! start "
fend   = "! end "
literalinclude = False

PYHelpers = {}


def add_all_helpers(library):
    """Create helper functions.
    Create helpers for all types.
    """
    symtab = library.symtab
    global literalinclude
    literalinclude = library.options.literalinclude2
    fmt = util.Scope(library.fmtdict)
    fmt.c_lstart = ""
    fmt.c_lend = ""
    fmt.f_lstart = ""
    fmt.f_lend = ""
    statements.add_json_fc_helpers(fmt)
    add_external_helpers(fmt, symtab)
    for ntypemap in symtab.typemaps.values():
        if ntypemap.sgroup == "native":
            add_to_PyList_helper(fmt, ntypemap)
            add_to_PyList_helper_vector(fmt, ntypemap)

def add_external_helpers(fmt, symtab):
    """Create helper which have generated names.
    For example, code uses format entries
    C_prefix, C_memory_dtor_function,
    F_array_type, C_array_type

    Some helpers are written in C, but called by Fortran.
    Since the names are external, mangle with C_prefix to avoid
    confict with other Shroud wrapped libraries.

    Args:
        fmt - format dictionary
        symtab - 
    """
    ########################################
    # Python
    ########################################
    name = "py_capsule_dtor"
    fmt.hname = name
    fmt.hnamefunc = wformat("FREE_{hname}", fmt)
    PYHelpers[name] = dict(
        c_fmtname=fmt.hnamefunc,
        source=wformat(
            """
// helper {hname}
// Release memory in PyCapsule.
// Used with native arrays.
static void {hnamefunc}(PyObject *obj)
{{+
void *in = PyCapsule_GetPointer(obj, {nullptr});
if (in != {nullptr}) {{+
{stdlib}free(in);
-}}
-}}""",
            fmt,
        ),
    )
    
    ########################################
    # char *
    name = "get_from_object_char"
    fmt.hname = name
    fmt.hnamefunc = fmt.PY_helper_prefix + name
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t {PY_typedef_converter} *value)", fmt)
    PYHelpers[name] = dict(
        c_fmtname=fmt.hnamefunc,
        dependent_helpers=["PY_converter_type"],
        proto=fmt.hnameproto + ";",
        source=wformat("""
// helper {hname}
// Converter from PyObject to char *.
// The returned status will be 1 for a successful conversion
// and 0 if the conversion has failed.
// value.obj is unused.
// value.dataobj - object which holds the data.
// If same as obj argument, its refcount is incremented.
// value.data is owned by value.dataobj and must be copied to be preserved.
// Caller must use Py_XDECREF(value.dataobj).
{PY_helper_static}{hnameproto}
{{+
size_t size = 0;
char *out;
if (PyUnicode_Check(obj)) {{+
^#if PY_MAJOR_VERSION >= 3
PyObject *strobj = PyUnicode_AsUTF8String(obj);
out = PyBytes_AS_STRING(strobj);
size = PyBytes_GET_SIZE(strobj);
value->dataobj = strobj;  // steal reference
^#else
PyObject *strobj = PyUnicode_AsUTF8String(obj);
out = PyString_AsString(strobj);
size = PyString_Size(obj);
value->dataobj = strobj;  // steal reference
^#endif
^#if PY_MAJOR_VERSION < 3
-}} else if (PyString_Check(obj)) {{+
out = PyString_AsString(obj);
size = PyString_Size(obj);
value->dataobj = obj;
Py_INCREF(obj);
^#endif
-}} else if (PyBytes_Check(obj)) {{+
out = PyBytes_AS_STRING(obj);
size = PyBytes_GET_SIZE(obj);
value->dataobj = obj;
Py_INCREF(obj);
-}} else if (PyByteArray_Check(obj)) {{+
out = PyByteArray_AS_STRING(obj);
size = PyByteArray_GET_SIZE(obj);
value->dataobj = obj;
Py_INCREF(obj);
-}} else if (obj == Py_None) {{+
out = NULL;
size = 0;
value->dataobj = NULL;
-}} else {{+
PyErr_Format(PyExc_TypeError,\t "argument should be string or None, not %.200s",\t Py_TYPE(obj)->tp_name);
return 0;
-}}
value->obj = {nullptr};
value->data = out;
value->size = size;
return 1;
-}}
""", fmt),
    )
    # There are no 'list' or 'numpy' version of these functions.
    # Use the one-true-version get_from_object_char.
    PYHelpers['get_from_object_char_list'] = dict(
        c_fmtname=fmt.hnamefunc,
        dependent_helpers=[name],
    )
    PYHelpers['get_from_object_char_numpy'] = dict(
        c_fmtname=fmt.hnamefunc,
        dependent_helpers=[name],
    )

    ##########
    # Generate C or C++ version of helper.
    ##########
    # 'char *' needs a custom handler because of the nature
    # of NULL terminated strings.
    ntypemap = symtab.lookup_typemap("char")
    fmt.fcn_suffix = "char"
    fmt.fcn_type = "string"
    fmt.c_type = "char *"
    fmt.Py_ctor = ntypemap.PY_ctor.format(ctor_expr="in[i]")
    fmt.c_const=""  # XXX issues with struct.yaml test, remove const.
    fmt.hname = "to_PyList_char"
    PYHelpers["to_PyList_char"] = create_to_PyList(fmt)

    ########################################
    name = "fill_from_PyObject_char"
    fmt.hname = name
    fmt.hnamefunc = fmt.PY_helper_prefix + name
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t const char *name,\t char *in,\t Py_ssize_t insize)", fmt)
    PYHelpers[name] = dict(
        c_fmtname=fmt.hnamefunc,
        dependent_helpers=["get_from_object_char"],
        c_include=["<string.h>"],
        cxx_include=["<cstring>"],
        proto=fmt.hnameproto + ";",
        source=wformat("""
// helper {hname}
// Fill existing char array from PyObject.
// Return 0 on success, -1 on error.
{PY_helper_static}{hnameproto}
{{+
{PY_typedef_converter} value;
int i = {PY_helper_prefix}get_from_object_char(obj, &value);
if (i == 0) {{+
Py_DECREF(obj);
return -1;
-}}
if (value.data == {nullptr}) {{+
in[0] = '\\0';
-}} else {{+
{stdlib}strncpy\t(in,\t {cast_static}char *{cast1}value.data{cast2},\t insize);
Py_DECREF(value.dataobj);
-}}
return 0;
-}}""", fmt),
    )

    ########################################
    # char **
    name = "get_from_object_charptr"
    fmt.size_var="size"
    fmt.c_var="in"
    fmt.hname = name
    fmt.hnamefunc = fmt.PY_helper_prefix + name
    PYHelpers[name] = create_get_from_object_list_charptr(fmt)
    # There are no 'list' or 'numpy' version of these functions.
    # Use the one-true-version SHROUD_get_from_object_charptr.
    PYHelpers['get_from_object_charptr_list'] = dict(
        c_fmtname=fmt.hnamefunc,
        dependent_helpers=[name],
    )
    PYHelpers['get_from_object_charptr_numpy'] = dict(
        c_fmtname=fmt.hnamefunc,
        dependent_helpers=[name],
    )

    ########################################
    PYHelpers['PY_converter_type'] = dict(
        scope="pwrap_impl",
        c_include=["<stddef.h>"],
        cxx_include=["<cstddef>"],
        # obj may be the argument passed into a function or
        # it may be a PyCapsule for locally allocated memory.
        source=wformat("""
// helper PY_converter_type
// Store PyObject and pointer to the data it contains.
// name - used in error messages
// obj  - A mutable object which holds the data.
//        For example, a NumPy array, Python array.
//        But not a list or str object.
// dataobj - converter allocated memory.
//           Decrement dataobj to release memory.
//           For example, extracted from a list or str.
// data  - C accessable pointer to data which is in obj or dataobj.
// size  - number of items in data (not number of bytes).
typedef struct {{+
const char *name;
PyObject *obj;
PyObject *dataobj;
void *data;   // points into obj.
size_t size;
-}} {PY_typedef_converter};""", fmt)
    )
    
######################################################################

def add_to_PyList_helper(fmt, ntypemap):
    """Add helpers to work with Python lists.
    Several helpers are created based on the type of arg.
    Used with sgroup="native" types.

    Args:
        fmt      - util.Scope, parent is newlibrary
        ntypemap - typemap.Typemap
    """
    flat_name = ntypemap.flat_name
    fmt.c_type = ntypemap.c_type
    fmt.numpy_type = ntypemap.PYN_typenum

    ########################################
    # Used with intent(out)
    name = "to_PyList_" + flat_name
    if ntypemap.PY_ctor is not None:
        fmt.hname = name
        fmt.fcn_suffix = flat_name
        ctor_expr = "in[i]"
        if ntypemap.py_ctype is not None:
            ctor_expr = ntypemap.pytype_to_pyctor.format(ctor_expr=ctor_expr)
        fmt.Py_ctor = ntypemap.PY_ctor.format(ctor_expr=ctor_expr)
        fmt.c_const="const "
        helper = create_to_PyList(fmt)
        PYHelpers[name] = create_to_PyList(fmt)

    ########################################
    # Used with intent(inout)
    name = "update_PyList_" + flat_name
    if ntypemap.PY_ctor is not None:
        ctor_expr = "in[i]"
        if ntypemap.py_ctype is not None:
            ctor_expr = ntypemap.pytype_to_pyctor.format(ctor_expr=ctor_expr)
        fmt.Py_ctor = ntypemap.PY_ctor.format(ctor_expr=ctor_expr)
        fmt.hname = name
        fmt.hnameproto = wformat(
            "void {PY_helper_prefix}{hname}\t(PyObject *out, {c_type} *in, size_t size)", fmt)
        helper = dict(
            proto=fmt.hnameproto + ";",
            source=wformat(
                """
// helper {hname}
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
{PY_helper_static}{hnameproto}
{{+
for (size_t i = 0; i < size; ++i) {{+
PyObject *item = PyList_GET_ITEM(out, i);
Py_DECREF(item);
PyList_SET_ITEM(out, i, {Py_ctor});
-}}
-}}""", fmt),
        )
        PYHelpers[name] = helper

    ########################################
    # Used with intent(in), setter.
    # Return -1 on error.
    # Use a fixed text in PySequence_Fast.
    # If an error occurs, replace message with one which includes argument name.
    if ntypemap.PY_get:
        name = "fill_from_PyObject_" + flat_name + "_list"
        fmt.hname = name
        fmt.flat_name = flat_name
        fmt.fcn_type = ntypemap.c_type
        fmt.py_ctype = fmt.c_type
        fmt.work_ctor = "cvalue"
        if ntypemap.py_ctype is not None:
            fmt.py_ctype = ntypemap.py_ctype
            fmt.work_ctor = ntypemap.pytype_to_cxx.format(work_var=fmt.work_ctor)
        fmt.Py_get_obj = ntypemap.PY_get.format(py_var="obj")
        fmt.Py_get = ntypemap.PY_get.format(py_var="item")
        PYHelpers[name] = fill_from_PyObject_list(fmt)

        name = "fill_from_PyObject_" + flat_name + "_numpy"
        fmt.hname = name
        PYHelpers[name] = fill_from_PyObject_numpy(fmt)

    ########################################
    # Function called by typemap.PY_get_converter for NumPy.
    name = "get_from_object_{}_numpy".format(flat_name)
    fmt.py_tmp = "array"
    fmt.c_type = ntypemap.c_type
    fmt.numpy_type = ntypemap.PYN_typenum
    fmt.hname = name
    fmt.hnamefunc = fmt.PY_helper_prefix + name
    fmt.hnameproto = wformat(
        "int {hnamefunc}\t(PyObject *obj,\t {PY_typedef_converter} *value)", fmt)
    helper = dict(
        c_fmtname=fmt.hnamefunc,
        dependent_helpers=["PY_converter_type"],
        need_numpy=True,
        proto=fmt.hnameproto + ";",
        source=wformat("""
// helper {hname}
// Convert PyObject to {c_type} pointer.
{PY_helper_static}{hnameproto}
{{+
PyObject *{py_tmp} = PyArray_FROM_OTF(obj,\t {numpy_type},\t NPY_ARRAY_IN_ARRAY);
if ({py_tmp} == {nullptr}) {{+
PyErr_SetString(PyExc_ValueError,\t "must be a 1-D array of {c_type}");
return 0;
-}}
value->obj = {py_tmp};
value->dataobj = {nullptr};
value->data = PyArray_DATA({cast_reinterpret}PyArrayObject *{cast1}{py_tmp}{cast2});
value->size = PyArray_SIZE({cast_reinterpret}PyArrayObject *{cast1}{py_tmp}{cast2});
return 1;
-}}""", fmt),
    )
    PYHelpers[name] = helper

    ########################################
    # Function called by typemap.PY_get_converter for list.
    if ntypemap.PY_get:
        name = "get_from_object_{}_list".format(flat_name)
        fmt.size_var = "size"
        fmt.c_var = "in"
        fmt.fcn_suffix = flat_name
        fmt.Py_get = ntypemap.PY_get.format(py_var="item")
        fmt.hname = name
        fmt.hnamefunc = fmt.PY_helper_prefix + name
        PYHelpers[name] = create_get_from_object_list(fmt)

def fill_from_PyObject_list(fmt):
    """Create helper to convert list of PyObjects to existing C array.

    If passed a scalar, broadcast to array.
    """
    fmt.hnamefunc = wformat(
        "{PY_helper_prefix}fill_from_PyObject_{flat_name}_list", fmt)
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t const char *name,\t "
            "{c_type} *in,\t Py_ssize_t insize)", fmt)
    helper = dict(
        c_fmtname=fmt.hnamefunc,
        proto=fmt.hnameproto + ";",
        source=wformat(
                """
// helper {hname}
// Fill {c_type} array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
{PY_helper_static}{hnameproto}
{{+
{py_ctype} cvalue = {Py_get_obj};
if (!PyErr_Occurred()) {{+
// Broadcast scalar.
for (Py_ssize_t i = 0; i < insize; ++i) {{+
in[i] = {work_ctor};
-}}
return 0;
-}}
PyErr_Clear();

// Look for sequence.
PyObject *seq = PySequence_Fast(obj, "holder");
if (seq == NULL) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be iterable",\t name);
return -1;
-}}
Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
if (size > insize) {{+
size = insize;
-}}
for (Py_ssize_t i = 0; i < size; ++i) {{+
PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
cvalue = {Py_get};
if (PyErr_Occurred()) {{+
Py_DECREF(seq);
PyErr_Format(PyExc_TypeError,\t "argument '%s', index %d must be {fcn_type}",\t name,\t (int) i);
return -1;
-}}
in[i] = {work_ctor};
-}}
Py_DECREF(seq);
return 0;
-}}""", fmt),
    )
    return helper
    
def fill_from_PyObject_numpy(fmt):
    """Create helper to convert list of PyObjects to existing C array.

    If passed a scalar, broadcast to array.
    """
    fmt.hnamefunc = wformat(
        "{PY_helper_prefix}fill_from_PyObject_{flat_name}_numpy", fmt)
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t const char *name,\t {c_type} *in,\t Py_ssize_t insize)", fmt)
    fmt.py_tmp = "array"
    fmt.numpy_type
    helper = dict(
        c_fmtname=fmt.hnamefunc,
        proto=fmt.hnameproto + ";",
        need_numpy=True,
        source=wformat(
                """
// helper {hname}
// Fill {c_type} array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
{PY_helper_static}{hnameproto}
{{+
{py_ctype} cvalue = {Py_get_obj};
if (!PyErr_Occurred()) {{+
// Broadcast scalar.
for (Py_ssize_t i = 0; i < insize; ++i) {{+
in[i] = {work_ctor};
-}}
return 0;
-}}
PyErr_Clear();

PyObject *{py_tmp} = PyArray_FROM_OTF(obj,\t {numpy_type},\t NPY_ARRAY_IN_ARRAY);
if ({py_tmp} == {nullptr}) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be a 1-D array of {c_type}",\t name);
return -1;
-}}
PyArrayObject *pyarray = {cast_reinterpret}PyArrayObject *{cast1}{py_tmp}{cast2};

{c_type} *data = {cast_static}{c_type} *{cast1}PyArray_DATA(pyarray){cast2};
npy_intp size = PyArray_SIZE(pyarray);
if (size > insize) {{+
size = insize;
-}}
for (Py_ssize_t i = 0; i < size; ++i) {{+
in[i] = data[i];
-}}
Py_DECREF(pyarray);
return 0;
-}}""", fmt),
        )
    return helper
    
def create_to_PyList(fmt):
    """Create helper to convert C array to PyList of PyObjects.
    """
    fmt.hnamefunc = wformat(
        "{PY_helper_prefix}to_PyList_{fcn_suffix}", fmt)
    fmt.hnameproto = wformat(
        "PyObject *{hnamefunc}\t({c_const}{c_type} *in, size_t size)", fmt)
    helper = dict(
        c_fmtname=fmt.hnamefunc,
        proto=fmt.hnameproto + ";",
        source=wformat(
            """
// helper {hname}
// Convert {c_type} pointer to PyList of PyObjects.
{PY_helper_static}{hnameproto}
{{+
PyObject *out = PyList_New(size);
for (size_t i = 0; i < size; ++i) {{+
PyList_SET_ITEM(out, i, {Py_ctor});
-}}
return out;
-}}""", fmt),
    )
    return helper

def create_get_from_object_list(fmt):
    """ Convert PyObject to {c_type} pointer.
    Used with native types.
# XXX - convert empty list to NULL pointer.

    format fields:
       fcn_suffix - 
    """
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t {PY_typedef_converter} *value)", fmt)
    fmt.dtor_helper = PYHelpers["py_capsule_dtor"]["c_fmtname"]
    helper = dict(
        c_fmtname=fmt.hnamefunc,
        dependent_helpers=[
            "PY_converter_type",
            "py_capsule_dtor",
        ],
        c_include=["<stdlib.h>"],   # malloc/free
        cxx_include=["<cstdlib>"],  # malloc/free
        proto=fmt.hnameproto + ";",
        source=wformat("""
// helper {hname}
// Convert list of PyObject to array of {c_type}.
// Return 0 on error, 1 on success.
// Set Python exception on error.
{PY_helper_static}{hnameproto}
{{+
PyObject *seq = PySequence_Fast(obj, "holder");
if (seq == NULL) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be iterable",\t value->name);
return 0;
-}}
Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
{c_type} *in = {cast_static}{c_type} *{cast1}{stdlib}malloc(size * sizeof({c_type})){cast2};
for (Py_ssize_t i = 0; i < size; i++) {{+
PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
{c_type} cvalue = {Py_get};
if (PyErr_Occurred()) {{+
{stdlib}free(in);
Py_DECREF(seq);
PyErr_Format(PyExc_TypeError,\t "argument '%s', index %d must be {fcn_type}",\t value->name,\t (int) i);
return 0;
-}}
in[i] = {work_ctor};
-}}
Py_DECREF(seq);

value->obj = {nullptr};  // Do not save list object.
value->dataobj = PyCapsule_New(in, {nullptr}, {dtor_helper});
value->data = {cast_static}{c_type} *{cast1}{c_var}{cast2};
value->size = size;
return 1;
-}}""", fmt),
    )
    return helper

def create_get_from_object_list_charptr(fmt):
    """ Convert PyObject to an char **.
    ["one", "two"]
    helper get_from_object_charptr

    Loop over all strings in the sequence object and
    convert using get_from_object_char helper
    which deals with unicode.
    All string values are copied into new memory.

    format fields:
       fcn_suffix - 
    """
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t {PY_typedef_converter} *value)", fmt)
    fmt.__helper = PYHelpers["get_from_object_char"]["c_fmtname"]
    helper = dict(
        c_fmtname=fmt.hnamefunc,
        dependent_helpers=[
            "PY_converter_type",
            "get_from_object_char",
        ],
        c_include=["<stdlib.h>"],   # malloc/free
        cxx_include=["<cstdlib>"],  # malloc/free
        proto=fmt.hnameproto + ";",
        source=wformat("""

// helper FREE_{hname}
static void FREE_{hname}(PyObject *obj)
{{+
char **in = {cast_static}char **{cast1}PyCapsule_GetPointer(obj, {nullptr}){cast2};
if (in == {nullptr})
+return;-
size_t *size = {cast_static}size_t *{cast1}PyCapsule_GetContext(obj){cast2};
if (size == {nullptr})
+return;-
for (size_t i=0; i < *size; ++i) {{+
if (in[i] == {nullptr})
+continue;-
{stdlib}free(in[i]);
-}}
{stdlib}free(in);
{stdlib}free(size);
-}}

// helper {hname}
// Convert obj into an array of char * (i.e. char **).
{PY_helper_static}{hnameproto}
{{+
PyObject *seq = PySequence_Fast(obj, "holder");
if (seq == NULL) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be iterable",\t value->name);
return -1;
-}}
Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
char **in = {cast_static}char **{cast1}{stdlib}calloc(size, sizeof(char *)){cast2};
PyObject *dataobj = PyCapsule_New(in, {nullptr}, FREE_{hname});
size_t *size_context = {cast_static}size_t *{cast1}malloc(sizeof(size_t)){cast2};
*size_context = size;
int ierr = PyCapsule_SetContext(dataobj, size_context);
// XXX - check error
{PY_typedef_converter} itemvalue = {PY_value_init};
for (Py_ssize_t i = 0; i < size; i++) {{+
PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
ierr = {__helper}(item, &itemvalue);
if (ierr == 0) {{+
Py_XDECREF(itemvalue.dataobj);
Py_DECREF(dataobj);
Py_DECREF(seq);
PyErr_Format(PyExc_TypeError,\t "argument '%s', index %d must be {fcn_type}",\t value->name,\t (int) i);
return 0;
-}}
if (itemvalue.data != {nullptr}) {{+
in[i] = strdup({cast_static}char *{cast1}itemvalue.data{cast2});
-}}
Py_XDECREF(itemvalue.dataobj);
-}}
Py_DECREF(seq);

value->obj = {nullptr};
value->dataobj = dataobj;
value->data = in;
value->size = {size_var};
return 1;
-}}""", fmt),
    )
    return helper

def add_to_PyList_helper_vector(fmt, ntypemap):
    """Add helpers to work with Python lists.
    Several helpers are created based on the type of arg.
    Used with sgroup="native" types.

    Args:
        fmt      - util.Scope
        ntypemap - typemap.Typemap
    """
    flat_name = ntypemap.flat_name
    fmt.c_type = ntypemap.c_type
    fmt.cxx_type = ntypemap.cxx_type
    
    # Used with intent(out)
    name = "to_PyList_vector_" + flat_name
    ctor = ntypemap.PY_ctor
    if ctor is None:
        ctor = "XXXPy_ctor"
    ctor_expr = "in[i]"
    if ntypemap.py_ctype is not None:
        ctor_expr = ntypemap.pytype_to_pyctor.format(ctor_expr=ctor_expr)
    fmt.Py_ctor = ctor.format(ctor_expr=ctor_expr)
    fmt.hname = name
    fmt.hnamefunc = wformat("{PY_helper_prefix}{hname}", fmt)
    fmt.hnameproto = wformat("PyObject *{hnamefunc}\t(std::vector<{c_type}> & in)", fmt)
    helper = dict(
        c_fmtname=fmt.hnamefunc,
        proto=fmt.hnameproto + ";",
        source=wformat(
            """
// helper {hname}
{PY_helper_static}{hnameproto}
{{+
size_t size = in.size();
PyObject *out = PyList_New(size);
for (size_t i = 0; i < size; ++i) {{+
PyList_SET_ITEM(out, i, {Py_ctor});
-}}
return out;
-}}""",
            fmt,
        ),
    )
    PYHelpers[name] = helper

    # Used with intent(inout)
    name = "update_PyList_vector_" + flat_name
    ctor = ntypemap.PY_ctor
    if ctor is None:
        ctor = "XXXPy_ctor"
    ctor_expr = "in[i]"
    if ntypemap.py_ctype is not None:
        ctor_expr = ntypemap.pytype_to_pyctor.format(ctor_expr=ctor_expr)
    fmt.Py_ctor = ctor.format(ctor_expr=ctor_expr)
    fmt.hname = name
    fmt.hnamefunc = wformat(
        "{PY_helper_prefix}{hname}", fmt)
    fmt.hnameproto = wformat(
        "void {hnamefunc}\t(PyObject *out, {c_type} *in, size_t size)", fmt)
    helper = dict(
        c_fmtname=fmt.hnamefunc,
        proto=fmt.hnameproto + ";",
        source=wformat(
            """
// helper {hname}
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
{PY_helper_static}{hnameproto}
{{+
for (size_t i = 0; i < size; ++i) {{+
PyObject *item = PyList_GET_ITEM(out, i);
Py_DECREF(item);
PyList_SET_ITEM(out, i, {Py_ctor});
-}}
-}}""",
            fmt,
        ),
    )
    PYHelpers[name] = helper

    # used with intent(in)
    # Return -1 on error.
    # Convert an empty list into a NULL pointer.
    # Use a fixed text in PySequence_Fast.
    # If an error occurs, replace message with one which includes argument name.
    name = "create_from_PyObject_vector_" + flat_name
    get = ntypemap.PY_get
    if get is None:
        get = "XXXPy_get"
    py_var = "item"
    fmt.Py_get = get.format(py_var=py_var)
    fmt.py_ctype = fmt.c_type;
    fmt.work_ctor = "cvalue"
    if ntypemap.py_ctype is not None:
        fmt.py_ctype = ntypemap.py_ctype
        fmt.work_ctor = ntypemap.pytype_to_cxx.format(work_var=fmt.work_ctor)
    fmt.hname = name
    fmt.hnamefunc= wformat(
        "{PY_helper_prefix}{hname}", fmt)
    fmt.hnameproto = wformat(
        "int {hnamefunc}\t(PyObject *obj,\t const char *name,\t std::vector<{cxx_type}> & in)", fmt)
    helper = dict(
        c_fmtname=fmt.hnamefunc,
##-        cxx_include=["<cstdlib>"],  # malloc/free
        cxx_proto=fmt.hnameproto + ";",
        cxx_source=wformat(
            """
// helper {hname}
// Convert obj into an array of type {cxx_type}
// Return -1 on error.
{PY_helper_static}{hnameproto}
{{+
PyObject *seq = PySequence_Fast(obj, "holder");
if (seq == NULL) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be iterable",\t name);
return -1;
-}}
Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
for (Py_ssize_t i = 0; i < size; i++) {{+
PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
{py_ctype} cvalue = {Py_get};
if (PyErr_Occurred()) {{+
Py_DECREF(seq);
PyErr_Format(PyExc_ValueError,\t "argument '%s', index %d must be {c_type}",\t name,\t (int) i);
return -1;
-}}
in.push_back({work_ctor});
-}}
Py_DECREF(seq);
return 0;
-}}""",
            fmt,
        ),
    )
    PYHelpers[name] = helper

"""
http://effbot.org/zone/python-capi-sequences.htm
if (PyList_Check(seq))
        for (i = 0; i < len; i++) {
            item = PyList_GET_ITEM(seq, i);
            ...
        }
    else
        for (i = 0; i < len; i++) {
            item = PyTuple_GET_ITEM(seq, i);
            ...
        }
"""
    
########################################
# Routines to dump helper routines to a file.

def gather_helpers(fp, wrapper, helpers, keys):
    """Dump helpers in human readable format.
    Dump selected keys in a format which can be used with sphinx
    literalinclude. Dump the other keys as JSON.
    Use with testing.
    """
    for name in sorted(helpers.keys()):
        helper = helpers[name]
        out = {}
        output = []
        for key, value in helper.items():
            if key in keys:
                output.append("")
                output.append("##### start {} {}".format(name, key))
                if isinstance(value, list):
                    output.extend(value)
                else:
                    output.append(value)
                output.append("##### end {} {}".format(name, key))
            else:
                out[key] = value

        print("\n----------", name, "----------", file=fp)
        json.dump(out, fp, sort_keys=True, indent=4, separators=(',', ': '))
        print("", file=fp)
        wrapper.write_lines(fp, output)

    return

c_lines = ["source", "c_source", "cxx_source"]
f_lines = ["derived_type", "interface", "f_source"]
fc_lines = c_lines + f_lines

def write_c_helpers(fp):
    wrapper = util.WrapperMixin()
    wrapper.linelen = 72
    wrapper.indent = 0
    wrapper.cont = ""
    output = gather_helpers(fp, wrapper, PYHelpers, c_lines)

def apply_fmtdict_from_helpers(helper, fmt):
    """Apply fmtdict field from helpers
    """
    name = helper["name"]
    if name.startswith("h_helper_"):
        name = name[9:]
    fmt.hname = name

    if literalinclude:
        fmt.c_lstart = "{}helper {}\n".format(cstart, name)
        fmt.c_lend = "\n{}helper {}".format(cend, name)
        fmt.f_lstart = "{}helper {}\n".format(fstart, name)
        fmt.f_lend = "\n{}helper {}".format(fend, name)
    
    if "fmtdict" in helper:
        for key, value in helper["fmtdict"].items():
            setattr(fmt, key, wformat(value, fmt))

    # Merge list into a single string
    for field in fc_lines:
        if field in helper:
            helper[field] = [wformat(line, fmt) for line in helper[field]]
            
    for field in ["c_fmtname", "f_fmtname", "proto"]:
        if field in helper:
            helper[field] = wformat(helper[field], fmt)
    

cmake = """
# Setup Shroud
# This file defines:
#  SHROUD_FOUND - If Shroud was found

if(NOT SHROUD_EXECUTABLE)
    MESSAGE(FATAL_ERROR "Could not find Shroud. Shroud requires explicit SHROUD_EXECUTABLE.")
endif()

message(STATUS "Found SHROUD: ${SHROUD_EXECUTABLE}")

add_custom_target(generate)
set(SHROUD_FOUND TRUE)

#
# Setup targets to generate code.
#
# Each package can create their own ${PROJECT}_generate target
#  add_dependencies(generate  ${PROJECT}_generate)

##------------------------------------------------------------------------------
## add_shroud( YAML_INPUT_FILE file
##             DEPENDS_SOURCE file1 ... filen
##             DEPENDS_BINARY file1 ... filen
##             C_FORTRAN_OUTPUT_DIR dir
##             PYTHON_OUTPUT_DIR dir
##             LUA_OUTPUT_DIR dir
##             YAML_OUTPUT_DIR dir
##             CFILES file
##             FFILES file
## )
##
##  YAML_INPUT_FILE - yaml input file to shroud. Required.
##  DEPENDS_SOURCE  - splicer files in the source directory
##  DEPENDS_BINARY  - splicer files in the binary directory
##  C_FORTRAN_OUTPUT_DIR - directory for C and Fortran wrapper output files.
##  PYTHON_OUTPUT_DIR - directory for Python wrapper output files.
##  LUA_OUTPUT_DIR  - directory for Lua wrapper output files.
##  YAML_OUTPUT_DIR - directory for YAML output files.
##                    Defaults to CMAKE_CURRENT_SOURCE_DIR
##  CFILES          - Output file with list of generated C/C++ files
##  FFILES          - Output file with list of generated Fortran files
##
## Add a target generate_${basename} where basename is generated from
## YAML_INPUT_FILE.  It is then added as a dependency to the generate target.
##
##------------------------------------------------------------------------------

macro(add_shroud)

    # Decide where the output files should be written.
    # For now all files are written into the source directory.
    # This allows them to be source controlled and does not require a library user
    # to generate them.  All they have to do is compile them.
    #set(SHROUD_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(SHROUD_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

    set(options)
    set(singleValueArgs
        YAML_INPUT_FILE
        C_FORTRAN_OUTPUT_DIR
        PYTHON_OUTPUT_DIR
        LUA_OUTPUT_DIR
        YAML_OUTPUT_DIR
        CFILES
        FFILES
    )
    set(multiValueArgs DEPENDS_SOURCE DEPENDS_BINARY )

    ## parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    # make sure YAML_INPUT_FILE is defined
    if(NOT arg_YAML_INPUT_FILE)
      message(FATAL_ERROR "add_shroud macro must define YAML_INPUT_FILE")
    endif()
    get_filename_component(_basename ${arg_YAML_INPUT_FILE} NAME_WE)

    if(arg_C_FORTRAN_OUTPUT_DIR)
      set(SHROUD_C_FORTRAN_OUTPUT_DIR --outdir-c-fortran ${arg_C_FORTRAN_OUTPUT_DIR})
    endif()

    if(arg_PYTHON_OUTPUT_DIR)
      set(SHROUD_PYTHON_OUTPUT_DIR --outdir-python ${arg_PYTHON_OUTPUT_DIR})
    endif()

    if(arg_LUA_OUTPUT_DIR)
      set(SHROUD_LUA_OUTPUT_DIR --outdir-lua ${arg_LUA_OUTPUT_DIR})
    endif()

    if(arg_YAML_OUTPUT_DIR)
      set(SHROUD_YAML_OUTPUT_DIR --outdir-yaml ${arg_YAML_OUTPUT_DIR})
    else()
      set(SHROUD_YAML_OUTPUT_DIR --outdir-yaml ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    if(arg_CFILES)
      set(SHROUD_CFILES ${arg_CFILES})
    else()
      set(SHROUD_CFILES ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.cfiles)
    endif()

    if(arg_FFILES)
      set(SHROUD_FFILES ${arg_FFILES})
    else()
      set(SHROUD_FFILES ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.ffiles)
    endif()

    # convert DEPENDS to full paths
    set(shroud_depends)
    foreach (_file ${arg_DEPENDS_SOURCE})
        list(APPEND shroud_depends "${CMAKE_CURRENT_SOURCE_DIR}/${_file}")
    endforeach ()
    foreach (_file ${arg_DEPENDS_BINARY})
        list(APPEND shroud_depends "${CMAKE_CURRENT_BINARY_DIR}/${_file}")
    endforeach ()

    set(_timestamp  ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.time)

    set(_cmd
        ${SHROUD_EXECUTABLE}
        --logdir ${CMAKE_CURRENT_BINARY_DIR}
        ${SHROUD_C_FORTRAN_OUTPUT_DIR}
        ${SHROUD_PYTHON_OUTPUT_DIR}
        ${SHROUD_LUA_OUTPUT_DIR}
        ${SHROUD_YAML_OUTPUT_DIR}
        # path controls where to search for splicer files listed in YAML_INPUT_FILE
        --path ${CMAKE_CURRENT_BINARY_DIR}
        --path ${CMAKE_CURRENT_SOURCE_DIR}
        --cfiles ${SHROUD_CFILES}
        --ffiles ${SHROUD_FFILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/${arg_YAML_INPUT_FILE}
    )

    add_custom_command(
        OUTPUT  ${_timestamp}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${arg_YAML_INPUT_FILE} ${shroud_depends}
        COMMAND ${_cmd}
        COMMAND ${CMAKE_COMMAND} -E touch ${_timestamp}
        COMMENT "Running shroud ${arg_YAML_INPUT_FILE}"
        WORKING_DIRECTORY ${SHROUD_OUTPUT_DIR}
    )

    # Create target to process this Shroud file
    add_custom_target(generate_${_basename}    DEPENDS ${_timestamp})

    add_dependencies(generate generate_${_basename})
endmacro(add_shroud)
"""
