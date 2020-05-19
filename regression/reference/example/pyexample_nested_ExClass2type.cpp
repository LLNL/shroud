// pyexample_nested_ExClass2type.cpp
// This file is generated by Shroud 0.11.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pyUserLibrarymodule.hpp"
// splicer begin namespace.example::nested.class.ExClass2.impl.include
// splicer end namespace.example::nested.class.ExClass2.impl.include

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
#endif
// splicer begin namespace.example::nested.class.ExClass2.impl.C_definition
// splicer end namespace.example::nested.class.ExClass2.impl.C_definition
// splicer begin namespace.example::nested.class.ExClass2.impl.additional_methods
// splicer end namespace.example::nested.class.ExClass2.impl.additional_methods
static void
PP_ExClass2_tp_dealloc (PP_ExClass2 *self)
{
// splicer begin namespace.example::nested.class.ExClass2.type.dealloc
    PyErr_SetString(PyExc_NotImplementedError, "dealloc");
    return;
// splicer end namespace.example::nested.class.ExClass2.type.dealloc
}
static int
PP_ExClass2_tp_print (PP_ExClass2 *self, FILE *fp, int flags)
{
// splicer begin namespace.example::nested.class.ExClass2.type.print
    PyErr_SetString(PyExc_NotImplementedError, "print");
    return -1;
// splicer end namespace.example::nested.class.ExClass2.type.print
}
static int
PP_ExClass2_tp_compare (PP_ExClass2 *self, PyObject *)
{
// splicer begin namespace.example::nested.class.ExClass2.type.compare
    PyErr_SetString(PyExc_NotImplementedError, "compare");
    return -1;
// splicer end namespace.example::nested.class.ExClass2.type.compare
}
static PyObject *
PP_ExClass2_tp_getattr (PP_ExClass2 *self, char *name)
{
// splicer begin namespace.example::nested.class.ExClass2.type.getattr
    PyErr_SetString(PyExc_NotImplementedError, "getattr");
    return nullptr;
// splicer end namespace.example::nested.class.ExClass2.type.getattr
}
static int
PP_ExClass2_tp_setattr (PP_ExClass2 *self, char *name, PyObject *value)
{
// splicer begin namespace.example::nested.class.ExClass2.type.setattr
    PyErr_SetString(PyExc_NotImplementedError, "setattr");
    return -1;
// splicer end namespace.example::nested.class.ExClass2.type.setattr
}
static PyObject *
PP_ExClass2_tp_getattro (PP_ExClass2 *self, PyObject *name)
{
// splicer begin namespace.example::nested.class.ExClass2.type.getattro
    PyErr_SetString(PyExc_NotImplementedError, "getattro");
    return nullptr;
// splicer end namespace.example::nested.class.ExClass2.type.getattro
}
static int
PP_ExClass2_tp_setattro (PP_ExClass2 *self, PyObject *name, PyObject *value)
{
// splicer begin namespace.example::nested.class.ExClass2.type.setattro
    PyErr_SetString(PyExc_NotImplementedError, "setattro");
    return -1;
// splicer end namespace.example::nested.class.ExClass2.type.setattro
}
static PyObject *
PP_ExClass2_tp_repr (PP_ExClass2 *self)
{
// splicer begin namespace.example::nested.class.ExClass2.type.repr
    PyErr_SetString(PyExc_NotImplementedError, "repr");
    return nullptr;
// splicer end namespace.example::nested.class.ExClass2.type.repr
}
static long
PP_ExClass2_tp_hash (PP_ExClass2 *self)
{
// splicer begin namespace.example::nested.class.ExClass2.type.hash
    PyErr_SetString(PyExc_NotImplementedError, "hash");
    return -1;
// splicer end namespace.example::nested.class.ExClass2.type.hash
}
static PyObject *
PP_ExClass2_tp_call (PP_ExClass2 *self, PyObject *args, PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.type.call
    PyErr_SetString(PyExc_NotImplementedError, "call");
    return nullptr;
// splicer end namespace.example::nested.class.ExClass2.type.call
}
static PyObject *
PP_ExClass2_tp_str (PP_ExClass2 *self)
{
// splicer begin namespace.example::nested.class.ExClass2.type.str
    PyErr_SetString(PyExc_NotImplementedError, "str");
    return nullptr;
// splicer end namespace.example::nested.class.ExClass2.type.str
}
static PyObject *
PP_ExClass2_tp_alloc (PyTypeObject *type, Py_ssize_t nitems)
{
// splicer begin namespace.example::nested.class.ExClass2.type.alloc
    PyErr_SetString(PyExc_NotImplementedError, "alloc");
    return nullptr;
// splicer end namespace.example::nested.class.ExClass2.type.alloc
}
static PyObject *
PP_ExClass2_tp_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.type.new
    PyErr_SetString(PyExc_NotImplementedError, "new");
    return nullptr;
// splicer end namespace.example::nested.class.ExClass2.type.new
}
static void
PP_ExClass2_tp_free (void *op)
{
// splicer begin namespace.example::nested.class.ExClass2.type.free
    PyErr_SetString(PyExc_NotImplementedError, "free");
    return;
// splicer end namespace.example::nested.class.ExClass2.type.free
}
static void
PP_ExClass2_tp_del (PP_ExClass2 *self)
{
// splicer begin namespace.example::nested.class.ExClass2.type.del
    PP_SHROUD_release_memory(self->idtor, self->obj);
    self->obj = nullptr;
// splicer end namespace.example::nested.class.ExClass2.type.del
}

// ----------------------------------------
// Function:  ExClass2
// Exact:     py_default
// ----------------------------------------
// Argument:  const string * name +intent(in)+len_trim(trim_name)
// Requested: py_string_*_in
// Match:     py_string_in
/**
 * \brief constructor
 *
 */
static int
PP_ExClass2_tp_init(
  PP_ExClass2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.method.ctor
    char * name;
    const char *SHT_kwlist[] = {
        "name",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:ctor",
        const_cast<char **>(SHT_kwlist), &name))
        return -1;

    // post_declare
    const std::string SH_name(name);

    self->obj = new example::nested::ExClass2(&SH_name);
    if (self->obj == nullptr) {
        PyErr_NoMemory();
        return -1;
    }
    self->idtor = 2;
    return 0;
// splicer end namespace.example::nested.class.ExClass2.method.ctor
}

// ----------------------------------------
// Function:  const string & getName +deref(result-as-arg)+len(aa_exclass2_get_name_length({F_this}%{F_derived_member}))
// Requested: py_string_result
// Match:     py_default
static char PP_getName__doc__[] =
"documentation"
;

static PyObject *
PP_getName(
  PP_ExClass2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.class.ExClass2.method.get_name
    PyObject * SHTPy_rv = nullptr;

    const std::string & SHCXX_rv = self->obj->getName();

    // post_call
    SHTPy_rv = PyString_FromStringAndSize(SHCXX_rv.data(),
        SHCXX_rv.size());

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.class.ExClass2.method.get_name
}

// ----------------------------------------
// Function:  const string & getName2 +deref(allocatable)
// Requested: py_string_result
// Match:     py_default
static char PP_getName2__doc__[] =
"documentation"
;

static PyObject *
PP_getName2(
  PP_ExClass2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.class.ExClass2.method.get_name2
    PyObject * SHTPy_rv = nullptr;

    const std::string & SHCXX_rv = self->obj->getName2();

    // post_call
    SHTPy_rv = PyString_FromStringAndSize(SHCXX_rv.data(),
        SHCXX_rv.size());

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.class.ExClass2.method.get_name2
}

// ----------------------------------------
// Function:  string & getName3 +deref(allocatable)
// Requested: py_string_result
// Match:     py_default
static char PP_getName3__doc__[] =
"documentation"
;

static PyObject *
PP_getName3(
  PP_ExClass2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.class.ExClass2.method.get_name3
    PyObject * SHTPy_rv = nullptr;

    std::string & SHCXX_rv = self->obj->getName3();

    // post_call
    SHTPy_rv = PyString_FromStringAndSize(SHCXX_rv.data(),
        SHCXX_rv.size());

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.class.ExClass2.method.get_name3
}

// ----------------------------------------
// Function:  string & getName4 +deref(allocatable)
// Requested: py_string_result
// Match:     py_default
static char PP_getName4__doc__[] =
"documentation"
;

static PyObject *
PP_getName4(
  PP_ExClass2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.class.ExClass2.method.get_name4
    PyObject * SHTPy_rv = nullptr;

    std::string & SHCXX_rv = self->obj->getName4();

    // post_call
    SHTPy_rv = PyString_FromStringAndSize(SHCXX_rv.data(),
        SHCXX_rv.size());

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.class.ExClass2.method.get_name4
}

// ----------------------------------------
// Function:  int GetNameLength
// Requested: py_native_scalar_result
// Match:     py_default
static char PP_GetNameLength__doc__[] =
"documentation"
;

/**
 * \brief helper function for Fortran
 *
 */
static PyObject *
PP_GetNameLength(
  PP_ExClass2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.class.ExClass2.method.get_name_length
    PyObject * SHTPy_rv = nullptr;

    int SHCXX_rv = self->obj->GetNameLength();

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.class.ExClass2.method.get_name_length
}

// ----------------------------------------
// Function:  ExClass1 * get_class1
// Exact:     py_shadow_result
// ----------------------------------------
// Argument:  const ExClass1 * in +intent(in)
// Requested: py_shadow_*_in
// Match:     py_shadow_in
static char PP_get_class1__doc__[] =
"documentation"
;

static PyObject *
PP_get_class1(
  PP_ExClass2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.method.get_class1
    PP_ExClass1 * SHPy_in;
    const char *SHT_kwlist[] = {
        "in",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:get_class1",
        const_cast<char **>(SHT_kwlist), &PP_ExClass1_Type, &SHPy_in))
        return nullptr;

    // post_declare
    const example::nested::ExClass1 * in =
        SHPy_in ? SHPy_in->obj : nullptr;

    example::nested::ExClass1 * SHCXX_rv = self->obj->get_class1(in);

    // post_call
    PP_ExClass1 * SHTPy_rv =
        PyObject_New(PP_ExClass1, &PP_ExClass1_Type);
    SHTPy_rv->obj = SHCXX_rv;

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.class.ExClass2.method.get_class1
}

// ----------------------------------------
// Function:  void * declare
// Exact:     py_default
// ----------------------------------------
// Argument:  TypeID type +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
// ----------------------------------------
// Argument:  SidreLength len=1 +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
static char PP_declare_1__doc__[] =
"documentation"
;

static PyObject *
PP_declare_1(
  PP_ExClass2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.method.declare
    Py_ssize_t SH_nargs = 0;
    int type;
    SIDRE_SidreLength len;
    const char *SHT_kwlist[] = {
        "type",
        "len",
        nullptr };

    if (args != nullptr) SH_nargs += PyTuple_Size(args);
    if (kwds != nullptr) SH_nargs += PyDict_Size(args);
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|l:declare",
        const_cast<char **>(SHT_kwlist), &type, &len))
        return nullptr;
    switch (SH_nargs) {
    case 1:
        {
            // post_parse
            TypeID SH_type = getTypeID(type);

            self->obj->declare(SH_type);
            break;
        }
    case 2:
        {
            // post_parse
            TypeID SH_type = getTypeID(type);

            self->obj->declare(SH_type, len);
            break;
        }
    default:
        PyErr_SetString(PyExc_ValueError, "Wrong number of arguments");
        return nullptr;
    }
    Py_RETURN_NONE;
// splicer end namespace.example::nested.class.ExClass2.method.declare
}

// ----------------------------------------
// Function:  void destroyall
// Exact:     py_default
static char PP_destroyall__doc__[] =
"documentation"
;

static PyObject *
PP_destroyall(
  PP_ExClass2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.class.ExClass2.method.destroyall
    self->obj->destroyall();
    Py_RETURN_NONE;
// splicer end namespace.example::nested.class.ExClass2.method.destroyall
}

// ----------------------------------------
// Function:  TypeID getTypeID
// Requested: py_native_scalar_result
// Match:     py_default
static char PP_getTypeID__doc__[] =
"documentation"
;

static PyObject *
PP_getTypeID(
  PP_ExClass2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.class.ExClass2.method.get_type_id
    PyObject * SHTPy_rv = nullptr;

    TypeID SHCXX_rv = self->obj->getTypeID();

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.class.ExClass2.method.get_type_id
}

// ----------------------------------------
// Function:  void setValue
// Exact:     py_default
// ----------------------------------------
// Argument:  int value +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
static PyObject *
PP_setValue_int(
  PP_ExClass2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.method.set_value_int
    int value;
    const char *SHT_kwlist[] = {
        "value",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:setValue",
        const_cast<char **>(SHT_kwlist), &value))
        return nullptr;

    self->obj->setValue<int>(value);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.class.ExClass2.method.set_value_int
}

// ----------------------------------------
// Function:  void setValue
// Exact:     py_default
// ----------------------------------------
// Argument:  long value +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
static PyObject *
PP_setValue_long(
  PP_ExClass2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.method.set_value_long
    long value;
    const char *SHT_kwlist[] = {
        "value",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l:setValue",
        const_cast<char **>(SHT_kwlist), &value))
        return nullptr;

    self->obj->setValue<long>(value);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.class.ExClass2.method.set_value_long
}

// ----------------------------------------
// Function:  void setValue
// Exact:     py_default
// ----------------------------------------
// Argument:  float value +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
static PyObject *
PP_setValue_float(
  PP_ExClass2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.method.set_value_float
    float value;
    const char *SHT_kwlist[] = {
        "value",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "f:setValue",
        const_cast<char **>(SHT_kwlist), &value))
        return nullptr;

    self->obj->setValue<float>(value);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.class.ExClass2.method.set_value_float
}

// ----------------------------------------
// Function:  void setValue
// Exact:     py_default
// ----------------------------------------
// Argument:  double value +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
static PyObject *
PP_setValue_double(
  PP_ExClass2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.method.set_value_double
    double value;
    const char *SHT_kwlist[] = {
        "value",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d:setValue",
        const_cast<char **>(SHT_kwlist), &value))
        return nullptr;

    self->obj->setValue<double>(value);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.class.ExClass2.method.set_value_double
}

// ----------------------------------------
// Function:  int getValue
// Requested: py_native_scalar_result
// Match:     py_default
static PyObject *
PP_getValue_int(
  PP_ExClass2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.class.ExClass2.method.get_value_int
    PyObject * SHTPy_rv = nullptr;

    int SHCXX_rv = self->obj->getValue<int>();

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.class.ExClass2.method.get_value_int
}

// ----------------------------------------
// Function:  double getValue
// Requested: py_native_scalar_result
// Match:     py_default
static PyObject *
PP_getValue_double(
  PP_ExClass2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.class.ExClass2.method.get_value_double
    PyObject * SHTPy_rv = nullptr;

    double SHCXX_rv = self->obj->getValue<double>();

    // post_call
    SHTPy_rv = PyFloat_FromDouble(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.class.ExClass2.method.get_value_double
}

static char PP_setValue__doc__[] =
"documentation"
;

static PyObject *
PP_setValue(
  PP_ExClass2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.method.set_value
    Py_ssize_t SHT_nargs = 0;
    if (args != nullptr) SHT_nargs += PyTuple_Size(args);
    if (kwds != nullptr) SHT_nargs += PyDict_Size(args);
    PyObject *rvobj;
    if (SHT_nargs == 1) {
        rvobj = PP_setValue_int(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs == 1) {
        rvobj = PP_setValue_long(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs == 1) {
        rvobj = PP_setValue_float(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs == 1) {
        rvobj = PP_setValue_double(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
    return nullptr;
// splicer end namespace.example::nested.class.ExClass2.method.set_value
}

static char PP_getValue__doc__[] =
"documentation"
;

static PyObject *
PP_getValue(
  PP_ExClass2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.class.ExClass2.method.get_value
    Py_ssize_t SHT_nargs = 0;
    if (args != nullptr) SHT_nargs += PyTuple_Size(args);
    if (kwds != nullptr) SHT_nargs += PyDict_Size(args);
    PyObject *rvobj;
    if (SHT_nargs == 0) {
        rvobj = PP_getValue_int(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs == 0) {
        rvobj = PP_getValue_double(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
    return nullptr;
// splicer end namespace.example::nested.class.ExClass2.method.get_value
}
// splicer begin namespace.example::nested.class.ExClass2.impl.after_methods
// splicer end namespace.example::nested.class.ExClass2.impl.after_methods
static PyMethodDef PP_ExClass2_methods[] = {
    {"getName", (PyCFunction)PP_getName, METH_NOARGS,
        PP_getName__doc__},
    {"getName2", (PyCFunction)PP_getName2, METH_NOARGS,
        PP_getName2__doc__},
    {"getName3", (PyCFunction)PP_getName3, METH_NOARGS,
        PP_getName3__doc__},
    {"getName4", (PyCFunction)PP_getName4, METH_NOARGS,
        PP_getName4__doc__},
    {"GetNameLength", (PyCFunction)PP_GetNameLength, METH_NOARGS,
        PP_GetNameLength__doc__},
    {"get_class1", (PyCFunction)PP_get_class1,
        METH_VARARGS|METH_KEYWORDS, PP_get_class1__doc__},
    {"declare", (PyCFunction)PP_declare_1, METH_VARARGS|METH_KEYWORDS,
        PP_declare_1__doc__},
    {"destroyall", (PyCFunction)PP_destroyall, METH_NOARGS,
        PP_destroyall__doc__},
    {"getTypeID", (PyCFunction)PP_getTypeID, METH_NOARGS,
        PP_getTypeID__doc__},
    {"setValue", (PyCFunction)PP_setValue, METH_VARARGS|METH_KEYWORDS,
        PP_setValue__doc__},
    {"getValue", (PyCFunction)PP_getValue, METH_VARARGS|METH_KEYWORDS,
        PP_getValue__doc__},
    // splicer begin namespace.example::nested.class.ExClass2.PyMethodDef
    // splicer end namespace.example::nested.class.ExClass2.PyMethodDef
    {nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
};

static char ExClass2__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PP_ExClass2_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "userlibrary.example.nested.ExClass2",                       /* tp_name */
    sizeof(PP_ExClass2),         /* tp_basicsize */
    0,                              /* tp_itemsize */
    /* Methods to implement standard operations */
    (destructor)PP_ExClass2_tp_dealloc,                 /* tp_dealloc */
    (printfunc)PP_ExClass2_tp_print,                   /* tp_print */
    (getattrfunc)PP_ExClass2_tp_getattr,                 /* tp_getattr */
    (setattrfunc)PP_ExClass2_tp_setattr,                 /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
    nullptr,                               /* tp_reserved */
#else
    (cmpfunc)PP_ExClass2_tp_compare,                     /* tp_compare */
#endif
    (reprfunc)PP_ExClass2_tp_repr,                    /* tp_repr */
    /* Method suites for standard classes */
    nullptr,                              /* tp_as_number */
    nullptr,                              /* tp_as_sequence */
    nullptr,                              /* tp_as_mapping */
    /* More standard operations (here for binary compatibility) */
    (hashfunc)PP_ExClass2_tp_hash,                    /* tp_hash */
    (ternaryfunc)PP_ExClass2_tp_call,                 /* tp_call */
    (reprfunc)PP_ExClass2_tp_str,                    /* tp_str */
    (getattrofunc)PP_ExClass2_tp_getattro,                /* tp_getattro */
    (setattrofunc)PP_ExClass2_tp_setattro,                /* tp_setattro */
    /* Functions to access object as input/output buffer */
    nullptr,                              /* tp_as_buffer */
    /* Flags to define presence of optional/expanded features */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    ExClass2__doc__,         /* tp_doc */
    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    (traverseproc)nullptr,                /* tp_traverse */
    /* delete references to contained objects */
    (inquiry)nullptr,                     /* tp_clear */
    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    (richcmpfunc)nullptr,                 /* tp_richcompare */
    /* weak reference enabler */
    0,                              /* tp_weaklistoffset */
    /* Added in release 2.2 */
    /* Iterators */
    (getiterfunc)nullptr,                 /* tp_iter */
    (iternextfunc)nullptr,                /* tp_iternext */
    /* Attribute descriptor and subclassing stuff */
    PP_ExClass2_methods,                             /* tp_methods */
    nullptr,                              /* tp_members */
    nullptr,                             /* tp_getset */
    nullptr,                              /* tp_base */
    nullptr,                              /* tp_dict */
    (descrgetfunc)nullptr,                /* tp_descr_get */
    (descrsetfunc)nullptr,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PP_ExClass2_tp_init,                   /* tp_init */
    (allocfunc)PP_ExClass2_tp_alloc,                  /* tp_alloc */
    (newfunc)PP_ExClass2_tp_new,                    /* tp_new */
    (freefunc)PP_ExClass2_tp_free,                   /* tp_free */
    (inquiry)nullptr,                     /* tp_is_gc */
    nullptr,                              /* tp_bases */
    nullptr,                              /* tp_mro */
    nullptr,                              /* tp_cache */
    nullptr,                              /* tp_subclasses */
    nullptr,                              /* tp_weaklist */
    (destructor)PP_ExClass2_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)nullptr,                  /* tp_finalize */
#endif
};
