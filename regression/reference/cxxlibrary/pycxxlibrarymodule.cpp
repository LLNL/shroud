// pycxxlibrarymodule.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pycxxlibrarymodule.hpp"
#define PY_ARRAY_UNIQUE_SYMBOL SHROUD_CXXLIBRARY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// splicer begin include
// splicer end include

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

// splicer begin C_definition
// splicer end C_definition
PyObject *PY_error_obj;
PyArray_Descr *PY_Cstruct1_array_descr;
// splicer begin additional_functions
// splicer end additional_functions

// ----------------------------------------
// Function:  int passStructByReference
// Requested: py_native_scalar_result
// Match:     py_default
// ----------------------------------------
// Argument:  Cstruct1 & arg +intent(inout)
// Exact:     py_struct_&_inout_numpy
static char PY_passStructByReference__doc__[] =
"documentation"
;

/**
 * Argument is modified by library, defaults to intent(inout).
 */
static PyObject *
PY_passStructByReference(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.pass_struct_by_reference
    Cstruct1 *arg;
    PyObject * SHTPy_arg = nullptr;
    PyArrayObject * SHPy_arg = nullptr;
    const char *SHT_kwlist[] = {
        "arg",
        nullptr };
    int SHCXX_rv;
    PyObject *SHTPy_rv = nullptr;  // return value object

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "O:passStructByReference", const_cast<char **>(SHT_kwlist), 
        &SHTPy_arg))
        return nullptr;

    // post_parse
    Py_INCREF(PY_Cstruct1_array_descr);
    SHPy_arg = reinterpret_cast<PyArrayObject *>(PyArray_FromAny(
        SHTPy_arg, PY_Cstruct1_array_descr, 0, 1, NPY_ARRAY_IN_ARRAY,
        nullptr));
    if (SHPy_arg == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "arg must be a 1-D array of CXX_cstruct1");
        goto fail;
    }

    // pre_call
    arg = static_cast<Cstruct1 *>(PyArray_DATA(SHPy_arg));

    SHCXX_rv = passStructByReference(*arg);

    // post_call
    SHTPy_rv = Py_BuildValue("iO", SHCXX_rv, SHPy_arg);

    return SHTPy_rv;

fail:
    Py_XDECREF(SHPy_arg);
    return nullptr;
// splicer end function.pass_struct_by_reference
}

// ----------------------------------------
// Function:  int passStructByReferenceIn
// Requested: py_native_scalar_result
// Match:     py_default
// ----------------------------------------
// Argument:  const Cstruct1 & arg +intent(in)
// Exact:     py_struct_&_in_numpy
static char PY_passStructByReferenceIn__doc__[] =
"documentation"
;

/**
 * const defaults to intent(in)
 */
static PyObject *
PY_passStructByReferenceIn(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.pass_struct_by_reference_in
    Cstruct1 *arg;
    PyObject * SHTPy_arg = nullptr;
    PyArrayObject * SHPy_arg = nullptr;
    const char *SHT_kwlist[] = {
        "arg",
        nullptr };
    int SHCXX_rv;
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "O:passStructByReferenceIn", const_cast<char **>(SHT_kwlist), 
        &SHTPy_arg))
        return nullptr;

    // post_parse
    Py_INCREF(PY_Cstruct1_array_descr);
    SHPy_arg = reinterpret_cast<PyArrayObject *>(PyArray_FromAny(
        SHTPy_arg, PY_Cstruct1_array_descr, 0, 1, NPY_ARRAY_IN_ARRAY,
        nullptr));
    if (SHPy_arg == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "arg must be a 1-D array of CXX_cstruct1");
        goto fail;
    }

    // pre_call
    arg = static_cast<Cstruct1 *>(PyArray_DATA(SHPy_arg));

    SHCXX_rv = passStructByReferenceIn(*arg);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    // cleanup
    Py_DECREF(SHPy_arg);

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHPy_arg);
    return nullptr;
// splicer end function.pass_struct_by_reference_in
}

// ----------------------------------------
// Function:  void passStructByReferenceInout
// Exact:     py_default
// ----------------------------------------
// Argument:  Cstruct1 & arg +intent(inout)
// Exact:     py_struct_&_inout_numpy
static char PY_passStructByReferenceInout__doc__[] =
"documentation"
;

static PyObject *
PY_passStructByReferenceInout(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.pass_struct_by_reference_inout
    Cstruct1 *arg;
    PyObject * SHTPy_arg = nullptr;
    PyArrayObject * SHPy_arg = nullptr;
    const char *SHT_kwlist[] = {
        "arg",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "O:passStructByReferenceInout",
        const_cast<char **>(SHT_kwlist), &SHTPy_arg))
        return nullptr;

    // post_parse
    Py_INCREF(PY_Cstruct1_array_descr);
    SHPy_arg = reinterpret_cast<PyArrayObject *>(PyArray_FromAny(
        SHTPy_arg, PY_Cstruct1_array_descr, 0, 1, NPY_ARRAY_IN_ARRAY,
        nullptr));
    if (SHPy_arg == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "arg must be a 1-D array of CXX_cstruct1");
        goto fail;
    }

    // pre_call
    arg = static_cast<Cstruct1 *>(PyArray_DATA(SHPy_arg));

    passStructByReferenceInout(*arg);
    return (PyObject *) SHPy_arg;

fail:
    Py_XDECREF(SHPy_arg);
    return nullptr;
// splicer end function.pass_struct_by_reference_inout
}

// ----------------------------------------
// Function:  void passStructByReferenceOut
// Exact:     py_default
// ----------------------------------------
// Argument:  Cstruct1 & arg +intent(out)
// Exact:     py_struct_&_out_numpy
static char PY_passStructByReferenceOut__doc__[] =
"documentation"
;

static PyObject *
PY_passStructByReferenceOut(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin function.pass_struct_by_reference_out
    Cstruct1 *arg;
    PyArrayObject * SHPy_arg = nullptr;

    // post_parse
    Py_INCREF(PY_Cstruct1_array_descr);
    SHPy_arg = reinterpret_cast<PyArrayObject *>(PyArray_NewFromDescr(
        &PyArray_Type, PY_Cstruct1_array_descr, 0, nullptr, nullptr,
        nullptr, 0, nullptr));
    if (SHPy_arg == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "arg must be a 1-D array of CXX_cstruct1");
        goto fail;
    }

    // pre_call
    arg = static_cast<Cstruct1 *>(PyArray_DATA(SHPy_arg));

    passStructByReferenceOut(*arg);
    return (PyObject *) SHPy_arg;

fail:
    Py_XDECREF(SHPy_arg);
    return nullptr;
// splicer end function.pass_struct_by_reference_out
}

// ----------------------------------------
// Function:  int passStructByReferenceCls
// Requested: py_native_scalar_result
// Match:     py_default
// ----------------------------------------
// Argument:  Cstruct1_cls & arg +intent(inout)
// Exact:     py_struct_&_inout_class
static char PY_passStructByReferenceCls__doc__[] =
"documentation"
;

static PyObject *
PY_passStructByReferenceCls(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.pass_struct_by_reference_cls
    PY_Cstruct1_cls * SHPy_arg;
    const char *SHT_kwlist[] = {
        "arg",
        nullptr };
    PyObject *SHTPy_rv = nullptr;  // return value object

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "O!:passStructByReferenceCls", const_cast<char **>(SHT_kwlist), 
        &PY_Cstruct1_cls_Type, &SHPy_arg))
        return nullptr;

    // post_declare
    Cstruct1_cls * arg = SHPy_arg ? SHPy_arg->obj : nullptr;

    int SHCXX_rv = passStructByReferenceCls(*arg);

    // post_call
    SHTPy_rv = Py_BuildValue("iO", SHCXX_rv, SHPy_arg);

    return SHTPy_rv;
// splicer end function.pass_struct_by_reference_cls
}

// ----------------------------------------
// Function:  int passStructByReferenceInCls
// Requested: py_native_scalar_result
// Match:     py_default
// ----------------------------------------
// Argument:  const Cstruct1_cls & arg +intent(in)
// Exact:     py_struct_&_in_class
static char PY_passStructByReferenceInCls__doc__[] =
"documentation"
;

/**
 * const defaults to intent(in)
 */
static PyObject *
PY_passStructByReferenceInCls(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.pass_struct_by_reference_in_cls
    PY_Cstruct1_cls * SHPy_arg;
    const char *SHT_kwlist[] = {
        "arg",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "O!:passStructByReferenceInCls",
        const_cast<char **>(SHT_kwlist), &PY_Cstruct1_cls_Type,
        &SHPy_arg))
        return nullptr;

    // post_declare
    const Cstruct1_cls * arg = SHPy_arg ? SHPy_arg->obj : nullptr;

    int SHCXX_rv = passStructByReferenceInCls(*arg);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.pass_struct_by_reference_in_cls
}

// ----------------------------------------
// Function:  void passStructByReferenceInoutCls
// Exact:     py_default
// ----------------------------------------
// Argument:  Cstruct1_cls & arg +intent(inout)
// Exact:     py_struct_&_inout_class
static char PY_passStructByReferenceInoutCls__doc__[] =
"documentation"
;

static PyObject *
PY_passStructByReferenceInoutCls(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.pass_struct_by_reference_inout_cls
    PY_Cstruct1_cls * SHPy_arg;
    const char *SHT_kwlist[] = {
        "arg",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "O!:passStructByReferenceInoutCls",
        const_cast<char **>(SHT_kwlist), &PY_Cstruct1_cls_Type,
        &SHPy_arg))
        return nullptr;

    // post_declare
    Cstruct1_cls * arg = SHPy_arg ? SHPy_arg->obj : nullptr;

    passStructByReferenceInoutCls(*arg);
    return (PyObject *) SHPy_arg;
// splicer end function.pass_struct_by_reference_inout_cls
}

// ----------------------------------------
// Function:  void passStructByReferenceOutCls
// Exact:     py_default
// ----------------------------------------
// Argument:  Cstruct1_cls & arg +intent(out)
// Exact:     py_struct_&_out_class
static char PY_passStructByReferenceOutCls__doc__[] =
"documentation"
;

static PyObject *
PY_passStructByReferenceOutCls(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin function.pass_struct_by_reference_out_cls
    Cstruct1_cls *arg = nullptr;
    PyObject *SHPy_arg = nullptr;

    // pre_call
    arg = new Cstruct1_cls;

    passStructByReferenceOutCls(*arg);

    // post_call
    SHPy_arg = PP_Cstruct1_cls_to_Object_idtor(arg, 0);
    if (SHPy_arg == nullptr) goto fail;

    return (PyObject *) SHPy_arg;

fail:
    Py_XDECREF(SHPy_arg);
    return nullptr;
// splicer end function.pass_struct_by_reference_out_cls
}

// ----------------------------------------
// Function:  bool defaultPtrIsNULL
// Requested: py_bool_scalar_result
// Match:     py_bool_result
// ----------------------------------------
// Argument:  double * data=nullptr +intent(in)+rank(1)
// Exact:     py_native_*_in_pointer_numpy
static char PY_defaultPtrIsNULL_1__doc__[] =
"documentation"
;

static PyObject *
PY_defaultPtrIsNULL_1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.default_ptr_is_null
    Py_ssize_t SH_nargs = 0;
    double * data;
    PyObject * SHTPy_data;
    PyArrayObject * SHPy_data = nullptr;
    const char *SHT_kwlist[] = {
        "data",
        nullptr };
    bool SHCXX_rv;
    PyObject * SHTPy_rv = nullptr;

    if (args != nullptr) SH_nargs += PyTuple_Size(args);
    if (kwds != nullptr) SH_nargs += PyDict_Size(args);
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:defaultPtrIsNULL",
        const_cast<char **>(SHT_kwlist), &SHTPy_data))
        return nullptr;
    switch (SH_nargs) {
    case 0:
        SHCXX_rv = defaultPtrIsNULL();
        break;
    case 1:
        {
            // post_parse
            SHPy_data = reinterpret_cast<PyArrayObject *>
                (PyArray_FROM_OTF(SHTPy_data, NPY_DOUBLE,
                NPY_ARRAY_IN_ARRAY));
            if (SHPy_data == nullptr) {
                PyErr_SetString(PyExc_ValueError,
                    "data must be a 1-D array of double");
                goto fail;
            }

            // pre_call
            data = static_cast<double *>(PyArray_DATA(SHPy_data));

            SHCXX_rv = defaultPtrIsNULL(data);
            break;
        }
    default:
        PyErr_SetString(PyExc_ValueError, "Wrong number of arguments");
        return nullptr;
    }

    // post_call
    SHTPy_rv = PyBool_FromLong(SHCXX_rv);
    if (SHTPy_rv == nullptr) goto fail;

    // cleanup
    Py_XDECREF(SHPy_data);

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHPy_data);
    Py_XDECREF(SHTPy_rv);
    return nullptr;
// splicer end function.default_ptr_is_null
}

// ----------------------------------------
// Function:  void defaultArgsInOut
// Exact:     py_default
// ----------------------------------------
// Argument:  int in1 +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
// ----------------------------------------
// Argument:  int * out1 +intent(out)
// Exact:     py_native_*_out
// ----------------------------------------
// Argument:  int * out2 +intent(out)
// Exact:     py_native_*_out
// ----------------------------------------
// Argument:  bool flag=false +intent(in)+value
// Requested: py_bool_scalar_in
// Match:     py_bool_in
static char PY_defaultArgsInOut_1__doc__[] =
"documentation"
;

static PyObject *
PY_defaultArgsInOut_1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.default_args_in_out
    Py_ssize_t SH_nargs = 0;
    int in1;
    int out1;
    int out2;
    bool flag;
    PyObject * SHPy_flag;
    const char *SHT_kwlist[] = {
        "in1",
        "flag",
        nullptr };
    PyObject *SHTPy_rv = nullptr;  // return value object

    if (args != nullptr) SH_nargs += PyTuple_Size(args);
    if (kwds != nullptr) SH_nargs += PyDict_Size(args);
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "i|O!:defaultArgsInOut", const_cast<char **>(SHT_kwlist), &in1,
        &PyBool_Type, &SHPy_flag))
        return nullptr;
    switch (SH_nargs) {
    case 1:
        defaultArgsInOut(in1, &out1, &out2);
        break;
    case 2:
        {
            // pre_call
            flag = PyObject_IsTrue(SHPy_flag);

            defaultArgsInOut(in1, &out1, &out2, flag);
            break;
        }
    default:
        PyErr_SetString(PyExc_ValueError, "Wrong number of arguments");
        return nullptr;
    }

    // post_call
    SHTPy_rv = Py_BuildValue("ii", out1, out2);

    return SHTPy_rv;
// splicer end function.default_args_in_out
}
static PyMethodDef PY_methods[] = {
{"passStructByReference", (PyCFunction)PY_passStructByReference,
    METH_VARARGS|METH_KEYWORDS, PY_passStructByReference__doc__},
{"passStructByReferenceIn", (PyCFunction)PY_passStructByReferenceIn,
    METH_VARARGS|METH_KEYWORDS, PY_passStructByReferenceIn__doc__},
{"passStructByReferenceInout",
    (PyCFunction)PY_passStructByReferenceInout,
    METH_VARARGS|METH_KEYWORDS, PY_passStructByReferenceInout__doc__},
{"passStructByReferenceOut", (PyCFunction)PY_passStructByReferenceOut,
    METH_NOARGS, PY_passStructByReferenceOut__doc__},
{"passStructByReferenceCls", (PyCFunction)PY_passStructByReferenceCls,
    METH_VARARGS|METH_KEYWORDS, PY_passStructByReferenceCls__doc__},
{"passStructByReferenceInCls",
    (PyCFunction)PY_passStructByReferenceInCls,
    METH_VARARGS|METH_KEYWORDS, PY_passStructByReferenceInCls__doc__},
{"passStructByReferenceInoutCls",
    (PyCFunction)PY_passStructByReferenceInoutCls,
    METH_VARARGS|METH_KEYWORDS,
    PY_passStructByReferenceInoutCls__doc__},
{"passStructByReferenceOutCls",
    (PyCFunction)PY_passStructByReferenceOutCls, METH_NOARGS,
    PY_passStructByReferenceOutCls__doc__},
{"defaultPtrIsNULL", (PyCFunction)PY_defaultPtrIsNULL_1,
    METH_VARARGS|METH_KEYWORDS, PY_defaultPtrIsNULL_1__doc__},
{"defaultArgsInOut", (PyCFunction)PY_defaultArgsInOut_1,
    METH_VARARGS|METH_KEYWORDS, PY_defaultArgsInOut_1__doc__},
{nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
};

// Create PyArray_Descr for Cstruct1
static PyArray_Descr *PY_Cstruct1_create_array_descr()
{
    int ierr;
    PyObject *obj = nullptr;
    PyObject * lnames = nullptr;
    PyObject * ldescr = nullptr;
    PyObject * dict = nullptr;
    PyArray_Descr *dtype = nullptr;

    lnames = PyList_New(2);
    if (lnames == nullptr) goto fail;
    ldescr = PyList_New(2);
    if (ldescr == nullptr) goto fail;

    // ifield
    obj = PyString_FromString("ifield");
    if (obj == nullptr) goto fail;
    PyList_SET_ITEM(lnames, 0, obj);
    obj = (PyObject *) PyArray_DescrFromType(NPY_INT);
    if (obj == nullptr) goto fail;
    PyList_SET_ITEM(ldescr, 0, obj);

    // dfield
    obj = PyString_FromString("dfield");
    if (obj == nullptr) goto fail;
    PyList_SET_ITEM(lnames, 1, obj);
    obj = (PyObject *) PyArray_DescrFromType(NPY_DOUBLE);
    if (obj == nullptr) goto fail;
    PyList_SET_ITEM(ldescr, 1, obj);
    obj = nullptr;

    dict = PyDict_New();
    if (dict == nullptr) goto fail;
    ierr = PyDict_SetItemString(dict, "names", lnames);
    if (ierr == -1) goto fail;
    lnames = nullptr;
    ierr = PyDict_SetItemString(dict, "formats", ldescr);
    if (ierr == -1) goto fail;
    ldescr = nullptr;
    ierr = PyArray_DescrAlignConverter(dict, &dtype);
    if (ierr == 0) goto fail;
    return dtype;
fail:
    Py_XDECREF(obj);
    if (lnames != nullptr) {
        for (int i=0; i < 2; i++) {
            Py_XDECREF(PyList_GET_ITEM(lnames, i));
        }
        Py_DECREF(lnames);
    }
    if (ldescr != nullptr) {
        for (int i=0; i < 2; i++) {
            Py_XDECREF(PyList_GET_ITEM(ldescr, i));
        }
        Py_DECREF(ldescr);
    }
    Py_XDECREF(dict);
    Py_XDECREF(dtype);
    return nullptr;
}

/*
 * initcxxlibrary - Initialization function for the module
 * *must* be called initcxxlibrary
 */
static char PY__doc__[] =
"library documentation"
;

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#if PY_MAJOR_VERSION >= 3
static int cxxlibrary_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int cxxlibrary_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cxxlibrary", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    nullptr, /* m_reload */
    cxxlibrary_traverse, /* m_traverse */
    cxxlibrary_clear, /* m_clear */
    NULL  /* m_free */
};

#define RETVAL m
#define INITERROR return nullptr
#else
#define RETVAL
#define INITERROR return
#endif

extern "C" PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_cxxlibrary(void)
#else
initcxxlibrary(void)
#endif
{
    PyObject *m = nullptr;
    const char * error_name = "cxxlibrary.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("cxxlibrary", PY_methods,
        PY__doc__,
        (PyObject*)nullptr,PYTHON_API_VERSION);
#endif
    if (m == nullptr)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

    import_array();

    // Cstruct1_cls
    PY_Cstruct1_cls_Type.tp_new   = PyType_GenericNew;
    PY_Cstruct1_cls_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Cstruct1_cls_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Cstruct1_cls_Type);
    PyModule_AddObject(m, "Cstruct1_cls", (PyObject *)&PY_Cstruct1_cls_Type);

    // Define PyArray_Descr for structs
    PY_Cstruct1_array_descr = PY_Cstruct1_create_array_descr();
    PyModule_AddObject(m, "Cstruct1_dtype", 
        (PyObject *) PY_Cstruct1_array_descr);

    PY_error_obj = PyErr_NewException((char *) error_name, nullptr, nullptr);
    if (PY_error_obj == nullptr)
        return RETVAL;
    st->error = PY_error_obj;
    PyModule_AddObject(m, "Error", st->error);

    // splicer begin C_init_body
    // splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module cxxlibrary");
    return RETVAL;
}

