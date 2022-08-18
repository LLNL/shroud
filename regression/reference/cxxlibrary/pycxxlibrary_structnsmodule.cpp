// pycxxlibrary_structnsmodule.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pycxxlibrarymodule.hpp"
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SHROUD_CXXLIBRARY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// splicer begin namespace.structns.include
// splicer end namespace.structns.include

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

// splicer begin namespace.structns.C_definition
// splicer end namespace.structns.C_definition
PyArray_Descr *PY_Cstruct1_array_descr;
// splicer begin namespace.structns.additional_functions
// splicer end namespace.structns.additional_functions

// ----------------------------------------
// Function:  int passStructByReference
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  Cstruct1 & arg
// Attrs:     +intent(inout)
// Exact:     py_inout_struct_&_numpy
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
// splicer begin namespace.structns.function.pass_struct_by_reference
    structns::Cstruct1 *arg;
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
            "arg must be a 0-D array of CXX_cstruct1");
        goto fail;
    }

    // pre_call
    arg = static_cast<structns::Cstruct1 *>(PyArray_DATA(SHPy_arg));

    SHCXX_rv = structns::passStructByReference(*arg);

    // post_call
    SHTPy_rv = Py_BuildValue("iO", SHCXX_rv, SHPy_arg);

    return SHTPy_rv;

fail:
    Py_XDECREF(SHPy_arg);
    return nullptr;
// splicer end namespace.structns.function.pass_struct_by_reference
}

// ----------------------------------------
// Function:  int passStructByReferenceIn
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  const Cstruct1 & arg
// Attrs:     +intent(in)
// Exact:     py_in_struct_&_numpy
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
// splicer begin namespace.structns.function.pass_struct_by_reference_in
    structns::Cstruct1 *arg;
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
            "arg must be a 0-D array of CXX_cstruct1");
        goto fail;
    }

    // pre_call
    arg = static_cast<structns::Cstruct1 *>(PyArray_DATA(SHPy_arg));

    SHCXX_rv = structns::passStructByReferenceIn(*arg);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    // cleanup
    Py_DECREF(SHPy_arg);

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHPy_arg);
    return nullptr;
// splicer end namespace.structns.function.pass_struct_by_reference_in
}

// ----------------------------------------
// Function:  void passStructByReferenceInout
// Attrs:     +intent(subroutine)
// Exact:     py_default
// ----------------------------------------
// Argument:  Cstruct1 & arg +intent(inout)
// Attrs:     +intent(inout)
// Exact:     py_inout_struct_&_numpy
static char PY_passStructByReferenceInout__doc__[] =
"documentation"
;

static PyObject *
PY_passStructByReferenceInout(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.structns.function.pass_struct_by_reference_inout
    structns::Cstruct1 *arg;
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
            "arg must be a 0-D array of CXX_cstruct1");
        goto fail;
    }

    // pre_call
    arg = static_cast<structns::Cstruct1 *>(PyArray_DATA(SHPy_arg));

    structns::passStructByReferenceInout(*arg);
    return (PyObject *) SHPy_arg;

fail:
    Py_XDECREF(SHPy_arg);
    return nullptr;
// splicer end namespace.structns.function.pass_struct_by_reference_inout
}

// ----------------------------------------
// Function:  void passStructByReferenceOut
// Attrs:     +intent(subroutine)
// Exact:     py_default
// ----------------------------------------
// Argument:  Cstruct1 & arg +intent(out)
// Attrs:     +intent(out)
// Exact:     py_out_struct_&_numpy
static char PY_passStructByReferenceOut__doc__[] =
"documentation"
;

static PyObject *
PY_passStructByReferenceOut(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.structns.function.pass_struct_by_reference_out
    structns::Cstruct1 *arg;
    PyArrayObject * SHPy_arg = nullptr;

    // post_parse
    Py_INCREF(PY_Cstruct1_array_descr);
    SHPy_arg = reinterpret_cast<PyArrayObject *>(PyArray_NewFromDescr(
        &PyArray_Type, PY_Cstruct1_array_descr, 0, nullptr, nullptr,
        nullptr, 0, nullptr));
    if (SHPy_arg == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "arg must be a 0-D array of CXX_cstruct1");
        goto fail;
    }

    // pre_call
    arg = static_cast<structns::Cstruct1 *>(PyArray_DATA(SHPy_arg));

    structns::passStructByReferenceOut(*arg);
    return (PyObject *) SHPy_arg;

fail:
    Py_XDECREF(SHPy_arg);
    return nullptr;
// splicer end namespace.structns.function.pass_struct_by_reference_out
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

#if PY_MAJOR_VERSION >= 3
static char PY__doc__[] =
"XXX submodule doc"  //"library documentation"
;

struct module_state {
    PyObject *error;
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cxxlibrary.structns", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    nullptr, /* m_reload */
//    cxxlibrary_traverse, /* m_traverse */
//    cxxlibrary_clear, /* m_clear */
    nullptr, /* m_traverse */
    nullptr, /* m_clear */
    nullptr  /* m_free */
};
#endif
#define RETVAL nullptr

PyObject *PY_init_cxxlibrary_structns(void)
{
    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3((char *) "structns", PY_methods, nullptr);
#endif
    if (m == nullptr)
        return nullptr;


    // Define PyArray_Descr for structs
    PY_Cstruct1_array_descr = PY_Cstruct1_create_array_descr();
    PyModule_AddObject(m, "Cstruct1_dtype", 
        (PyObject *) PY_Cstruct1_array_descr);

    return m;
}

