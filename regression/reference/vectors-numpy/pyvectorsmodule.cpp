// pyvectorsmodule.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
//
// All rights reserved.
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
#include "pyvectorsmodule.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "vectors.hpp"

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
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#endif

// splicer begin C_definition
// splicer end C_definition
PyObject *PY_error_obj;
// splicer begin additional_functions
// splicer end additional_functions

static char PY_vector_sum__doc__[] =
"documentation"
;

static PyObject *
PY_vector_sum(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int vector_sum(const std::vector<int> & arg +dimension(:)+intent(in))
// splicer begin function.vector_sum
    PyObject * SHTPy_arg;
    PyArrayObject * SHPy_arg = NULL;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };
    PyObject * SHTPy_rv = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:vector_sum",
        const_cast<char **>(SHT_kwlist), &SHTPy_arg))
        return NULL;

    // post_parse
    SHPy_arg = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        SHTPy_arg, NPY_INT, NPY_ARRAY_IN_ARRAY));
    if (SHPy_arg == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "arg must be a 1-D array of int");
        goto fail;
    }
    {
        // pre_call
        int * SHData_arg = static_cast<int *>(PyArray_DATA(SHPy_arg));
        std::vector<int> SH_arg(SHData_arg,
            SHData_arg+PyArray_SIZE(SHPy_arg));

        int rv = vector_sum(SH_arg);

        // post_call
        SHTPy_rv = PyInt_FromLong(rv);

        return (PyObject *) SHTPy_rv;
    }

fail:
    Py_XDECREF(SHPy_arg);
    return NULL;
// splicer end function.vector_sum
}

static char PY_vector_iota_out__doc__[] =
"documentation"
;

/**
 * \brief Copy vector into Fortran input array
 *
 */
static PyObject *
PY_vector_iota_out(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void vector_iota_out(std::vector<int> & arg +dimension(:)+intent(out))
// splicer begin function.vector_iota_out
    std::vector<int> * SH_arg = NULL;
    PyObject * SHPy_arg = NULL;
    PyObject *SHC_arg = NULL;

    {
        // pre_call
        SH_arg = new std::vector<int>;
        if (SH_arg == NULL) {
            PyErr_NoMemory();
            goto fail;
        }

        vector_iota_out(*SH_arg);

        // post_call
        npy_intp SHD_arg[1];
        SHD_arg[0] = SH_arg->size();
        SHPy_arg = PyArray_SimpleNewFromData(1, SHD_arg, NPY_INT,
            SH_arg->data());
        if (SHPy_arg == NULL) goto fail;
        SHC_arg = PyCapsule_New(SH_arg, "PY_array_dtor", 
            PY_SHROUD_capsule_destructor);
        if (SHC_arg == NULL) goto fail;
        PyCapsule_SetContext(SHC_arg, PY_SHROUD_fetch_context(1));
        if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>
            (SHPy_arg), SHC_arg) < 0) goto fail;

        return (PyObject *) SHPy_arg;
    }

fail:
    if (SH_arg != NULL) {
        PY_SHROUD_release_memory(1, SH_arg);
    }
    Py_XDECREF(SHPy_arg);
    Py_XDECREF(SHC_arg);
    return NULL;
// splicer end function.vector_iota_out
}

static char PY_ReturnVectorAlloc__doc__[] =
"documentation"
;

/**
 * Implement iota function.
 * Return a vector as an ALLOCATABLE array.
 * Copy results into the new array.
 */
static PyObject *
PY_ReturnVectorAlloc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// std::vector<int> ReturnVectorAlloc(int n +intent(in)+value) +deref(allocatable)
// splicer begin function.return_vector_alloc
    int n;
    const char *SHT_kwlist[] = {
        "n",
        NULL };
    std::vector<int> * rv = NULL;
    PyObject * SHTPy_rv = NULL;
    PyObject *SHC_rv = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:ReturnVectorAlloc",
        const_cast<char **>(SHT_kwlist), &n))
        return NULL;

    // result pre_call
    rv = new std::vector<int>;
    if (rv == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    *rv = ReturnVectorAlloc(n);

    // post_call
    npy_intp SHD_rv[1];
    SHD_rv[0] = rv->size();
    SHTPy_rv = PyArray_SimpleNewFromData(1, SHD_rv, NPY_INT,
        rv->data());
    if (SHTPy_rv == NULL) goto fail;
    SHC_rv = PyCapsule_New(rv, "PY_array_dtor", 
        PY_SHROUD_capsule_destructor);
    if (SHC_rv == NULL) goto fail;
    PyCapsule_SetContext(SHC_rv, PY_SHROUD_fetch_context(1));
    if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>
        (SHTPy_rv), SHC_rv) < 0) goto fail;

    return (PyObject *) SHTPy_rv;

fail:
    if (rv != NULL) {
        PY_SHROUD_release_memory(1, rv);
    }
    Py_XDECREF(SHTPy_rv);
    Py_XDECREF(SHC_rv);
    return NULL;
// splicer end function.return_vector_alloc
}
static PyMethodDef PY_methods[] = {
{"vector_sum", (PyCFunction)PY_vector_sum, METH_VARARGS|METH_KEYWORDS,
    PY_vector_sum__doc__},
{"vector_iota_out", (PyCFunction)PY_vector_iota_out, METH_NOARGS,
    PY_vector_iota_out__doc__},
{"ReturnVectorAlloc", (PyCFunction)PY_ReturnVectorAlloc,
    METH_VARARGS|METH_KEYWORDS, PY_ReturnVectorAlloc__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

/*
 * initvectors - Initialization function for the module
 * *must* be called initvectors
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
static int vectors_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int vectors_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "vectors", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
    vectors_traverse, /* m_traverse */
    vectors_clear, /* m_clear */
    NULL  /* m_free */
};

#define RETVAL m
#define INITERROR return NULL
#else
#define RETVAL
#define INITERROR return
#endif

extern "C" PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_vectors(void)
#else
initvectors(void)
#endif
{
    PyObject *m = NULL;
    const char * error_name = "vectors.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("vectors", PY_methods,
        PY__doc__,
        (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

    import_array();

    PY_error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
    if (PY_error_obj == NULL)
        return RETVAL;
    st->error = PY_error_obj;
    PyModule_AddObject(m, "Error", st->error);

    // splicer begin C_init_body
    // splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module vectors");
    return RETVAL;
}

