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

// Convert obj into an array of type int
// Return -1 on error.
static int SHROUD_from_PyObject_vector_int(PyObject *obj,
    const char *name, std::vector<int> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}

static PyObject *SHROUD_to_PyList_vector_double(std::vector<double> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
    return out;
}

static PyObject *SHROUD_to_PyList_vector_int(std::vector<int> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}

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
    const char *SHT_kwlist[] = {
        "arg",
        NULL };
    PyObject * SHTPy_rv = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:vector_sum",
        const_cast<char **>(SHT_kwlist), &SHTPy_arg))
        return NULL;
    {
        // pre_call
        std::vector<int> SH_arg;
        if (SHROUD_from_PyObject_vector_int(SHTPy_arg, "arg",
            SH_arg) == -1)
            goto fail;

        int SHCXX_rv = vector_sum(SH_arg);

        // post_call
        SHTPy_rv = PyInt_FromLong(SHCXX_rv);

        return (PyObject *) SHTPy_rv;
    }

fail:
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
    PyObject * SHPy_arg = NULL;

    {
        // pre_call
        std::vector<int> SH_arg;

        vector_iota_out(SH_arg);

        // post_call
        SHPy_arg = SHROUD_to_PyList_vector_int(SH_arg);
        if (SHPy_arg == NULL) goto fail;

        return (PyObject *) SHPy_arg;
    }

fail:
    Py_XDECREF(SHPy_arg);
    return NULL;
// splicer end function.vector_iota_out
}

static char PY_vector_iota_out_d__doc__[] =
"documentation"
;

/**
 * \brief Copy vector into Fortran input array
 *
 */
static PyObject *
PY_vector_iota_out_d(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void vector_iota_out_d(std::vector<double> & arg +dimension(:)+intent(out))
// splicer begin function.vector_iota_out_d
    PyObject * SHPy_arg = NULL;

    {
        // pre_call
        std::vector<double> SH_arg;

        vector_iota_out_d(SH_arg);

        // post_call
        SHPy_arg = SHROUD_to_PyList_vector_double(SH_arg);
        if (SHPy_arg == NULL) goto fail;

        return (PyObject *) SHPy_arg;
    }

fail:
    Py_XDECREF(SHPy_arg);
    return NULL;
// splicer end function.vector_iota_out_d
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
// std::vector<int> ReturnVectorAlloc(int n +intent(in)+value) +dimension(:)
// splicer begin function.return_vector_alloc
    int n;
    const char *SHT_kwlist[] = {
        "n",
        NULL };
    PyObject * SHTPy_rv = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:ReturnVectorAlloc",
        const_cast<char **>(SHT_kwlist), &n))
        return NULL;

    std::vector<int> SHCXX_rv = ReturnVectorAlloc(n);

    // post_call
    SHTPy_rv = SHROUD_to_PyList_vector_int(SHCXX_rv);
    if (SHTPy_rv == NULL) goto fail;

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return NULL;
// splicer end function.return_vector_alloc
}
static PyMethodDef PY_methods[] = {
{"vector_sum", (PyCFunction)PY_vector_sum, METH_VARARGS|METH_KEYWORDS,
    PY_vector_sum__doc__},
{"vector_iota_out", (PyCFunction)PY_vector_iota_out, METH_NOARGS,
    PY_vector_iota_out__doc__},
{"vector_iota_out_d", (PyCFunction)PY_vector_iota_out_d, METH_NOARGS,
    PY_vector_iota_out_d__doc__},
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

