// pypointersmodule.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
// All rights reserved.
//
// This file is part of Shroud.  For details, see
// https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the disclaimer (as noted below)
//   in the documentation and/or other materials provided with the
//   distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
// LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// #######################################################################
#include "pypointersmodule.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "pointers.hpp"

// splicer begin include
// splicer end include

#ifdef __cplusplus
#define SHROUD_UNUSED(param)
#else
#define SHROUD_UNUSED(param) param
#endif

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#endif

// splicer begin C_definition
// splicer end C_definition
PyObject *PY_error_obj;
// splicer begin additional_functions
// splicer end additional_functions

static char PY_intargs__doc__[] =
"documentation"
;

static PyObject *
PY_intargs(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void intargs(const int argin +intent(in)+value, int * arginout +intent(inout), int * argout +intent(out))
// splicer begin function.intargs
    int argin;
    int arginout;
    const char *SHT_kwlist[] = {
        "argin",
        "arginout",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii:intargs",
        const_cast<char **>(SHT_kwlist), &argin, &arginout))
        return NULL;

    // pre_call
    int argout;  // intent(out)

    intargs(argin, &arginout, &argout);

    // post_call
    PyObject * SHTPy_rv = Py_BuildValue("ii", arginout, argout);

    return SHTPy_rv;
// splicer end function.intargs
}

static char PY_cos_doubles__doc__[] =
"documentation"
;

/**
 * \brief compute cos of IN and save in OUT
 *
 * allocate OUT same type as IN implied size of array
 */
static PyObject *
PY_cos_doubles(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void cos_doubles(double * in +dimension(:)+intent(in), double * out +allocatable(mold=in)+intent(out), int sizein +implied(size(in))+intent(in)+value)
// splicer begin function.cos_doubles
    PyObject * SHTPy_in;
    PyArrayObject * SHPy_in = NULL;
    PyArrayObject * SHPy_out = NULL;
    const char *SHT_kwlist[] = {
        "in",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:cos_doubles",
        const_cast<char **>(SHT_kwlist), &SHTPy_in))
        return NULL;

    // post_parse
    SHPy_in = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        SHTPy_in, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (SHPy_in == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "in must be a 1-D array of double");
        goto fail;
    }
    {
        // pre_call
        double * in = static_cast<double *>(PyArray_DATA(SHPy_in));
        SHPy_out = reinterpret_cast<PyArrayObject *>
            (PyArray_NewLikeArray(SHPy_in, NPY_CORDER, NULL, 0));
        if (SHPy_out == NULL)
            goto fail;
        double * out = static_cast<double *>(PyArray_DATA(SHPy_out));
        int sizein = PyArray_SIZE(SHPy_in);

        cos_doubles(in, out, sizein);

        // cleanup
        Py_DECREF(SHPy_in);

        return (PyObject *) SHPy_out;
    }

fail:
    Py_XDECREF(SHPy_in);
    Py_XDECREF(SHPy_out);
    return NULL;
// splicer end function.cos_doubles
}

static char PY_truncate_to_int__doc__[] =
"documentation"
;

/**
 * \brief truncate IN argument and save in OUT
 *
 * allocate OUT different type as IN
 * implied size of array
 */
static PyObject *
PY_truncate_to_int(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void truncate_to_int(double * in +dimension(:)+intent(in), int * out +allocatable(mold=in)+intent(out), int sizein +implied(size(in))+intent(in)+value)
// splicer begin function.truncate_to_int
    PyObject * SHTPy_in;
    PyArrayObject * SHPy_in = NULL;
    PyArrayObject * SHPy_out = NULL;
    const char *SHT_kwlist[] = {
        "in",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:truncate_to_int",
        const_cast<char **>(SHT_kwlist), &SHTPy_in))
        return NULL;

    // post_parse
    SHPy_in = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        SHTPy_in, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (SHPy_in == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "in must be a 1-D array of double");
        goto fail;
    }
    {
        // pre_call
        double * in = static_cast<double *>(PyArray_DATA(SHPy_in));
        PyArray_Descr * SHDPy_out = PyArray_DescrFromType(NPY_INT);
        SHPy_out = reinterpret_cast<PyArrayObject *>
            (PyArray_NewLikeArray(SHPy_in, NPY_CORDER, SHDPy_out, 0));
        if (SHPy_out == NULL)
            goto fail;
        int * out = static_cast<int *>(PyArray_DATA(SHPy_out));
        int sizein = PyArray_SIZE(SHPy_in);

        truncate_to_int(in, out, sizein);

        // cleanup
        Py_DECREF(SHPy_in);

        return (PyObject *) SHPy_out;
    }

fail:
    Py_XDECREF(SHPy_in);
    Py_XDECREF(SHPy_out);
    return NULL;
// splicer end function.truncate_to_int
}

static char PY_increment__doc__[] =
"documentation"
;

/**
 * \brief None
 *
 * array with intent(INOUT)
 */
static PyObject *
PY_increment(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void increment(int * array +dimension(:)+intent(inout), int sizein +implied(size(array))+intent(in)+value)
// splicer begin function.increment
    PyObject * SHTPy_array;
    PyArrayObject * SHPy_array = NULL;
    const char *SHT_kwlist[] = {
        "array",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:increment",
        const_cast<char **>(SHT_kwlist), &SHTPy_array))
        return NULL;

    // post_parse
    SHPy_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        SHTPy_array, NPY_INT, NPY_ARRAY_INOUT_ARRAY));
    if (SHPy_array == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "array must be a 1-D array of int");
        goto fail;
    }
    {
        // pre_call
        int * array = static_cast<int *>(PyArray_DATA(SHPy_array));
        int sizein = PyArray_SIZE(SHPy_array);

        increment(array, sizein);
        return (PyObject *) SHPy_array;
    }

fail:
    return NULL;
// splicer end function.increment
}

static char PY_get_values__doc__[] =
"documentation"
;

/**
 * \brief fill values into array
 *
 * The function knows how long the array must be.
 * Fortran will treat the dimension as assumed-length.
 * The Python wrapper will create a NumPy array so it must
 * have an explicit dimension (not assumed-length).
 */
static PyObject *
PY_get_values(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void get_values(int * nvalues +intent(out), int * values +dimension(3)+intent(out))
// splicer begin function.get_values
    PyArrayObject * SHPy_values = NULL;

    // post_parse
    npy_intp SHD_values[1] = { 3 };
    SHPy_values = reinterpret_cast<PyArrayObject *>
        (PyArray_SimpleNew(1, SHD_values, NPY_INT));
    if (SHPy_values == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "values must be a 1-D array of int");
        goto fail;
    }
    {
        // pre_call
        int nvalues;  // intent(out)
        int * values = static_cast<int *>(PyArray_DATA(SHPy_values));

        get_values(&nvalues, values);

        // post_call
        PyObject * SHTPy_rv = Py_BuildValue("iO", nvalues, SHPy_values);

        return SHTPy_rv;
    }

fail:
    Py_XDECREF(SHPy_values);
    return NULL;
// splicer end function.get_values
}
static PyMethodDef PY_methods[] = {
{"intargs", (PyCFunction)PY_intargs, METH_VARARGS|METH_KEYWORDS,
    PY_intargs__doc__},
{"cos_doubles", (PyCFunction)PY_cos_doubles, METH_VARARGS|METH_KEYWORDS,
    PY_cos_doubles__doc__},
{"truncate_to_int", (PyCFunction)PY_truncate_to_int,
    METH_VARARGS|METH_KEYWORDS, PY_truncate_to_int__doc__},
{"increment", (PyCFunction)PY_increment, METH_VARARGS|METH_KEYWORDS,
    PY_increment__doc__},
{"get_values", (PyCFunction)PY_get_values, METH_NOARGS,
    PY_get_values__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

/*
 * initpointers - Initialization function for the module
 * *must* be called initpointers
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
static int pointers_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int pointers_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "pointers", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
    pointers_traverse, /* m_traverse */
    pointers_clear, /* m_clear */
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
PyInit_pointers(void)
#else
initpointers(void)
#endif
{
    PyObject *m = NULL;
    const char * error_name = "pointers.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("pointers", PY_methods,
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
        Py_FatalError("can't initialize module pointers");
    return RETVAL;
}

