// pyClibrarymodule.c
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
#include "pyClibrarymodule.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// splicer begin include
// splicer end include

// splicer begin C_definition
// splicer end C_definition
PyObject *PY_error_obj;
// splicer begin additional_functions
// splicer end additional_functions

static char PY_function1__doc__[] =
"documentation"
;

static PyObject *
PY_function1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void Function1()
// splicer begin function.function1
    Function1();
    Py_RETURN_NONE;
// splicer end function.function1
}

static char PY_function2__doc__[] =
"documentation"
;

static PyObject *
PY_function2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// double Function2(double arg1 +intent(in)+value, int arg2 +intent(in)+value)
// splicer begin function.function2
    double arg1;
    int arg2;
    char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "di:Function2",
        SHT_kwlist,
        &arg1, &arg2))
    {
        return NULL;
    }
    double SHT_rv = Function2(arg1, arg2);
    PyObject * SHTPy_rv = PyFloat_FromDouble(SHT_rv);
    return (PyObject *) SHTPy_rv;
// splicer end function.function2
}

static char PY_sum__doc__[] =
"documentation"
;

static PyObject *
PY_sum(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void Sum(int len +implied(size(values))+intent(in)+value, int * values +dimension(:)+intent(in), int * result +intent(out))
// splicer begin function.sum
    PyObject * SHPy_values;
    PyArrayObject * SHAPy_values = NULL;
    char *SHT_kwlist[] = {
        "values",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:Sum", SHT_kwlist,
        &SHPy_values))
    {
        return NULL;
    }
    SHAPy_values = (PyArrayObject *) PyArray_FROM_OTF(SHPy_values,
        NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (SHAPy_values == NULL) {
        PyErr_SetString(PyExc_ValueError, "values must be a 1-D array of int");
        goto fail;
    }
    int * values = PyArray_DATA(SHAPy_values);
    int result;  // intent(out)
    int len = PyArray_SIZE(SHAPy_values);
    Sum(len, values, &result);
    PyObject * SHPy_result = PyInt_FromLong(result);
    Py_DECREF(SHAPy_values);
    return (PyObject *) SHPy_result;

fail:
    Py_XDECREF(SHAPy_values);
    return NULL;
// splicer end function.sum
}

static char PY_function3__doc__[] =
"documentation"
;

static PyObject *
PY_function3(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// bool Function3(bool arg +intent(in)+value)
// splicer begin function.function3
    PyObject * SHPy_arg;
    char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:Function3",
        SHT_kwlist,
        &PyBool_Type, &SHPy_arg))
    {
        return NULL;
    }
    bool arg = PyObject_IsTrue(SHPy_arg);
    bool SHT_rv = Function3(arg);
    PyObject * SHTPy_rv = PyBool_FromLong(SHT_rv);
    return (PyObject *) SHTPy_rv;
// splicer end function.function3
}

static char PY_function3b__doc__[] =
"documentation"
;

static PyObject *
PY_function3b(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void Function3b(const bool arg1 +intent(in)+value, bool * arg2 +intent(out), bool * arg3 +intent(inout))
// splicer begin function.function3b
    PyObject * SHPy_arg1;
    PyObject * SHPy_arg3;
    char *SHT_kwlist[] = {
        "arg1",
        "arg3",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!:Function3b",
        SHT_kwlist,
        &PyBool_Type, &SHPy_arg1, &PyBool_Type, &SHPy_arg3))
    {
        return NULL;
    }
    bool arg1 = PyObject_IsTrue(SHPy_arg1);
    bool arg3 = PyObject_IsTrue(SHPy_arg3);
    bool arg2;  // intent(out)
    Function3b(arg1, &arg2, &arg3);
    PyObject * SHPy_arg2 = PyBool_FromLong(arg2);
    SHPy_arg3 = PyBool_FromLong(arg3);
    PyObject * SHTPy_rv = Py_BuildValue("OO", SHPy_arg2, SHPy_arg3);
    return SHTPy_rv;
// splicer end function.function3b
}

static char PY_function4a__doc__[] =
"documentation"
;

static PyObject *
PY_function4a(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// char * Function4a +len(30)(const char * arg1 +intent(in), const char * arg2 +intent(in))
// splicer begin function.function4a
    const char * arg1;
    const char * arg2;
    char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss:Function4a",
        SHT_kwlist,
        &arg1, &arg2))
    {
        return NULL;
    }
    char * SHT_rv = Function4a(arg1, arg2);
    PyObject * SHTPy_rv = PyString_FromString(SHT_rv);
    return (PyObject *) SHTPy_rv;
// splicer end function.function4a
}

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
    char *SHT_kwlist[] = {
        "argin",
        "arginout",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii:intargs",
        SHT_kwlist,
        &argin, &arginout))
    {
        return NULL;
    }
    int argout;  // intent(out)
    intargs(argin, &arginout, &argout);
    PyObject * SHTPy_rv = Py_BuildValue("ii", arginout, argout);
    return SHTPy_rv;
// splicer end function.intargs
}
static PyMethodDef PY_methods[] = {
{"Function1", (PyCFunction)PY_function1, METH_NOARGS,
    PY_function1__doc__},
{"Function2", (PyCFunction)PY_function2, METH_VARARGS|METH_KEYWORDS,
    PY_function2__doc__},
{"Sum", (PyCFunction)PY_sum, METH_VARARGS|METH_KEYWORDS, PY_sum__doc__},
{"Function3", (PyCFunction)PY_function3, METH_VARARGS|METH_KEYWORDS,
    PY_function3__doc__},
{"Function3b", (PyCFunction)PY_function3b, METH_VARARGS|METH_KEYWORDS,
    PY_function3b__doc__},
{"Function4a", (PyCFunction)PY_function4a, METH_VARARGS|METH_KEYWORDS,
    PY_function4a__doc__},
{"intargs", (PyCFunction)PY_intargs, METH_VARARGS|METH_KEYWORDS,
    PY_intargs__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

/*
 * initclibrary - Initialization function for the module
 * *must* be called initclibrary
 */
static char PY__doc__[] =
"library documentation"
;

struct module_state {
    PyObject *error;
};

#ifdef IS_PY3K
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#ifdef IS_PY3K
static int clibrary_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int clibrary_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "clibrary", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
    clibrary_traverse, /* m_traverse */
    clibrary_clear, /* m_clear */
    NULL  /* m_free */
};

#define RETVAL m
#define INITERROR return NULL
#else
#define RETVAL
#define INITERROR return
#endif

#ifdef __cplusplus
extern "C" {
#endif
PyMODINIT_FUNC
MOD_INITBASIS(void)
{
    PyObject *m = NULL;
    const char * error_name = "clibrary.Error";

// splicer begin C_init_locals
// splicer end C_init_locals


    /* Create the module and add the functions */
#ifdef IS_PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("clibrary", PY_methods,
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
        Py_FatalError("can't initialize module clibrary");
    return RETVAL;
}
#ifdef __cplusplus
}
#endif

