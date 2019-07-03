// pystructmodule.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pystructmodule.hpp"
#include "struct.h"

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

static char PY_passStructByValue__doc__[] =
"documentation"
;

static PyObject *
PY_passStructByValue(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int passStructByValue(Cstruct1 arg +intent(in)+value)
// splicer begin function.pass_struct_by_value
    PY_Cstruct1 * SHPy_arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };
    PyObject * SHTPy_rv = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:passStructByValue",
        const_cast<char **>(SHT_kwlist), &PY_Cstruct1_Type, &SHPy_arg))
        return NULL;

    // post_parse
    Cstruct1 * arg = SHPy_arg ? SHPy_arg->obj : NULL;

    int rv = passStructByValue(*arg);

    // post_call
    SHTPy_rv = PyInt_FromLong(rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.pass_struct_by_value
}

static char PY_passStruct1__doc__[] =
"documentation"
;

static PyObject *
PY_passStruct1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int passStruct1(Cstruct1 * arg +intent(in))
// splicer begin function.pass_struct1
    PY_Cstruct1 * SHPy_arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };
    PyObject * SHTPy_rv = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:passStruct1",
        const_cast<char **>(SHT_kwlist), &PY_Cstruct1_Type, &SHPy_arg))
        return NULL;

    // post_parse
    Cstruct1 * arg = SHPy_arg ? SHPy_arg->obj : NULL;

    int rv = passStruct1(arg);

    // post_call
    SHTPy_rv = PyInt_FromLong(rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.pass_struct1
}

static char PY_passStruct2__doc__[] =
"documentation"
;

/**
 * Pass name argument which will build a bufferify function.
 */
static PyObject *
PY_passStruct2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int passStruct2(Cstruct1 * s1 +intent(in), char * outbuf +charlen(LENOUTBUF)+intent(out))
// splicer begin function.pass_struct2
    PY_Cstruct1 * SHPy_s1;
    const char *SHT_kwlist[] = {
        "s1",
        NULL };
    PyObject *SHTPy_rv = NULL;  // return value object

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:passStruct2",
        const_cast<char **>(SHT_kwlist), &PY_Cstruct1_Type, &SHPy_s1))
        return NULL;

    // post_parse
    Cstruct1 * s1 = SHPy_s1 ? SHPy_s1->obj : NULL;

    // pre_call
    char outbuf[LENOUTBUF];  // intent(out)

    int rv = passStruct2(s1, outbuf);

    // post_call
    SHTPy_rv = Py_BuildValue("is", rv, outbuf);

    return SHTPy_rv;
// splicer end function.pass_struct2
}

static char PY_acceptStructOutPtr__doc__[] =
"documentation"
;

/**
 * Pass name argument which will build a bufferify function.
 */
static PyObject *
PY_acceptStructOutPtr(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void acceptStructOutPtr(Cstruct1 * arg +intent(out), int i +intent(in)+value, double d +intent(in)+value)
// splicer begin function.accept_struct_out_ptr
    PY_Cstruct1 * SHPy_arg = NULL;
    int i;
    double d;
    const char *SHT_kwlist[] = {
        "i",
        "d",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "id:acceptStructOutPtr", const_cast<char **>(SHT_kwlist), &i,
        &d))
        return NULL;
    {
        // pre_call
        Cstruct1 * arg = new Cstruct1;

        acceptStructOutPtr(arg, i, d);

        // post_call
        SHPy_arg = PyObject_New(PY_Cstruct1, &PY_Cstruct1_Type);
        if (SHPy_arg == NULL) goto fail;
        SHPy_arg->obj = arg;
        SHPy_arg->idtor = 0;

        return (PyObject *) SHPy_arg;
    }

fail:
    Py_XDECREF(SHPy_arg);
    return NULL;
// splicer end function.accept_struct_out_ptr
}

static char PY_acceptStructInOutPtr__doc__[] =
"documentation"
;

static PyObject *
PY_acceptStructInOutPtr(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void acceptStructInOutPtr(Cstruct1 * arg +intent(inout))
// splicer begin function.accept_struct_in_out_ptr
    PY_Cstruct1 * SHPy_arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "O!:acceptStructInOutPtr", const_cast<char **>(SHT_kwlist), 
        &PY_Cstruct1_Type, &SHPy_arg))
        return NULL;

    // post_parse
    Cstruct1 * arg = SHPy_arg ? SHPy_arg->obj : NULL;

    acceptStructInOutPtr(arg);
    return (PyObject *) SHPy_arg;
// splicer end function.accept_struct_in_out_ptr
}

static char PY_returnStructByValue__doc__[] =
"documentation"
;

static PyObject *
PY_returnStructByValue(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// Cstruct1 returnStructByValue(int i +intent(in)+value, double d +intent(in)+value)
// splicer begin function.return_struct_by_value
    int i;
    double d;
    const char *SHT_kwlist[] = {
        "i",
        "d",
        NULL };
    Cstruct1 * rv = NULL;
    PY_Cstruct1 *SHTPy_rv = NULL;  // struct_result_class

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "id:returnStructByValue", const_cast<char **>(SHT_kwlist), &i,
        &d))
        return NULL;

    rv = new Cstruct1;
    if (rv == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    *rv = returnStructByValue(i, d);

    // post_call
    SHTPy_rv = PyObject_New(PY_Cstruct1, &PY_Cstruct1_Type);
    if (SHTPy_rv == NULL) goto fail;
    SHTPy_rv->obj = rv;
    SHTPy_rv->idtor = 1;

    return (PyObject *) SHTPy_rv;

fail:
    if (rv != NULL) {
        PY_SHROUD_release_memory(1, rv);
    }
    Py_XDECREF(SHTPy_rv);
    return NULL;
// splicer end function.return_struct_by_value
}

static char PY_returnStructPtr1__doc__[] =
"documentation"
;

/**
 * \brief Return a pointer to a struct
 *
 * Does not generate a bufferify C wrapper.
 */
static PyObject *
PY_returnStructPtr1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// Cstruct1 * returnStructPtr1(int i +intent(in)+value, double d +intent(in)+value)
// splicer begin function.return_struct_ptr1
    int i;
    double d;
    const char *SHT_kwlist[] = {
        "i",
        "d",
        NULL };
    PY_Cstruct1 *SHTPy_rv = NULL;  // struct_result_class

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "id:returnStructPtr1",
        const_cast<char **>(SHT_kwlist), &i, &d))
        return NULL;

    Cstruct1 * SHCXX_rv = returnStructPtr1(i, d);

    // post_call
    SHTPy_rv = PyObject_New(PY_Cstruct1, &PY_Cstruct1_Type);
    if (SHTPy_rv == NULL) goto fail;
    SHTPy_rv->obj = SHCXX_rv;
    SHTPy_rv->idtor = 0;

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return NULL;
// splicer end function.return_struct_ptr1
}

static char PY_returnStructPtr2__doc__[] =
"documentation"
;

/**
 * \brief Return a pointer to a struct
 *
 * Generates a bufferify C wrapper function.
 */
static PyObject *
PY_returnStructPtr2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// Cstruct1 * returnStructPtr2(int i +intent(in)+value, double d +intent(in)+value, char * outbuf +charlen(LENOUTBUF)+intent(out))
// splicer begin function.return_struct_ptr2
    int i;
    double d;
    const char *SHT_kwlist[] = {
        "i",
        "d",
        NULL };
    PY_Cstruct1 *SHTPy_rv = NULL;  // struct_result_class
    PyObject *SHPyResult = NULL;  // return value object

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "id:returnStructPtr2",
        const_cast<char **>(SHT_kwlist), &i, &d))
        return NULL;

    // pre_call
    char outbuf[LENOUTBUF];  // intent(out)

    Cstruct1 * SHCXX_rv = returnStructPtr2(i, d, outbuf);

    // post_call
    SHTPy_rv = PyObject_New(PY_Cstruct1, &PY_Cstruct1_Type);
    if (SHTPy_rv == NULL) goto fail;
    SHTPy_rv->obj = SHCXX_rv;
    SHTPy_rv->idtor = 0;
    SHPyResult = Py_BuildValue("Os", SHTPy_rv, outbuf);

    return SHPyResult;

fail:
    Py_XDECREF(SHTPy_rv);
    return NULL;
// splicer end function.return_struct_ptr2
}
static PyMethodDef PY_methods[] = {
{"passStructByValue", (PyCFunction)PY_passStructByValue,
    METH_VARARGS|METH_KEYWORDS, PY_passStructByValue__doc__},
{"passStruct1", (PyCFunction)PY_passStruct1, METH_VARARGS|METH_KEYWORDS,
    PY_passStruct1__doc__},
{"passStruct2", (PyCFunction)PY_passStruct2, METH_VARARGS|METH_KEYWORDS,
    PY_passStruct2__doc__},
{"acceptStructOutPtr", (PyCFunction)PY_acceptStructOutPtr,
    METH_VARARGS|METH_KEYWORDS, PY_acceptStructOutPtr__doc__},
{"acceptStructInOutPtr", (PyCFunction)PY_acceptStructInOutPtr,
    METH_VARARGS|METH_KEYWORDS, PY_acceptStructInOutPtr__doc__},
{"returnStructByValue", (PyCFunction)PY_returnStructByValue,
    METH_VARARGS|METH_KEYWORDS, PY_returnStructByValue__doc__},
{"returnStructPtr1", (PyCFunction)PY_returnStructPtr1,
    METH_VARARGS|METH_KEYWORDS, PY_returnStructPtr1__doc__},
{"returnStructPtr2", (PyCFunction)PY_returnStructPtr2,
    METH_VARARGS|METH_KEYWORDS, PY_returnStructPtr2__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

/*
 * initstruct - Initialization function for the module
 * *must* be called initstruct
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
static int struct_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int struct_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "struct", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
    struct_traverse, /* m_traverse */
    struct_clear, /* m_clear */
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
PyInit_cstruct(void)
#else
initcstruct(void)
#endif
{
    PyObject *m = NULL;
    const char * error_name = "struct.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("cstruct", PY_methods,
        PY__doc__,
        (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

    // Cstruct1
    PY_Cstruct1_Type.tp_new   = PyType_GenericNew;
    PY_Cstruct1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Cstruct1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Cstruct1_Type);
    PyModule_AddObject(m, "Cstruct1", (PyObject *)&PY_Cstruct1_Type);

    PY_error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
    if (PY_error_obj == NULL)
        return RETVAL;
    st->error = PY_error_obj;
    PyModule_AddObject(m, "Error", st->error);

    // splicer begin C_init_body
    // splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module cstruct");
    return RETVAL;
}

