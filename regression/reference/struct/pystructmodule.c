// pystructmodule.c
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pystructmodule.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
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
PyArray_Descr *PY_Cstruct1_array_descr;
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
    PyObject * SHTPy_arg = NULL;
    PyArrayObject * SHPy_arg = NULL;
    char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:passStructByValue",
        SHT_kwlist, &SHTPy_arg))
        return NULL;

    // post_parse
    Py_INCREF(PY_Cstruct1_array_descr);
    SHPy_arg = (PyArrayObject *) PyArray_FromAny(SHTPy_arg,
        PY_Cstruct1_array_descr, 0, 1, NPY_ARRAY_IN_ARRAY, NULL);
    if (SHPy_arg == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "arg must be a 1-D array of Cstruct1");
        goto fail;
    }

    // pre_call
    Cstruct1 * arg = PyArray_DATA(SHPy_arg);

    int SHC_rv = passStructByValue(*arg);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    // cleanup
    Py_DECREF(SHPy_arg);

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHPy_arg);
    return NULL;
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
    PyObject * SHTPy_arg = NULL;
    PyArrayObject * SHPy_arg = NULL;
    char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:passStruct1",
        SHT_kwlist, &SHTPy_arg))
        return NULL;

    // post_parse
    Py_INCREF(PY_Cstruct1_array_descr);
    SHPy_arg = (PyArrayObject *) PyArray_FromAny(SHTPy_arg,
        PY_Cstruct1_array_descr, 0, 1, NPY_ARRAY_IN_ARRAY, NULL);
    if (SHPy_arg == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "arg must be a 1-D array of Cstruct1");
        goto fail;
    }

    // pre_call
    Cstruct1 * arg = PyArray_DATA(SHPy_arg);

    int SHC_rv = passStruct1(arg);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    // cleanup
    Py_DECREF(SHPy_arg);

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHPy_arg);
    return NULL;
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
    PyObject * SHTPy_s1 = NULL;
    PyArrayObject * SHPy_s1 = NULL;
    char *SHT_kwlist[] = {
        "s1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:passStruct2",
        SHT_kwlist, &SHTPy_s1))
        return NULL;

    // post_parse
    Py_INCREF(PY_Cstruct1_array_descr);
    SHPy_s1 = (PyArrayObject *) PyArray_FromAny(SHTPy_s1,
        PY_Cstruct1_array_descr, 0, 1, NPY_ARRAY_IN_ARRAY, NULL);
    if (SHPy_s1 == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "s1 must be a 1-D array of Cstruct1");
        goto fail;
    }

    // pre_call
    Cstruct1 * s1 = PyArray_DATA(SHPy_s1);
    char outbuf[LENOUTBUF];  // intent(out)

    int SHC_rv = passStruct2(s1, outbuf);

    // post_call
    PyObject * SHTPy_rv = Py_BuildValue("is", SHC_rv, outbuf);

    // cleanup
    Py_DECREF(SHPy_s1);

    return SHTPy_rv;

fail:
    Py_XDECREF(SHPy_s1);
    return NULL;
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
    PyArrayObject * SHPy_arg = NULL;
    int i;
    double d;
    char *SHT_kwlist[] = {
        "i",
        "d",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "id:acceptStructOutPtr", SHT_kwlist, &i, &d))
        return NULL;

    // post_parse
    Py_INCREF(PY_Cstruct1_array_descr);
    SHPy_arg = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type,
        PY_Cstruct1_array_descr, 0, NULL, NULL, NULL, 0, NULL);
    if (SHPy_arg == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "arg must be a 1-D array of Cstruct1");
        goto fail;
    }

    // pre_call
    Cstruct1 *arg = PyArray_DATA(SHPy_arg);

    acceptStructOutPtr(arg, i, d);
    return (PyObject *) SHPy_arg;

fail:
    Py_XDECREF(SHPy_arg);
    return NULL;
// splicer end function.accept_struct_out_ptr
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
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

// Create PyArray_Descr for Cstruct1
static PyArray_Descr *PY_Cstruct1_create_array_descr(void)
{
    int ierr;
    PyObject *obj = NULL;
    PyObject * lnames = NULL;
    PyObject * ldescr = NULL;
    PyObject * dict = NULL;
    PyArray_Descr *dtype = NULL;

    lnames = PyList_New(2);
    if (lnames == NULL) goto fail;
    ldescr = PyList_New(2);
    if (ldescr == NULL) goto fail;

    // ifield
    obj = PyString_FromString("ifield");
    if (obj == NULL) goto fail;
    PyList_SET_ITEM(lnames, 0, obj);
    obj = (PyObject *) PyArray_DescrFromType(NPY_INT);
    if (obj == NULL) goto fail;
    PyList_SET_ITEM(ldescr, 0, obj);

    // dfield
    obj = PyString_FromString("dfield");
    if (obj == NULL) goto fail;
    PyList_SET_ITEM(lnames, 1, obj);
    obj = (PyObject *) PyArray_DescrFromType(NPY_DOUBLE);
    if (obj == NULL) goto fail;
    PyList_SET_ITEM(ldescr, 1, obj);
    obj = NULL;

    dict = PyDict_New();
    if (dict == NULL) goto fail;
    ierr = PyDict_SetItemString(dict, "names", lnames);
    if (ierr == -1) goto fail;
    lnames = NULL;
    ierr = PyDict_SetItemString(dict, "formats", ldescr);
    if (ierr == -1) goto fail;
    ldescr = NULL;
    ierr = PyArray_DescrAlignConverter(dict, &dtype);
    if (ierr == 0) goto fail;
    return dtype;
fail:
    Py_XDECREF(obj);
    if (lnames != NULL) {
        for (int i=0; i < 2; i++) {
            Py_XDECREF(PyList_GET_ITEM(lnames, i));
        }
        Py_DECREF(lnames);
    }
    if (ldescr != NULL) {
        for (int i=0; i < 2; i++) {
            Py_XDECREF(PyList_GET_ITEM(ldescr, i));
        }
        Py_DECREF(ldescr);
    }
    Py_XDECREF(dict);
    Py_XDECREF(dtype);
    return NULL;
}

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

PyMODINIT_FUNC
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

    import_array();

    // Define PyArray_Descr for structs
    PY_Cstruct1_array_descr = PY_Cstruct1_create_array_descr();
    PyModule_AddObject(m, "Cstruct1_dtype", 
        (PyObject *) PY_Cstruct1_array_descr);

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

