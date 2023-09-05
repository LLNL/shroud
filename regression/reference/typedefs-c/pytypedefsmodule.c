// pytypedefsmodule.c
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pytypedefsmodule.h"
#define PY_ARRAY_UNIQUE_SYMBOL SHROUD_TYPEDEFS_ARRAY_API
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
PyArray_Descr *PY_Struct1Rename_array_descr;
// splicer begin additional_functions
// splicer end additional_functions

// ----------------------------------------
// Function:  TypeID typefunc
// Attrs:     +intent(function)
// Exact:     py_function_native_scalar
// ----------------------------------------
// Argument:  TypeID arg +value
// Attrs:     +intent(in)
// Exact:     py_in_native_scalar
static char PY_typefunc__doc__[] =
"documentation"
;

static PyObject *
PY_typefunc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.typefunc
    int arg;
    char *SHT_kwlist[] = {
        "arg",
        NULL };
    PyObject * SHTPy_rv = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:typefunc",
        SHT_kwlist, &arg))
        return NULL;

    TypeID SHCXX_rv = typefunc(arg);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.typefunc
}

// ----------------------------------------
// Function:  void typestruct
// Attrs:     +intent(subroutine)
// Exact:     py_default
// ----------------------------------------
// Argument:  Struct1Rename * arg1
// Attrs:     +intent(inout)
// Exact:     py_inout_struct_*_list
static char PY_typestruct__doc__[] =
"documentation"
;

static PyObject *
PY_typestruct(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.typestruct
    char *SHT_kwlist[] = {
        "arg1",
        NULL };
    PyObject * SHPy_arg1 = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:typestruct",
        SHT_kwlist, &arg1))
        return NULL;

    typestruct(&arg1);

    // post_call
    SHPy_arg1 = Py_BuildValue("O", arg1);

    return (PyObject *) SHPy_arg1;
// splicer end function.typestruct
}
static PyMethodDef PY_methods[] = {
{"typefunc", (PyCFunction)PY_typefunc, METH_VARARGS|METH_KEYWORDS,
    PY_typefunc__doc__},
{"typestruct", (PyCFunction)PY_typestruct, METH_VARARGS|METH_KEYWORDS,
    PY_typestruct__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

// start PY_Struct1Rename_create_array_descr
// Create PyArray_Descr for Struct1Rename
static PyArray_Descr *PY_Struct1Rename_create_array_descr(void)
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

    // i
    obj = PyString_FromString("i");
    if (obj == NULL) goto fail;
    PyList_SET_ITEM(lnames, 0, obj);
    obj = (PyObject *) PyArray_DescrFromType(NPY_INT);
    if (obj == NULL) goto fail;
    PyList_SET_ITEM(ldescr, 0, obj);

    // d
    obj = PyString_FromString("d");
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
// end PY_Struct1Rename_create_array_descr

/*
 * inittypedefs - Initialization function for the module
 * *must* be called inittypedefs
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
static int typedefs_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int typedefs_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "typedefs", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
    typedefs_traverse, /* m_traverse */
    typedefs_clear, /* m_clear */
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
PyInit_typedefs(void)
#else
inittypedefs(void)
#endif
{
    PyObject *m = NULL;
    const char * error_name = "typedefs.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("typedefs", PY_methods,
        PY__doc__,
        (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

    import_array();

    // Define PyArray_Descr for structs
    PY_Struct1Rename_array_descr = PY_Struct1Rename_create_array_descr();
    PyModule_AddObject(m, "Struct1Rename_dtype", 
        (PyObject *) PY_Struct1Rename_array_descr);

    PY_error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
    if (PY_error_obj == NULL)
        return RETVAL;
    st->error = PY_error_obj;
    PyModule_AddObject(m, "Error", st->error);

    // splicer begin C_init_body
    // splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module typedefs");
    return RETVAL;
}

