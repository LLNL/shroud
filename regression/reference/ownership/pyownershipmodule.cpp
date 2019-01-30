// pyownershipmodule.cpp
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
#include "pyownershipmodule.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "ownership.hpp"

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

static char PY_ReturnIntPtrScalar__doc__[] =
"documentation"
;

static PyObject *
PY_ReturnIntPtrScalar(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// int * ReturnIntPtrScalar() +deref(scalar)
// splicer begin function.return_int_ptr_scalar
    int * SHC_rv = ReturnIntPtrScalar();

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(*SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.return_int_ptr_scalar
}

static char PY_ReturnIntPtrPointer__doc__[] =
"documentation"
;

static PyObject *
PY_ReturnIntPtrPointer(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// int * ReturnIntPtrPointer() +deref(pointer)
// splicer begin function.return_int_ptr_pointer
    int * SHC_rv = ReturnIntPtrPointer();

    // post_call
    PyObject * SHTPy_rv = PyArray_SimpleNewFromData(0, NULL, NPY_INT,
        SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.return_int_ptr_pointer
}

static char PY_ReturnIntPtrDimPointer__doc__[] =
"documentation"
;

static PyObject *
PY_ReturnIntPtrDimPointer(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// int * ReturnIntPtrDimPointer(int * len +hidden+intent(out)) +deref(pointer)+dimension(len)
// splicer begin function.return_int_ptr_dim_pointer
    // pre_call
    int len;  // intent(out)

    int * SHC_rv = ReturnIntPtrDimPointer(&len);

    // post_call
    npy_intp SHD_ReturnIntPtrDimPointer[1] = { len };
    PyObject * SHTPy_rv = PyArray_SimpleNewFromData(1,
        SHD_ReturnIntPtrDimPointer, NPY_INT, SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.return_int_ptr_dim_pointer
}

static char PY_ReturnIntPtrDimAlloc__doc__[] =
"documentation"
;

static PyObject *
PY_ReturnIntPtrDimAlloc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// int * ReturnIntPtrDimAlloc(int * len +hidden+intent(out)) +deref(allocatable)+dimension(len)
// splicer begin function.return_int_ptr_dim_alloc
    // pre_call
    int len;  // intent(out)

    int * SHC_rv = ReturnIntPtrDimAlloc(&len);

    // post_call
    npy_intp SHD_ReturnIntPtrDimAlloc[1] = { len };
    PyObject * SHTPy_rv = PyArray_SimpleNewFromData(1,
        SHD_ReturnIntPtrDimAlloc, NPY_INT, SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.return_int_ptr_dim_alloc
}

static char PY_ReturnIntPtrDimDefault__doc__[] =
"documentation"
;

static PyObject *
PY_ReturnIntPtrDimDefault(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// int * ReturnIntPtrDimDefault(int * len +hidden+intent(out)) +dimension(len)
// splicer begin function.return_int_ptr_dim_default
    // pre_call
    int len;  // intent(out)

    int * SHC_rv = ReturnIntPtrDimDefault(&len);

    // post_call
    npy_intp SHD_ReturnIntPtrDimDefault[1] = { len };
    PyObject * SHTPy_rv = PyArray_SimpleNewFromData(1,
        SHD_ReturnIntPtrDimDefault, NPY_INT, SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.return_int_ptr_dim_default
}

static char PY_ReturnIntPtrDimPointerNew__doc__[] =
"documentation"
;

static PyObject *
PY_ReturnIntPtrDimPointerNew(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// int * ReturnIntPtrDimPointerNew(int * len +hidden+intent(out)) +deref(pointer)+dimension(len)+owner(caller)
// splicer begin function.return_int_ptr_dim_pointer_new
    // pre_call
    int len;  // intent(out)

    int * SHC_rv = ReturnIntPtrDimPointerNew(&len);

    // post_call
    npy_intp SHD_ReturnIntPtrDimPointerNew[1] = { len };
    PyObject * SHTPy_rv = PyArray_SimpleNewFromData(1,
        SHD_ReturnIntPtrDimPointerNew, NPY_INT, SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.return_int_ptr_dim_pointer_new
}

static char PY_ReturnIntPtrDimAllocNew__doc__[] =
"documentation"
;

static PyObject *
PY_ReturnIntPtrDimAllocNew(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// int * ReturnIntPtrDimAllocNew(int * len +hidden+intent(out)) +deref(allocatable)+dimension(len)+owner(caller)
// splicer begin function.return_int_ptr_dim_alloc_new
    // pre_call
    int len;  // intent(out)

    int * SHC_rv = ReturnIntPtrDimAllocNew(&len);

    // post_call
    npy_intp SHD_ReturnIntPtrDimAllocNew[1] = { len };
    PyObject * SHTPy_rv = PyArray_SimpleNewFromData(1,
        SHD_ReturnIntPtrDimAllocNew, NPY_INT, SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.return_int_ptr_dim_alloc_new
}

static char PY_ReturnIntPtrDimDefaultNew__doc__[] =
"documentation"
;

static PyObject *
PY_ReturnIntPtrDimDefaultNew(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// int * ReturnIntPtrDimDefaultNew(int * len +hidden+intent(out)) +dimension(len)+owner(caller)
// splicer begin function.return_int_ptr_dim_default_new
    // pre_call
    int len;  // intent(out)

    int * SHC_rv = ReturnIntPtrDimDefaultNew(&len);

    // post_call
    npy_intp SHD_ReturnIntPtrDimDefaultNew[1] = { len };
    PyObject * SHTPy_rv = PyArray_SimpleNewFromData(1,
        SHD_ReturnIntPtrDimDefaultNew, NPY_INT, SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.return_int_ptr_dim_default_new
}

static char PY_createClassStatic__doc__[] =
"documentation"
;

static PyObject *
PY_createClassStatic(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void createClassStatic(int flag +intent(in)+value)
// splicer begin function.create_class_static
    int flag;
    const char *SHT_kwlist[] = {
        "flag",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:createClassStatic",
        const_cast<char **>(SHT_kwlist), &flag))
        return NULL;

    createClassStatic(flag);
    Py_RETURN_NONE;
// splicer end function.create_class_static
}

static char PY_getClassStatic__doc__[] =
"documentation"
;

static PyObject *
PY_getClassStatic(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// Class1 * getClassStatic() +owner(library)
// splicer begin function.get_class_static
    Class1 * SHCXX_rv = getClassStatic();

    // post_call
    PY_Class1 * SHTPy_rv = PyObject_New(PY_Class1, &PY_Class1_Type);
    SHTPy_rv->obj = SHCXX_rv;

    return (PyObject *) SHTPy_rv;
// splicer end function.get_class_static
}

static char PY_getClassNew__doc__[] =
"documentation"
;

/**
 * \brief Return pointer to new Class1 instance.
 *
 */
static PyObject *
PY_getClassNew(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// Class1 * getClassNew(int flag +intent(in)+value) +owner(caller)
// splicer begin function.get_class_new
    int flag;
    const char *SHT_kwlist[] = {
        "flag",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:getClassNew",
        const_cast<char **>(SHT_kwlist), &flag))
        return NULL;

    Class1 * SHCXX_rv = getClassNew(flag);

    // post_call
    PY_Class1 * SHTPy_rv = PyObject_New(PY_Class1, &PY_Class1_Type);
    SHTPy_rv->obj = SHCXX_rv;

    return (PyObject *) SHTPy_rv;
// splicer end function.get_class_new
}
static PyMethodDef PY_methods[] = {
{"ReturnIntPtrScalar", (PyCFunction)PY_ReturnIntPtrScalar, METH_NOARGS,
    PY_ReturnIntPtrScalar__doc__},
{"ReturnIntPtrPointer", (PyCFunction)PY_ReturnIntPtrPointer,
    METH_NOARGS, PY_ReturnIntPtrPointer__doc__},
{"ReturnIntPtrDimPointer", (PyCFunction)PY_ReturnIntPtrDimPointer,
    METH_NOARGS, PY_ReturnIntPtrDimPointer__doc__},
{"ReturnIntPtrDimAlloc", (PyCFunction)PY_ReturnIntPtrDimAlloc,
    METH_NOARGS, PY_ReturnIntPtrDimAlloc__doc__},
{"ReturnIntPtrDimDefault", (PyCFunction)PY_ReturnIntPtrDimDefault,
    METH_NOARGS, PY_ReturnIntPtrDimDefault__doc__},
{"ReturnIntPtrDimPointerNew", (PyCFunction)PY_ReturnIntPtrDimPointerNew,
    METH_NOARGS, PY_ReturnIntPtrDimPointerNew__doc__},
{"ReturnIntPtrDimAllocNew", (PyCFunction)PY_ReturnIntPtrDimAllocNew,
    METH_NOARGS, PY_ReturnIntPtrDimAllocNew__doc__},
{"ReturnIntPtrDimDefaultNew", (PyCFunction)PY_ReturnIntPtrDimDefaultNew,
    METH_NOARGS, PY_ReturnIntPtrDimDefaultNew__doc__},
{"createClassStatic", (PyCFunction)PY_createClassStatic,
    METH_VARARGS|METH_KEYWORDS, PY_createClassStatic__doc__},
{"getClassStatic", (PyCFunction)PY_getClassStatic, METH_NOARGS,
    PY_getClassStatic__doc__},
{"getClassNew", (PyCFunction)PY_getClassNew, METH_VARARGS|METH_KEYWORDS,
    PY_getClassNew__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

/*
 * initownership - Initialization function for the module
 * *must* be called initownership
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
static int ownership_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int ownership_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ownership", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
    ownership_traverse, /* m_traverse */
    ownership_clear, /* m_clear */
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
PyInit_ownership(void)
#else
initownership(void)
#endif
{
    PyObject *m = NULL;
    const char * error_name = "ownership.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("ownership", PY_methods,
        PY__doc__,
        (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

    import_array();

    // Class1
    PY_Class1_Type.tp_new   = PyType_GenericNew;
    PY_Class1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Class1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Class1_Type);
    PyModule_AddObject(m, "Class1", (PyObject *)&PY_Class1_Type);


    PY_error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
    if (PY_error_obj == NULL)
        return RETVAL;
    st->error = PY_error_obj;
    PyModule_AddObject(m, "Error", st->error);

    // splicer begin C_init_body
    // splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module ownership");
    return RETVAL;
}

