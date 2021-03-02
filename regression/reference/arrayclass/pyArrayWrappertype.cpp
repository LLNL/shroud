// pyArrayWrappertype.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pyarrayclassmodule.hpp"
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SHROUD_ARRAYCLASS_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// splicer begin class.ArrayWrapper.impl.include
// splicer end class.ArrayWrapper.impl.include

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
// splicer begin class.ArrayWrapper.impl.C_definition
// splicer end class.ArrayWrapper.impl.C_definition
// splicer begin class.ArrayWrapper.impl.additional_methods
// splicer end class.ArrayWrapper.impl.additional_methods
static void
PY_ArrayWrapper_tp_del (PY_ArrayWrapper *self)
{
// splicer begin class.ArrayWrapper.type.del
    PY_SHROUD_release_memory(self->idtor, self->obj);
    self->obj = nullptr;
// splicer end class.ArrayWrapper.type.del
}

// ----------------------------------------
// Function:  ArrayWrapper
// Attrs:     +intent(result)
// Exact:     py_default
static int
PY_ArrayWrapper_tp_init(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.ctor
    self->obj = new ArrayWrapper();
    if (self->obj == nullptr) {
        PyErr_NoMemory();
        return -1;
    }
    self->idtor = 1;
    return 0;
// splicer end class.ArrayWrapper.method.ctor
}

// ----------------------------------------
// Function:  void setSize
// Exact:     py_default
// ----------------------------------------
// Argument:  int size +value
// Attrs:     +intent(in)
// Requested: py_native_scalar_in
// Match:     py_default
static char PY_setSize__doc__[] =
"documentation"
;

static PyObject *
PY_setSize(
  PY_ArrayWrapper *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin class.ArrayWrapper.method.set_size
    int size;
    const char *SHT_kwlist[] = {
        "size",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:setSize",
        const_cast<char **>(SHT_kwlist), &size))
        return nullptr;

    self->obj->setSize(size);
    Py_RETURN_NONE;
// splicer end class.ArrayWrapper.method.set_size
}

// ----------------------------------------
// Function:  int getSize
// Attrs:     +intent(result)
// Requested: py_native_scalar_result
// Match:     py_default
static char PY_getSize__doc__[] =
"documentation"
;

static PyObject *
PY_getSize(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.get_size
    PyObject * SHTPy_rv = nullptr;

    int SHCXX_rv = self->obj->getSize();

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end class.ArrayWrapper.method.get_size
}

// ----------------------------------------
// Function:  void fillSize
// Exact:     py_default
// ----------------------------------------
// Argument:  int & size +intent(out)
// Attrs:     +intent(out)
// Exact:     py_native_&_out
static char PY_fillSize__doc__[] =
"documentation"
;

static PyObject *
PY_fillSize(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.fill_size
    int size;
    PyObject * SHPy_size = nullptr;

    self->obj->fillSize(size);

    // post_call
    SHPy_size = PyInt_FromLong(size);

    return (PyObject *) SHPy_size;
// splicer end class.ArrayWrapper.method.fill_size
}

// ----------------------------------------
// Function:  void allocate
// Exact:     py_default
static char PY_allocate__doc__[] =
"documentation"
;

static PyObject *
PY_allocate(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.allocate
    self->obj->allocate();
    Py_RETURN_NONE;
// splicer end class.ArrayWrapper.method.allocate
}

// ----------------------------------------
// Function:  double * getArray +deref(pointer)+dimension(getSize())
// Attrs:     +deref(pointer)+intent(result)
// Exact:     py_native_*_result_pointer_numpy
static char PY_getArray__doc__[] =
"documentation"
;

static PyObject *
PY_getArray(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.get_array
    npy_intp SHD_rv[1];
    PyObject * SHTPy_rv = nullptr;

    double * SHCXX_rv = self->obj->getArray();

    // post_call
    SHD_rv[0] = self->obj->getSize();
    SHTPy_rv = PyArray_SimpleNewFromData(1, SHD_rv, NPY_DOUBLE,
        SHCXX_rv);
    if (SHTPy_rv == nullptr) goto fail;

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return nullptr;
// splicer end class.ArrayWrapper.method.get_array
}

// ----------------------------------------
// Function:  double * getArrayConst +deref(pointer)+dimension(getSize())
// Attrs:     +deref(pointer)+intent(result)
// Exact:     py_native_*_result_pointer_numpy
static char PY_getArrayConst__doc__[] =
"documentation"
;

static PyObject *
PY_getArrayConst(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.get_array_const
    npy_intp SHD_rv[1];
    PyObject * SHTPy_rv = nullptr;

    double * SHCXX_rv = self->obj->getArrayConst();

    // post_call
    SHD_rv[0] = self->obj->getSize();
    SHTPy_rv = PyArray_SimpleNewFromData(1, SHD_rv, NPY_DOUBLE,
        SHCXX_rv);
    if (SHTPy_rv == nullptr) goto fail;

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return nullptr;
// splicer end class.ArrayWrapper.method.get_array_const
}

// ----------------------------------------
// Function:  const double * getArrayC +deref(pointer)+dimension(getSize())
// Attrs:     +deref(pointer)+intent(result)
// Exact:     py_native_*_result_pointer_numpy
static char PY_getArrayC__doc__[] =
"documentation"
;

static PyObject *
PY_getArrayC(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.get_array_c
    npy_intp SHD_rv[1];
    PyObject * SHTPy_rv = nullptr;

    const double * SHCXX_rv = self->obj->getArrayC();

    // post_call
    SHD_rv[0] = self->obj->getSize();
    SHTPy_rv = PyArray_SimpleNewFromData(1, SHD_rv, NPY_DOUBLE,
        const_cast<double *>(SHCXX_rv));
    if (SHTPy_rv == nullptr) goto fail;

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return nullptr;
// splicer end class.ArrayWrapper.method.get_array_c
}

// ----------------------------------------
// Function:  const double * getArrayConstC +deref(pointer)+dimension(getSize())
// Attrs:     +deref(pointer)+intent(result)
// Exact:     py_native_*_result_pointer_numpy
static char PY_getArrayConstC__doc__[] =
"documentation"
;

static PyObject *
PY_getArrayConstC(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.get_array_const_c
    npy_intp SHD_rv[1];
    PyObject * SHTPy_rv = nullptr;

    const double * SHCXX_rv = self->obj->getArrayConstC();

    // post_call
    SHD_rv[0] = self->obj->getSize();
    SHTPy_rv = PyArray_SimpleNewFromData(1, SHD_rv, NPY_DOUBLE,
        const_cast<double *>(SHCXX_rv));
    if (SHTPy_rv == nullptr) goto fail;

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return nullptr;
// splicer end class.ArrayWrapper.method.get_array_const_c
}

// ----------------------------------------
// Function:  void fetchArrayPtr
// Exact:     py_default
// ----------------------------------------
// Argument:  double * * array +deref(pointer)+dimension(isize)+intent(out)
// Attrs:     +deref(pointer)+intent(out)
// Exact:     py_native_**_out_pointer_numpy
// ----------------------------------------
// Argument:  int * isize +hidden
// Attrs:     +intent(inout)
// Exact:     py_native_*_inout
static char PY_fetchArrayPtr__doc__[] =
"documentation"
;

static PyObject *
PY_fetchArrayPtr(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.fetch_array_ptr
    double *array;
    npy_intp SHD_array[1];
    PyObject *SHPy_array = nullptr;
    int isize;

    self->obj->fetchArrayPtr(&array, &isize);

    // post_call
    SHD_array[0] = isize;
    SHPy_array = PyArray_SimpleNewFromData(1, SHD_array, NPY_DOUBLE,
        array);
    if (SHPy_array == nullptr) goto fail;

    return (PyObject *) SHPy_array;

fail:
    Py_XDECREF(SHPy_array);
    return nullptr;
// splicer end class.ArrayWrapper.method.fetch_array_ptr
}

// ----------------------------------------
// Function:  void fetchArrayRef
// Exact:     py_default
// ----------------------------------------
// Argument:  double * & array +deref(pointer)+dimension(isize)+intent(out)
// Attrs:     +deref(pointer)+intent(out)
// Exact:     py_native_*&_out_pointer_numpy
// ----------------------------------------
// Argument:  int & isize +hidden
// Attrs:     +intent(inout)
// Exact:     py_native_&_inout
static char PY_fetchArrayRef__doc__[] =
"documentation"
;

static PyObject *
PY_fetchArrayRef(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.fetch_array_ref
    double *array;
    npy_intp SHD_array[1];
    PyObject *SHPy_array = nullptr;
    int isize;

    self->obj->fetchArrayRef(array, isize);

    // post_call
    SHD_array[0] = isize;
    SHPy_array = PyArray_SimpleNewFromData(1, SHD_array, NPY_DOUBLE,
        array);
    if (SHPy_array == nullptr) goto fail;

    return (PyObject *) SHPy_array;

fail:
    Py_XDECREF(SHPy_array);
    return nullptr;
// splicer end class.ArrayWrapper.method.fetch_array_ref
}

// ----------------------------------------
// Function:  void fetchArrayPtrConst
// Exact:     py_default
// ----------------------------------------
// Argument:  const double * * array +deref(pointer)+dimension(isize)+intent(out)
// Attrs:     +deref(pointer)+intent(out)
// Exact:     py_native_**_out_pointer_numpy
// ----------------------------------------
// Argument:  int * isize +hidden
// Attrs:     +intent(inout)
// Exact:     py_native_*_inout
static char PY_fetchArrayPtrConst__doc__[] =
"documentation"
;

static PyObject *
PY_fetchArrayPtrConst(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.fetch_array_ptr_const
    const double *array;
    npy_intp SHD_array[1];
    PyObject *SHPy_array = nullptr;
    int isize;

    self->obj->fetchArrayPtrConst(&array, &isize);

    // post_call
    SHD_array[0] = isize;
    SHPy_array = PyArray_SimpleNewFromData(1, SHD_array, NPY_DOUBLE,
        const_cast<double *>(array));
    if (SHPy_array == nullptr) goto fail;

    return (PyObject *) SHPy_array;

fail:
    Py_XDECREF(SHPy_array);
    return nullptr;
// splicer end class.ArrayWrapper.method.fetch_array_ptr_const
}

// ----------------------------------------
// Function:  void fetchArrayRefConst
// Exact:     py_default
// ----------------------------------------
// Argument:  const double * & array +deref(pointer)+dimension(isize)+intent(out)
// Attrs:     +deref(pointer)+intent(out)
// Exact:     py_native_*&_out_pointer_numpy
// ----------------------------------------
// Argument:  int & isize +hidden
// Attrs:     +intent(inout)
// Exact:     py_native_&_inout
static char PY_fetchArrayRefConst__doc__[] =
"documentation"
;

static PyObject *
PY_fetchArrayRefConst(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.fetch_array_ref_const
    const double *array;
    npy_intp SHD_array[1];
    PyObject *SHPy_array = nullptr;
    int isize;

    self->obj->fetchArrayRefConst(array, isize);

    // post_call
    SHD_array[0] = isize;
    SHPy_array = PyArray_SimpleNewFromData(1, SHD_array, NPY_DOUBLE,
        const_cast<double *>(array));
    if (SHPy_array == nullptr) goto fail;

    return (PyObject *) SHPy_array;

fail:
    Py_XDECREF(SHPy_array);
    return nullptr;
// splicer end class.ArrayWrapper.method.fetch_array_ref_const
}

// ----------------------------------------
// Function:  void fetchVoidPtr
// Exact:     py_default
// ----------------------------------------
// Argument:  void * * array +intent(out)
// Attrs:     +intent(out)
// Exact:     py_void_**_out
static char PY_fetchVoidPtr__doc__[] =
"documentation"
;

static PyObject *
PY_fetchVoidPtr(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.fetch_void_ptr
    void *array;
    PyObject * SHPy_array = nullptr;

    self->obj->fetchVoidPtr(&array);

    // post_call
    SHPy_array = PyCapsule_New(array, NULL, NULL);

    return (PyObject *) SHPy_array;
// splicer end class.ArrayWrapper.method.fetch_void_ptr
}

// ----------------------------------------
// Function:  void fetchVoidRef
// Exact:     py_default
// ----------------------------------------
// Argument:  void * & array +intent(out)
// Attrs:     +intent(out)
// Exact:     py_void_*&_out
static char PY_fetchVoidRef__doc__[] =
"documentation"
;

static PyObject *
PY_fetchVoidRef(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.fetch_void_ref
    void *array;
    PyObject * SHPy_array = nullptr;

    self->obj->fetchVoidRef(array);

    // post_call
    SHPy_array = PyCapsule_New(array, NULL, NULL);

    return (PyObject *) SHPy_array;
// splicer end class.ArrayWrapper.method.fetch_void_ref
}

// ----------------------------------------
// Function:  bool checkPtr
// Attrs:     +intent(result)
// Requested: py_bool_scalar_result
// Match:     py_bool_result
// ----------------------------------------
// Argument:  void * array +value
// Attrs:     +intent(in)
// Exact:     py_void_*_in
static char PY_checkPtr__doc__[] =
"documentation"
;

static PyObject *
PY_checkPtr(
  PY_ArrayWrapper *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin class.ArrayWrapper.method.check_ptr
    void * array;
    PyObject *SHPy_array;
    const char *SHT_kwlist[] = {
        "array",
        nullptr };
    bool SHCXX_rv;
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:checkPtr",
        const_cast<char **>(SHT_kwlist), &SHPy_array))
        return nullptr;

    // post_parse
    array = PyCapsule_GetPointer(SHPy_array, NULL);
    if (PyErr_Occurred())
        goto fail;

    SHCXX_rv = self->obj->checkPtr(array);

    // post_call
    SHTPy_rv = PyBool_FromLong(SHCXX_rv);
    if (SHTPy_rv == nullptr) goto fail;

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return nullptr;
// splicer end class.ArrayWrapper.method.check_ptr
}

// ----------------------------------------
// Function:  double sumArray
// Attrs:     +intent(result)
// Requested: py_native_scalar_result
// Match:     py_default
static char PY_sumArray__doc__[] =
"documentation"
;

static PyObject *
PY_sumArray(
  PY_ArrayWrapper *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.ArrayWrapper.method.sum_array
    PyObject * SHTPy_rv = nullptr;

    double SHCXX_rv = self->obj->sumArray();

    // post_call
    SHTPy_rv = PyFloat_FromDouble(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end class.ArrayWrapper.method.sum_array
}
// splicer begin class.ArrayWrapper.impl.after_methods
// splicer end class.ArrayWrapper.impl.after_methods
static PyMethodDef PY_ArrayWrapper_methods[] = {
    {"setSize", (PyCFunction)PY_setSize, METH_VARARGS|METH_KEYWORDS,
        PY_setSize__doc__},
    {"getSize", (PyCFunction)PY_getSize, METH_NOARGS,
        PY_getSize__doc__},
    {"fillSize", (PyCFunction)PY_fillSize, METH_NOARGS,
        PY_fillSize__doc__},
    {"allocate", (PyCFunction)PY_allocate, METH_NOARGS,
        PY_allocate__doc__},
    {"getArray", (PyCFunction)PY_getArray, METH_NOARGS,
        PY_getArray__doc__},
    {"getArrayConst", (PyCFunction)PY_getArrayConst, METH_NOARGS,
        PY_getArrayConst__doc__},
    {"getArrayC", (PyCFunction)PY_getArrayC, METH_NOARGS,
        PY_getArrayC__doc__},
    {"getArrayConstC", (PyCFunction)PY_getArrayConstC, METH_NOARGS,
        PY_getArrayConstC__doc__},
    {"fetchArrayPtr", (PyCFunction)PY_fetchArrayPtr, METH_NOARGS,
        PY_fetchArrayPtr__doc__},
    {"fetchArrayRef", (PyCFunction)PY_fetchArrayRef, METH_NOARGS,
        PY_fetchArrayRef__doc__},
    {"fetchArrayPtrConst", (PyCFunction)PY_fetchArrayPtrConst,
        METH_NOARGS, PY_fetchArrayPtrConst__doc__},
    {"fetchArrayRefConst", (PyCFunction)PY_fetchArrayRefConst,
        METH_NOARGS, PY_fetchArrayRefConst__doc__},
    {"fetchVoidPtr", (PyCFunction)PY_fetchVoidPtr, METH_NOARGS,
        PY_fetchVoidPtr__doc__},
    {"fetchVoidRef", (PyCFunction)PY_fetchVoidRef, METH_NOARGS,
        PY_fetchVoidRef__doc__},
    {"checkPtr", (PyCFunction)PY_checkPtr, METH_VARARGS|METH_KEYWORDS,
        PY_checkPtr__doc__},
    {"sumArray", (PyCFunction)PY_sumArray, METH_NOARGS,
        PY_sumArray__doc__},
    // splicer begin class.ArrayWrapper.PyMethodDef
    // splicer end class.ArrayWrapper.PyMethodDef
    {nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
};

static char ArrayWrapper__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_ArrayWrapper_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "arrayclass.ArrayWrapper",                       /* tp_name */
    sizeof(PY_ArrayWrapper),         /* tp_basicsize */
    0,                              /* tp_itemsize */
    /* Methods to implement standard operations */
    (destructor)nullptr,                 /* tp_dealloc */
    (printfunc)nullptr,                   /* tp_print */
    (getattrfunc)nullptr,                 /* tp_getattr */
    (setattrfunc)nullptr,                 /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
    nullptr,                               /* tp_reserved */
#else
    (cmpfunc)nullptr,                     /* tp_compare */
#endif
    (reprfunc)nullptr,                    /* tp_repr */
    /* Method suites for standard classes */
    nullptr,                              /* tp_as_number */
    nullptr,                              /* tp_as_sequence */
    nullptr,                              /* tp_as_mapping */
    /* More standard operations (here for binary compatibility) */
    (hashfunc)nullptr,                    /* tp_hash */
    (ternaryfunc)nullptr,                 /* tp_call */
    (reprfunc)nullptr,                    /* tp_str */
    (getattrofunc)nullptr,                /* tp_getattro */
    (setattrofunc)nullptr,                /* tp_setattro */
    /* Functions to access object as input/output buffer */
    nullptr,                              /* tp_as_buffer */
    /* Flags to define presence of optional/expanded features */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    ArrayWrapper__doc__,         /* tp_doc */
    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    (traverseproc)nullptr,                /* tp_traverse */
    /* delete references to contained objects */
    (inquiry)nullptr,                     /* tp_clear */
    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    (richcmpfunc)nullptr,                 /* tp_richcompare */
    /* weak reference enabler */
    0,                              /* tp_weaklistoffset */
    /* Added in release 2.2 */
    /* Iterators */
    (getiterfunc)nullptr,                 /* tp_iter */
    (iternextfunc)nullptr,                /* tp_iternext */
    /* Attribute descriptor and subclassing stuff */
    PY_ArrayWrapper_methods,                             /* tp_methods */
    nullptr,                              /* tp_members */
    nullptr,                             /* tp_getset */
    nullptr,                              /* tp_base */
    nullptr,                              /* tp_dict */
    (descrgetfunc)nullptr,                /* tp_descr_get */
    (descrsetfunc)nullptr,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PY_ArrayWrapper_tp_init,                   /* tp_init */
    (allocfunc)nullptr,                  /* tp_alloc */
    (newfunc)nullptr,                    /* tp_new */
    (freefunc)nullptr,                   /* tp_free */
    (inquiry)nullptr,                     /* tp_is_gc */
    nullptr,                              /* tp_bases */
    nullptr,                              /* tp_mro */
    nullptr,                              /* tp_cache */
    nullptr,                              /* tp_subclasses */
    nullptr,                              /* tp_weaklist */
    (destructor)PY_ArrayWrapper_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)nullptr,                  /* tp_finalize */
#endif
};
