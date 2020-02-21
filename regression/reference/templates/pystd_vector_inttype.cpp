// pystd_vector_inttype.cpp
// This file is generated by Shroud 0.11.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pytemplatesmodule.hpp"
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SHROUD_TEMPLATES_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
// splicer begin namespace.std.class.vector.impl.include
// splicer end namespace.std.class.vector.impl.include

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
// splicer begin namespace.std.class.vector.impl.C_definition
// splicer end namespace.std.class.vector.impl.C_definition
// splicer begin namespace.std.class.vector.impl.additional_methods
// splicer end namespace.std.class.vector.impl.additional_methods
static void
PY_vector_int_tp_del (PY_vector_int *self)
{
// splicer begin namespace.std.class.vector.type.del
    PY_SHROUD_release_memory(self->idtor, self->obj);
    self->obj = NULL;
// splicer end namespace.std.class.vector.type.del
}

static int
PY_vector_int_tp_init(
  PY_vector_int *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.std.class.vector.method.ctor
    self->obj = new std::vector<int>();
    if (self->obj == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->idtor = 1;
    return 0;
// splicer end namespace.std.class.vector.method.ctor
}

static char PY_push_back__doc__[] =
"documentation"
;

static PyObject *
PY_push_back(
  PY_vector_int *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.std.class.vector.method.push_back
    int value;
    const char *SHT_kwlist[] = {
        "value",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:push_back",
        const_cast<char **>(SHT_kwlist), &value))
        return NULL;
    self->obj->push_back(value);
    Py_RETURN_NONE;
// splicer end namespace.std.class.vector.method.push_back
}

static char PY_at__doc__[] =
"documentation"
;

static PyObject *
PY_at(
  PY_vector_int *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.std.class.vector.method.at
    size_t n;
    const char *SHT_kwlist[] = {
        "n",
        NULL };
    PyObject * SHTPy_rv = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n:at",
        const_cast<char **>(SHT_kwlist), &n))
        return NULL;
    int & SHCXX_rv = self->obj->at(n);
    SHTPy_rv = PyArray_SimpleNewFromData(0, NULL, NPY_INT, &SHCXX_rv);
    if (SHTPy_rv == NULL) goto fail;
    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return NULL;
// splicer end namespace.std.class.vector.method.at
}
// splicer begin namespace.std.class.vector.impl.after_methods
// splicer end namespace.std.class.vector.impl.after_methods
static PyMethodDef PY_vector_int_methods[] = {
    {"push_back", (PyCFunction)PY_push_back, METH_VARARGS|METH_KEYWORDS,
        PY_push_back__doc__},
    {"at", (PyCFunction)PY_at, METH_VARARGS|METH_KEYWORDS,
        PY_at__doc__},
    // splicer begin namespace.std.class.vector.PyMethodDef
    // splicer end namespace.std.class.vector.PyMethodDef
    {NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

static char vector_int__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_vector_int_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "templates.vector_int",                       /* tp_name */
    sizeof(PY_vector_int),         /* tp_basicsize */
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
    vector_int__doc__,         /* tp_doc */
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
    PY_vector_int_methods,                             /* tp_methods */
    nullptr,                              /* tp_members */
    nullptr,                             /* tp_getset */
    nullptr,                              /* tp_base */
    nullptr,                              /* tp_dict */
    (descrgetfunc)nullptr,                /* tp_descr_get */
    (descrsetfunc)nullptr,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PY_vector_int_tp_init,                   /* tp_init */
    (allocfunc)nullptr,                  /* tp_alloc */
    (newfunc)nullptr,                    /* tp_new */
    (freefunc)nullptr,                   /* tp_free */
    (inquiry)nullptr,                     /* tp_is_gc */
    nullptr,                              /* tp_bases */
    nullptr,                              /* tp_mro */
    nullptr,                              /* tp_cache */
    nullptr,                              /* tp_subclasses */
    nullptr,                              /* tp_weaklist */
    (destructor)PY_vector_int_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)nullptr,                  /* tp_finalize */
#endif
};
