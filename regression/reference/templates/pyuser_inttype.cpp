// pyuser_inttype.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
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

// splicer begin class.user.impl.include
// splicer end class.user.impl.include

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
// splicer begin class.user.impl.C_definition
// splicer end class.user.impl.C_definition
// splicer begin class.user.impl.additional_methods
// splicer end class.user.impl.additional_methods
static void
PY_user_int_tp_del (PY_user_int *self)
{
// splicer begin class.user.type.del
    PY_SHROUD_release_memory(self->idtor, self->obj);
    self->obj = nullptr;
// splicer end class.user.type.del
}

// ----------------------------------------
// Function:  void nested
// Exact:     py_default
// ----------------------------------------
// Argument:  int arg1 +value
// Attrs:     +intent(in)
// Requested: py_native_scalar_in
// Match:     py_default
// ----------------------------------------
// Argument:  double arg2 +value
// Attrs:     +intent(in)
// Requested: py_native_scalar_in
// Match:     py_default
static char PY_nested_double__doc__[] =
"documentation"
;

static PyObject *
PY_nested_double(
  PY_user_int *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin class.user.method.nested_double
    int arg1;
    double arg2;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "id:nested",
        const_cast<char **>(SHT_kwlist), &arg1, &arg2))
        return nullptr;

    self->obj->nested<double>(arg1, arg2);
    Py_RETURN_NONE;
// splicer end class.user.method.nested_double
}
// splicer begin class.user.impl.after_methods
// splicer end class.user.impl.after_methods
static PyMethodDef PY_user_int_methods[] = {
    {"nested_double", (PyCFunction)PY_nested_double,
        METH_VARARGS|METH_KEYWORDS, PY_nested_double__doc__},
    // splicer begin class.user.PyMethodDef
    // splicer end class.user.PyMethodDef
    {nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
};

static char user_int__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_user_int_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "templates.user_int",                       /* tp_name */
    sizeof(PY_user_int),         /* tp_basicsize */
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
    user_int__doc__,         /* tp_doc */
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
    PY_user_int_methods,                             /* tp_methods */
    nullptr,                              /* tp_members */
    nullptr,                             /* tp_getset */
    nullptr,                              /* tp_base */
    nullptr,                              /* tp_dict */
    (descrgetfunc)nullptr,                /* tp_descr_get */
    (descrsetfunc)nullptr,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)0,                   /* tp_init */
    (allocfunc)nullptr,                  /* tp_alloc */
    (newfunc)nullptr,                    /* tp_new */
    (freefunc)nullptr,                   /* tp_free */
    (inquiry)nullptr,                     /* tp_is_gc */
    nullptr,                              /* tp_bases */
    nullptr,                              /* tp_mro */
    nullptr,                              /* tp_cache */
    nullptr,                              /* tp_subclasses */
    nullptr,                              /* tp_weaklist */
    (destructor)PY_user_int_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)nullptr,                  /* tp_finalize */
#endif
};
