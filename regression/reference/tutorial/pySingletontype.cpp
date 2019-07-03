// pySingletontype.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pyTutorialmodule.hpp"
#include "tutorial.hpp"
// splicer begin class.Singleton.impl.include
// splicer end class.Singleton.impl.include

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
// splicer begin class.Singleton.impl.C_definition
// splicer end class.Singleton.impl.C_definition
// splicer begin class.Singleton.impl.additional_methods
// splicer end class.Singleton.impl.additional_methods
static void
PY_Singleton_tp_del (PY_Singleton *self)
{
// splicer begin class.Singleton.type.del
    if (self->dtor != NULL) {
         self->dtor->dtor(static_cast<void *>(self->obj));
    }
    self->obj = NULL;
// splicer end class.Singleton.type.del
}

static char PY_singleton_getReference__doc__[] =
"documentation"
;

static PyObject *
PY_singleton_getReference(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// static Singleton & getReference()
// splicer begin class.Singleton.method.get_reference
    Singleton & SHCXX_rv = Singleton::getReference();

    // post_call
    PY_Singleton * SHTPy_rv =
        PyObject_New(PY_Singleton, &PY_Singleton_Type);
    SHTPy_rv->obj = &SHCXX_rv;

    return (PyObject *) SHTPy_rv;
// splicer end class.Singleton.method.get_reference
}
// splicer begin class.Singleton.impl.after_methods
// splicer end class.Singleton.impl.after_methods
static PyMethodDef PY_Singleton_methods[] = {
    {"getReference", (PyCFunction)PY_singleton_getReference,
        METH_STATIC|METH_NOARGS, PY_singleton_getReference__doc__},
    // splicer begin class.Singleton.PyMethodDef
    // splicer end class.Singleton.PyMethodDef
    {NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

static char Singleton__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_Singleton_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "tutorial.Singleton",                       /* tp_name */
    sizeof(PY_Singleton),         /* tp_basicsize */
    0,                              /* tp_itemsize */
    /* Methods to implement standard operations */
    (destructor)0,                 /* tp_dealloc */
    (printfunc)0,                   /* tp_print */
    (getattrfunc)0,                 /* tp_getattr */
    (setattrfunc)0,                 /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
    0,                               /* tp_reserved */
#else
    (cmpfunc)0,                     /* tp_compare */
#endif
    (reprfunc)0,                    /* tp_repr */
    /* Method suites for standard classes */
    0,                              /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    /* More standard operations (here for binary compatibility) */
    (hashfunc)0,                    /* tp_hash */
    (ternaryfunc)0,                 /* tp_call */
    (reprfunc)0,                    /* tp_str */
    (getattrofunc)0,                /* tp_getattro */
    (setattrofunc)0,                /* tp_setattro */
    /* Functions to access object as input/output buffer */
    0,                              /* tp_as_buffer */
    /* Flags to define presence of optional/expanded features */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    Singleton__doc__,         /* tp_doc */
    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    (traverseproc)0,                /* tp_traverse */
    /* delete references to contained objects */
    (inquiry)0,                     /* tp_clear */
    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    (richcmpfunc)0,                 /* tp_richcompare */
    /* weak reference enabler */
    0,                              /* tp_weaklistoffset */
    /* Added in release 2.2 */
    /* Iterators */
    (getiterfunc)0,                 /* tp_iter */
    (iternextfunc)0,                /* tp_iternext */
    /* Attribute descriptor and subclassing stuff */
    PY_Singleton_methods,                             /* tp_methods */
    0,                              /* tp_members */
    0,                             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    (descrgetfunc)0,                /* tp_descr_get */
    (descrsetfunc)0,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)0,                   /* tp_init */
    (allocfunc)0,                  /* tp_alloc */
    (newfunc)0,                    /* tp_new */
    (freefunc)0,                   /* tp_free */
    (inquiry)0,                     /* tp_is_gc */
    0,                              /* tp_bases */
    0,                              /* tp_mro */
    0,                              /* tp_cache */
    0,                              /* tp_subclasses */
    0,                              /* tp_weaklist */
    (destructor)PY_Singleton_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)0,                  /* tp_finalize */
#endif
};
