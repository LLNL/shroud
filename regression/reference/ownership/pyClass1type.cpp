// pyClass1type.cpp
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
#include "ownership.hpp"
// splicer begin class.Class1.impl.include
// splicer end class.Class1.impl.include

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
// splicer begin class.Class1.impl.C_definition
// splicer end class.Class1.impl.C_definition
// splicer begin class.Class1.impl.additional_methods
// splicer end class.Class1.impl.additional_methods
static void
PY_Class1_tp_del (PY_Class1 *self)
{
// splicer begin class.Class1.type.del
    if (self->dtor != NULL) {
         self->dtor->dtor(static_cast<void *>(self->obj));
    }
    self->obj = NULL;
// splicer end class.Class1.type.del
}
// splicer begin class.Class1.impl.after_methods
// splicer end class.Class1.impl.after_methods

static PyObject *PY_Class1_flag_getter(PY_Class1 *self,
    void *SHROUD_UNUSED(closure))
{
    PyObject * rv = PyInt_FromLong(self->obj->m_flag);
    return rv;
}

static PyGetSetDef PY_Class1_getset[] = {
    {(char *)"flag", (getter)PY_Class1_flag_getter, (setter)NULL,
        NULL, NULL},
    // splicer begin class.Class1.PyGetSetDef
    // splicer end class.Class1.PyGetSetDef
    {NULL}            /* sentinel */
};
static PyMethodDef PY_Class1_methods[] = {
    // splicer begin class.Class1.PyMethodDef
    // splicer end class.Class1.PyMethodDef
    {NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

static char Class1__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_Class1_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ownership.Class1",                       /* tp_name */
    sizeof(PY_Class1),         /* tp_basicsize */
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
    Class1__doc__,         /* tp_doc */
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
    PY_Class1_methods,                             /* tp_methods */
    0,                              /* tp_members */
    PY_Class1_getset,                             /* tp_getset */
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
    (destructor)PY_Class1_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)0,                  /* tp_finalize */
#endif
};
