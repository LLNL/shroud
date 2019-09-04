// pyCstruct1type.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pynsmodule.hpp"
#include "namespace.hpp"
// splicer begin namespace.outer.class.Cstruct1.impl.include
// splicer end namespace.outer.class.Cstruct1.impl.include

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
// splicer begin namespace.outer.class.Cstruct1.impl.C_definition
// splicer end namespace.outer.class.Cstruct1.impl.C_definition
// splicer begin namespace.outer.class.Cstruct1.impl.additional_methods
// splicer end namespace.outer.class.Cstruct1.impl.additional_methods
static void
PY_Cstruct1_tp_del (PY_Cstruct1 *self)
{
// splicer begin namespace.outer.class.Cstruct1.type.del
    PY_SHROUD_release_memory(self->idtor, self->obj);
    self->obj = NULL;
// splicer end namespace.outer.class.Cstruct1.type.del
}

static int
PY_Cstruct1_tp_init(
  PY_Cstruct1 *self,
  PyObject *args,
  PyObject *kwds)
{
// Cstruct1(int ifield +intent(in), double dfield +intent(in)) +name(Cstruct1_ctor)
// splicer begin namespace.outer.class.Cstruct1.method.cstruct1_ctor
    int ifield;
    double dfield;
    const char *SHT_kwlist[] = {
        "ifield",
        "dfield",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "id:Cstruct1_ctor",
        const_cast<char **>(SHT_kwlist), &ifield, &dfield))
        return -1;

    self->obj = new outer::Cstruct1;
    if (self->obj == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->idtor = 1;
    // initialize fields
    outer::Cstruct1 *SH_obj = self->obj;
    SH_obj->ifield = ifield;
    SH_obj->dfield = dfield;
    return 0;
// splicer end namespace.outer.class.Cstruct1.method.cstruct1_ctor
}
// splicer begin namespace.outer.class.Cstruct1.impl.after_methods
// splicer end namespace.outer.class.Cstruct1.impl.after_methods

static PyObject *PY_Cstruct1_ifield_getter(PY_Cstruct1 *self,
    void *SHROUD_UNUSED(closure))
{
    PyObject * rv = PyInt_FromLong(self->obj->ifield);
    return rv;
}

static int PY_Cstruct1_ifield_setter(PY_Cstruct1 *self, PyObject *value,
    void *SHROUD_UNUSED(closure))
{
    int rv = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
        return -1;
    }
    self->obj->ifield = rv;
    return 0;
}

static PyObject *PY_Cstruct1_dfield_getter(PY_Cstruct1 *self,
    void *SHROUD_UNUSED(closure))
{
    PyObject * rv = PyFloat_FromDouble(self->obj->dfield);
    return rv;
}

static int PY_Cstruct1_dfield_setter(PY_Cstruct1 *self, PyObject *value,
    void *SHROUD_UNUSED(closure))
{
    double rv = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }
    self->obj->dfield = rv;
    return 0;
}

static PyGetSetDef PY_Cstruct1_getset[] = {
    {(char *)"ifield", (getter)PY_Cstruct1_ifield_getter,
        (setter)PY_Cstruct1_ifield_setter, NULL, NULL},
    {(char *)"dfield", (getter)PY_Cstruct1_dfield_getter,
        (setter)PY_Cstruct1_dfield_setter, NULL, NULL},
    // splicer begin namespace.outer.class.Cstruct1.PyGetSetDef
    // splicer end namespace.outer.class.Cstruct1.PyGetSetDef
    {NULL}            /* sentinel */
};
static PyMethodDef PY_Cstruct1_methods[] = {
    // splicer begin namespace.outer.class.Cstruct1.PyMethodDef
    // splicer end namespace.outer.class.Cstruct1.PyMethodDef
    {NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

static char Cstruct1__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_Cstruct1_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ns.outer.Cstruct1",                       /* tp_name */
    sizeof(PY_Cstruct1),         /* tp_basicsize */
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
    Cstruct1__doc__,         /* tp_doc */
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
    PY_Cstruct1_methods,                             /* tp_methods */
    0,                              /* tp_members */
    PY_Cstruct1_getset,                             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    (descrgetfunc)0,                /* tp_descr_get */
    (descrsetfunc)0,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PY_Cstruct1_tp_init,                   /* tp_init */
    (allocfunc)0,                  /* tp_alloc */
    (newfunc)0,                    /* tp_new */
    (freefunc)0,                   /* tp_free */
    (inquiry)0,                     /* tp_is_gc */
    0,                              /* tp_bases */
    0,                              /* tp_mro */
    0,                              /* tp_cache */
    0,                              /* tp_subclasses */
    0,                              /* tp_weaklist */
    (destructor)PY_Cstruct1_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)0,                  /* tp_finalize */
#endif
};
