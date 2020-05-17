// pyClass2type.cpp
// This file is generated by Shroud 0.11.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pyforwardmodule.hpp"
// splicer begin class.Class2.impl.include
// splicer end class.Class2.impl.include

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
// splicer begin class.Class2.impl.C_definition
// splicer end class.Class2.impl.C_definition
// splicer begin class.Class2.impl.additional_methods
// splicer end class.Class2.impl.additional_methods
static void
PY_Class2_tp_del (PY_Class2 *self)
{
// splicer begin class.Class2.type.del
    PY_SHROUD_release_memory(self->idtor, self->obj);
    self->obj = nullptr;
// splicer end class.Class2.type.del
}

// ----------------------------------------
// Function:  Class2
// Exact:     py_default
static int
PY_Class2_tp_init(
  PY_Class2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.Class2.method.ctor
    self->obj = new forward::Class2();
    if (self->obj == nullptr) {
        PyErr_NoMemory();
        return -1;
    }
    self->idtor = 1;
    return 0;
// splicer end class.Class2.method.ctor
}

// ----------------------------------------
// Function:  void func1
// Exact:     py_default
// ----------------------------------------
// Argument:  tutorial::Class1 * arg +intent(in)
// Requested: py_shadow_*_in
// Match:     py_shadow_in
static char PY_func1__doc__[] =
"documentation"
;

static PyObject *
PY_func1(
  PY_Class2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin class.Class2.method.func1
    TUT_Class1 arg;
    const char *SHT_kwlist[] = {
        "arg",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:func1",
        const_cast<char **>(SHT_kwlist), &arg))
        return nullptr;

    // post_declare
    tutorial::Class1 * SH_arg = SHPy_arg ? SHPy_arg->obj : nullptr;

    self->obj->func1(SH_arg);
    Py_RETURN_NONE;
// splicer end class.Class2.method.func1
}

// ----------------------------------------
// Function:  void acceptClass3
// Exact:     py_default
// ----------------------------------------
// Argument:  Class3 * arg +intent(in)
// Requested: py_shadow_*_in
// Match:     py_shadow_in
static char PY_acceptClass3__doc__[] =
"documentation"
;

static PyObject *
PY_acceptClass3(
  PY_Class2 *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin class.Class2.method.accept_class3
    PY_Class3 * SHPy_arg;
    const char *SHT_kwlist[] = {
        "arg",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:acceptClass3",
        const_cast<char **>(SHT_kwlist), &PY_Class3_Type, &SHPy_arg))
        return nullptr;

    // post_declare
    forward::Class3 * arg = SHPy_arg ? SHPy_arg->obj : nullptr;

    self->obj->acceptClass3(arg);
    Py_RETURN_NONE;
// splicer end class.Class2.method.accept_class3
}
// splicer begin class.Class2.impl.after_methods
// splicer end class.Class2.impl.after_methods
static PyMethodDef PY_Class2_methods[] = {
    {"func1", (PyCFunction)PY_func1, METH_VARARGS|METH_KEYWORDS,
        PY_func1__doc__},
    {"acceptClass3", (PyCFunction)PY_acceptClass3,
        METH_VARARGS|METH_KEYWORDS, PY_acceptClass3__doc__},
    // splicer begin class.Class2.PyMethodDef
    // splicer end class.Class2.PyMethodDef
    {nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
};

static char Class2__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_Class2_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "forward.Class2",                       /* tp_name */
    sizeof(PY_Class2),         /* tp_basicsize */
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
    Class2__doc__,         /* tp_doc */
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
    PY_Class2_methods,                             /* tp_methods */
    nullptr,                              /* tp_members */
    nullptr,                             /* tp_getset */
    nullptr,                              /* tp_base */
    nullptr,                              /* tp_dict */
    (descrgetfunc)nullptr,                /* tp_descr_get */
    (descrsetfunc)nullptr,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PY_Class2_tp_init,                   /* tp_init */
    (allocfunc)nullptr,                  /* tp_alloc */
    (newfunc)nullptr,                    /* tp_new */
    (freefunc)nullptr,                   /* tp_free */
    (inquiry)nullptr,                     /* tp_is_gc */
    nullptr,                              /* tp_bases */
    nullptr,                              /* tp_mro */
    nullptr,                              /* tp_cache */
    nullptr,                              /* tp_subclasses */
    nullptr,                              /* tp_weaklist */
    (destructor)PY_Class2_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)nullptr,                  /* tp_finalize */
#endif
};
