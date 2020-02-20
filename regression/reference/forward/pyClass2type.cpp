// pyClass2type.cpp
// This is generated code, do not edit
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
    self->obj = NULL;
// splicer end class.Class2.type.del
}

static int
PY_Class2_tp_init(
  PY_Class2 *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// Class2()
// splicer begin class.Class2.method.ctor
    self->obj = new forward::Class2();
    if (self->obj == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->idtor = 1;
    return 0;
// splicer end class.Class2.method.ctor
}

static char PY_func1__doc__[] =
"documentation"
;

static PyObject *
PY_func1(
  PY_Class2 *self,
  PyObject *args,
  PyObject *kwds)
{
// void func1(tutorial::Class1 * arg +intent(in))
// splicer begin class.Class2.method.func1
    TUT_Class1 arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:func1",
        const_cast<char **>(SHT_kwlist), &arg))
        return NULL;

    // post_parse
    tutorial::Class1 * SH_arg = SHPy_arg ? SHPy_arg->obj : NULL;

    self->obj->func1(SH_arg);
    Py_RETURN_NONE;
// splicer end class.Class2.method.func1
}

static char PY_acceptClass3__doc__[] =
"documentation"
;

static PyObject *
PY_acceptClass3(
  PY_Class2 *self,
  PyObject *args,
  PyObject *kwds)
{
// void acceptClass3(Class3 * arg +intent(in))
// splicer begin class.Class2.method.accept_class3
    PY_Class3 * SHPy_arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:acceptClass3",
        const_cast<char **>(SHT_kwlist), &PY_Class3_Type, &SHPy_arg))
        return NULL;

    // post_parse
    forward::Class3 * arg = SHPy_arg ? SHPy_arg->obj : NULL;

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
    {NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

static char Class2__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_Class2_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "forward.Class2",                       /* tp_name */
    sizeof(PY_Class2),         /* tp_basicsize */
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
    Class2__doc__,         /* tp_doc */
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
    PY_Class2_methods,                             /* tp_methods */
    0,                              /* tp_members */
    0,                             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    (descrgetfunc)0,                /* tp_descr_get */
    (descrsetfunc)0,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PY_Class2_tp_init,                   /* tp_init */
    (allocfunc)0,                  /* tp_alloc */
    (newfunc)0,                    /* tp_new */
    (freefunc)0,                   /* tp_free */
    (inquiry)0,                     /* tp_is_gc */
    0,                              /* tp_bases */
    0,                              /* tp_mro */
    0,                              /* tp_cache */
    0,                              /* tp_subclasses */
    0,                              /* tp_weaklist */
    (destructor)PY_Class2_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)0,                  /* tp_finalize */
#endif
};
