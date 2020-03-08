// pyCstruct_ptrtype.cpp
// This file is generated by Shroud 0.11.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pystructmodule.hpp"
// splicer begin class.Cstruct_ptr.impl.include
// splicer end class.Cstruct_ptr.impl.include

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

// Keep track of the PyObject and pointer to the data it contains.
typedef struct {
    PyObject *obj;
    void *data;   // points into obj.
} SHROUD_converter_value;

// Helper - converter to PyObject to char *.
static int SHROUD_get_from_object_char(PyObject *obj,
    SHROUD_converter_value *value)
{
    char *out;
    if (PyUnicode_Check(obj))
    {
#if PY_MAJOR_VERSION >= 3
        PyObject *strobj = PyUnicode_AsUTF8String(obj);
        out = PyBytes_AS_STRING(strobj);
        value->obj = strobj;  // steal reference
#else
        PyObject *strobj = PyUnicode_AsUTF8String(obj);
        out = PyString_AsString(strobj);
        value->obj = strobj;  // steal reference
#endif
#if PY_MAJOR_VERSION >= 3
    } else if (PyByteArray_Check(obj)) {
        out = PyBytes_AS_STRING(obj);
        value->obj = obj;
        Py_INCREF(obj);
#else
    } else if (PyString_Check(obj)) {
        out = PyString_AsString(obj);
        value->obj = obj;
        Py_INCREF(obj);
#endif
    } else if (obj == Py_None) {
        out = NULL;
        value->obj = NULL;
    } else {
        PyErr_SetString(PyExc_ValueError, "argument must be a string");
        return 0;
    }
    value->data = out;
    return 1;
}

// splicer begin class.Cstruct_ptr.impl.C_definition
// splicer end class.Cstruct_ptr.impl.C_definition
// splicer begin class.Cstruct_ptr.impl.additional_methods
// splicer end class.Cstruct_ptr.impl.additional_methods
static void
PY_Cstruct_ptr_tp_del (PY_Cstruct_ptr *self)
{
// splicer begin class.Cstruct_ptr.type.del
    PY_SHROUD_release_memory(self->idtor, self->obj);
    self->obj = nullptr;
    // Python objects for members.
    Py_XDECREF(self->cfield_obj);
// splicer end class.Cstruct_ptr.type.del
}

static int
PY_Cstruct_ptr_tp_init(
  PY_Cstruct_ptr *self,
  PyObject *args,
  PyObject *kwds)
{
// Cstruct_ptr(char * cfield +intent(in)+optional(0)) +name(Cstruct_ptr_ctor)
// splicer begin class.Cstruct_ptr.method.cstruct_ptr_ctor
    SHROUD_converter_value SHPy_cfield;
    const char *SHT_kwlist[] = {
        "cfield",
        nullptr };

    SHPy_cfield.obj = nullptr;
    SHPy_cfield.data = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&:Cstruct_ptr_ctor",
        const_cast<char **>(SHT_kwlist), SHROUD_get_from_object_char,
        &SHPy_cfield))
        return -1;

    self->obj = new Cstruct_ptr;
    if (self->obj == nullptr) {
        PyErr_NoMemory();
        return -1;
    }
    self->idtor = 2;
    // initialize fields
    Cstruct_ptr *SH_obj = self->obj;
    SH_obj->cfield = static_cast<char *>(SHPy_cfield.data);
    self->cfield_obj = SHPy_cfield.obj;  // steal reference
    return 0;
// splicer end class.Cstruct_ptr.method.cstruct_ptr_ctor
}
// splicer begin class.Cstruct_ptr.impl.after_methods
// splicer end class.Cstruct_ptr.impl.after_methods

static PyObject *PY_Cstruct_ptr_cfield_getter(PY_Cstruct_ptr *self,
    void *SHROUD_UNUSED(closure))
{
    if (self->obj->cfield == nullptr) {
        Py_RETURN_NONE;
    }
    if (self->cfield_obj != nullptr) {
        Py_INCREF(self->cfield_obj);
        return self->cfield_obj;
    }
    PyObject * rv = PyString_FromString(self->obj->cfield);
    return rv;
}

static int PY_Cstruct_ptr_cfield_setter(PY_Cstruct_ptr *self, PyObject *value,
    void *SHROUD_UNUSED(closure))
{
    SHROUD_converter_value cvalue;
    Py_XDECREF(self->cfield_obj);
    if (SHROUD_get_from_object_char(value, &cvalue) == 0) {
        self->obj->cfield = NULL;
        self->cfield_obj = NULL;
        // XXXX set error
        return -1;
    }
    self->obj->cfield = static_cast<char *>(cvalue.data);
    self->cfield_obj = cvalue.obj;  // steal reference
    return 0;
}

static PyGetSetDef PY_Cstruct_ptr_getset[] = {
    {(char *)"cfield", (getter)PY_Cstruct_ptr_cfield_getter,
        (setter)PY_Cstruct_ptr_cfield_setter, nullptr, nullptr},
    // splicer begin class.Cstruct_ptr.PyGetSetDef
    // splicer end class.Cstruct_ptr.PyGetSetDef
    {nullptr}            /* sentinel */
};
static PyMethodDef PY_Cstruct_ptr_methods[] = {
    // splicer begin class.Cstruct_ptr.PyMethodDef
    // splicer end class.Cstruct_ptr.PyMethodDef
    {nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
};

static char Cstruct_ptr__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_Cstruct_ptr_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "cstruct.Cstruct_ptr",                       /* tp_name */
    sizeof(PY_Cstruct_ptr),         /* tp_basicsize */
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
    Cstruct_ptr__doc__,         /* tp_doc */
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
    PY_Cstruct_ptr_methods,                             /* tp_methods */
    nullptr,                              /* tp_members */
    PY_Cstruct_ptr_getset,                             /* tp_getset */
    nullptr,                              /* tp_base */
    nullptr,                              /* tp_dict */
    (descrgetfunc)nullptr,                /* tp_descr_get */
    (descrsetfunc)nullptr,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PY_Cstruct_ptr_tp_init,                   /* tp_init */
    (allocfunc)nullptr,                  /* tp_alloc */
    (newfunc)nullptr,                    /* tp_new */
    (freefunc)nullptr,                   /* tp_free */
    (inquiry)nullptr,                     /* tp_is_gc */
    nullptr,                              /* tp_bases */
    nullptr,                              /* tp_mro */
    nullptr,                              /* tp_cache */
    nullptr,                              /* tp_subclasses */
    nullptr,                              /* tp_weaklist */
    (destructor)PY_Cstruct_ptr_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)nullptr,                  /* tp_finalize */
#endif
};