// pyCstruct_numpytype.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pystructmodule.hpp"
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SHROUD_STRUCT_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// splicer begin class.Cstruct_numpy.impl.include
// splicer end class.Cstruct_numpy.impl.include

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
// splicer begin class.Cstruct_numpy.impl.C_definition
// splicer end class.Cstruct_numpy.impl.C_definition
// splicer begin class.Cstruct_numpy.impl.additional_methods
// splicer end class.Cstruct_numpy.impl.additional_methods
static void
PY_Cstruct_numpy_tp_del (PY_Cstruct_numpy *self)
{
// splicer begin class.Cstruct_numpy.type.del
    PY_SHROUD_release_memory(self->idtor, self->obj);
    self->obj = nullptr;
    // Python objects for members.
    Py_XDECREF(self->ivalue_obj);
    Py_XDECREF(self->dvalue_obj);
    // Python objects for members.
    Py_XDECREF(self->ivalue_dataobj);
    Py_XDECREF(self->dvalue_dataobj);
// splicer end class.Cstruct_numpy.type.del
}

// ----------------------------------------
// Function:  Cstruct_numpy +name(Cstruct_numpy_ctor)
// Attrs:     +intent(ctor)
// Exact:     py_default
// ----------------------------------------
// Argument:  int nitems
// Attrs:     +intent(in)
// Requested: py_ctor_native_scalar_numpy
// Match:     py_ctor_native
// ----------------------------------------
// Argument:  int * ivalue +dimension(nitems)
// Attrs:     +intent(in)
// Requested: py_ctor_native_*_numpy
// Match:     py_ctor_native_*
// ----------------------------------------
// Argument:  double * dvalue +dimension(nitems)
// Attrs:     +intent(in)
// Requested: py_ctor_native_*_numpy
// Match:     py_ctor_native_*
static int
PY_Cstruct_numpy_tp_init(
  PY_Cstruct_numpy *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin class.Cstruct_numpy.method.Cstruct_numpy_ctor
    int nitems = 0;
    STR_SHROUD_converter_value SHValue_ivalue = {NULL, NULL, NULL, NULL, 0};
    SHValue_ivalue.name = "ivalue";
    STR_SHROUD_converter_value SHValue_dvalue = {NULL, NULL, NULL, NULL, 0};
    SHValue_dvalue.name = "dvalue";
    const char *SHT_kwlist[] = {
        "nitems",
        "ivalue",
        "dvalue",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "|iO&O&:Cstruct_numpy_ctor", const_cast<char **>(SHT_kwlist), 
        &nitems, STR_SHROUD_get_from_object_int_numpy, &SHValue_ivalue,
        STR_SHROUD_get_from_object_double_numpy, &SHValue_dvalue))
        return -1;

    self->obj = new Cstruct_numpy;
    if (self->obj == nullptr) {
        PyErr_NoMemory();
        return -1;
    }
    self->idtor = 4;

    // post_call - initialize fields
    Cstruct_numpy *SH_obj = self->obj;
    SH_obj->nitems = nitems;
    SH_obj->ivalue = static_cast<int *>(SHValue_ivalue.data);
    self->ivalue_obj = SHValue_ivalue.obj;  // steal reference
    SH_obj->dvalue = static_cast<double *>(SHValue_dvalue.data);
    self->dvalue_obj = SHValue_dvalue.obj;  // steal reference

    return 0;
// splicer end class.Cstruct_numpy.method.Cstruct_numpy_ctor
}
// splicer begin class.Cstruct_numpy.impl.after_methods
// splicer end class.Cstruct_numpy.impl.after_methods

// Requested: py_descr_native_scalar
// Match:     py_descr_native
static PyObject *PY_Cstruct_numpy_nitems_getter(PY_Cstruct_numpy *self,
    void *SHROUD_UNUSED(closure))
{
    PyObject * rv = PyInt_FromLong(self->obj->nitems);
    return rv;
}

// Requested: py_descr_native_scalar
// Match:     py_descr_native
static int PY_Cstruct_numpy_nitems_setter(PY_Cstruct_numpy *self, PyObject *value,
    void *SHROUD_UNUSED(closure))
{
    int rv = PyInt_AsLong(value);
    if (PyErr_Occurred()) {
        return -1;
    }
    self->obj->nitems = rv;
    return 0;
}

// Exact:     py_descr_native_*_numpy
static PyObject *PY_Cstruct_numpy_ivalue_getter(PY_Cstruct_numpy *self,
    void *SHROUD_UNUSED(closure))
{
    if (self->obj->ivalue == nullptr) {
        Py_RETURN_NONE;
    }
    if (self->ivalue_obj != nullptr) {
        Py_INCREF(self->ivalue_obj);
        return self->ivalue_obj;
    }
    npy_intp dims[1] = { self->obj->nitems };
    PyObject *rv = PyArray_SimpleNewFromData(1, dims, NPY_INT,
        self->obj->ivalue);
    if (rv != nullptr) {
        Py_INCREF(rv);
        self->ivalue_obj = rv;
    }
    return rv;
}

// Exact:     py_descr_native_*_numpy
static int PY_Cstruct_numpy_ivalue_setter(PY_Cstruct_numpy *self, PyObject *value,
    void *SHROUD_UNUSED(closure))
{
    STR_SHROUD_converter_value cvalue;
    Py_XDECREF(self->ivalue_obj);
    if (STR_SHROUD_get_from_object_int_numpy(value, &cvalue) == 0) {
        self->obj->ivalue = nullptr;
        self->ivalue_obj = nullptr;
        // XXXX set error
        return -1;
    }
    self->obj->ivalue = static_cast<int *>(cvalue.data);
    self->ivalue_obj = cvalue.obj;  // steal reference
    return 0;
}

// Exact:     py_descr_native_*_numpy
static PyObject *PY_Cstruct_numpy_dvalue_getter(PY_Cstruct_numpy *self,
    void *SHROUD_UNUSED(closure))
{
    if (self->obj->dvalue == nullptr) {
        Py_RETURN_NONE;
    }
    if (self->dvalue_obj != nullptr) {
        Py_INCREF(self->dvalue_obj);
        return self->dvalue_obj;
    }
    npy_intp dims[1] = { self->obj->nitems };
    PyObject *rv = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
        self->obj->dvalue);
    if (rv != nullptr) {
        Py_INCREF(rv);
        self->dvalue_obj = rv;
    }
    return rv;
}

// Exact:     py_descr_native_*_numpy
static int PY_Cstruct_numpy_dvalue_setter(PY_Cstruct_numpy *self, PyObject *value,
    void *SHROUD_UNUSED(closure))
{
    STR_SHROUD_converter_value cvalue;
    Py_XDECREF(self->dvalue_obj);
    if (STR_SHROUD_get_from_object_double_numpy(value, &cvalue) == 0) {
        self->obj->dvalue = nullptr;
        self->dvalue_obj = nullptr;
        // XXXX set error
        return -1;
    }
    self->obj->dvalue = static_cast<double *>(cvalue.data);
    self->dvalue_obj = cvalue.obj;  // steal reference
    return 0;
}

static PyGetSetDef PY_Cstruct_numpy_getset[] = {
    {(char *)"nitems", (getter)PY_Cstruct_numpy_nitems_getter,
        (setter)PY_Cstruct_numpy_nitems_setter, nullptr, nullptr},
    {(char *)"ivalue", (getter)PY_Cstruct_numpy_ivalue_getter,
        (setter)PY_Cstruct_numpy_ivalue_setter, nullptr, nullptr},
    {(char *)"dvalue", (getter)PY_Cstruct_numpy_dvalue_getter,
        (setter)PY_Cstruct_numpy_dvalue_setter, nullptr, nullptr},
    // splicer begin class.Cstruct_numpy.PyGetSetDef
    // splicer end class.Cstruct_numpy.PyGetSetDef
    {nullptr}            /* sentinel */
};
static PyMethodDef PY_Cstruct_numpy_methods[] = {
    // splicer begin class.Cstruct_numpy.PyMethodDef
    // splicer end class.Cstruct_numpy.PyMethodDef
    {nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
};

static char Cstruct_numpy__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_Cstruct_numpy_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "cstruct.Cstruct_numpy",                       /* tp_name */
    sizeof(PY_Cstruct_numpy),         /* tp_basicsize */
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
    Cstruct_numpy__doc__,         /* tp_doc */
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
    PY_Cstruct_numpy_methods,                             /* tp_methods */
    nullptr,                              /* tp_members */
    PY_Cstruct_numpy_getset,                             /* tp_getset */
    nullptr,                              /* tp_base */
    nullptr,                              /* tp_dict */
    (descrgetfunc)nullptr,                /* tp_descr_get */
    (descrsetfunc)nullptr,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PY_Cstruct_numpy_tp_init,                   /* tp_init */
    (allocfunc)nullptr,                  /* tp_alloc */
    (newfunc)nullptr,                    /* tp_new */
    (freefunc)nullptr,                   /* tp_free */
    (inquiry)nullptr,                     /* tp_is_gc */
    nullptr,                              /* tp_bases */
    nullptr,                              /* tp_mro */
    nullptr,                              /* tp_cache */
    nullptr,                              /* tp_subclasses */
    nullptr,                              /* tp_weaklist */
    (destructor)PY_Cstruct_numpy_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)nullptr,                  /* tp_finalize */
#endif
};
