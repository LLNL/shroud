// pyCstruct_ptrtype.c
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pystructmodule.h"
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SHROUD_STRUCT_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
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
// splicer begin class.Cstruct_ptr.impl.C_definition
// splicer end class.Cstruct_ptr.impl.C_definition
// splicer begin class.Cstruct_ptr.impl.additional_methods
// splicer end class.Cstruct_ptr.impl.additional_methods
static void
PY_Cstruct_ptr_tp_del (PY_Cstruct_ptr *self)
{
// splicer begin class.Cstruct_ptr.type.del
    PY_SHROUD_release_memory(self->idtor, self->obj);
    self->obj = NULL;
    // Python objects for members.
    Py_XDECREF(self->cfield_obj);
    Py_XDECREF(self->const_dvalue_obj);
    // Python objects for members.
    Py_XDECREF(self->cfield_dataobj);
    Py_XDECREF(self->const_dvalue_dataobj);
// splicer end class.Cstruct_ptr.type.del
}

// ----------------------------------------
// Function:  Cstruct_ptr +name(Cstruct_ptr_ctor)
// Exact:     py_default
// ----------------------------------------
// Argument:  char * cfield +intent(in)
// Requested: py_ctor_char_*_numpy
// Match:     py_ctor_char_*
// ----------------------------------------
// Argument:  const double * const_dvalue +intent(in)
// Requested: py_ctor_native_*_numpy
// Match:     py_ctor_native_*
static int
PY_Cstruct_ptr_tp_init(
  PY_Cstruct_ptr *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin class.Cstruct_ptr.method.cstruct_ptr_ctor
    STR_SHROUD_converter_value SHValue_cfield = {NULL, NULL, NULL, NULL, 0};
    SHValue_cfield.name = "cfield";
    STR_SHROUD_converter_value SHValue_const_dvalue = {NULL, NULL, NULL, NULL, 0};
    SHValue_const_dvalue.name = "const_dvalue";
    char *SHT_kwlist[] = {
        "cfield",
        "const_dvalue",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "|O&O&:Cstruct_ptr_ctor", SHT_kwlist, 
        STR_SHROUD_get_from_object_char, &SHValue_cfield,
        STR_SHROUD_get_from_object_double_numpy, &SHValue_const_dvalue))
        return -1;

    self->obj = malloc(sizeof(Cstruct_ptr));
    if (self->obj == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->idtor = 2;

    // post_call - initialize fields
    Cstruct_ptr *SH_obj = self->obj;
    SH_obj->cfield = (char *) SHValue_cfield.data;
    self->cfield_obj = SHValue_cfield.obj;  // steal reference
    SH_obj->const_dvalue = (double *) SHValue_const_dvalue.data;
    self->const_dvalue_obj = SHValue_const_dvalue.obj;  // steal reference

    return 0;
// splicer end class.Cstruct_ptr.method.cstruct_ptr_ctor
}
// splicer begin class.Cstruct_ptr.impl.after_methods
// splicer end class.Cstruct_ptr.impl.after_methods

// Requested: py_descr_char_*_numpy
// Match:     py_descr_char_*
static PyObject *PY_Cstruct_ptr_cfield_getter(PY_Cstruct_ptr *self,
    void *SHROUD_UNUSED(closure))
{
    if (self->obj->cfield == NULL) {
        Py_RETURN_NONE;
    }
    PyObject * rv = PyString_FromString(self->obj->cfield);
    return rv;
}

// Requested: py_descr_char_*_numpy
// Match:     py_descr_char_*
static int PY_Cstruct_ptr_cfield_setter(PY_Cstruct_ptr *self, PyObject *value,
    void *SHROUD_UNUSED(closure))
{
    STR_SHROUD_converter_value cvalue;
    Py_XDECREF(self->cfield_dataobj);
    if (STR_SHROUD_get_from_object_char(value, &cvalue) == 0) {
        self->obj->cfield = NULL;
        self->cfield_dataobj = NULL;
        return -1;
    }
    self->obj->cfield = (char *) cvalue.data;
    self->cfield_dataobj = cvalue.dataobj;  // steal reference
    return 0;
}

// Exact:     py_descr_native_*_numpy
static PyObject *PY_Cstruct_ptr_const_dvalue_getter(PY_Cstruct_ptr *self,
    void *SHROUD_UNUSED(closure))
{
    if (self->obj->const_dvalue == NULL) {
        Py_RETURN_NONE;
    }
    if (self->const_dvalue_obj != NULL) {
        Py_INCREF(self->const_dvalue_obj);
        return self->const_dvalue_obj;
    }
    npy_intp dims[1] = { 1 };
    PyObject *rv = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
        (double *) self->obj->const_dvalue);
    if (rv != NULL) {
        Py_INCREF(rv);
        self->const_dvalue_obj = rv;
    }
    return rv;
}

// Exact:     py_descr_native_*_numpy
static int PY_Cstruct_ptr_const_dvalue_setter(PY_Cstruct_ptr *self, PyObject *value,
    void *SHROUD_UNUSED(closure))
{
    STR_SHROUD_converter_value cvalue;
    Py_XDECREF(self->const_dvalue_obj);
    if (STR_SHROUD_get_from_object_double_numpy(value, &cvalue) == 0) {
        self->obj->const_dvalue = NULL;
        self->const_dvalue_obj = NULL;
        // XXXX set error
        return -1;
    }
    self->obj->const_dvalue = (double *) cvalue.data;
    self->const_dvalue_obj = cvalue.obj;  // steal reference
    return 0;
}

static PyGetSetDef PY_Cstruct_ptr_getset[] = {
    {(char *)"cfield", (getter)PY_Cstruct_ptr_cfield_getter,
        (setter)PY_Cstruct_ptr_cfield_setter, NULL, NULL},
    {(char *)"const_dvalue", (getter)PY_Cstruct_ptr_const_dvalue_getter,
        (setter)PY_Cstruct_ptr_const_dvalue_setter, NULL, NULL},
    // splicer begin class.Cstruct_ptr.PyGetSetDef
    // splicer end class.Cstruct_ptr.PyGetSetDef
    {NULL}            /* sentinel */
};
static PyMethodDef PY_Cstruct_ptr_methods[] = {
    // splicer begin class.Cstruct_ptr.PyMethodDef
    // splicer end class.Cstruct_ptr.PyMethodDef
    {NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

static char Cstruct_ptr__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_Cstruct_ptr_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cstruct.Cstruct_ptr",                       /* tp_name */
    sizeof(PY_Cstruct_ptr),         /* tp_basicsize */
    0,                              /* tp_itemsize */
    /* Methods to implement standard operations */
    (destructor)NULL,                 /* tp_dealloc */
    (printfunc)NULL,                   /* tp_print */
    (getattrfunc)NULL,                 /* tp_getattr */
    (setattrfunc)NULL,                 /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
    NULL,                               /* tp_reserved */
#else
    (cmpfunc)NULL,                     /* tp_compare */
#endif
    (reprfunc)NULL,                    /* tp_repr */
    /* Method suites for standard classes */
    NULL,                              /* tp_as_number */
    NULL,                              /* tp_as_sequence */
    NULL,                              /* tp_as_mapping */
    /* More standard operations (here for binary compatibility) */
    (hashfunc)NULL,                    /* tp_hash */
    (ternaryfunc)NULL,                 /* tp_call */
    (reprfunc)NULL,                    /* tp_str */
    (getattrofunc)NULL,                /* tp_getattro */
    (setattrofunc)NULL,                /* tp_setattro */
    /* Functions to access object as input/output buffer */
    NULL,                              /* tp_as_buffer */
    /* Flags to define presence of optional/expanded features */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    Cstruct_ptr__doc__,         /* tp_doc */
    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    (traverseproc)NULL,                /* tp_traverse */
    /* delete references to contained objects */
    (inquiry)NULL,                     /* tp_clear */
    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    (richcmpfunc)NULL,                 /* tp_richcompare */
    /* weak reference enabler */
    0,                              /* tp_weaklistoffset */
    /* Added in release 2.2 */
    /* Iterators */
    (getiterfunc)NULL,                 /* tp_iter */
    (iternextfunc)NULL,                /* tp_iternext */
    /* Attribute descriptor and subclassing stuff */
    PY_Cstruct_ptr_methods,                             /* tp_methods */
    NULL,                              /* tp_members */
    PY_Cstruct_ptr_getset,                             /* tp_getset */
    NULL,                              /* tp_base */
    NULL,                              /* tp_dict */
    (descrgetfunc)NULL,                /* tp_descr_get */
    (descrsetfunc)NULL,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PY_Cstruct_ptr_tp_init,                   /* tp_init */
    (allocfunc)NULL,                  /* tp_alloc */
    (newfunc)NULL,                    /* tp_new */
    (freefunc)NULL,                   /* tp_free */
    (inquiry)NULL,                     /* tp_is_gc */
    NULL,                              /* tp_bases */
    NULL,                              /* tp_mro */
    NULL,                              /* tp_cache */
    NULL,                              /* tp_subclasses */
    NULL,                              /* tp_weaklist */
    (destructor)PY_Cstruct_ptr_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)NULL,                  /* tp_finalize */
#endif
};
