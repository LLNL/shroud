// pystructmodule.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#ifndef PYSTRUCTMODULE_H
#define PYSTRUCTMODULE_H

#include <Python.h>

// cxx_header
#include "struct.h"

// splicer begin header.include
// splicer end header.include

// helper PY_converter_type
// Store PyObject and pointer to the data it contains.
// name - used in error messages
// obj  - A mutable object which holds the data.
//        For example, a NumPy array, Python array.
//        But not a list or str object.
// dataobj - converter allocated memory.
//           Decrement dataobj to release memory.
//           For example, extracted from a list or str.
// data  - C accessable pointer to data which is in obj or dataobj.
// size  - number of items in data (not number of bytes).
typedef struct {
    const char *name;
    PyObject *obj;
    PyObject *dataobj;
    void *data;   // points into obj.
    size_t size;
} STR_SHROUD_converter_value;

// Helper functions.
int STR_SHROUD_get_from_object_char(PyObject *obj,
    STR_SHROUD_converter_value *value);
int STR_SHROUD_fill_from_PyObject_char(PyObject *obj, const char *name,
    char *in, Py_ssize_t insize);
int STR_SHROUD_fill_from_PyObject_int_list(PyObject *obj,
    const char *name, int *in, Py_ssize_t insize);
PyObject *STR_SHROUD_to_PyList_int(const int *in, size_t size);

// utility functions
extern void PY_SHROUD_release_memory(int icontext, void *ptr);
extern void *PY_SHROUD_fetch_context(int icontext);
extern void PY_SHROUD_capsule_destructor(PyObject *cap);

// ------------------------------
extern PyTypeObject PY_Arrays1_Type;
// splicer begin class.Arrays1.C_declaration
// splicer end class.Arrays1.C_declaration

typedef struct {
PyObject_HEAD
    Arrays1 * obj;
    int idtor;
    // Python objects for members.
    PyObject *name_obj;
    PyObject *count_obj;
    // Python objects for members.
    PyObject *name_dataobj;
    PyObject *count_dataobj;
    // splicer begin class.Arrays1.C_object
    // splicer end class.Arrays1.C_object
} PY_Arrays1;

extern const char *PY_Arrays1_capsule_name;
PyObject *PP_Arrays1_to_Object_idtor(Arrays1 *addr, int idtor);
PyObject *PP_Arrays1_to_Object(Arrays1 *addr);
int PP_Arrays1_from_Object(PyObject *obj, void **addr);
// ------------------------------

// splicer begin header.C_declaration
// splicer end header.C_declaration

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_cstruct(void);
#else
PyMODINIT_FUNC initcstruct(void);
#endif

#endif  /* PYSTRUCTMODULE_H */
