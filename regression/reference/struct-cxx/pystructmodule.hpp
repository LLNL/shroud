// pystructmodule.hpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#ifndef PYSTRUCTMODULE_HPP
#define PYSTRUCTMODULE_HPP
#include <Python.h>
// splicer begin header.include
// splicer end header.include

// utility functions
typedef struct {
    const char *name;
    void (*dtor)(void *ptr);
} blah;
extern blah PY_array_destructor_context[];
extern void PY_array_destructor_function(PyObject *cap);

// ------------------------------
class Cstruct1;  // forward declare
extern PyTypeObject PY_Cstruct1_Type;
// splicer begin class.Cstruct1.C_declaration
// splicer end class.Cstruct1.C_declaration

typedef struct {
PyObject_HEAD
    Cstruct1 * obj;
    blah * dtor;
    // splicer begin class.Cstruct1.C_object
    // splicer end class.Cstruct1.C_object
} PY_Cstruct1;

extern const char *PY_Cstruct1_capsule_name;
PyObject *PP_Cstruct1_to_Object(Cstruct1 *addr);
int PP_Cstruct1_from_Object(PyObject *obj, void **addr);
// ------------------------------

// splicer begin header.C_declaration
// splicer end header.C_declaration

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
extern "C" PyMODINIT_FUNC PyInit_cstruct(void);
#else
extern "C" PyMODINIT_FUNC initcstruct(void);
#endif

#endif  /* PYSTRUCTMODULE_HPP */
