// pyownershipmodule.hpp
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
#ifndef PYOWNERSHIPMODULE_HPP
#define PYOWNERSHIPMODULE_HPP
#include <Python.h>
// splicer begin header.include
// splicer end header.include

// utility functions
typedef struct {
    const char *name;
    void (*dtor)(void *ptr);
} blah;

// ------------------------------
class Class1;  // forward declare
extern PyTypeObject PY_Class1_Type;
// splicer begin class.Class1.C_declaration
// splicer end class.Class1.C_declaration

typedef struct {
PyObject_HEAD
    Class1 * obj;
    blah * dtor;
    // splicer begin class.Class1.C_object
    // splicer end class.Class1.C_object
} PY_Class1;

extern const char *PY_Class1_capsule_name;
PyObject *PP_Class1_to_Object(Class1 *addr);
int PP_Class1_from_Object(PyObject *obj, void **addr);
// ------------------------------

// splicer begin header.C_declaration
// splicer end header.C_declaration

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
extern "C" PyMODINIT_FUNC PyInit_ownership(void);
#else
extern "C" PyMODINIT_FUNC initownership(void);
#endif

#endif  /* PYOWNERSHIPMODULE_HPP */
