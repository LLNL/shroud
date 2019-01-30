// pytestnamesmodule.hpp
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
#ifndef PYTESTNAMESMODULE_HPP
#define PYTESTNAMESMODULE_HPP
#include <Python.h>
// splicer begin header.include
// splicer end header.include

// forward declare classes
class Names;
class Names2;

extern PyTypeObject PY_Names_Type;
extern PyTypeObject PY_Names2_Type;

// splicer begin header.C_declaration
// splicer end header.C_declaration

// helper functions
extern const char *PY_Names_capsule_name;
extern const char *PY_Names2_capsule_name;
PyObject *PP_Names_to_Object(Names *addr);
int PP_Names_from_Object(PyObject *obj, void **addr);
PyObject *PP_Names2_to_Object(Names2 *addr);
int PP_Names2_from_Object(PyObject *obj, void **addr);

// splicer begin class.Names.C_declaration
// splicer end class.Names.C_declaration

typedef struct {
PyObject_HEAD
    Names * obj;
    // splicer begin class.Names.C_object
    // splicer end class.Names.C_object
} PY_Names;
// splicer begin class.Names2.C_declaration
// splicer end class.Names2.C_declaration

typedef struct {
PyObject_HEAD
    Names2 * obj;
    // splicer begin class.Names2.C_object
    // splicer end class.Names2.C_object
} PY_Names2;

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
extern "C" PyMODINIT_FUNC PyInit_testnames(void);
#else
extern "C" PyMODINIT_FUNC inittestnames(void);
#endif

#endif  /* PYTESTNAMESMODULE_HPP */
