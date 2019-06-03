// pypreprocessmodule.hpp
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
#ifndef PYPREPROCESSMODULE_HPP
#define PYPREPROCESSMODULE_HPP
#include <Python.h>
// splicer begin header.include
// splicer end header.include

// forward declare classes
class User1;

extern PyTypeObject PY_User1_Type;

// splicer begin header.C_declaration
// splicer end header.C_declaration

// helper functions
extern const char *PY_User1_capsule_name;
PyObject *PP_User1_to_Object(User1 *addr);
int PP_User1_from_Object(PyObject *obj, void **addr);

// splicer begin class.User1.C_declaration
// splicer end class.User1.C_declaration

typedef struct {
PyObject_HEAD
    User1 * obj;
    // splicer begin class.User1.C_object
    // splicer end class.User1.C_object
} PY_User1;

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
extern "C" PyMODINIT_FUNC PyInit_preprocess(void);
#else
extern "C" PyMODINIT_FUNC initpreprocess(void);
#endif

#endif  /* PYPREPROCESSMODULE_HPP */
