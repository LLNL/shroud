// pyTutorialmodule.hpp
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
#ifndef PYTUTORIALMODULE_HPP
#define PYTUTORIALMODULE_HPP
#include <Python.h>
// splicer begin header.include
// splicer end header.include

// forward declare classes
namespace tutorial {
    class struct1;
}
namespace tutorial {
    class Class1;
}
class Singleton;

extern PyTypeObject PY_Class1_Type;
extern PyTypeObject PY_Singleton_Type;

// splicer begin header.C_declaration
// splicer end header.C_declaration

// helper functions
extern const char *PY_Class1_capsule_name;
extern const char *PY_Singleton_capsule_name;
extern const char * PY_array_destructor_context[];
extern void PY_array_destructor_function(PyObject *cap);
PyObject *PP_Class1_to_Object(tutorial::Class1 *addr);
int PP_Class1_from_Object(PyObject *obj, void **addr);
PyObject *PP_Singleton_to_Object(Singleton *addr);
int PP_Singleton_from_Object(PyObject *obj, void **addr);

// splicer begin class.Class1.C_declaration
// splicer end class.Class1.C_declaration

typedef struct {
PyObject_HEAD
    tutorial::Class1 * obj;
    // splicer begin class.Class1.C_object
    // splicer end class.Class1.C_object
} PY_Class1;
// splicer begin class.Singleton.C_declaration
// splicer end class.Singleton.C_declaration

typedef struct {
PyObject_HEAD
    Singleton * obj;
    // splicer begin class.Singleton.C_object
    // splicer end class.Singleton.C_object
} PY_Singleton;

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
extern "C" PyMODINIT_FUNC PyInit_tutorial(void);
#else
extern "C" PyMODINIT_FUNC inittutorial(void);
#endif

#endif  /* PYTUTORIALMODULE_HPP */
