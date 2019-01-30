// pyUserLibrarymodule.hpp
// This is generated code, do not edit
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
#ifndef PYUSERLIBRARYMODULE_HPP
#define PYUSERLIBRARYMODULE_HPP
#include <Python.h>
// splicer begin header.include
// splicer end header.include

// forward declare classes
namespace example {
    namespace nested {
        class ExClass1;
    }
}
namespace example {
    namespace nested {
        class ExClass2;
    }
}

extern PyTypeObject PP_ExClass1_Type;
extern PyTypeObject PP_ExClass2_Type;

// splicer begin header.C_declaration
// splicer end header.C_declaration

// helper functions
extern const char *PY_ExClass1_capsule_name;
extern const char *PY_ExClass2_capsule_name;
PyObject *PP_ExClass1_to_Object(example::nested::ExClass1 *addr);
int PP_ExClass1_from_Object(PyObject *obj, void **addr);
PyObject *PP_ExClass2_to_Object(example::nested::ExClass2 *addr);
int PP_ExClass2_from_Object(PyObject *obj, void **addr);

// splicer begin class.ExClass1.C_declaration
// splicer end class.ExClass1.C_declaration

typedef struct {
PyObject_HEAD
    example::nested::ExClass1 * obj;
    // splicer begin class.ExClass1.C_object
    // splicer end class.ExClass1.C_object
} PP_ExClass1;
// splicer begin class.ExClass2.C_declaration
// splicer end class.ExClass2.C_declaration

typedef struct {
PyObject_HEAD
    example::nested::ExClass2 * obj;
    // splicer begin class.ExClass2.C_object
    // splicer end class.ExClass2.C_object
} PP_ExClass2;

extern PyObject *PP_error_obj;

#if PY_MAJOR_VERSION >= 3
extern "C" PyMODINIT_FUNC PyInit_userlibrary(void);
#else
extern "C" PyMODINIT_FUNC inituserlibrary(void);
#endif

#endif  /* PYUSERLIBRARYMODULE_HPP */
