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

// utility functions
typedef struct {
    const char *name;
    void (*dtor)(void *ptr);
} PY_SHROUD_dtor_context;

extern PY_SHROUD_dtor_context PY_SHROUD_capsule_context[];
extern void PY_SHROUD_capsule_destructor(PyObject *cap);
extern void PY_SHROUD_release_memory(int icontext, void *ptr);

// ------------------------------
class Names;  // forward declare
extern PyTypeObject PY_Names_Type;
// splicer begin class.Names.C_declaration
// splicer end class.Names.C_declaration

typedef struct {
PyObject_HEAD
    Names * myobj;
    int mydtor;
    // splicer begin class.Names.C_object
    // splicer end class.Names.C_object
} PY_Names;

extern const char *PY_Names_capsule_name;
PyObject *PP_Names_to_Object(Names *addr);
int PP_Names_from_Object(PyObject *obj, void **addr);

// ------------------------------
class Names2;  // forward declare
extern PyTypeObject PY_Names2_Type;
// splicer begin class.Names2.C_declaration
// splicer end class.Names2.C_declaration

typedef struct {
PyObject_HEAD
    Names2 * myobj;
    int mydtor;
    // splicer begin class.Names2.C_object
    // splicer end class.Names2.C_object
} PY_Names2;

extern const char *PY_Names2_capsule_name;
PyObject *PP_Names2_to_Object(Names2 *addr);
int PP_Names2_from_Object(PyObject *obj, void **addr);

// ------------------------------
namespace std {
    class vector;  // forward declare
}
extern PyTypeObject PY_Vvv1_Type;
// splicer begin class.vector.C_declaration
// splicer end class.vector.C_declaration

typedef struct {
PyObject_HEAD
    std::Vvv1 * myobj;
    int mydtor;
    // splicer begin class.vector.C_object
    // splicer end class.vector.C_object
} PY_Vvv1;

extern const char *PY_Vvv1_capsule_name;
PyObject *PP_Vvv1_to_Object(std::Vvv1 *addr);
int PP_Vvv1_from_Object(PyObject *obj, void **addr);

// ------------------------------
namespace std {
    class vector;  // forward declare
}
extern PyTypeObject PY_vector_double_Type;
// splicer begin class.vector.C_declaration
// splicer end class.vector.C_declaration

typedef struct {
PyObject_HEAD
    std::vector_double * myobj;
    int mydtor;
    // splicer begin class.vector.C_object
    // splicer end class.vector.C_object
} PY_vector_double;

extern const char *PY_vector_double_capsule_name;
PyObject *PP_vector_double_to_Object(std::vector_double *addr);
int PP_vector_double_from_Object(PyObject *obj, void **addr);

// ------------------------------
namespace std {
    class vector;  // forward declare
}
extern PyTypeObject PY_vector_instantiation5_Type;
// splicer begin class.vector.C_declaration
// splicer end class.vector.C_declaration

typedef struct {
PyObject_HEAD
    std::vector_instantiation5 * myobj;
    int mydtor;
    // splicer begin class.vector.C_object
    // splicer end class.vector.C_object
} PY_vector_instantiation5;

extern const char *PY_vector_instantiation5_capsule_name;
PyObject *PP_vector_instantiation5_to_Object(std::vector_instantiation5 *addr);
int PP_vector_instantiation5_from_Object(PyObject *obj, void **addr);

// ------------------------------
namespace std {
    class vector;  // forward declare
}
extern PyTypeObject PY_vector_instantiation3_Type;
// splicer begin class.vector.C_declaration
// splicer end class.vector.C_declaration

typedef struct {
PyObject_HEAD
    std::vector_instantiation3 * myobj;
    int mydtor;
    // splicer begin class.vector.C_object
    // splicer end class.vector.C_object
} PY_vector_instantiation3;

extern const char *PY_vector_instantiation3_capsule_name;
PyObject *PP_vector_instantiation3_to_Object(std::vector_instantiation3 *addr);
int PP_vector_instantiation3_from_Object(PyObject *obj, void **addr);

// ------------------------------
class twoTs;  // forward declare
extern PyTypeObject PY_twoTs_0_Type;
// splicer begin class.twoTs.C_declaration
// splicer end class.twoTs.C_declaration

typedef struct {
PyObject_HEAD
    twoTs_0 * myobj;
    int mydtor;
    // splicer begin class.twoTs.C_object
    // splicer end class.twoTs.C_object
} PY_twoTs_0;

extern const char *PY_twoTs_0_capsule_name;
PyObject *PP_twoTs_0_to_Object(twoTs_0 *addr);
int PP_twoTs_0_from_Object(PyObject *obj, void **addr);

// ------------------------------
class twoTs;  // forward declare
extern PyTypeObject PY_twoTs_instantiation4_Type;
// splicer begin class.twoTs.C_declaration
// splicer end class.twoTs.C_declaration

typedef struct {
PyObject_HEAD
    twoTs_instantiation4 * myobj;
    int mydtor;
    // splicer begin class.twoTs.C_object
    // splicer end class.twoTs.C_object
} PY_twoTs_instantiation4;

extern const char *PY_twoTs_instantiation4_capsule_name;
PyObject *PP_twoTs_instantiation4_to_Object(twoTs_instantiation4 *addr);
int PP_twoTs_instantiation4_from_Object(PyObject *obj, void **addr);
// ------------------------------

// splicer begin header.C_declaration
// splicer end header.C_declaration

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
extern "C" PyMODINIT_FUNC PyInit_testnames(void);
#else
extern "C" PyMODINIT_FUNC inittestnames(void);
#endif

#endif  /* PYTESTNAMESMODULE_HPP */
