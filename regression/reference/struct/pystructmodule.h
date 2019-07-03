// pystructmodule.h
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#ifndef PYSTRUCTMODULE_H
#define PYSTRUCTMODULE_H
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

// splicer begin header.C_declaration
// splicer end header.C_declaration

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_cstruct(void);
#else
PyMODINIT_FUNC initcstruct(void);
#endif

#endif  /* PYSTRUCTMODULE_H */
