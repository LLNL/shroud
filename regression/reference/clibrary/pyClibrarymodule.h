// pyClibrarymodule.h
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
#ifndef PYCLIBRARYMODULE_H
#define PYCLIBRARYMODULE_H
#include <Python.h>
// splicer begin header.include
// splicer end header.include

// splicer begin header.C_declaration
// splicer end header.C_declaration

// helper functions


extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_clibrary(void);
#else
PyMODINIT_FUNC initclibrary(void);
#endif

#endif  /* PYCLIBRARYMODULE_H */
