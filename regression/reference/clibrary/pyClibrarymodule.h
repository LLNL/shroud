// pyClibrarymodule.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#ifndef PYCLIBRARYMODULE_H
#define PYCLIBRARYMODULE_H

#include <Python.h>

// cxx_header
#include "clibrary.h"

// splicer begin header.include
// splicer end header.include

// splicer begin header.C_declaration
// splicer end header.C_declaration

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_clibrary(void);
#else
PyMODINIT_FUNC initclibrary(void);
#endif

#endif  /* PYCLIBRARYMODULE_H */
