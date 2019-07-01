// pystructutil.c
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pystructmodule.h"
#include "struct.h"



// Code used to release arrays for NumPy objects
// via a Capsule base object with a destructor.
// Context strings
const char * PY_array_destructor_context[] = {
    "Cstruct1 *",
    NULL
};

// destructor function for PyCapsule
void PY_array_destructor_function(PyObject *cap)
{
    void *ptr = PyCapsule_GetPointer(cap, "PY_array_dtor");
    const char * context = PyCapsule_GetContext(cap);
    if (context == PY_array_destructor_context[0]) {
        free(ptr);
    } else {
        // no such destructor
    }
}
