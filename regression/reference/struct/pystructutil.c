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



// destructor function for PyCapsule
void PY_array_destructor_function(PyObject *cap)
{
    void *ptr = PyCapsule_GetPointer(cap, "PY_array_dtor");
    PY_SHROUD_dtor_context * context = PyCapsule_GetContext(cap);
    context->dtor(ptr);
}

// 0 - c Cstruct1 *
static void PY_array_destructor_function_0(void *ptr)
{
    free(ptr);
}

// Code used to release arrays for NumPy objects
// via a Capsule base object with a destructor.
// Context strings
PY_SHROUD_dtor_context PY_array_destructor_context[] = {
    {"c Cstruct1 *", PY_array_destructor_function_0},
    {NULL, NULL}
};
