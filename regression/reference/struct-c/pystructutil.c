// pystructutil.c
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pystructmodule.h"


// ----------------------------------------
typedef struct {
    const char *name;
    void (*dtor)(void *ptr);
} PY_SHROUD_dtor_context;

// 0 - --none--
static void PY_SHROUD_capsule_destructor_0(void *ptr)
{
    // Do not release
}

// 1 - c Cstruct1 *
static void PY_SHROUD_capsule_destructor_1(void *ptr)
{
    free(ptr);
}

// 2 - c const Cstruct1 *
static void PY_SHROUD_capsule_destructor_2(void *ptr)
{
    free(ptr);
}

// Code used to release arrays for NumPy objects
// via a Capsule base object with a destructor.
// Context strings
static PY_SHROUD_dtor_context PY_SHROUD_capsule_context[] = {
    {"--none--", PY_SHROUD_capsule_destructor_0},
    {"c Cstruct1 *", PY_SHROUD_capsule_destructor_1},
    {"c const Cstruct1 *", PY_SHROUD_capsule_destructor_2},
    {NULL, NULL}
};

// Release memory based on icontext.
void PY_SHROUD_release_memory(int icontext, void *ptr)
{
    PY_SHROUD_capsule_context[icontext].dtor(ptr);
}

//Fetch garbage collection context.
void *PY_SHROUD_fetch_context(int icontext)
{
    return PY_SHROUD_capsule_context + icontext;
}

// destructor function for PyCapsule
void PY_SHROUD_capsule_destructor(PyObject *cap)
{
    void *ptr = PyCapsule_GetPointer(cap, "PY_array_dtor");
    PY_SHROUD_dtor_context * context = PyCapsule_GetContext(cap);
    context->dtor(ptr);
}
