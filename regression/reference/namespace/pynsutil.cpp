// pynsutil.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pynsmodule.hpp"
const char *PY_Cstruct1_capsule_name = "Cstruct1";
const char *PY_ClassWork_capsule_name = "ClassWork";


PyObject *PP_Cstruct1_to_Object(outer::Cstruct1 *addr)
{
    // splicer begin namespace.outer.class.Cstruct1.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Cstruct1_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Cstruct1_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end namespace.outer.class.Cstruct1.utility.to_object
}

int PP_Cstruct1_from_Object(PyObject *obj, void **addr)
{
    // splicer begin namespace.outer.class.Cstruct1.utility.from_object
    if (obj->ob_type != &PY_Cstruct1_Type) {
        // raise exception
        return 0;
    }
    PY_Cstruct1 * self = (PY_Cstruct1 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end namespace.outer.class.Cstruct1.utility.from_object
}

PyObject *PP_ClassWork_to_Object(nswork::ClassWork *addr)
{
    // splicer begin namespace.nswork.class.ClassWork.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_ClassWork_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_ClassWork_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end namespace.nswork.class.ClassWork.utility.to_object
}

int PP_ClassWork_from_Object(PyObject *obj, void **addr)
{
    // splicer begin namespace.nswork.class.ClassWork.utility.from_object
    if (obj->ob_type != &PY_ClassWork_Type) {
        // raise exception
        return 0;
    }
    PY_ClassWork * self = (PY_ClassWork *) obj;
    *addr = self->obj;
    return 1;
    // splicer end namespace.nswork.class.ClassWork.utility.from_object
}

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

// 1 - cxx outer::Cstruct1 *
static void PY_SHROUD_capsule_destructor_1(void *ptr)
{
    outer::Cstruct1 * cxx_ptr = static_cast<outer::Cstruct1 *>(ptr);
    delete cxx_ptr;
}

// Code used to release arrays for NumPy objects
// via a Capsule base object with a destructor.
// Context strings
static PY_SHROUD_dtor_context PY_SHROUD_capsule_context[] = {
    {"--none--", PY_SHROUD_capsule_destructor_0},
    {"cxx outer::Cstruct1 *", PY_SHROUD_capsule_destructor_1},
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
    PY_SHROUD_dtor_context * context = static_cast<PY_SHROUD_dtor_context *>
        (PyCapsule_GetContext(cap));
    context->dtor(ptr);
}
