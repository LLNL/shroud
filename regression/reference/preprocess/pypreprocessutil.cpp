// pypreprocessutil.cpp
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
#include "pypreprocessmodule.hpp"
const char *PY_User1_capsule_name = "User1";
#ifdef USE_USER2
const char *PY_User2_capsule_name = "User2";
#endif  // ifdef USE_USER2


PyObject *PP_User1_to_Object(User1 *addr)
{
    // splicer begin class.User1.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_User1_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_User1_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.User1.utility.to_object
}

int PP_User1_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.User1.utility.from_object
    if (obj->ob_type != &PY_User1_Type) {
        // raise exception
        return 0;
    }
    PY_User1 * self = (PY_User1 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.User1.utility.from_object
}

#ifdef USE_USER2
PyObject *PP_User2_to_Object(User2 *addr)
{
    // splicer begin class.User2.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_User2_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_User2_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.User2.utility.to_object
}

int PP_User2_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.User2.utility.from_object
    if (obj->ob_type != &PY_User2_Type) {
        // raise exception
        return 0;
    }
    PY_User2 * self = (PY_User2 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.User2.utility.from_object
}
#endif  // ifdef USE_USER2

// ----------------------------------------
typedef struct {
    const char *name;
    void (*dtor)(void *ptr);
} PY_SHROUD_dtor_context;

// Code used to release arrays for NumPy objects
// via a Capsule base object with a destructor.
// Context strings
static PY_SHROUD_dtor_context PY_SHROUD_capsule_context[] = {
    {NULL, NULL}
};

// Release memory based on icontext.
void PY_SHROUD_release_memory(int icontext, void *ptr)
{
    if (icontext != -1) {
        PY_SHROUD_capsule_context[icontext].dtor(ptr);
    }
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
