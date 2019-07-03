// pyUserLibraryutil.cpp
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
#include "pyUserLibrarymodule.hpp"

const char *PY_ExClass1_capsule_name = "ExClass1";
const char *PY_ExClass2_capsule_name = "ExClass2";


PyObject *PP_ExClass1_to_Object(example::nested::ExClass1 *addr)
{
    // splicer begin class.ExClass1.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_ExClass1_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PP_ExClass1_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.ExClass1.utility.to_object
}

int PP_ExClass1_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.ExClass1.utility.from_object
    if (obj->ob_type != &PP_ExClass1_Type) {
        // raise exception
        return 0;
    }
    PP_ExClass1 * self = (PP_ExClass1 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.ExClass1.utility.from_object
}

PyObject *PP_ExClass2_to_Object(example::nested::ExClass2 *addr)
{
    // splicer begin class.ExClass2.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_ExClass2_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PP_ExClass2_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.ExClass2.utility.to_object
}

int PP_ExClass2_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.ExClass2.utility.from_object
    if (obj->ob_type != &PP_ExClass2_Type) {
        // raise exception
        return 0;
    }
    PP_ExClass2 * self = (PP_ExClass2 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.ExClass2.utility.from_object
}

// destructor function for PyCapsule
void PP_SHROUD_capsule_destructor(PyObject *cap)
{
    void *ptr = PyCapsule_GetPointer(cap, "PP_array_dtor");
    PP_SHROUD_dtor_context * context = static_cast<PP_SHROUD_dtor_context *>
        (PyCapsule_GetContext(cap));
    context->dtor(ptr);
}
// Release memory based on icontext.
void PP_SHROUD_release_memory(int icontext, void *ptr)
{
    if (icontext != -1) {
        PP_SHROUD_capsule_context[icontext].dtor(ptr);
    }
}

// 0 - cxx example::nested::ExClass1 *
static void PP_SHROUD_capsule_destructor_0(void *ptr)
{
    example::nested::ExClass1 * cxx_ptr =
        static_cast<example::nested::ExClass1 *>(ptr);
    delete cxx_ptr;
}

// 1 - cxx example::nested::ExClass2 *
static void PP_SHROUD_capsule_destructor_1(void *ptr)
{
    example::nested::ExClass2 * cxx_ptr =
        static_cast<example::nested::ExClass2 *>(ptr);
    delete cxx_ptr;
}

// Code used to release arrays for NumPy objects
// via a Capsule base object with a destructor.
// Context strings
PP_SHROUD_dtor_context PP_SHROUD_capsule_context[] = {
    {"cxx example::nested::ExClass1 *", PP_SHROUD_capsule_destructor_0},
    {"cxx example::nested::ExClass2 *", PP_SHROUD_capsule_destructor_1},
    {NULL, NULL}
};
