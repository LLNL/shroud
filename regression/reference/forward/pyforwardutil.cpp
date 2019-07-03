// pyforwardutil.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include "pyforwardmodule.hpp"
#include "tutorial.hpp"

const char *PY_Class3_capsule_name = "Class3";
const char *PY_Class2_capsule_name = "Class2";


PyObject *PP_Class3_to_Object(tutorial::Class3 *addr)
{
    // splicer begin class.Class3.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Class3_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Class3_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.Class3.utility.to_object
}

int PP_Class3_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Class3.utility.from_object
    if (obj->ob_type != &PY_Class3_Type) {
        // raise exception
        return 0;
    }
    PY_Class3 * self = (PY_Class3 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Class3.utility.from_object
}

PyObject *PP_Class2_to_Object(tutorial::Class2 *addr)
{
    // splicer begin class.Class2.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Class2_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Class2_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.Class2.utility.to_object
}

int PP_Class2_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Class2.utility.from_object
    if (obj->ob_type != &PY_Class2_Type) {
        // raise exception
        return 0;
    }
    PY_Class2 * self = (PY_Class2 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Class2.utility.from_object
}

// destructor function for PyCapsule
void PY_array_destructor_function(PyObject *cap)
{
    void *ptr = PyCapsule_GetPointer(cap, "PY_array_dtor");
    blah * context = static_cast<blah *>(PyCapsule_GetContext(cap));
    context->dtor(ptr);
}

// 0 - cxx tutorial::Class2 *
static void PY_array_destructor_function_0(void *ptr)
{
    tutorial::Class2 * cxx_ptr = static_cast<tutorial::Class2 *>(ptr);
    delete cxx_ptr;
}

// Code used to release arrays for NumPy objects
// via a Capsule base object with a destructor.
// Context strings
blah PY_array_destructor_context[] = {
    {"cxx tutorial::Class2 *", PY_array_destructor_function_0},
    {NULL, NULL}
};
