// pyTutorialhelper.cpp
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
#include "pyTutorialmodule.hpp"
#include "tutorial.hpp"

const char *PY_Class1_capsule_name = "Class1";
const char *PY_Singleton_capsule_name = "Singleton";


PyObject *PP_Class1_to_Object(tutorial::Class1 *addr)
{
    // splicer begin class.Class1.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Class1_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Class1_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.Class1.helper.to_object
}

int PP_Class1_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Class1.helper.from_object
    if (obj->ob_type != &PY_Class1_Type) {
        // raise exception
        return 0;
    }
    PY_Class1 * self = (PY_Class1 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Class1.helper.from_object
}

PyObject *PP_Singleton_to_Object(Singleton *addr)
{
    // splicer begin class.Singleton.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Singleton_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Singleton_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.Singleton.helper.to_object
}

int PP_Singleton_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Singleton.helper.from_object
    if (obj->ob_type != &PY_Singleton_Type) {
        // raise exception
        return 0;
    }
    PY_Singleton * self = (PY_Singleton *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Singleton.helper.from_object
}

// Code used to release arrays for NumPy objects
// via a Capsule base object with a destructor.
// Context strings
const char * PY_array_destructor_context[] = {
    "tutorial::struct1 *",
    NULL
};

// destructor function for PyCapsule
void PY_array_destructor_function(PyObject *cap)
{
    void *ptr = PyCapsule_GetPointer(cap, "PY_array_dtor");
    const char * context = static_cast<const char *>
        (PyCapsule_GetContext(cap));
    if (context == PY_array_destructor_context[0]) {
        tutorial::struct1 * cxx_ptr =
            static_cast<tutorial::struct1 *>(ptr);
        delete cxx_ptr;
    } else {
        // no such destructor
    }
}
