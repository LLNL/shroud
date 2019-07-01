// pystructutil.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pystructmodule.hpp"
#include "struct.h"

const char *PY_Cstruct1_capsule_name = "Cstruct1";


PyObject *PP_Cstruct1_to_Object(Cstruct1 *addr)
{
    // splicer begin class.Cstruct1.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Cstruct1_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Cstruct1_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.Cstruct1.utility.to_object
}

int PP_Cstruct1_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Cstruct1.utility.from_object
    if (obj->ob_type != &PY_Cstruct1_Type) {
        // raise exception
        return 0;
    }
    PY_Cstruct1 * self = (PY_Cstruct1 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Cstruct1.utility.from_object
}

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
    const char * context = static_cast<const char *>
        (PyCapsule_GetContext(cap));
    if (context == PY_array_destructor_context[0]) {
        Cstruct1 * cxx_ptr = static_cast<Cstruct1 *>(ptr);
        delete cxx_ptr;
    } else {
        // no such destructor
    }
}
