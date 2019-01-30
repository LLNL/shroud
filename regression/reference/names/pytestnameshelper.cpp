// pytestnameshelper.cpp
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
#include "pytestnamesmodule.hpp"
const char *PY_Names_capsule_name = "Names";
const char *PY_Names2_capsule_name = "Names2";


PyObject *PP_Names_to_Object(Names *addr)
{
    // splicer begin class.Names.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Names_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Names_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.Names.helper.to_object
}

int PP_Names_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Names.helper.from_object
    if (obj->ob_type != &PY_Names_Type) {
        // raise exception
        return 0;
    }
    PY_Names * self = (PY_Names *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Names.helper.from_object
}

PyObject *PP_Names2_to_Object(Names2 *addr)
{
    // splicer begin class.Names2.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Names2_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Names2_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.Names2.helper.to_object
}

int PP_Names2_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Names2.helper.from_object
    if (obj->ob_type != &PY_Names2_Type) {
        // raise exception
        return 0;
    }
    PY_Names2 * self = (PY_Names2 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Names2.helper.from_object
}
