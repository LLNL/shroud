// pypreprocesshelper.cpp
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
const char *PY_User2_capsule_name = "User2";


PyObject *PP_User1_to_Object(User1 *addr)
{
    // splicer begin class.User1.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_User1_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_User1_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.User1.helper.to_object
}

int PP_User1_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.User1.helper.from_object
    if (obj->ob_type != &PY_User1_Type) {
        // raise exception
        return 0;
    }
    PY_User1 * self = (PY_User1 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.User1.helper.from_object
}

PyObject *PP_User2_to_Object(User2 *addr)
{
    // splicer begin class.User2.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_User2_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_User2_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.User2.helper.to_object
}

int PP_User2_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.User2.helper.from_object
    if (obj->ob_type != &PY_User2_Type) {
        // raise exception
        return 0;
    }
    PY_User2 * self = (PY_User2 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.User2.helper.from_object
}
