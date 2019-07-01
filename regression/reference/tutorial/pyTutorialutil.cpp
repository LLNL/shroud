// pyTutorialutil.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pyTutorialmodule.hpp"
const char *PY_Class1_capsule_name = "Class1";
const char *PY_Singleton_capsule_name = "Singleton";


PyObject *PP_Class1_to_Object(tutorial::Class1 *addr)
{
    // splicer begin class.Class1.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Class1_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Class1_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.Class1.utility.to_object
}

int PP_Class1_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Class1.utility.from_object
    if (obj->ob_type != &PY_Class1_Type) {
        // raise exception
        return 0;
    }
    PY_Class1 * self = (PY_Class1 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Class1.utility.from_object
}

PyObject *PP_Singleton_to_Object(Singleton *addr)
{
    // splicer begin class.Singleton.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Singleton_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Singleton_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.Singleton.utility.to_object
}

int PP_Singleton_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Singleton.utility.from_object
    if (obj->ob_type != &PY_Singleton_Type) {
        // raise exception
        return 0;
    }
    PY_Singleton * self = (PY_Singleton *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Singleton.utility.from_object
}
