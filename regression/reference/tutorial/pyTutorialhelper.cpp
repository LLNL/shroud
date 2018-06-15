// pyTutorialhelper.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
// All rights reserved.
//
// This file is part of Shroud.  For details, see
// https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the disclaimer (as noted below)
//   in the documentation and/or other materials provided with the
//   distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
// LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
