// pytestnameshelper.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
