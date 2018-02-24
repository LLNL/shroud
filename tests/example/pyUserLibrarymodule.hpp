// pyUserLibrarymodule.hpp
// This is generated code, do not edit
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
#ifndef PYUSERLIBRARYMODULE_HPP
#define PYUSERLIBRARYMODULE_HPP
#include <Python.h>
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#ifdef __cplusplus
#define SHROUD_UNUSED(param)
#else
#define SHROUD_UNUSED(param) param
#endif

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#endif

// splicer begin header.include
// splicer end header.include

namespace example {
namespace nested {
extern PyTypeObject PP_ExClass1_Type;
extern PyTypeObject PP_ExClass2_Type;
// splicer begin header.C_declaration
// splicer end header.C_declaration

// helper functions
extern const char *PY_ExClass1_capsule_name;
extern const char *PY_ExClass2_capsule_name;
PyObject *PP_ExClass1_to_Object(ExClass1 *addr);
int PP_ExClass1_from_Object(PyObject *obj, void **addr);
PyObject *PP_ExClass2_to_Object(ExClass2 *addr);
int PP_ExClass2_from_Object(PyObject *obj, void **addr);

// splicer begin class.ExClass1.C_declaration
// splicer end class.ExClass1.C_declaration

typedef struct {
PyObject_HEAD
    ExClass1 * obj;
    // splicer begin class.ExClass1.C_object
    // splicer end class.ExClass1.C_object
} PP_ExClass1;
// splicer begin class.ExClass2.C_declaration
// splicer end class.ExClass2.C_declaration

typedef struct {
PyObject_HEAD
    ExClass2 * obj;
    // splicer begin class.ExClass2.C_object
    // splicer end class.ExClass2.C_object
} PP_ExClass2;

extern PyObject *PP_error_obj;

extern "C" {
#ifdef IS_PY3K
PyMODINIT_FUNC PyInit_userlibrary(void);
#else
PyMODINIT_FUNC inituserlibrary(void);
#endif
}   // extern "C"


}  // namespace nested
}  // namespace example
#endif  /* PYUSERLIBRARYMODULE_HPP */
