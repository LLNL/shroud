// pytypesmodule.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include "pytypesmodule.hpp"
#include "types.hpp"

// splicer begin include
// splicer end include

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

// splicer begin C_definition
// splicer end C_definition
PyObject *PY_error_obj;
// splicer begin additional_functions
// splicer end additional_functions

static char PY_short_func__doc__[] =
"documentation"
;

static PyObject *
PY_short_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// short short_func(short arg1 +intent(in)+value)
// splicer begin function.short_func
    short arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "h:short_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    short SHC_rv = short_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.short_func
}

static char PY_int_func__doc__[] =
"documentation"
;

static PyObject *
PY_int_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int int_func(int arg1 +intent(in)+value)
// splicer begin function.int_func
    int arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:int_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    int SHC_rv = int_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.int_func
}

static char PY_long_func__doc__[] =
"documentation"
;

static PyObject *
PY_long_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// long long_func(long arg1 +intent(in)+value)
// splicer begin function.long_func
    long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l:long_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    long SHC_rv = long_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.long_func
}

static char PY_long_long_func__doc__[] =
"documentation"
;

static PyObject *
PY_long_long_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// long long long_long_func(long long arg1 +intent(in)+value)
// splicer begin function.long_long_func
    long long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L:long_long_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    long long SHC_rv = long_long_func(arg1);

    // post_call
    PyObject * SHTPy_rv = Py_BuildValue("L", SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.long_long_func
}

static char PY_short_int_func__doc__[] =
"documentation"
;

static PyObject *
PY_short_int_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// short int short_int_func(short int arg1 +intent(in)+value)
// splicer begin function.short_int_func
    short arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "h:short_int_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    short SHC_rv = short_int_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.short_int_func
}

static char PY_long_int_func__doc__[] =
"documentation"
;

static PyObject *
PY_long_int_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// long int long_int_func(long int arg1 +intent(in)+value)
// splicer begin function.long_int_func
    long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l:long_int_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    long SHC_rv = long_int_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.long_int_func
}

static char PY_long_long_int_func__doc__[] =
"documentation"
;

static PyObject *
PY_long_long_int_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// long long int long_long_int_func(long long int arg1 +intent(in)+value)
// splicer begin function.long_long_int_func
    long long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L:long_long_int_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    long long SHC_rv = long_long_int_func(arg1);

    // post_call
    PyObject * SHTPy_rv = Py_BuildValue("L", SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.long_long_int_func
}

static char PY_unsigned_func__doc__[] =
"documentation"
;

static PyObject *
PY_unsigned_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// unsigned unsigned_func(unsigned arg1 +intent(in)+value)
// splicer begin function.unsigned_func
    unsigned int arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:unsigned_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    unsigned int SHC_rv = unsigned_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.unsigned_func
}

static char PY_ushort_func__doc__[] =
"documentation"
;

static PyObject *
PY_ushort_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// unsigned short ushort_func(unsigned short arg1 +intent(in)+value)
// splicer begin function.ushort_func
    unsigned short arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "h:ushort_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    unsigned short SHC_rv = ushort_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.ushort_func
}

static char PY_uint_func__doc__[] =
"documentation"
;

static PyObject *
PY_uint_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// unsigned int uint_func(unsigned int arg1 +intent(in)+value)
// splicer begin function.uint_func
    unsigned int arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:uint_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    unsigned int SHC_rv = uint_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.uint_func
}

static char PY_ulong_func__doc__[] =
"documentation"
;

static PyObject *
PY_ulong_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// unsigned long ulong_func(unsigned long arg1 +intent(in)+value)
// splicer begin function.ulong_func
    unsigned long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l:ulong_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    unsigned long SHC_rv = ulong_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.ulong_func
}

static char PY_ulong_long_func__doc__[] =
"documentation"
;

static PyObject *
PY_ulong_long_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// unsigned long long ulong_long_func(unsigned long long arg1 +intent(in)+value)
// splicer begin function.ulong_long_func
    unsigned long long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L:ulong_long_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    unsigned long long SHC_rv = ulong_long_func(arg1);

    // post_call
    PyObject * SHTPy_rv = Py_BuildValue("L", SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.ulong_long_func
}

static char PY_ulong_int_func__doc__[] =
"documentation"
;

static PyObject *
PY_ulong_int_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// unsigned long int ulong_int_func(unsigned long int arg1 +intent(in)+value)
// splicer begin function.ulong_int_func
    unsigned long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l:ulong_int_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    unsigned long SHC_rv = ulong_int_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.ulong_int_func
}

static char PY_int8_func__doc__[] =
"documentation"
;

static PyObject *
PY_int8_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int8_t int8_func(int8_t arg1 +intent(in)+value)
// splicer begin function.int8_func
    int8_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:int8_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    int8_t SHC_rv = int8_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.int8_func
}

static char PY_int16_func__doc__[] =
"documentation"
;

static PyObject *
PY_int16_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int16_t int16_func(int16_t arg1 +intent(in)+value)
// splicer begin function.int16_func
    int16_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:int16_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    int16_t SHC_rv = int16_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.int16_func
}

static char PY_int32_func__doc__[] =
"documentation"
;

static PyObject *
PY_int32_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int32_t int32_func(int32_t arg1 +intent(in)+value)
// splicer begin function.int32_func
    int32_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:int32_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    int32_t SHC_rv = int32_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.int32_func
}

static char PY_int64_func__doc__[] =
"documentation"
;

static PyObject *
PY_int64_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int64_t int64_func(int64_t arg1 +intent(in)+value)
// splicer begin function.int64_func
    int64_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:int64_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    int64_t SHC_rv = int64_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.int64_func
}

static char PY_uint8_func__doc__[] =
"documentation"
;

static PyObject *
PY_uint8_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// uint8_t uint8_func(uint8_t arg1 +intent(in)+value)
// splicer begin function.uint8_func
    uint8_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:uint8_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    uint8_t SHC_rv = uint8_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.uint8_func
}

static char PY_uint16_func__doc__[] =
"documentation"
;

static PyObject *
PY_uint16_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// uint16_t uint16_func(uint16_t arg1 +intent(in)+value)
// splicer begin function.uint16_func
    uint16_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:uint16_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    uint16_t SHC_rv = uint16_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.uint16_func
}

static char PY_uint32_func__doc__[] =
"documentation"
;

static PyObject *
PY_uint32_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// uint32_t uint32_func(uint32_t arg1 +intent(in)+value)
// splicer begin function.uint32_func
    uint32_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:uint32_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    uint32_t SHC_rv = uint32_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.uint32_func
}

static char PY_uint64_func__doc__[] =
"documentation"
;

static PyObject *
PY_uint64_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// uint64_t uint64_func(uint64_t arg1 +intent(in)+value)
// splicer begin function.uint64_func
    uint64_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:uint64_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    uint64_t SHC_rv = uint64_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.uint64_func
}

static char PY_size_func__doc__[] =
"documentation"
;

static PyObject *
PY_size_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// size_t size_func(size_t arg1 +intent(in)+value)
// splicer begin function.size_func
    size_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:size_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    size_t SHC_rv = size_func(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromSize_t(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.size_func
}
static PyMethodDef PY_methods[] = {
{"short_func", (PyCFunction)PY_short_func, METH_VARARGS|METH_KEYWORDS,
    PY_short_func__doc__},
{"int_func", (PyCFunction)PY_int_func, METH_VARARGS|METH_KEYWORDS,
    PY_int_func__doc__},
{"long_func", (PyCFunction)PY_long_func, METH_VARARGS|METH_KEYWORDS,
    PY_long_func__doc__},
{"long_long_func", (PyCFunction)PY_long_long_func,
    METH_VARARGS|METH_KEYWORDS, PY_long_long_func__doc__},
{"short_int_func", (PyCFunction)PY_short_int_func,
    METH_VARARGS|METH_KEYWORDS, PY_short_int_func__doc__},
{"long_int_func", (PyCFunction)PY_long_int_func,
    METH_VARARGS|METH_KEYWORDS, PY_long_int_func__doc__},
{"long_long_int_func", (PyCFunction)PY_long_long_int_func,
    METH_VARARGS|METH_KEYWORDS, PY_long_long_int_func__doc__},
{"unsigned_func", (PyCFunction)PY_unsigned_func,
    METH_VARARGS|METH_KEYWORDS, PY_unsigned_func__doc__},
{"ushort_func", (PyCFunction)PY_ushort_func, METH_VARARGS|METH_KEYWORDS,
    PY_ushort_func__doc__},
{"uint_func", (PyCFunction)PY_uint_func, METH_VARARGS|METH_KEYWORDS,
    PY_uint_func__doc__},
{"ulong_func", (PyCFunction)PY_ulong_func, METH_VARARGS|METH_KEYWORDS,
    PY_ulong_func__doc__},
{"ulong_long_func", (PyCFunction)PY_ulong_long_func,
    METH_VARARGS|METH_KEYWORDS, PY_ulong_long_func__doc__},
{"ulong_int_func", (PyCFunction)PY_ulong_int_func,
    METH_VARARGS|METH_KEYWORDS, PY_ulong_int_func__doc__},
{"int8_func", (PyCFunction)PY_int8_func, METH_VARARGS|METH_KEYWORDS,
    PY_int8_func__doc__},
{"int16_func", (PyCFunction)PY_int16_func, METH_VARARGS|METH_KEYWORDS,
    PY_int16_func__doc__},
{"int32_func", (PyCFunction)PY_int32_func, METH_VARARGS|METH_KEYWORDS,
    PY_int32_func__doc__},
{"int64_func", (PyCFunction)PY_int64_func, METH_VARARGS|METH_KEYWORDS,
    PY_int64_func__doc__},
{"uint8_func", (PyCFunction)PY_uint8_func, METH_VARARGS|METH_KEYWORDS,
    PY_uint8_func__doc__},
{"uint16_func", (PyCFunction)PY_uint16_func, METH_VARARGS|METH_KEYWORDS,
    PY_uint16_func__doc__},
{"uint32_func", (PyCFunction)PY_uint32_func, METH_VARARGS|METH_KEYWORDS,
    PY_uint32_func__doc__},
{"uint64_func", (PyCFunction)PY_uint64_func, METH_VARARGS|METH_KEYWORDS,
    PY_uint64_func__doc__},
{"size_func", (PyCFunction)PY_size_func, METH_VARARGS|METH_KEYWORDS,
    PY_size_func__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

/*
 * inittypes - Initialization function for the module
 * *must* be called inittypes
 */
static char PY__doc__[] =
"library documentation"
;

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#if PY_MAJOR_VERSION >= 3
static int types_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int types_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "types", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
    types_traverse, /* m_traverse */
    types_clear, /* m_clear */
    NULL  /* m_free */
};

#define RETVAL m
#define INITERROR return NULL
#else
#define RETVAL
#define INITERROR return
#endif

extern "C" PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_types(void)
#else
inittypes(void)
#endif
{
    PyObject *m = NULL;
    const char * error_name = "types.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("types", PY_methods,
        PY__doc__,
        (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL)
        return RETVAL;
    struct module_state *st = GETSTATE(m);


    PY_error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
    if (PY_error_obj == NULL)
        return RETVAL;
    st->error = PY_error_obj;
    PyModule_AddObject(m, "Error", st->error);

    // splicer begin C_init_body
    // splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module types");
    return RETVAL;
}

