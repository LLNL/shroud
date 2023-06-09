// pytypesmodule.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pytypesmodule.hpp"

// splicer begin include
// splicer end include

#ifdef __cplusplus
#define SHROUD_UNUSED(param)
#else
#define SHROUD_UNUSED(param) param
#endif

#if PY_MAJOR_VERSION >= 3
#define PyInt_AsLong PyLong_AsLong
#define PyInt_FromLong PyLong_FromLong
#define PyInt_FromSize_t PyLong_FromSize_t
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#endif

// splicer begin C_definition
// splicer end C_definition
PyObject *PY_error_obj;
// splicer begin additional_functions
// splicer end additional_functions

// ----------------------------------------
// Function:  short short_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  short arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_short_func__doc__[] =
"documentation"
;

static PyObject *
PY_short_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.short_func
    short arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "h:short_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    short SHCXX_rv = short_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.short_func
}

// ----------------------------------------
// Function:  int int_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  int arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_int_func__doc__[] =
"documentation"
;

static PyObject *
PY_int_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.int_func
    int arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:int_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    int SHCXX_rv = int_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.int_func
}

// ----------------------------------------
// Function:  long long_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  long arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_long_func__doc__[] =
"documentation"
;

static PyObject *
PY_long_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.long_func
    long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l:long_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    long SHCXX_rv = long_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.long_func
}

// ----------------------------------------
// Function:  long long long_long_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  long long arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_long_long_func__doc__[] =
"documentation"
;

static PyObject *
PY_long_long_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.long_long_func
    long long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L:long_long_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    long long SHCXX_rv = long_long_func(arg1);

    // post_call
    SHTPy_rv = Py_BuildValue("L", SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.long_long_func
}

// ----------------------------------------
// Function:  short int short_int_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  short int arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_short_int_func__doc__[] =
"documentation"
;

static PyObject *
PY_short_int_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.short_int_func
    short arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "h:short_int_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    short SHCXX_rv = short_int_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.short_int_func
}

// ----------------------------------------
// Function:  long int long_int_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  long int arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_long_int_func__doc__[] =
"documentation"
;

static PyObject *
PY_long_int_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.long_int_func
    long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l:long_int_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    long SHCXX_rv = long_int_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.long_int_func
}

// ----------------------------------------
// Function:  long long int long_long_int_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  long long int arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_long_long_int_func__doc__[] =
"documentation"
;

static PyObject *
PY_long_long_int_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.long_long_int_func
    long long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L:long_long_int_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    long long SHCXX_rv = long_long_int_func(arg1);

    // post_call
    SHTPy_rv = Py_BuildValue("L", SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.long_long_int_func
}

// ----------------------------------------
// Function:  unsigned unsigned_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  unsigned arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_unsigned_func__doc__[] =
"documentation"
;

static PyObject *
PY_unsigned_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.unsigned_func
    unsigned int arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "I:unsigned_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    unsigned int SHCXX_rv = unsigned_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.unsigned_func
}

// ----------------------------------------
// Function:  unsigned short ushort_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  unsigned short arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_ushort_func__doc__[] =
"documentation"
;

static PyObject *
PY_ushort_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.ushort_func
    unsigned short arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "H:ushort_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    unsigned short SHCXX_rv = ushort_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.ushort_func
}

// ----------------------------------------
// Function:  unsigned int uint_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  unsigned int arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_uint_func__doc__[] =
"documentation"
;

static PyObject *
PY_uint_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.uint_func
    unsigned int arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "I:uint_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    unsigned int SHCXX_rv = uint_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.uint_func
}

// ----------------------------------------
// Function:  unsigned long ulong_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  unsigned long arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_ulong_func__doc__[] =
"documentation"
;

static PyObject *
PY_ulong_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.ulong_func
    unsigned long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "k:ulong_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    unsigned long SHCXX_rv = ulong_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.ulong_func
}

// ----------------------------------------
// Function:  unsigned long long ulong_long_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  unsigned long long arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_ulong_long_func__doc__[] =
"documentation"
;

static PyObject *
PY_ulong_long_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.ulong_long_func
    unsigned long long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K:ulong_long_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    unsigned long long SHCXX_rv = ulong_long_func(arg1);

    // post_call
    SHTPy_rv = Py_BuildValue("K", SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.ulong_long_func
}

// ----------------------------------------
// Function:  unsigned long int ulong_int_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  unsigned long int arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_ulong_int_func__doc__[] =
"documentation"
;

static PyObject *
PY_ulong_int_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.ulong_int_func
    unsigned long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "k:ulong_int_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    unsigned long SHCXX_rv = ulong_int_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.ulong_int_func
}

// ----------------------------------------
// Function:  int8_t int8_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  int8_t arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_int8_func__doc__[] =
"documentation"
;

static PyObject *
PY_int8_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.int8_func
    int8_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:int8_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    int8_t SHCXX_rv = int8_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.int8_func
}

// ----------------------------------------
// Function:  int16_t int16_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  int16_t arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_int16_func__doc__[] =
"documentation"
;

static PyObject *
PY_int16_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.int16_func
    int16_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:int16_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    int16_t SHCXX_rv = int16_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.int16_func
}

// ----------------------------------------
// Function:  int32_t int32_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  int32_t arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_int32_func__doc__[] =
"documentation"
;

static PyObject *
PY_int32_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.int32_func
    int32_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:int32_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    int32_t SHCXX_rv = int32_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.int32_func
}

// ----------------------------------------
// Function:  int64_t int64_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  int64_t arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_int64_func__doc__[] =
"documentation"
;

static PyObject *
PY_int64_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.int64_func
    int64_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L:int64_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    int64_t SHCXX_rv = int64_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.int64_func
}

// ----------------------------------------
// Function:  uint8_t uint8_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  uint8_t arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_uint8_func__doc__[] =
"documentation"
;

static PyObject *
PY_uint8_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.uint8_func
    uint8_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:uint8_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    uint8_t SHCXX_rv = uint8_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.uint8_func
}

// ----------------------------------------
// Function:  uint16_t uint16_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  uint16_t arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_uint16_func__doc__[] =
"documentation"
;

static PyObject *
PY_uint16_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.uint16_func
    uint16_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:uint16_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    uint16_t SHCXX_rv = uint16_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.uint16_func
}

// ----------------------------------------
// Function:  uint32_t uint32_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  uint32_t arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_uint32_func__doc__[] =
"documentation"
;

static PyObject *
PY_uint32_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.uint32_func
    uint32_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:uint32_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    uint32_t SHCXX_rv = uint32_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.uint32_func
}

// ----------------------------------------
// Function:  uint64_t uint64_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  uint64_t arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_uint64_func__doc__[] =
"documentation"
;

static PyObject *
PY_uint64_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.uint64_func
    uint64_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L:uint64_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    uint64_t SHCXX_rv = uint64_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.uint64_func
}

// ----------------------------------------
// Function:  size_t size_func
// Attrs:     +intent(function)
// Requested: py_function_native_scalar
// Match:     py_default
// ----------------------------------------
// Argument:  size_t arg1 +value
// Attrs:     +intent(in)
// Requested: py_in_native_scalar
// Match:     py_default
static char PY_size_func__doc__[] =
"documentation"
;

static PyObject *
PY_size_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.size_func
    size_t arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n:size_func",
        const_cast<char **>(SHT_kwlist), &arg1))
        return nullptr;

    size_t SHCXX_rv = size_func(arg1);

    // post_call
    SHTPy_rv = PyInt_FromSize_t(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.size_func
}

// ----------------------------------------
// Function:  bool bool_func
// Attrs:     +intent(function)
// Requested: py_function_bool_scalar
// Match:     py_function_bool
// ----------------------------------------
// Argument:  bool arg +value
// Attrs:     +intent(in)
// Requested: py_in_bool_scalar
// Match:     py_in_bool
static char PY_bool_func__doc__[] =
"documentation"
;

static PyObject *
PY_bool_func(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.bool_func
    bool arg;
    PyObject * SHPy_arg;
    const char *SHT_kwlist[] = {
        "arg",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:bool_func",
        const_cast<char **>(SHT_kwlist), &PyBool_Type, &SHPy_arg))
        return nullptr;

    // pre_call
    arg = PyObject_IsTrue(SHPy_arg);

    bool SHCXX_rv = bool_func(arg);

    // post_call
    SHTPy_rv = PyBool_FromLong(SHCXX_rv);
    if (SHTPy_rv == nullptr) goto fail;

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return nullptr;
// splicer end function.bool_func
}

// ----------------------------------------
// Function:  bool returnBoolAndOthers
// Attrs:     +intent(function)
// Requested: py_function_bool_scalar
// Match:     py_function_bool
// ----------------------------------------
// Argument:  int * flag +intent(out)
// Attrs:     +intent(out)
// Exact:     py_out_native_*
static char PY_returnBoolAndOthers__doc__[] =
"documentation"
;

/**
 * \brief Function which returns bool with other intent(out) arguments
 *
 * Python treats bool differently since Py_BuildValue does not support
 * bool until Python 3.3.
 * Must create a PyObject with PyBool_FromLong then include that object
 * in call to Py_BuildValue as type 'O'.  But since two return values
 * are being created, function return and argument flag, rename first
 * local C variable to avoid duplicate names in wrapper.
 */
static PyObject *
PY_returnBoolAndOthers(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin function.returnBoolAndOthers
    int flag;
    PyObject * SHTPy_rv = nullptr;
    PyObject *SHPyResult = nullptr;  // return value object

    bool SHCXX_rv = returnBoolAndOthers(&flag);

    // post_call
    SHTPy_rv = PyBool_FromLong(SHCXX_rv);
    if (SHTPy_rv == nullptr) goto fail;
    SHPyResult = Py_BuildValue("Oi", SHTPy_rv, flag);

    return SHPyResult;

fail:
    Py_XDECREF(SHTPy_rv);
    return nullptr;
// splicer end function.returnBoolAndOthers
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
{"bool_func", (PyCFunction)PY_bool_func, METH_VARARGS|METH_KEYWORDS,
    PY_bool_func__doc__},
{"returnBoolAndOthers", (PyCFunction)PY_returnBoolAndOthers,
    METH_NOARGS, PY_returnBoolAndOthers__doc__},
{nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
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
    nullptr, /* m_reload */
    types_traverse, /* m_traverse */
    types_clear, /* m_clear */
    NULL  /* m_free */
};

#define RETVAL m
#define INITERROR return nullptr
#else
#define RETVAL
#define INITERROR return
#endif

extern "C" PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_shtypes(void)
#else
initshtypes(void)
#endif
{
    PyObject *m = nullptr;
    const char * error_name = "types.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("shtypes", PY_methods,
        PY__doc__,
        (PyObject*)nullptr,PYTHON_API_VERSION);
#endif
    if (m == nullptr)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

    PY_error_obj = PyErr_NewException((char *) error_name, nullptr, nullptr);
    if (PY_error_obj == nullptr)
        return RETVAL;
    st->error = PY_error_obj;
    PyModule_AddObject(m, "Error", st->error);

    // splicer begin C_init_body
    // splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module shtypes");
    return RETVAL;
}

