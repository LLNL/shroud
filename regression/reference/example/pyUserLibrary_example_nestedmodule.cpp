// pyUserLibrary_example_nestedmodule.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pyUserLibrarymodule.hpp"
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SHROUD_USERLIBRARY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// splicer begin namespace.example::nested.include
// splicer end namespace.example::nested.include

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

// splicer begin namespace.example::nested.C_definition
// splicer end namespace.example::nested.C_definition
// splicer begin namespace.example::nested.additional_functions
// splicer end namespace.example::nested.additional_functions

// ----------------------------------------
// Function:  void local_function1
// Statement: py_default
static char PP_local_function1__doc__[] =
"documentation"
;

static PyObject *
PP_local_function1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.function.local_function1
    example::nested::local_function1();
    Py_RETURN_NONE;
// splicer end namespace.example::nested.function.local_function1
}

// ----------------------------------------
// Function:  bool isNameValid
// Statement: py_function_bool_scalar
// ----------------------------------------
// Argument:  const std::string & name
// Statement: py_in_string_&
static char PP_isNameValid__doc__[] =
"documentation"
;

static PyObject *
PP_isNameValid(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.isNameValid
    char * name;
    const char *SHT_kwlist[] = {
        "name",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:isNameValid",
        const_cast<char **>(SHT_kwlist), &name))
        return nullptr;

    // post_declare
    const std::string SH_name(name);

    bool SHCXX_rv = example::nested::isNameValid(SH_name);

    // post_call
    SHTPy_rv = PyBool_FromLong(SHCXX_rv);
    if (SHTPy_rv == nullptr) goto fail;

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return nullptr;
// splicer end namespace.example::nested.function.isNameValid
}

// ----------------------------------------
// Function:  bool isInitialized
// Statement: py_function_bool_scalar
static char PP_isInitialized__doc__[] =
"documentation"
;

static PyObject *
PP_isInitialized(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.function.isInitialized
    PyObject * SHTPy_rv = nullptr;

    bool SHCXX_rv = example::nested::isInitialized();

    // post_call
    SHTPy_rv = PyBool_FromLong(SHCXX_rv);
    if (SHTPy_rv == nullptr) goto fail;

    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return nullptr;
// splicer end namespace.example::nested.function.isInitialized
}

// ----------------------------------------
// Function:  void test_names
// Statement: py_default
// ----------------------------------------
// Argument:  const std::string & name
// Statement: py_in_string_&
static PyObject *
PP_test_names(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.test_names
    char * name;
    const char *SHT_kwlist[] = {
        "name",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:test_names",
        const_cast<char **>(SHT_kwlist), &name))
        return nullptr;

    // post_declare
    const std::string SH_name(name);

    example::nested::test_names(SH_name);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.function.test_names
}

// ----------------------------------------
// Function:  void test_names
// Statement: py_default
// ----------------------------------------
// Argument:  const std::string & name
// Statement: py_in_string_&
// ----------------------------------------
// Argument:  int flag
// Statement: py_in_native_scalar
static PyObject *
PP_test_names_flag(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.test_names_flag
    char * name;
    int flag;
    const char *SHT_kwlist[] = {
        "name",
        "flag",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "si:test_names",
        const_cast<char **>(SHT_kwlist), &name, &flag))
        return nullptr;

    // post_declare
    const std::string SH_name(name);

    example::nested::test_names(SH_name, flag);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.function.test_names_flag
}

// ----------------------------------------
// Function:  void testoptional
// Statement: py_default
// ----------------------------------------
// Argument:  int i=1
// Statement: py_in_native_scalar
// ----------------------------------------
// Argument:  long j=2
// Statement: py_in_native_scalar
static char PP_testoptional_2__doc__[] =
"documentation"
;

static PyObject *
PP_testoptional_2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.testoptional
    Py_ssize_t SH_nargs = 0;
    int i;
    long j;
    const char *SHT_kwlist[] = {
        "i",
        "j",
        nullptr };

    if (args != nullptr) SH_nargs += PyTuple_Size(args);
    if (kwds != nullptr) SH_nargs += PyDict_Size(args);
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|il:testoptional",
        const_cast<char **>(SHT_kwlist), &i, &j))
        return nullptr;
    switch (SH_nargs) {
    case 0:
        example::nested::testoptional();
        break;
    case 1:
        example::nested::testoptional(i);
        break;
    case 2:
        example::nested::testoptional(i, j);
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Wrong number of arguments");
        return nullptr;
    }
    Py_RETURN_NONE;
// splicer end namespace.example::nested.function.testoptional
}

// ----------------------------------------
// Function:  size_t test_size_t
// Statement: py_function_native_scalar
static char PP_test_size_t__doc__[] =
"documentation"
;

static PyObject *
PP_test_size_t(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.function.test_size_t
    PyObject * SHTPy_rv = nullptr;

    size_t SHCXX_rv = example::nested::test_size_t();

    // post_call
    SHTPy_rv = PyInt_FromSize_t(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.function.test_size_t
}

// ----------------------------------------
// Function:  void testmpi
// Statement: py_default
// ----------------------------------------
// Argument:  MPI_Comm comm
// Statement: py_in_unknown_scalar
#ifdef HAVE_MPI
static PyObject *
PP_testmpi_mpi(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.testmpi_mpi
    MPI_Fint comm;
    const char *SHT_kwlist[] = {
        "comm",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:testmpi",
        const_cast<char **>(SHT_kwlist), &comm))
        return nullptr;

    // post_declare
    MPI_Comm SH_comm = MPI_Comm_f2c(comm);

    example::nested::testmpi(SH_comm);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.function.testmpi_mpi
}
#endif // ifdef HAVE_MPI

// ----------------------------------------
// Function:  void testmpi
// Statement: py_default
#ifndef HAVE_MPI
static PyObject *
PP_testmpi_serial(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin namespace.example::nested.function.testmpi_serial
    example::nested::testmpi();
    Py_RETURN_NONE;
// splicer end namespace.example::nested.function.testmpi_serial
}
#endif // ifndef HAVE_MPI

// ----------------------------------------
// Function:  void FuncPtr1
// Statement: py_default
// ----------------------------------------
// Argument:  void ( * get)(void)
// Exact:     py_default
static char PP_FuncPtr1__doc__[] =
"documentation"
;

/**
 * \brief subroutine
 *
 */
static PyObject *
PP_FuncPtr1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.FuncPtr1
    void ( * get)(void);
    const char *SHT_kwlist[] = {
        "get",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:FuncPtr1",
        const_cast<char **>(SHT_kwlist), &get))
        return nullptr;

    example::nested::FuncPtr1(get);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.function.FuncPtr1
}

// ----------------------------------------
// Function:  void FuncPtr2
// Statement: py_default
// ----------------------------------------
// Argument:  double * ( * get)(void)
// Exact:     py_default
static char PP_FuncPtr2__doc__[] =
"documentation"
;

/**
 * \brief return a pointer
 *
 */
static PyObject *
PP_FuncPtr2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.FuncPtr2
    double * ( * get)(void);
    const char *SHT_kwlist[] = {
        "get",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d:FuncPtr2",
        const_cast<char **>(SHT_kwlist), &get))
        return nullptr;

    example::nested::FuncPtr2(get);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.function.FuncPtr2
}

// ----------------------------------------
// Function:  void FuncPtr3
// Statement: py_default
// ----------------------------------------
// Argument:  double ( * get)(int i +value, int +value)
// Exact:     py_default
static char PP_FuncPtr3__doc__[] =
"documentation"
;

/**
 * \brief abstract argument
 *
 */
static PyObject *
PP_FuncPtr3(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.FuncPtr3
    double ( * get)(int i, int);
    const char *SHT_kwlist[] = {
        "get",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d:FuncPtr3",
        const_cast<char **>(SHT_kwlist), &get))
        return nullptr;

    example::nested::FuncPtr3(get);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.function.FuncPtr3
}

// ----------------------------------------
// Function:  void FuncPtr5
// Statement: py_default
// ----------------------------------------
// Argument:  void ( * get)(int verylongname1 +value, int verylongname2 +value, int verylongname3 +value, int verylongname4 +value, int verylongname5 +value, int verylongname6 +value, int verylongname7 +value, int verylongname8 +value, int verylongname9 +value, int verylongname10 +value)
// Exact:     py_default
static char PP_FuncPtr5__doc__[] =
"documentation"
;

static PyObject *
PP_FuncPtr5(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.FuncPtr5
    void ( * get)(int verylongname1, int verylongname2,
        int verylongname3, int verylongname4, int verylongname5,
        int verylongname6, int verylongname7, int verylongname8,
        int verylongname9, int verylongname10);
    const char *SHT_kwlist[] = {
        "get",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:FuncPtr5",
        const_cast<char **>(SHT_kwlist), &get))
        return nullptr;

    example::nested::FuncPtr5(get);
    Py_RETURN_NONE;
// splicer end namespace.example::nested.function.FuncPtr5
}

// ----------------------------------------
// Function:  void verylongfunctionname1
// Statement: py_default
// ----------------------------------------
// Argument:  int * verylongname1 +intent(inout)
// Statement: py_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname2 +intent(inout)
// Statement: py_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname3 +intent(inout)
// Statement: py_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname4 +intent(inout)
// Statement: py_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname5 +intent(inout)
// Statement: py_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname6 +intent(inout)
// Statement: py_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname7 +intent(inout)
// Statement: py_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname8 +intent(inout)
// Statement: py_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname9 +intent(inout)
// Statement: py_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname10 +intent(inout)
// Statement: py_inout_native_*
static char PP_verylongfunctionname1__doc__[] =
"documentation"
;

static PyObject *
PP_verylongfunctionname1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.verylongfunctionname1
    int verylongname1;
    int verylongname2;
    int verylongname3;
    int verylongname4;
    int verylongname5;
    int verylongname6;
    int verylongname7;
    int verylongname8;
    int verylongname9;
    int verylongname10;
    const char *SHT_kwlist[] = {
        "verylongname1",
        "verylongname2",
        "verylongname3",
        "verylongname4",
        "verylongname5",
        "verylongname6",
        "verylongname7",
        "verylongname8",
        "verylongname9",
        "verylongname10",
        nullptr };
    PyObject *SHTPy_rv = nullptr;  // return value object

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "iiiiiiiiii:verylongfunctionname1",
        const_cast<char **>(SHT_kwlist), &verylongname1, &verylongname2,
        &verylongname3, &verylongname4, &verylongname5, &verylongname6,
        &verylongname7, &verylongname8, &verylongname9,
        &verylongname10))
        return nullptr;

    example::nested::verylongfunctionname1(&verylongname1,
        &verylongname2, &verylongname3, &verylongname4, &verylongname5,
        &verylongname6, &verylongname7, &verylongname8, &verylongname9,
        &verylongname10);

    // post_call
    SHTPy_rv = Py_BuildValue("iiiiiiiiii", verylongname1, verylongname2,
        verylongname3, verylongname4, verylongname5, verylongname6,
        verylongname7, verylongname8, verylongname9, verylongname10);

    return SHTPy_rv;
// splicer end namespace.example::nested.function.verylongfunctionname1
}

// ----------------------------------------
// Function:  int verylongfunctionname2
// Statement: py_function_native_scalar
// ----------------------------------------
// Argument:  int verylongname1
// Statement: py_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname2
// Statement: py_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname3
// Statement: py_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname4
// Statement: py_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname5
// Statement: py_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname6
// Statement: py_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname7
// Statement: py_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname8
// Statement: py_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname9
// Statement: py_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname10
// Statement: py_in_native_scalar
static char PP_verylongfunctionname2__doc__[] =
"documentation"
;

static PyObject *
PP_verylongfunctionname2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.verylongfunctionname2
    int verylongname1;
    int verylongname2;
    int verylongname3;
    int verylongname4;
    int verylongname5;
    int verylongname6;
    int verylongname7;
    int verylongname8;
    int verylongname9;
    int verylongname10;
    const char *SHT_kwlist[] = {
        "verylongname1",
        "verylongname2",
        "verylongname3",
        "verylongname4",
        "verylongname5",
        "verylongname6",
        "verylongname7",
        "verylongname8",
        "verylongname9",
        "verylongname10",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "iiiiiiiiii:verylongfunctionname2",
        const_cast<char **>(SHT_kwlist), &verylongname1, &verylongname2,
        &verylongname3, &verylongname4, &verylongname5, &verylongname6,
        &verylongname7, &verylongname8, &verylongname9,
        &verylongname10))
        return nullptr;

    int SHCXX_rv = example::nested::verylongfunctionname2(verylongname1,
        verylongname2, verylongname3, verylongname4, verylongname5,
        verylongname6, verylongname7, verylongname8, verylongname9,
        verylongname10);

    // post_call
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end namespace.example::nested.function.verylongfunctionname2
}

// ----------------------------------------
// Function:  void cos_doubles
// Statement: py_default
// ----------------------------------------
// Argument:  double * in +intent(in)+rank(2)
// Statement: py_in_native_*_numpy
// ----------------------------------------
// Argument:  double * out +dimension(shape(in))+intent(out)
// Statement: py_out_native_*_numpy
// ----------------------------------------
// Argument:  int sizein +implied(size(in))
// Exact:     py_default
static char PP_cos_doubles__doc__[] =
"documentation"
;

/**
 * \brief Test multidimensional arrays with allocatable
 *
 */
static PyObject *
PP_cos_doubles(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.cos_doubles
    double * in;
    PyObject * SHTPy_in;
    PyArrayObject * SHPy_in = nullptr;
    double * out;
    npy_intp SHD_out[1];
    PyArrayObject * SHPy_out = nullptr;
    int sizein;
    const char *SHT_kwlist[] = {
        "in",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:cos_doubles",
        const_cast<char **>(SHT_kwlist), &SHTPy_in))
        return nullptr;

    // post_parse
    SHPy_in = reinterpret_cast<PyArrayObject *>
        (PyArray_ContiguousFromObject(SHTPy_in, NPY_DOUBLE, 2, 2));
    if (SHPy_in == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "in must be a 2-D array of double");
        goto fail;
    }
    SHD_out[0] = shape(in);
    SHPy_out = reinterpret_cast<PyArrayObject *>
        (PyArray_SimpleNew(1, SHD_out, NPY_DOUBLE));
    if (SHPy_out == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "out must be a 1-D array of double");
        goto fail;
    }

    // pre_call
    in = static_cast<double *>(PyArray_DATA(SHPy_in));
    out = static_cast<double *>(PyArray_DATA(SHPy_out));
    sizein = PyArray_SIZE(SHPy_in);

    example::nested::cos_doubles(in, out, sizein);

    // cleanup
    Py_DECREF(SHPy_in);

    return (PyObject *) SHPy_out;

fail:
    Py_XDECREF(SHPy_in);
    Py_XDECREF(SHPy_out);
    return nullptr;
// splicer end namespace.example::nested.function.cos_doubles
}

static char PP_test_names__doc__[] =
"documentation"
;

static PyObject *
PP_test_names(
  PyObject *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.test_names
    Py_ssize_t SHT_nargs = 0;
    if (args != nullptr) SHT_nargs += PyTuple_Size(args);
    if (kwds != nullptr) SHT_nargs += PyDict_Size(args);
    PyObject *rvobj;
    if (SHT_nargs == 1) {
        rvobj = PP_test_names(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs == 2) {
        rvobj = PP_test_names_flag(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
    return nullptr;
// splicer end namespace.example::nested.function.test_names
}

static char PP_testmpi__doc__[] =
"documentation"
;

static PyObject *
PP_testmpi(
  PyObject *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin namespace.example::nested.function.testmpi
    Py_ssize_t SHT_nargs = 0;
    if (args != nullptr) SHT_nargs += PyTuple_Size(args);
    if (kwds != nullptr) SHT_nargs += PyDict_Size(args);
    PyObject *rvobj;
#ifdef HAVE_MPI
    if (SHT_nargs == 1) {
        rvobj = PP_testmpi_mpi(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
#endif // ifdef HAVE_MPI
#ifndef HAVE_MPI
    if (SHT_nargs == 0) {
        rvobj = PP_testmpi_serial(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
#endif // ifndef HAVE_MPI
    PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
    return nullptr;
// splicer end namespace.example::nested.function.testmpi
}
static PyMethodDef PP_methods[] = {
{"local_function1", (PyCFunction)PP_local_function1, METH_NOARGS,
    PP_local_function1__doc__},
{"isNameValid", (PyCFunction)PP_isNameValid, METH_VARARGS|METH_KEYWORDS,
    PP_isNameValid__doc__},
{"isInitialized", (PyCFunction)PP_isInitialized, METH_NOARGS,
    PP_isInitialized__doc__},
{"testoptional", (PyCFunction)PP_testoptional_2,
    METH_VARARGS|METH_KEYWORDS, PP_testoptional_2__doc__},
{"test_size_t", (PyCFunction)PP_test_size_t, METH_NOARGS,
    PP_test_size_t__doc__},
{"FuncPtr1", (PyCFunction)PP_FuncPtr1, METH_VARARGS|METH_KEYWORDS,
    PP_FuncPtr1__doc__},
{"FuncPtr2", (PyCFunction)PP_FuncPtr2, METH_VARARGS|METH_KEYWORDS,
    PP_FuncPtr2__doc__},
{"FuncPtr3", (PyCFunction)PP_FuncPtr3, METH_VARARGS|METH_KEYWORDS,
    PP_FuncPtr3__doc__},
{"FuncPtr5", (PyCFunction)PP_FuncPtr5, METH_VARARGS|METH_KEYWORDS,
    PP_FuncPtr5__doc__},
{"verylongfunctionname1", (PyCFunction)PP_verylongfunctionname1,
    METH_VARARGS|METH_KEYWORDS, PP_verylongfunctionname1__doc__},
{"verylongfunctionname2", (PyCFunction)PP_verylongfunctionname2,
    METH_VARARGS|METH_KEYWORDS, PP_verylongfunctionname2__doc__},
{"cos_doubles", (PyCFunction)PP_cos_doubles, METH_VARARGS|METH_KEYWORDS,
    PP_cos_doubles__doc__},
{"test_names", (PyCFunction)PP_test_names, METH_VARARGS|METH_KEYWORDS,
    PP_test_names__doc__},
{"testmpi", (PyCFunction)PP_testmpi, METH_VARARGS|METH_KEYWORDS,
    PP_testmpi__doc__},
{nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
};

#if PY_MAJOR_VERSION >= 3
static char PP__doc__[] =
"XXX submodule doc"  //"library documentation"
;

struct module_state {
    PyObject *error;
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "userlibrary.example.nested", /* m_name */
    PP__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PP_methods, /* m_methods */
    nullptr, /* m_reload */
//    userlibrary_traverse, /* m_traverse */
//    userlibrary_clear, /* m_clear */
    nullptr, /* m_traverse */
    nullptr, /* m_clear */
    nullptr  /* m_free */
};
#endif
#define RETVAL nullptr

PyObject *PP_init_userlibrary_example_nested(void)
{
    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3((char *) "nested", PP_methods, nullptr);
#endif
    if (m == nullptr)
        return nullptr;


    // ExClass1
    PP_ExClass1_Type.tp_new   = PyType_GenericNew;
    PP_ExClass1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PP_ExClass1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PP_ExClass1_Type);
    PyModule_AddObject(m, "ExClass1", (PyObject *)&PP_ExClass1_Type);

    // ExClass2
    PP_ExClass2_Type.tp_new   = PyType_GenericNew;
    PP_ExClass2_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PP_ExClass2_Type) < 0)
        return RETVAL;
    Py_INCREF(&PP_ExClass2_Type);
    PyModule_AddObject(m, "ExClass2", (PyObject *)&PP_ExClass2_Type);

    return m;
}

