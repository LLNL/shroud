// pyTutorialmodule.cpp
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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "tutorial.hpp"

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

static char PY_Function1__doc__[] =
"documentation"
;

static PyObject *
PY_Function1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void Function1()
// splicer begin function.function1
    tutorial::Function1();
    Py_RETURN_NONE;
// splicer end function.function1
}

static char PY_Function2__doc__[] =
"documentation"
;

static PyObject *
PY_Function2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// double Function2(double arg1 +intent(in)+value, int arg2 +intent(in)+value)
// splicer begin function.function2
    double arg1;
    int arg2;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "di:Function2",
        const_cast<char **>(SHT_kwlist), &arg1, &arg2))
        return NULL;

    double SHC_rv = tutorial::Function2(arg1, arg2);

    // post_call
    PyObject * SHTPy_rv = PyFloat_FromDouble(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.function2
}

static char PY_Sum__doc__[] =
"documentation"
;

static PyObject *
PY_Sum(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void Sum(size_t len +implied(size(values))+intent(in)+value, int * values +dimension(:)+intent(in), int * result +intent(out))
// splicer begin function.sum
    PyObject * SHTPy_values;
    PyArrayObject * SHPy_values = NULL;
    const char *SHT_kwlist[] = {
        "values",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:Sum",
        const_cast<char **>(SHT_kwlist), &SHTPy_values))
        return NULL;

    // post_parse
    SHPy_values = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        SHTPy_values, NPY_INT, NPY_ARRAY_IN_ARRAY));
    if (SHPy_values == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "values must be a 1-D array of int");
        goto fail;
    }
    {
        // pre_call
        int * values = static_cast<int *>(PyArray_DATA(SHPy_values));
        int result;  // intent(out)
        size_t len = PyArray_SIZE(SHPy_values);

        tutorial::Sum(len, values, &result);

        // post_call
        PyObject * SHPy_result = PyInt_FromLong(result);

        // cleanup
        Py_DECREF(SHPy_values);

        return (PyObject *) SHPy_result;
    }

fail:
    Py_XDECREF(SHPy_values);
    return NULL;
// splicer end function.sum
}

static char PY_TypeLongLong__doc__[] =
"documentation"
;

static PyObject *
PY_TypeLongLong(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// long long TypeLongLong(long long arg1 +intent(in)+value)
// splicer begin function.type_long_long
    long long arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L:TypeLongLong",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    long long SHC_rv = tutorial::TypeLongLong(arg1);

    // post_call
    PyObject * SHTPy_rv = Py_BuildValue("L", SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.type_long_long
}

static char PY_Function3__doc__[] =
"documentation"
;

static PyObject *
PY_Function3(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// bool Function3(bool arg +intent(in)+value)
// splicer begin function.function3
    PyObject * SHPy_arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:Function3",
        const_cast<char **>(SHT_kwlist), &PyBool_Type, &SHPy_arg))
        return NULL;

    // pre_call
    bool arg = PyObject_IsTrue(SHPy_arg);

    bool SHC_rv = tutorial::Function3(arg);

    // post_call
    PyObject * SHTPy_rv = PyBool_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.function3
}

static char PY_ReturnIntPtrDim__doc__[] =
"documentation"
;

static PyObject *
PY_ReturnIntPtrDim(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// int * ReturnIntPtrDim(int * len +hidden+intent(out)) +dimension(len)
// splicer begin function.return_int_ptr_dim
    // pre_call
    int len;  // intent(out)

    int * SHC_rv = tutorial::ReturnIntPtrDim(&len);

    // post_call
    npy_intp SHD_ReturnIntPtrDim[1] = { len };
    PyObject * SHTPy_rv = PyArray_SimpleNewFromData(1, SHD_ReturnIntPtrDim, NPY_INT, SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.return_int_ptr_dim
}

static char PY_Function4a__doc__[] =
"documentation"
;

static PyObject *
PY_Function4a(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// const std::string Function4a(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in)) +len(30)
// splicer begin function.function4a
    const char * arg1;
    const char * arg2;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss:Function4a",
        const_cast<char **>(SHT_kwlist), &arg1, &arg2))
        return NULL;

    // post_parse
    const std::string SH_arg1(arg1);
    const std::string SH_arg2(arg2);

    const std::string SHCXX_rv = tutorial::Function4a(SH_arg1, SH_arg2);

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.function4a
}

static char PY_Function4b__doc__[] =
"documentation"
;

static PyObject *
PY_Function4b(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// const std::string & Function4b(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in))
// splicer begin function.function4b
    const char * arg1;
    const char * arg2;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss:Function4b",
        const_cast<char **>(SHT_kwlist), &arg1, &arg2))
        return NULL;

    // post_parse
    const std::string SH_arg1(arg1);
    const std::string SH_arg2(arg2);

    const std::string & SHCXX_rv = tutorial::Function4b(SH_arg1,
        SH_arg2);

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.function4b
}

static char PY_Function5_arg1_arg2__doc__[] =
"documentation"
;

static PyObject *
PY_Function5_arg1_arg2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// double Function5(double arg1=3.1415 +intent(in)+value, bool arg2=true +intent(in)+value)
// splicer begin function.function5
    Py_ssize_t SH_nargs = 0;
    double arg1;
    PyObject * SHPy_arg2;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        NULL };
    double SHC_rv;

    if (args != NULL) SH_nargs += PyTuple_Size(args);
    if (kwds != NULL) SH_nargs += PyDict_Size(args);
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|dO!:Function5",
        const_cast<char **>(SHT_kwlist), &arg1, &PyBool_Type,
        &SHPy_arg2))
        return NULL;
    switch (SH_nargs) {
    case 0:
        SHC_rv = tutorial::Function5();
        break;
    case 1:
        SHC_rv = tutorial::Function5(arg1);
        break;
    case 2:
        {
            // pre_call
            bool arg2 = PyObject_IsTrue(SHPy_arg2);

            SHC_rv = tutorial::Function5(arg1, arg2);
            break;
        }
    }

    // post_call
    PyObject * SHTPy_rv = PyFloat_FromDouble(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.function5
}

static PyObject *
PY_Function6_from_name(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void Function6(const std::string & name +intent(in))
// splicer begin function.function6_from_name
    const char * name;
    const char *SHT_kwlist[] = {
        "name",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:Function6",
        const_cast<char **>(SHT_kwlist), &name))
        return NULL;

    // post_parse
    const std::string SH_name(name);

    tutorial::Function6(SH_name);
    Py_RETURN_NONE;
// splicer end function.function6_from_name
}

static PyObject *
PY_Function6_from_index(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void Function6(int indx +intent(in)+value)
// splicer begin function.function6_from_index
    int indx;
    const char *SHT_kwlist[] = {
        "indx",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:Function6",
        const_cast<char **>(SHT_kwlist), &indx))
        return NULL;

    tutorial::Function6(indx);
    Py_RETURN_NONE;
// splicer end function.function6_from_index
}

static char PY_Function9__doc__[] =
"documentation"
;

static PyObject *
PY_Function9(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void Function9(double arg +intent(in)+value)
// splicer begin function.function9
    double arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d:Function9",
        const_cast<char **>(SHT_kwlist), &arg))
        return NULL;

    tutorial::Function9(arg);
    Py_RETURN_NONE;
// splicer end function.function9
}

static PyObject *
PY_Function10_0(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void Function10()
// splicer begin function.function10_0
    tutorial::Function10();
    Py_RETURN_NONE;
// splicer end function.function10_0
}

static PyObject *
PY_Function10_1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void Function10(const std::string & name +intent(in), double arg2 +intent(in)+value)
// splicer begin function.function10_1
    const char * name;
    double arg2;
    const char *SHT_kwlist[] = {
        "name",
        "arg2",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sd:Function10",
        const_cast<char **>(SHT_kwlist), &name, &arg2))
        return NULL;

    // post_parse
    const std::string SH_name(name);

    tutorial::Function10(SH_name, arg2);
    Py_RETURN_NONE;
// splicer end function.function10_1
}

static PyObject *
PY_overload1_num_offset_stride(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int overload1(int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
// splicer begin function.overload1_num_offset_stride
    Py_ssize_t SH_nargs = 0;
    int num;
    int offset;
    int stride;
    const char *SHT_kwlist[] = {
        "num",
        "offset",
        "stride",
        NULL };
    int SHC_rv;

    if (args != NULL) SH_nargs += PyTuple_Size(args);
    if (kwds != NULL) SH_nargs += PyDict_Size(args);
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|ii:overload1",
        const_cast<char **>(SHT_kwlist), &num, &offset, &stride))
        return NULL;
    switch (SH_nargs) {
    case 1:
        SHC_rv = tutorial::overload1(num);
        break;
    case 2:
        SHC_rv = tutorial::overload1(num, offset);
        break;
    case 3:
        SHC_rv = tutorial::overload1(num, offset, stride);
        break;
    }

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.overload1_num_offset_stride
}

static PyObject *
PY_overload1_5(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int overload1(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
// splicer begin function.overload1_5
    Py_ssize_t SH_nargs = 0;
    double type;
    int num;
    int offset;
    int stride;
    const char *SHT_kwlist[] = {
        "type",
        "num",
        "offset",
        "stride",
        NULL };
    int SHC_rv;

    if (args != NULL) SH_nargs += PyTuple_Size(args);
    if (kwds != NULL) SH_nargs += PyDict_Size(args);
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "di|ii:overload1",
        const_cast<char **>(SHT_kwlist), &type, &num, &offset, &stride))
        return NULL;
    switch (SH_nargs) {
    case 2:
        SHC_rv = tutorial::overload1(type, num);
        break;
    case 3:
        SHC_rv = tutorial::overload1(type, num, offset);
        break;
    case 4:
        SHC_rv = tutorial::overload1(type, num, offset, stride);
        break;
    }

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.overload1_5
}

static char PY_typefunc__doc__[] =
"documentation"
;

static PyObject *
PY_typefunc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// TypeID typefunc(TypeID arg +intent(in)+value)
// splicer begin function.typefunc
    int arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:typefunc",
        const_cast<char **>(SHT_kwlist), &arg))
        return NULL;

    tutorial::TypeID SHC_rv = tutorial::typefunc(arg);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.typefunc
}

static char PY_enumfunc__doc__[] =
"documentation"
;

static PyObject *
PY_enumfunc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// EnumTypeID enumfunc(EnumTypeID arg +intent(in)+value)
// splicer begin function.enumfunc
    int arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:enumfunc",
        const_cast<char **>(SHT_kwlist), &arg))
        return NULL;

    // post_parse
    tutorial::EnumTypeID SH_arg = static_cast<tutorial::EnumTypeID>(arg);

    tutorial::EnumTypeID SHCXX_rv = tutorial::enumfunc(SH_arg);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.enumfunc
}

static char PY_colorfunc__doc__[] =
"documentation"
;

static PyObject *
PY_colorfunc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// Color colorfunc(Color arg +intent(in)+value)
// splicer begin function.colorfunc
    int arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:colorfunc",
        const_cast<char **>(SHT_kwlist), &arg))
        return NULL;

    // post_parse
    tutorial::Color SH_arg = static_cast<tutorial::Color>(arg);

    tutorial::Color SHCXX_rv = tutorial::colorfunc(SH_arg);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.colorfunc
}

static char PY_getMinMax__doc__[] =
"documentation"
;

static PyObject *
PY_getMinMax(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void getMinMax(int & min +intent(out), int & max +intent(out))
// splicer begin function.get_min_max
    // pre_call
    int min;  // intent(out)
    int max;  // intent(out)

    tutorial::getMinMax(min, max);

    // post_call
    PyObject * SHTPy_rv = Py_BuildValue("ii", min, max);

    return SHTPy_rv;
// splicer end function.get_min_max
}

static char PY_directionFunc__doc__[] =
"documentation"
;

static PyObject *
PY_directionFunc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// Class1::DIRECTION directionFunc(Class1::DIRECTION arg +intent(in)+value)
// splicer begin function.direction_func
    int arg;
    const char *SHT_kwlist[] = {
        "arg",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:directionFunc",
        const_cast<char **>(SHT_kwlist), &arg))
        return NULL;

    // post_parse
    tutorial::Class1::DIRECTION SH_arg = static_cast<tutorial::
        Class1::DIRECTION>(arg);

    tutorial::Class1::DIRECTION SHCXX_rv = tutorial::
        directionFunc(SH_arg);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.direction_func
}

static char PY_useclass__doc__[] =
"documentation"
;

static PyObject *
PY_useclass(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int useclass(const Class1 * arg1 +intent(in)+value)
// splicer begin function.useclass
    PY_Class1 * SHPy_arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:useclass",
        const_cast<char **>(SHT_kwlist), &PY_Class1_Type, &SHPy_arg1))
        return NULL;

    // post_parse
    const tutorial::Class1 * arg1 = SHPy_arg1 ? SHPy_arg1->obj : NULL;

    int SHC_rv = tutorial::useclass(arg1);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.useclass
}

static char PY_getclass3__doc__[] =
"documentation"
;

static PyObject *
PY_getclass3(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// Class1 * getclass3()
// splicer begin function.getclass3
    tutorial::Class1 * SHCXX_rv = tutorial::getclass3();

    // post_call
    PY_Class1 * SHTPy_rv = PyObject_New(PY_Class1, &PY_Class1_Type);
    SHTPy_rv->obj = SHCXX_rv;

    return (PyObject *) SHTPy_rv;
// splicer end function.getclass3
}

static char PY_LastFunctionCalled__doc__[] =
"documentation"
;

static PyObject *
PY_LastFunctionCalled(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const std::string & LastFunctionCalled() +len(30)
// splicer begin function.last_function_called
    const std::string & SHCXX_rv = tutorial::LastFunctionCalled();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.last_function_called
}

static char PY_Function10__doc__[] =
"documentation"
;

static PyObject *
PY_Function10(
  PyObject *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.function10
    Py_ssize_t SHT_nargs = 0;
    if (args != NULL) SHT_nargs += PyTuple_Size(args);
    if (kwds != NULL) SHT_nargs += PyDict_Size(args);
    PyObject *rvobj;
    if (SHT_nargs == 0) {
        rvobj = PY_Function10_0(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs == 2) {
        rvobj = PY_Function10_1(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
    return NULL;
// splicer end function.function10
}

static char PY_Function6__doc__[] =
"documentation"
;

static PyObject *
PY_Function6(
  PyObject *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.function6
    Py_ssize_t SHT_nargs = 0;
    if (args != NULL) SHT_nargs += PyTuple_Size(args);
    if (kwds != NULL) SHT_nargs += PyDict_Size(args);
    PyObject *rvobj;
    if (SHT_nargs == 1) {
        rvobj = PY_Function6_from_name(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs == 1) {
        rvobj = PY_Function6_from_index(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
    return NULL;
// splicer end function.function6
}

static char PY_overload1__doc__[] =
"documentation"
;

static PyObject *
PY_overload1(
  PyObject *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.overload1
    Py_ssize_t SHT_nargs = 0;
    if (args != NULL) SHT_nargs += PyTuple_Size(args);
    if (kwds != NULL) SHT_nargs += PyDict_Size(args);
    PyObject *rvobj;
    if (SHT_nargs >= 1 && SHT_nargs <= 3) {
        rvobj = PY_overload1_num_offset_stride(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs >= 2 && SHT_nargs <= 4) {
        rvobj = PY_overload1_5(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
    return NULL;
// splicer end function.overload1
}
static PyMethodDef PY_methods[] = {
{"Function1", (PyCFunction)PY_Function1, METH_NOARGS,
    PY_Function1__doc__},
{"Function2", (PyCFunction)PY_Function2, METH_VARARGS|METH_KEYWORDS,
    PY_Function2__doc__},
{"Sum", (PyCFunction)PY_Sum, METH_VARARGS|METH_KEYWORDS, PY_Sum__doc__},
{"TypeLongLong", (PyCFunction)PY_TypeLongLong,
    METH_VARARGS|METH_KEYWORDS, PY_TypeLongLong__doc__},
{"Function3", (PyCFunction)PY_Function3, METH_VARARGS|METH_KEYWORDS,
    PY_Function3__doc__},
{"ReturnIntPtrDim", (PyCFunction)PY_ReturnIntPtrDim, METH_NOARGS,
    PY_ReturnIntPtrDim__doc__},
{"Function4a", (PyCFunction)PY_Function4a, METH_VARARGS|METH_KEYWORDS,
    PY_Function4a__doc__},
{"Function4b", (PyCFunction)PY_Function4b, METH_VARARGS|METH_KEYWORDS,
    PY_Function4b__doc__},
{"Function5", (PyCFunction)PY_Function5_arg1_arg2,
    METH_VARARGS|METH_KEYWORDS, PY_Function5_arg1_arg2__doc__},
{"Function9", (PyCFunction)PY_Function9, METH_VARARGS|METH_KEYWORDS,
    PY_Function9__doc__},
{"typefunc", (PyCFunction)PY_typefunc, METH_VARARGS|METH_KEYWORDS,
    PY_typefunc__doc__},
{"enumfunc", (PyCFunction)PY_enumfunc, METH_VARARGS|METH_KEYWORDS,
    PY_enumfunc__doc__},
{"colorfunc", (PyCFunction)PY_colorfunc, METH_VARARGS|METH_KEYWORDS,
    PY_colorfunc__doc__},
{"getMinMax", (PyCFunction)PY_getMinMax, METH_NOARGS,
    PY_getMinMax__doc__},
{"directionFunc", (PyCFunction)PY_directionFunc,
    METH_VARARGS|METH_KEYWORDS, PY_directionFunc__doc__},
{"useclass", (PyCFunction)PY_useclass, METH_VARARGS|METH_KEYWORDS,
    PY_useclass__doc__},
{"getclass3", (PyCFunction)PY_getclass3, METH_NOARGS,
    PY_getclass3__doc__},
{"LastFunctionCalled", (PyCFunction)PY_LastFunctionCalled, METH_NOARGS,
    PY_LastFunctionCalled__doc__},
{"Function10", (PyCFunction)PY_Function10, METH_VARARGS|METH_KEYWORDS,
    PY_Function10__doc__},
{"Function6", (PyCFunction)PY_Function6, METH_VARARGS|METH_KEYWORDS,
    PY_Function6__doc__},
{"overload1", (PyCFunction)PY_overload1, METH_VARARGS|METH_KEYWORDS,
    PY_overload1__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

/*
 * inittutorial - Initialization function for the module
 * *must* be called inittutorial
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
static int tutorial_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int tutorial_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "tutorial", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
    tutorial_traverse, /* m_traverse */
    tutorial_clear, /* m_clear */
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
PyInit_tutorial(void)
#else
inittutorial(void)
#endif
{
    PyObject *m = NULL;
    const char * error_name = "tutorial.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("tutorial", PY_methods,
        PY__doc__,
        (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

    import_array();

    // struct1
    PY_struct1_Type.tp_new   = PyType_GenericNew;
    PY_struct1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_struct1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_struct1_Type);
    PyModule_AddObject(m, "struct1", (PyObject *)&PY_struct1_Type);


    // Class1
    PY_Class1_Type.tp_new   = PyType_GenericNew;
    PY_Class1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Class1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Class1_Type);
    PyModule_AddObject(m, "Class1", (PyObject *)&PY_Class1_Type);


    // Singleton
    PY_Singleton_Type.tp_new   = PyType_GenericNew;
    PY_Singleton_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Singleton_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Singleton_Type);
    PyModule_AddObject(m, "Singleton", (PyObject *)&PY_Singleton_Type);

    {
        // enumeration DIRECTION
        PyObject *tmp_value;
        tmp_value = PyLong_FromLong(tutorial::Class1::UP);
        PyDict_SetItemString((PyObject*) PY_Class1_Type.tp_dict, "UP", tmp_value);
        Py_DECREF(tmp_value);
        tmp_value = PyLong_FromLong(tutorial::Class1::DOWN);
        PyDict_SetItemString((PyObject*) PY_Class1_Type.tp_dict, "DOWN", tmp_value);
        Py_DECREF(tmp_value);
        tmp_value = PyLong_FromLong(tutorial::Class1::LEFT);
        PyDict_SetItemString((PyObject*) PY_Class1_Type.tp_dict, "LEFT", tmp_value);
        Py_DECREF(tmp_value);
        tmp_value = PyLong_FromLong(tutorial::Class1::RIGHT);
        PyDict_SetItemString((PyObject*) PY_Class1_Type.tp_dict, "RIGHT", tmp_value);
        Py_DECREF(tmp_value);
    }

    // enumeration Color
    PyModule_AddIntConstant(m, "RED", tutorial::RED);
    PyModule_AddIntConstant(m, "BLUE", tutorial::BLUE);
    PyModule_AddIntConstant(m, "WHITE", tutorial::WHITE);

    PY_error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
    if (PY_error_obj == NULL)
        return RETVAL;
    st->error = PY_error_obj;
    PyModule_AddObject(m, "Error", st->error);

    // splicer begin C_init_body
    // splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module tutorial");
    return RETVAL;
}

