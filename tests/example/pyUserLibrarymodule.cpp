// pyUserLibrarymodule.cpp
// This is generated code, do not edit
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
#include "pyUserLibrarymodule.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// splicer begin include
// splicer end include

namespace example {
namespace nested {

// splicer begin C_definition
// splicer end C_definition
PyObject *PP_error_obj;
// splicer begin additional_functions
// splicer end additional_functions

static char PP_local_function1__doc__[] =
"documentation"
;

static PyObject *
PP_local_function1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void local_function1()
// splicer begin function.local_function1
    local_function1();
    Py_RETURN_NONE;
// splicer end function.local_function1
}

static char PP_is_name_valid__doc__[] =
"documentation"
;

static PyObject *
PP_is_name_valid(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// bool isNameValid(const std::string & name +intent(in))
// splicer begin function.is_name_valid
    const char * name;
    const char *SHT_kwlist[] = {
        "name",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:isNameValid",
        const_cast<char **>(SHT_kwlist),
        &name))
        return NULL;

    // post_parse
    const std::string SH_name(name);

    bool SHCXX_rv = isNameValid(SH_name);

    // post_call
    PyObject * SHTPy_rv = PyBool_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.is_name_valid
}

static char PP_is_initialized__doc__[] =
"documentation"
;

static PyObject *
PP_is_initialized(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// bool isInitialized()
// splicer begin function.is_initialized
    bool SHCXX_rv = isInitialized();

    // post_call
    PyObject * SHTPy_rv = PyBool_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.is_initialized
}

static char PP_check_bool__doc__[] =
"documentation"
;

static PyObject *
PP_check_bool(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void checkBool(bool arg1 +intent(in)+value, bool * arg2 +intent(out), bool * arg3 +intent(inout))
// splicer begin function.check_bool
    PyObject * SHPy_arg1;
    PyObject * SHPy_arg3;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg3",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!:checkBool",
        const_cast<char **>(SHT_kwlist),
        &PyBool_Type, &SHPy_arg1, &PyBool_Type, &SHPy_arg3))
        return NULL;

    // pre_call
    bool arg1 = PyObject_IsTrue(SHPy_arg1);
    bool arg2;  // intent(out)
    bool arg3 = PyObject_IsTrue(SHPy_arg3);

    checkBool(arg1, &arg2, &arg3);

    // post_call
    PyObject * SHPy_arg2 = PyBool_FromLong(arg2);
    SHPy_arg3 = PyBool_FromLong(arg3);
    PyObject * SHTPy_rv = Py_BuildValue("OO", SHPy_arg2, SHPy_arg3);

    return SHTPy_rv;
// splicer end function.check_bool
}

static PyObject *
PP_test_names(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void test_names(const std::string & name +intent(in))
// splicer begin function.test_names
    const char * name;
    const char *SHT_kwlist[] = {
        "name",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:test_names",
        const_cast<char **>(SHT_kwlist),
        &name))
        return NULL;

    // post_parse
    const std::string SH_name(name);

    test_names(SH_name);
    Py_RETURN_NONE;
// splicer end function.test_names
}

static PyObject *
PP_test_names_flag(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void test_names(const std::string & name +intent(in), int flag +intent(in)+value)
// splicer begin function.test_names_flag
    const char * name;
    int flag;
    const char *SHT_kwlist[] = {
        "name",
        "flag",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "si:test_names",
        const_cast<char **>(SHT_kwlist),
        &name, &flag))
        return NULL;

    // post_parse
    const std::string SH_name(name);

    test_names(SH_name, flag);
    Py_RETURN_NONE;
// splicer end function.test_names_flag
}

static char PP_testoptional_2__doc__[] =
"documentation"
;

static PyObject *
PP_testoptional_2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void testoptional(int i=1 +intent(in)+value, long j=2 +intent(in)+value)
// splicer begin function.testoptional
    Py_ssize_t SH_nargs = 0;
    int i;
    long j;
    const char *SHT_kwlist[] = {
        "i",
        "j",
        NULL };

    if (args != NULL) SH_nargs += PyTuple_Size(args);
    if (kwds != NULL) SH_nargs += PyDict_Size(args);
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|il:testoptional",
        const_cast<char **>(SHT_kwlist),
        &i, &j))
        return NULL;
    switch (SH_nargs) {
    case 0:
        testoptional();
        break;
    case 1:
        testoptional(i);
        break;
    case 2:
        testoptional(i, j);
        break;
    }
    Py_RETURN_NONE;
// splicer end function.testoptional
}

static char PP_test_size_t__doc__[] =
"documentation"
;

static PyObject *
PP_test_size_t(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// size_t test_size_t()
// splicer begin function.test_size_t
    size_t SHCXX_rv = test_size_t();

    // post_call
    PyObject * SHTPy_rv = PyInt_FromSize_t(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.test_size_t
}

static char PP_testmpi__doc__[] =
"documentation"
;

static PyObject *
PP_testmpi(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void testmpi(MPI_Comm comm +intent(in)+value)
// splicer begin function.testmpi
    MPI_Fint comm;
    const char *SHT_kwlist[] = {
        "comm",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:testmpi",
        const_cast<char **>(SHT_kwlist),
        &comm))
        return NULL;

    // post_parse
    MPI_Comm SH_comm = MPI_Comm_f2c(comm);

    testmpi(SH_comm);
    Py_RETURN_NONE;
// splicer end function.testmpi
}

static char PP_testgroup1__doc__[] =
"documentation"
;

static PyObject *
PP_testgroup1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void testgroup1(DataGroup * grp +intent(in)+value)
// splicer begin function.testgroup1
    PyObject * SHPy_grp;
    const char *SHT_kwlist[] = {
        "grp",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:testgroup1",
        const_cast<char **>(SHT_kwlist),
        &FillInTypeForGroup, &SHPy_grp))
        return NULL;

    // post_parse
    axom::sidre::Group * grp = SHPy_grp ? SHPy_grp->obj : NULL;

    testgroup1(grp);
    Py_RETURN_NONE;
// splicer end function.testgroup1
}

static char PP_testgroup2__doc__[] =
"documentation"
;

static PyObject *
PP_testgroup2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void testgroup2(const DataGroup * grp +intent(in)+value)
// splicer begin function.testgroup2
    PyObject * SHPy_grp;
    const char *SHT_kwlist[] = {
        "grp",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:testgroup2",
        const_cast<char **>(SHT_kwlist),
        &FillInTypeForGroup, &SHPy_grp))
        return NULL;

    // post_parse
    const axom::sidre::Group * grp = SHPy_grp ? SHPy_grp->obj : NULL;

    testgroup2(grp);
    Py_RETURN_NONE;
// splicer end function.testgroup2
}

static char PP_func_ptr1__doc__[] =
"documentation"
;

static PyObject *
PP_func_ptr1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void FuncPtr1(void ( * get) +intent(in)+value())
// splicer begin function.func_ptr1
    void ( * get)();
    const char *SHT_kwlist[] = {
        "get",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:FuncPtr1",
        const_cast<char **>(SHT_kwlist),
        &get))
        return NULL;

    FuncPtr1(get);
    Py_RETURN_NONE;
// splicer end function.func_ptr1
}

static char PP_func_ptr2__doc__[] =
"documentation"
;

static PyObject *
PP_func_ptr2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void FuncPtr2(double * ( * get) +intent(in)())
// splicer begin function.func_ptr2
    double * ( * get)();
    const char *SHT_kwlist[] = {
        "get",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d:FuncPtr2",
        const_cast<char **>(SHT_kwlist),
        &get))
        return NULL;

    FuncPtr2(get);
    Py_RETURN_NONE;
// splicer end function.func_ptr2
}

static char PP_func_ptr3__doc__[] =
"documentation"
;

static PyObject *
PP_func_ptr3(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void FuncPtr3(double ( * get) +intent(in)+value(int i +value, int +value))
// splicer begin function.func_ptr3
    double ( * get)(int i, int);
    const char *SHT_kwlist[] = {
        "get",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d:FuncPtr3",
        const_cast<char **>(SHT_kwlist),
        &get))
        return NULL;

    FuncPtr3(get);
    Py_RETURN_NONE;
// splicer end function.func_ptr3
}

static char PP_func_ptr5__doc__[] =
"documentation"
;

static PyObject *
PP_func_ptr5(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void FuncPtr5(void ( * get) +intent(in)+value(int verylongname1 +value, int verylongname2 +value, int verylongname3 +value, int verylongname4 +value, int verylongname5 +value, int verylongname6 +value, int verylongname7 +value, int verylongname8 +value, int verylongname9 +value, int verylongname10 +value))
// splicer begin function.func_ptr5
    void ( * get)(int verylongname1, int verylongname2,
        int verylongname3, int verylongname4, int verylongname5,
        int verylongname6, int verylongname7, int verylongname8,
        int verylongname9, int verylongname10);
    const char *SHT_kwlist[] = {
        "get",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:FuncPtr5",
        const_cast<char **>(SHT_kwlist),
        &get))
        return NULL;

    FuncPtr5(get);
    Py_RETURN_NONE;
// splicer end function.func_ptr5
}

static char PP_verylongfunctionname1__doc__[] =
"documentation"
;

static PyObject *
PP_verylongfunctionname1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void verylongfunctionname1(int * verylongname1 +intent(inout), int * verylongname2 +intent(inout), int * verylongname3 +intent(inout), int * verylongname4 +intent(inout), int * verylongname5 +intent(inout), int * verylongname6 +intent(inout), int * verylongname7 +intent(inout), int * verylongname8 +intent(inout), int * verylongname9 +intent(inout), int * verylongname10 +intent(inout))
// splicer begin function.verylongfunctionname1
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
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "iiiiiiiiii:verylongfunctionname1",
        const_cast<char **>(SHT_kwlist),
        &verylongname1, &verylongname2, &verylongname3, &verylongname4,
        &verylongname5, &verylongname6, &verylongname7, &verylongname8,
        &verylongname9, &verylongname10))
        return NULL;

    verylongfunctionname1(&verylongname1, &verylongname2,
        &verylongname3, &verylongname4, &verylongname5, &verylongname6,
        &verylongname7, &verylongname8, &verylongname9,
        &verylongname10);

    // post_call
    PyObject * SHTPy_rv = Py_BuildValue("iiiiiiiiii", verylongname1,
        verylongname2, verylongname3, verylongname4, verylongname5,
        verylongname6, verylongname7, verylongname8, verylongname9,
        verylongname10);

    return SHTPy_rv;
// splicer end function.verylongfunctionname1
}

static char PP_verylongfunctionname2__doc__[] =
"documentation"
;

static PyObject *
PP_verylongfunctionname2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int verylongfunctionname2(int verylongname1 +intent(in)+value, int verylongname2 +intent(in)+value, int verylongname3 +intent(in)+value, int verylongname4 +intent(in)+value, int verylongname5 +intent(in)+value, int verylongname6 +intent(in)+value, int verylongname7 +intent(in)+value, int verylongname8 +intent(in)+value, int verylongname9 +intent(in)+value, int verylongname10 +intent(in)+value)
// splicer begin function.verylongfunctionname2
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
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "iiiiiiiiii:verylongfunctionname2",
        const_cast<char **>(SHT_kwlist),
        &verylongname1, &verylongname2, &verylongname3, &verylongname4,
        &verylongname5, &verylongname6, &verylongname7, &verylongname8,
        &verylongname9, &verylongname10))
        return NULL;

    int SHCXX_rv = verylongfunctionname2(verylongname1, verylongname2,
        verylongname3, verylongname4, verylongname5, verylongname6,
        verylongname7, verylongname8, verylongname9, verylongname10);

    // post_call
    PyObject * SHTPy_rv = PyInt_FromLong(SHCXX_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.verylongfunctionname2
}

static char PP_cos_doubles__doc__[] =
"documentation"
;

static PyObject *
PP_cos_doubles(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void cos_doubles(double * in +dimension(:,:)+intent(in), double * out +allocatable(mold=in)+dimension(:,:)+intent(out), int sizein +implied(size(in))+intent(in)+value)
// splicer begin function.cos_doubles
    PyObject * SHTPy_in;
    PyArrayObject * SHPy_in = NULL;
    PyArrayObject * SHPy_out = NULL;
    const char *SHT_kwlist[] = {
        "in",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:cos_doubles",
        const_cast<char **>(SHT_kwlist),
        &SHTPy_in))
        return NULL;

    // post_parse
    SHPy_in = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        SHTPy_in, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (SHPy_in == NULL) {
        PyErr_SetString(PyExc_ValueError,
            "in must be a 1-D array of double");
        goto fail;
    }
    {
        // pre_call
        double * in = static_cast<double *>(PyArray_DATA(SHPy_in));
        SHPy_out = reinterpret_cast<PyArrayObject *>
            (PyArray_NewLikeArray(SHPy_in, NPY_CORDER, NULL, 0));
        if (SHPy_out == NULL)
            goto fail;
        double * out = static_cast<double *>(PyArray_DATA(SHPy_out));
        int sizein = PyArray_SIZE(SHPy_in);

        cos_doubles(in, out, sizein);

        // cleanup
        Py_DECREF(SHPy_in);

        return (PyObject *) SHPy_out;
    }

fail:
    Py_XDECREF(SHPy_in);
    Py_XDECREF(SHPy_out);
    return NULL;
// splicer end function.cos_doubles
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
// splicer begin function.test_names
    Py_ssize_t SH_nargs = 0;
    if (args != NULL) SH_nargs += PyTuple_Size(args);
    if (kwds != NULL) SH_nargs += PyDict_Size(args);
    PyObject *rvobj;
    if (SH_nargs == 1) {
        rvobj = PP_test_names(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SH_nargs == 2) {
        rvobj = PP_test_names_flag(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
    return NULL;
// splicer end function.test_names
}
static PyMethodDef PP_methods[] = {
{"local_function1", (PyCFunction)PP_local_function1, METH_NOARGS,
    PP_local_function1__doc__},
{"isNameValid", (PyCFunction)PP_is_name_valid,
    METH_VARARGS|METH_KEYWORDS, PP_is_name_valid__doc__},
{"isInitialized", (PyCFunction)PP_is_initialized, METH_NOARGS,
    PP_is_initialized__doc__},
{"checkBool", (PyCFunction)PP_check_bool, METH_VARARGS|METH_KEYWORDS,
    PP_check_bool__doc__},
{"testoptional", (PyCFunction)PP_testoptional_2,
    METH_VARARGS|METH_KEYWORDS, PP_testoptional_2__doc__},
{"test_size_t", (PyCFunction)PP_test_size_t, METH_NOARGS,
    PP_test_size_t__doc__},
{"testmpi", (PyCFunction)PP_testmpi, METH_VARARGS|METH_KEYWORDS,
    PP_testmpi__doc__},
{"testgroup1", (PyCFunction)PP_testgroup1, METH_VARARGS|METH_KEYWORDS,
    PP_testgroup1__doc__},
{"testgroup2", (PyCFunction)PP_testgroup2, METH_VARARGS|METH_KEYWORDS,
    PP_testgroup2__doc__},
{"FuncPtr1", (PyCFunction)PP_func_ptr1, METH_VARARGS|METH_KEYWORDS,
    PP_func_ptr1__doc__},
{"FuncPtr2", (PyCFunction)PP_func_ptr2, METH_VARARGS|METH_KEYWORDS,
    PP_func_ptr2__doc__},
{"FuncPtr3", (PyCFunction)PP_func_ptr3, METH_VARARGS|METH_KEYWORDS,
    PP_func_ptr3__doc__},
{"FuncPtr5", (PyCFunction)PP_func_ptr5, METH_VARARGS|METH_KEYWORDS,
    PP_func_ptr5__doc__},
{"verylongfunctionname1", (PyCFunction)PP_verylongfunctionname1,
    METH_VARARGS|METH_KEYWORDS, PP_verylongfunctionname1__doc__},
{"verylongfunctionname2", (PyCFunction)PP_verylongfunctionname2,
    METH_VARARGS|METH_KEYWORDS, PP_verylongfunctionname2__doc__},
{"cos_doubles", (PyCFunction)PP_cos_doubles, METH_VARARGS|METH_KEYWORDS,
    PP_cos_doubles__doc__},
{"test_names", (PyCFunction)PP_test_names, METH_VARARGS|METH_KEYWORDS,
    PP_test_names__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

/*
 * inituserlibrary - Initialization function for the module
 * *must* be called inituserlibrary
 */
static char PP__doc__[] =
"library documentation"
;

struct module_state {
    PyObject *error;
};

#ifdef IS_PY3K
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#ifdef IS_PY3K
static int userlibrary_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int userlibrary_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "userlibrary", /* m_name */
    PP__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PP_methods, /* m_methods */
    NULL, /* m_reload */
    userlibrary_traverse, /* m_traverse */
    userlibrary_clear, /* m_clear */
    NULL  /* m_free */
};

#define RETVAL m
#define INITERROR return NULL
#else
#define RETVAL
#define INITERROR return
#endif

#ifdef __cplusplus
extern "C" {
#endif
PyMODINIT_FUNC
SHROUD_MOD_INIT(void)
{
    PyObject *m = NULL;
    const char * error_name = "userlibrary.Error";

// splicer begin C_init_locals
// splicer end C_init_locals


    /* Create the module and add the functions */
#ifdef IS_PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("userlibrary", PP_methods,
                       PP__doc__,
                       (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

    import_array();

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


    PP_error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
    if (PP_error_obj == NULL)
        return RETVAL;
    st->error = PP_error_obj;
    PyModule_AddObject(m, "Error", st->error);

// splicer begin C_init_body
// splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module userlibrary");
    return RETVAL;
}
#ifdef __cplusplus
}
#endif


}  // namespace nested
}  // namespace example
