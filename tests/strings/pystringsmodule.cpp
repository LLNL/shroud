// pystringsmodule.cpp
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
#include "pystringsmodule.hpp"
#include "strings.hpp"

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

static char PY_passChar__doc__[] =
"documentation"
;

static PyObject *
PY_passChar(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void passChar(char_scalar status +intent(in)+value)
// splicer begin function.pass_char
    char status;
    const char *SHT_kwlist[] = {
        "status",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "c:passChar",
        const_cast<char **>(SHT_kwlist), &status))
        return NULL;

    passChar(status);
    Py_RETURN_NONE;
// splicer end function.pass_char
}

static char PY_returnChar__doc__[] =
"documentation"
;

static PyObject *
PY_returnChar(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// char_scalar returnChar()
// splicer begin function.return_char
    char SHC_rv = returnChar();

    // post_call
    PyObject * SHTPy_rv = PyString_FromStringAndSize(&SHC_rv, 1);

    return (PyObject *) SHTPy_rv;
// splicer end function.return_char
}

static char PY_passCharPtrInOut__doc__[] =
"documentation"
;

static PyObject *
PY_passCharPtrInOut(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void passCharPtrInOut(char * s +intent(inout))
// splicer begin function.pass_char_ptr_in_out
    char * s;
    const char *SHT_kwlist[] = {
        "s",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:passCharPtrInOut",
        const_cast<char **>(SHT_kwlist), &s))
        return NULL;

    passCharPtrInOut(s);

    // post_call
    PyObject * SHPy_s = PyString_FromString(s);

    return (PyObject *) SHPy_s;
// splicer end function.pass_char_ptr_in_out
}

static char PY_getCharPtr1__doc__[] =
"documentation"
;

static PyObject *
PY_getCharPtr1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const char * getCharPtr1() +pure
// splicer begin function.get_char_ptr1
    const char * SHC_rv = getCharPtr1();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.get_char_ptr1
}

static char PY_getCharPtr2__doc__[] =
"documentation"
;

static PyObject *
PY_getCharPtr2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const char * getCharPtr2() +len(30)
// splicer begin function.get_char_ptr2
    const char * SHC_rv = getCharPtr2();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.get_char_ptr2
}

static char PY_getCharPtr3__doc__[] =
"documentation"
;

static PyObject *
PY_getCharPtr3(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const char * getCharPtr3()
// splicer begin function.get_char_ptr3
    const char * SHC_rv = getCharPtr3();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHC_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.get_char_ptr3
}

static char PY_getConstStringLen__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringLen(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string getConstStringLen() +len(30)
// splicer begin function.get_const_string_len
    const std::string SHCXX_rv = getConstStringLen();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_len
}

static char PY_getConstStringAsArg__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringAsArg(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string getConstStringAsArg()
// splicer begin function.get_const_string_as_arg
    const std::string SHCXX_rv = getConstStringAsArg();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_as_arg
}

static char PY_getConstStringAlloc__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringAlloc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const std::string getConstStringAlloc() +allocatable
// splicer begin function.get_const_string_alloc
    const std::string SHCXX_rv = getConstStringAlloc();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_alloc
}

static char PY_getConstStringRefPure__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringRefPure(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string & getConstStringRefPure() +pure
// splicer begin function.get_const_string_ref_pure
    const std::string & SHCXX_rv = getConstStringRefPure();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_ref_pure
}

static char PY_getConstStringRefLen__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringRefLen(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string & getConstStringRefLen() +len(30)
// splicer begin function.get_const_string_ref_len
    const std::string & SHCXX_rv = getConstStringRefLen();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_ref_len
}

static char PY_getConstStringRefAsArg__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringRefAsArg(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string & getConstStringRefAsArg()
// splicer begin function.get_const_string_ref_as_arg
    const std::string & SHCXX_rv = getConstStringRefAsArg();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_ref_as_arg
}

static char PY_getConstStringRefLenEmpty__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringRefLenEmpty(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string & getConstStringRefLenEmpty() +len(30)
// splicer begin function.get_const_string_ref_len_empty
    const std::string & SHCXX_rv = getConstStringRefLenEmpty();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_ref_len_empty
}

static char PY_getConstStringRefAlloc__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringRefAlloc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const std::string & getConstStringRefAlloc() +allocatable
// splicer begin function.get_const_string_ref_alloc
    const std::string & SHCXX_rv = getConstStringRefAlloc();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_ref_alloc
}

static char PY_getConstStringPtrLen__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringPtrLen(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string * getConstStringPtrLen() +len(30)
// splicer begin function.get_const_string_ptr_len
    const std::string * SHCXX_rv = getConstStringPtrLen();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv->c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_ptr_len
}

static char PY_getConstStringPtrAlloc__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringPtrAlloc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const std::string * getConstStringPtrAlloc() +allocatable
// splicer begin function.get_const_string_ptr_alloc
    const std::string * SHCXX_rv = getConstStringPtrAlloc();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv->c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_ptr_alloc
}

static char PY_getConstStringPtrOwnsAlloc__doc__[] =
"documentation"
;

static PyObject *
PY_getConstStringPtrOwnsAlloc(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const std::string * getConstStringPtrOwnsAlloc() +allocatable
// splicer begin function.get_const_string_ptr_owns_alloc
    const std::string * SHCXX_rv = getConstStringPtrOwnsAlloc();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv->c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_const_string_ptr_owns_alloc
}

static char PY_acceptStringConstReference__doc__[] =
"documentation"
;

static PyObject *
PY_acceptStringConstReference(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void acceptStringConstReference(const std::string & arg1 +intent(in))
// splicer begin function.accept_string_const_reference
    const char * arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "s:acceptStringConstReference",
        const_cast<char **>(SHT_kwlist), &arg1))
        return NULL;

    // post_parse
    const std::string SH_arg1(arg1);

    acceptStringConstReference(SH_arg1);
    Py_RETURN_NONE;
// splicer end function.accept_string_const_reference
}

static char PY_acceptStringReferenceOut__doc__[] =
"documentation"
;

static PyObject *
PY_acceptStringReferenceOut(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void acceptStringReferenceOut(std::string & arg1 +intent(out))
// splicer begin function.accept_string_reference_out
    // post_parse
    std::string SH_arg1;

    acceptStringReferenceOut(SH_arg1);

    // post_call
    PyObject * SHPy_arg1 = PyString_FromString(SH_arg1.c_str());

    return (PyObject *) SHPy_arg1;
// splicer end function.accept_string_reference_out
}

static char PY_acceptStringReference__doc__[] =
"documentation"
;

static PyObject *
PY_acceptStringReference(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void acceptStringReference(std::string & arg1 +intent(inout))
// splicer begin function.accept_string_reference
    char * arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "s:acceptStringReference", const_cast<char **>(SHT_kwlist), 
        &arg1))
        return NULL;

    // post_parse
    std::string SH_arg1(arg1);

    acceptStringReference(SH_arg1);

    // post_call
    PyObject * SHPy_arg1 = PyString_FromString(SH_arg1.c_str());

    return (PyObject *) SHPy_arg1;
// splicer end function.accept_string_reference
}

static char PY_acceptStringPointer__doc__[] =
"documentation"
;

static PyObject *
PY_acceptStringPointer(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void acceptStringPointer(std::string * arg1 +intent(inout))
// splicer begin function.accept_string_pointer
    char * arg1;
    const char *SHT_kwlist[] = {
        "arg1",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
        "s:acceptStringPointer", const_cast<char **>(SHT_kwlist), 
        &arg1))
        return NULL;

    // post_parse
    std::string SH_arg1(arg1);

    acceptStringPointer(&SH_arg1);

    // post_call
    PyObject * SHPy_arg1 = PyString_FromString(SH_arg1.c_str());

    return (PyObject *) SHPy_arg1;
// splicer end function.accept_string_pointer
}

static char PY_returnStrings__doc__[] =
"documentation"
;

static PyObject *
PY_returnStrings(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void returnStrings(std::string & arg1 +intent(out), std::string & arg2 +intent(out))
// splicer begin function.return_strings
    // post_parse
    std::string SH_arg1;
    std::string SH_arg2;

    returnStrings(SH_arg1, SH_arg2);

    // post_call
    PyObject * SHTPy_rv = Py_BuildValue("ss", SH_arg1.c_str(),
        SH_arg2.c_str());

    return SHTPy_rv;
// splicer end function.return_strings
}

static char PY_explicit1__doc__[] =
"documentation"
;

static PyObject *
PY_explicit1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void explicit1(char * name +intent(in)+len_trim(AAlen))
// splicer begin function.explicit1
    char * name;
    const char *SHT_kwlist[] = {
        "name",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:explicit1",
        const_cast<char **>(SHT_kwlist), &name))
        return NULL;

    explicit1(name);
    Py_RETURN_NONE;
// splicer end function.explicit1
}

static char PY_CpassChar__doc__[] =
"documentation"
;

static PyObject *
PY_CpassChar(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void CpassChar(char_scalar status +intent(in)+value)
// splicer begin function.cpass_char
    char status;
    const char *SHT_kwlist[] = {
        "status",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "c:CpassChar",
        const_cast<char **>(SHT_kwlist), &status))
        return NULL;

    CpassChar(status);
    Py_RETURN_NONE;
// splicer end function.cpass_char
}

static char PY_CreturnChar__doc__[] =
"documentation"
;

static PyObject *
PY_CreturnChar(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// char_scalar CreturnChar()
// splicer begin function.creturn_char
    char SHC_rv = CreturnChar();

    // post_call
    PyObject * SHTPy_rv = PyString_FromStringAndSize(&SHC_rv, 1);

    return (PyObject *) SHTPy_rv;
// splicer end function.creturn_char
}
static PyMethodDef PY_methods[] = {
{"passChar", (PyCFunction)PY_passChar, METH_VARARGS|METH_KEYWORDS,
    PY_passChar__doc__},
{"returnChar", (PyCFunction)PY_returnChar, METH_NOARGS,
    PY_returnChar__doc__},
{"passCharPtrInOut", (PyCFunction)PY_passCharPtrInOut,
    METH_VARARGS|METH_KEYWORDS, PY_passCharPtrInOut__doc__},
{"getCharPtr1", (PyCFunction)PY_getCharPtr1, METH_NOARGS,
    PY_getCharPtr1__doc__},
{"getCharPtr2", (PyCFunction)PY_getCharPtr2, METH_NOARGS,
    PY_getCharPtr2__doc__},
{"getCharPtr3", (PyCFunction)PY_getCharPtr3, METH_NOARGS,
    PY_getCharPtr3__doc__},
{"getConstStringLen", (PyCFunction)PY_getConstStringLen, METH_NOARGS,
    PY_getConstStringLen__doc__},
{"getConstStringAsArg", (PyCFunction)PY_getConstStringAsArg,
    METH_NOARGS, PY_getConstStringAsArg__doc__},
{"getConstStringAlloc", (PyCFunction)PY_getConstStringAlloc,
    METH_NOARGS, PY_getConstStringAlloc__doc__},
{"getConstStringRefPure", (PyCFunction)PY_getConstStringRefPure,
    METH_NOARGS, PY_getConstStringRefPure__doc__},
{"getConstStringRefLen", (PyCFunction)PY_getConstStringRefLen,
    METH_NOARGS, PY_getConstStringRefLen__doc__},
{"getConstStringRefAsArg", (PyCFunction)PY_getConstStringRefAsArg,
    METH_NOARGS, PY_getConstStringRefAsArg__doc__},
{"getConstStringRefLenEmpty", (PyCFunction)PY_getConstStringRefLenEmpty,
    METH_NOARGS, PY_getConstStringRefLenEmpty__doc__},
{"getConstStringRefAlloc", (PyCFunction)PY_getConstStringRefAlloc,
    METH_NOARGS, PY_getConstStringRefAlloc__doc__},
{"getConstStringPtrLen", (PyCFunction)PY_getConstStringPtrLen,
    METH_NOARGS, PY_getConstStringPtrLen__doc__},
{"getConstStringPtrAlloc", (PyCFunction)PY_getConstStringPtrAlloc,
    METH_NOARGS, PY_getConstStringPtrAlloc__doc__},
{"getConstStringPtrOwnsAlloc",
    (PyCFunction)PY_getConstStringPtrOwnsAlloc, METH_NOARGS,
    PY_getConstStringPtrOwnsAlloc__doc__},
{"acceptStringConstReference",
    (PyCFunction)PY_acceptStringConstReference,
    METH_VARARGS|METH_KEYWORDS, PY_acceptStringConstReference__doc__},
{"acceptStringReferenceOut", (PyCFunction)PY_acceptStringReferenceOut,
    METH_NOARGS, PY_acceptStringReferenceOut__doc__},
{"acceptStringReference", (PyCFunction)PY_acceptStringReference,
    METH_VARARGS|METH_KEYWORDS, PY_acceptStringReference__doc__},
{"acceptStringPointer", (PyCFunction)PY_acceptStringPointer,
    METH_VARARGS|METH_KEYWORDS, PY_acceptStringPointer__doc__},
{"returnStrings", (PyCFunction)PY_returnStrings, METH_NOARGS,
    PY_returnStrings__doc__},
{"explicit1", (PyCFunction)PY_explicit1, METH_VARARGS|METH_KEYWORDS,
    PY_explicit1__doc__},
{"CpassChar", (PyCFunction)PY_CpassChar, METH_VARARGS|METH_KEYWORDS,
    PY_CpassChar__doc__},
{"CreturnChar", (PyCFunction)PY_CreturnChar, METH_NOARGS,
    PY_CreturnChar__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

/*
 * initstrings - Initialization function for the module
 * *must* be called initstrings
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
static int strings_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int strings_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "strings", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
    strings_traverse, /* m_traverse */
    strings_clear, /* m_clear */
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
PyInit_strings(void)
#else
initstrings(void)
#endif
{
    PyObject *m = NULL;
    const char * error_name = "strings.Error";

// splicer begin C_init_locals
// splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("strings", PY_methods,
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
        Py_FatalError("can't initialize module strings");
    return RETVAL;
}

