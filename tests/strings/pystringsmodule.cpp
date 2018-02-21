// pystringsmodule.cpp
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
#include "pystringsmodule.hpp"

// splicer begin include
// splicer end include

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
        const_cast<char **>(SHT_kwlist),
        &status))
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
        const_cast<char **>(SHT_kwlist),
        &s))
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
// const char * getCharPtr2 +len(30)()
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

static char PY_getString1__doc__[] =
"documentation"
;

static PyObject *
PY_getString1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string & getString1() +pure
// splicer begin function.get_string1
    const std::string & SHCXX_rv = getString1();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_string1
}

static char PY_getString2__doc__[] =
"documentation"
;

static PyObject *
PY_getString2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string & getString2 +len(30)()
// splicer begin function.get_string2
    const std::string & SHCXX_rv = getString2();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_string2
}

static char PY_getString3__doc__[] =
"documentation"
;

static PyObject *
PY_getString3(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string & getString3()
// splicer begin function.get_string3
    const std::string & SHCXX_rv = getString3();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_string3
}

static char PY_getString2_empty__doc__[] =
"documentation"
;

static PyObject *
PY_getString2_empty(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string & getString2_empty +len(30)()
// splicer begin function.get_string2_empty
    const std::string & SHCXX_rv = getString2_empty();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_string2_empty
}

static char PY_getString5__doc__[] =
"documentation"
;

static PyObject *
PY_getString5(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string getString5 +len(30)()
// splicer begin function.get_string5
    const std::string SHCXX_rv = getString5();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_string5
}

static char PY_getString6__doc__[] =
"documentation"
;

static PyObject *
PY_getString6(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string getString6()
// splicer begin function.get_string6
    const std::string SHCXX_rv = getString6();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv.c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_string6
}

static char PY_getString7__doc__[] =
"documentation"
;

static PyObject *
PY_getString7(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// const string * getString7 +len(30)()
// splicer begin function.get_string7
    const std::string * SHCXX_rv = getString7();

    // post_call
    PyObject * SHTPy_rv = PyString_FromString(SHCXX_rv->c_str());

    return (PyObject *) SHTPy_rv;
// splicer end function.get_string7
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
        "s:acceptStringConstReference", const_cast<char **>(SHT_kwlist),
        &arg1))
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
        const_cast<char **>(SHT_kwlist),
        &name))
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
        const_cast<char **>(SHT_kwlist),
        &status))
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
{"getString1", (PyCFunction)PY_getString1, METH_NOARGS,
    PY_getString1__doc__},
{"getString2", (PyCFunction)PY_getString2, METH_NOARGS,
    PY_getString2__doc__},
{"getString3", (PyCFunction)PY_getString3, METH_NOARGS,
    PY_getString3__doc__},
{"getString2_empty", (PyCFunction)PY_getString2_empty, METH_NOARGS,
    PY_getString2_empty__doc__},
{"getString5", (PyCFunction)PY_getString5, METH_NOARGS,
    PY_getString5__doc__},
{"getString6", (PyCFunction)PY_getString6, METH_NOARGS,
    PY_getString6__doc__},
{"getString7", (PyCFunction)PY_getString7, METH_NOARGS,
    PY_getString7__doc__},
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

#ifdef IS_PY3K
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#ifdef IS_PY3K
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

#ifdef __cplusplus
extern "C" {
#endif
PyMODINIT_FUNC
SHROUD_MOD_INIT(void)
{
    PyObject *m = NULL;
    const char * error_name = "strings.Error";

// splicer begin C_init_locals
// splicer end C_init_locals


    /* Create the module and add the functions */
#ifdef IS_PY3K
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
#ifdef __cplusplus
}
#endif

