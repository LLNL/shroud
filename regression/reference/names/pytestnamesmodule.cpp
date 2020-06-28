// pytestnamesmodule.cpp
// This file is generated by Shroud 0.12.1. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pytestnamesmodule.hpp"

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
PyObject *PY_init_testnames_ns0_inner(void);
PyObject *PY_init_testnames_ns0(void);
PyObject *PY_init_testnames_ns1(void);
PyObject *PY_init_testnames_internal(void);
PyObject *PY_init_testnames_std(void);
// splicer begin additional_functions
// splicer end additional_functions

// ----------------------------------------
// Function:  void getName
// Exact:     py_default
// ----------------------------------------
// Argument:  char * name +intent(inout)+len(worklen)+len_trim(worktrim)
// Exact:     py_char_*_inout
static char PY_getName__doc__[] =
"documentation"
;

static PyObject *
PY_getName(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.get_name
    char * name;
    const char *SHT_kwlist[] = {
        "name",
        nullptr };
    PyObject * SHPy_name = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:getName",
        const_cast<char **>(SHT_kwlist), &name))
        return nullptr;

    getName(name);

    // post_call
    SHPy_name = PyString_FromString(name);

    return (PyObject *) SHPy_name;
// splicer end function.get_name
}

// ----------------------------------------
// Function:  void function1
// Exact:     py_default
static char PY_function1__doc__[] =
"documentation"
;

static PyObject *
PY_function1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin function.function1
// splicer code for function.function1
// splicer end function.function1
}

// ----------------------------------------
// Function:  void function2
// Exact:     py_default
static char PY_function2__doc__[] =
"documentation"
;

static PyObject *
PY_function2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin function.function2
    function2();
    Py_RETURN_NONE;
// splicer end function.function2
}

// ----------------------------------------
// Function:  void function3a
// Exact:     py_default
// ----------------------------------------
// Argument:  int i +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
static PyObject *
PY_function3a_0(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.function3a_0
    int i;
    const char *SHT_kwlist[] = {
        "i",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:function3a",
        const_cast<char **>(SHT_kwlist), &i))
        return nullptr;

    function3a(i);
    Py_RETURN_NONE;
// splicer end function.function3a_0
}

// ----------------------------------------
// Function:  void function3a
// Exact:     py_default
// ----------------------------------------
// Argument:  long i +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
static PyObject *
PY_function3a_1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.function3a_1
    long i;
    const char *SHT_kwlist[] = {
        "i",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l:function3a",
        const_cast<char **>(SHT_kwlist), &i))
        return nullptr;

    function3a(i);
    Py_RETURN_NONE;
// splicer end function.function3a_1
}

// ----------------------------------------
// Function:  int function4
// Requested: py_native_scalar_result
// Match:     py_default
// ----------------------------------------
// Argument:  const std::string & rv +intent(in)
// Exact:     py_string_&_in
static char PY_function4__doc__[] =
"documentation"
;

static PyObject *
PY_function4(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.function4
    char * rv;
    const char *SHT_kwlist[] = {
        "rv",
        nullptr };
    PyObject * SHTPy_rv = nullptr;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:function4",
        const_cast<char **>(SHT_kwlist), &rv))
        return nullptr;

    // post_declare
    const std::string SH_rv(rv);

    int ARG_rv = function4(SH_rv);

    // post_call
    SHTPy_rv = PyInt_FromLong(ARG_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.function4
}

// ----------------------------------------
// Function:  void function5 +name(fiveplus)
// Exact:     py_default
static char PY_fiveplus__doc__[] =
"documentation"
;

static PyObject *
PY_fiveplus(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin function.fiveplus
    fiveplus();
    Py_RETURN_NONE;
// splicer end function.fiveplus
}

// ----------------------------------------
// Function:  void TestMultilineSplicer
// Exact:     py_default
// ----------------------------------------
// Argument:  std::string & name +intent(inout)
// Exact:     py_string_&_inout
// ----------------------------------------
// Argument:  int * value +intent(out)
// Exact:     py_native_*_out
static char PY_TestMultilineSplicer__doc__[] =
"documentation"
;

/**
 * Use std::string argument to get bufferified function.
 */
static PyObject *
PY_TestMultilineSplicer(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.test_multiline_splicer
// py line 1
// py line 2
// splicer end function.test_multiline_splicer
}

// ----------------------------------------
// Function:  void FunctionTU
// Exact:     py_default
// ----------------------------------------
// Argument:  int arg1 +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
// ----------------------------------------
// Argument:  long arg2 +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
/**
 * \brief Function template with two template parameters.
 *
 */
static PyObject *
PY_name_instantiation1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.function_tu_0
    int arg1;
    long arg2;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "il:FunctionTU",
        const_cast<char **>(SHT_kwlist), &arg1, &arg2))
        return nullptr;

    FunctionTU<int, long>(arg1, arg2);
    Py_RETURN_NONE;
// splicer end function.function_tu_0
}

// ----------------------------------------
// Function:  void FunctionTU
// Exact:     py_default
// ----------------------------------------
// Argument:  float arg1 +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
// ----------------------------------------
// Argument:  double arg2 +intent(in)+value
// Requested: py_native_scalar_in
// Match:     py_default
/**
 * \brief Function template with two template parameters.
 *
 */
static PyObject *
PY_FunctionTU_instantiation2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.function_tu_instantiation2
    float arg1;
    double arg2;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "fd:FunctionTU",
        const_cast<char **>(SHT_kwlist), &arg1, &arg2))
        return nullptr;

    FunctionTU<float, double>(arg1, arg2);
    Py_RETURN_NONE;
// splicer end function.function_tu_instantiation2
}

// ----------------------------------------
// Function:  int UseImplWorker
// Requested: py_native_scalar_result
// Match:     py_default
static char PY_UseImplWorker_instantiation3__doc__[] =
"documentation"
;

/**
 * \brief Function which uses a templated T in the implemetation.
 *
 */
static PyObject *
PY_UseImplWorker_instantiation3(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin function.use_impl_worker_instantiation3
    PyObject * SHTPy_rv = nullptr;

    int ARG_rv = UseImplWorker<internal::ImplWorker1>();

    // post_call
    SHTPy_rv = PyInt_FromLong(ARG_rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.use_impl_worker_instantiation3
}

static char PY_function3a__doc__[] =
"documentation"
;

static PyObject *
PY_function3a(
  PyObject *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.function3a
    Py_ssize_t SHT_nargs = 0;
    if (args != nullptr) SHT_nargs += PyTuple_Size(args);
    if (kwds != nullptr) SHT_nargs += PyDict_Size(args);
    PyObject *rvobj;
    if (SHT_nargs == 1) {
        rvobj = PY_function3a_0(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs == 1) {
        rvobj = PY_function3a_1(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
    return nullptr;
// splicer end function.function3a
}

static char PY_FunctionTU__doc__[] =
"documentation"
;

static PyObject *
PY_FunctionTU(
  PyObject *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.function_tu
    Py_ssize_t SHT_nargs = 0;
    if (args != nullptr) SHT_nargs += PyTuple_Size(args);
    if (kwds != nullptr) SHT_nargs += PyDict_Size(args);
    PyObject *rvobj;
    if (SHT_nargs == 2) {
        rvobj = PY_name_instantiation1(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs == 2) {
        rvobj = PY_FunctionTU_instantiation2(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    PyErr_SetString(PyExc_TypeError, "wrong arguments multi-dispatch");
    return nullptr;
// splicer end function.function_tu
}
static PyMethodDef PY_methods[] = {
{"getName", (PyCFunction)PY_getName, METH_VARARGS|METH_KEYWORDS,
    PY_getName__doc__},
{"function1", (PyCFunction)PY_function1, METH_NOARGS,
    PY_function1__doc__},
{"function2", (PyCFunction)PY_function2, METH_NOARGS,
    PY_function2__doc__},
{"function4", (PyCFunction)PY_function4, METH_VARARGS|METH_KEYWORDS,
    PY_function4__doc__},
{"fiveplus", (PyCFunction)PY_fiveplus, METH_NOARGS, PY_fiveplus__doc__},
{"TestMultilineSplicer", (PyCFunction)PY_TestMultilineSplicer,
    METH_VARARGS|METH_KEYWORDS, PY_TestMultilineSplicer__doc__},
{"UseImplWorker_instantiation3",
    (PyCFunction)PY_UseImplWorker_instantiation3, METH_NOARGS,
    PY_UseImplWorker_instantiation3__doc__},
{"function3a", (PyCFunction)PY_function3a, METH_VARARGS|METH_KEYWORDS,
    PY_function3a__doc__},
{"FunctionTU", (PyCFunction)PY_FunctionTU, METH_VARARGS|METH_KEYWORDS,
    PY_FunctionTU__doc__},
{nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
};

/*
 * inittestnames - Initialization function for the module
 * *must* be called inittestnames
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
static int testnames_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int testnames_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "testnames", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    nullptr, /* m_reload */
    testnames_traverse, /* m_traverse */
    testnames_clear, /* m_clear */
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
PyInit_testnames(void)
#else
inittestnames(void)
#endif
{
    PyObject *m = nullptr;
    const char * error_name = "testnames.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("testnames", PY_methods,
        PY__doc__,
        (PyObject*)nullptr,PYTHON_API_VERSION);
#endif
    if (m == nullptr)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

    {
        PyObject *submodule = PY_init_testnames_ns0();
        if (submodule == nullptr)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "ns0", submodule);
    }

    {
        PyObject *submodule = PY_init_testnames_ns1();
        if (submodule == nullptr)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "ns1", submodule);
    }

    {
        PyObject *submodule = PY_init_testnames_internal();
        if (submodule == nullptr)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "internal", submodule);
    }

    {
        PyObject *submodule = PY_init_testnames_std();
        if (submodule == nullptr)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "std", submodule);
    }

    // Names2
    PY_Names2_Type.tp_new   = PyType_GenericNew;
    PY_Names2_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Names2_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Names2_Type);
    PyModule_AddObject(m, "Names2", (PyObject *)&PY_Names2_Type);

    // twoTs_0
    PY_twoTs_0_Type.tp_new   = PyType_GenericNew;
    PY_twoTs_0_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_twoTs_0_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_twoTs_0_Type);
    PyModule_AddObject(m, "twoTs_0", (PyObject *)&PY_twoTs_0_Type);

    // twoTs_instantiation4
    PY_twoTs_instantiation4_Type.tp_new   = PyType_GenericNew;
    PY_twoTs_instantiation4_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_twoTs_instantiation4_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_twoTs_instantiation4_Type);
    PyModule_AddObject(m, "twoTs_instantiation4", (PyObject *)&PY_twoTs_instantiation4_Type);

    // enum Color
    PyModule_AddIntConstant(m, "RED", RED);
    PyModule_AddIntConstant(m, "BLUE", BLUE);
    PyModule_AddIntConstant(m, "WHITE", WHITE);

    PY_error_obj = PyErr_NewException((char *) error_name, nullptr, nullptr);
    if (PY_error_obj == nullptr)
        return RETVAL;
    st->error = PY_error_obj;
    PyModule_AddObject(m, "Error", st->error);

    // splicer begin C_init_body
    // splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module testnames");
    return RETVAL;
}

