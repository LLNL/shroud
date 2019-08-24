// pytestnamesmodule.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
//
// All rights reserved.
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
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

static char PY_function1__doc__[] =
"documentation"
;

static PyObject *
PY_function1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void function1()
// splicer begin function.function1
    function1();
    Py_RETURN_NONE;
// splicer end function.function1
}

static char PY_function2__doc__[] =
"documentation"
;

static PyObject *
PY_function2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void function2()
// splicer begin function.function2
    function2();
    Py_RETURN_NONE;
// splicer end function.function2
}

static PyObject *
PY_function3a_0(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void function3a(int i +intent(in)+value)
// splicer begin function.function3a_0
    int i;
    const char *SHT_kwlist[] = {
        "i",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:function3a",
        const_cast<char **>(SHT_kwlist), &i))
        return NULL;

    function3a(i);
    Py_RETURN_NONE;
// splicer end function.function3a_0
}

static PyObject *
PY_function3a_1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// void function3a(long i +intent(in)+value)
// splicer begin function.function3a_1
    long i;
    const char *SHT_kwlist[] = {
        "i",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l:function3a",
        const_cast<char **>(SHT_kwlist), &i))
        return NULL;

    function3a(i);
    Py_RETURN_NONE;
// splicer end function.function3a_1
}

static char PY_function4__doc__[] =
"documentation"
;

static PyObject *
PY_function4(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// int function4(const std::string & rv +intent(in))
// splicer begin function.function4
    const char * rv;
    const char *SHT_kwlist[] = {
        "rv",
        NULL };
    PyObject * SHTPy_rv = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:function4",
        const_cast<char **>(SHT_kwlist), &rv))
        return NULL;

    // post_parse
    const std::string SH_rv(rv);

    int rv = function4(SH_rv);

    // post_call
    SHTPy_rv = PyInt_FromLong(rv);

    return (PyObject *) SHTPy_rv;
// splicer end function.function4
}

static char PY_fiveplus__doc__[] =
"documentation"
;

static PyObject *
PY_fiveplus(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void function5() +name(fiveplus)
// splicer begin function.fiveplus
    fiveplus();
    Py_RETURN_NONE;
// splicer end function.fiveplus
}

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
// void FunctionTU(int arg1 +intent(in)+value, long arg2 +intent(in)+value)
// splicer begin function.function_tu_0
    int arg1;
    long arg2;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "il:FunctionTU",
        const_cast<char **>(SHT_kwlist), &arg1, &arg2))
        return NULL;

    FunctionTU(arg1, arg2);
    Py_RETURN_NONE;
// splicer end function.function_tu_0
}

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
// void FunctionTU(float arg1 +intent(in)+value, double arg2 +intent(in)+value)
// splicer begin function.function_tu_instantiation2
    float arg1;
    double arg2;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "fd:FunctionTU",
        const_cast<char **>(SHT_kwlist), &arg1, &arg2))
        return NULL;

    FunctionTU(arg1, arg2);
    Py_RETURN_NONE;
// splicer end function.function_tu_instantiation2
}

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
// int UseImplWorker()
// splicer begin function.use_impl_worker_instantiation3
    PyObject * SHTPy_rv = NULL;

    int rv = UseImplWorker();

    // post_call
    SHTPy_rv = PyInt_FromLong(rv);

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
    if (args != NULL) SHT_nargs += PyTuple_Size(args);
    if (kwds != NULL) SHT_nargs += PyDict_Size(args);
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
    return NULL;
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
    if (args != NULL) SHT_nargs += PyTuple_Size(args);
    if (kwds != NULL) SHT_nargs += PyDict_Size(args);
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
    return NULL;
// splicer end function.function_tu
}
static PyMethodDef PY_methods[] = {
{"function1", (PyCFunction)PY_function1, METH_NOARGS,
    PY_function1__doc__},
{"function2", (PyCFunction)PY_function2, METH_NOARGS,
    PY_function2__doc__},
{"function4", (PyCFunction)PY_function4, METH_VARARGS|METH_KEYWORDS,
    PY_function4__doc__},
{"fiveplus", (PyCFunction)PY_fiveplus, METH_NOARGS, PY_fiveplus__doc__},
{"UseImplWorker_instantiation3",
    (PyCFunction)PY_UseImplWorker_instantiation3, METH_NOARGS,
    PY_UseImplWorker_instantiation3__doc__},
{"function3a", (PyCFunction)PY_function3a, METH_VARARGS|METH_KEYWORDS,
    PY_function3a__doc__},
{"FunctionTU", (PyCFunction)PY_FunctionTU, METH_VARARGS|METH_KEYWORDS,
    PY_FunctionTU__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
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
    NULL, /* m_reload */
    testnames_traverse, /* m_traverse */
    testnames_clear, /* m_clear */
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
PyInit_testnames(void)
#else
inittestnames(void)
#endif
{
    PyObject *m = NULL;
    const char * error_name = "testnames.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("testnames", PY_methods,
        PY__doc__,
        (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

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

    {
        PyObject *submodule = PY_init_testnames_ns0();
        if (submodule == NULL)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "ns0", submodule);
    }

    {
        PyObject *submodule = PY_init_testnames_ns1();
        if (submodule == NULL)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "ns1", submodule);
    }

    {
        PyObject *submodule = PY_init_testnames_internal();
        if (submodule == NULL)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "internal", submodule);
    }

    {
        PyObject *submodule = PY_init_testnames_std();
        if (submodule == NULL)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "std", submodule);
    }

    // enum Color
    PyModule_AddIntConstant(m, "RED", RED);
    PyModule_AddIntConstant(m, "BLUE", BLUE);
    PyModule_AddIntConstant(m, "WHITE", WHITE);

    PY_error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
    if (PY_error_obj == NULL)
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

