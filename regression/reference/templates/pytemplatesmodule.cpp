// pytemplatesmodule.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pytemplatesmodule.hpp"
#define PY_ARRAY_UNIQUE_SYMBOL SHROUD_TEMPLATES_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

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
PyObject *PY_init_templates_std(void);
PyObject *PY_init_templates_internal(void);
// splicer begin additional_functions
// splicer end additional_functions

/**
 * \brief Function template with two template parameters.
 *
 */
static PyObject *
PY_FunctionTU_0(
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
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "il:FunctionTU",
        const_cast<char **>(SHT_kwlist), &arg1, &arg2))
        return NULL;
    FunctionTU<int, long>(arg1, arg2);
    Py_RETURN_NONE;
// splicer end function.function_tu_0
}

/**
 * \brief Function template with two template parameters.
 *
 */
static PyObject *
PY_FunctionTU_1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *args,
  PyObject *kwds)
{
// splicer begin function.function_tu_1
    float arg1;
    double arg2;
    const char *SHT_kwlist[] = {
        "arg1",
        "arg2",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "fd:FunctionTU",
        const_cast<char **>(SHT_kwlist), &arg1, &arg2))
        return NULL;
    FunctionTU<float, double>(arg1, arg2);
    Py_RETURN_NONE;
// splicer end function.function_tu_1
}

static char PY_UseImplWorker_internal_ImplWorker1__doc__[] =
"documentation"
;

/**
 * \brief Function which uses a templated T in the implemetation.
 *
 */
static PyObject *
PY_UseImplWorker_internal_ImplWorker1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin function.use_impl_worker_internal_ImplWorker1
    PyObject * SHTPy_rv = NULL;

    int SHCXX_rv = UseImplWorker<internal::ImplWorker1>();
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);
    return (PyObject *) SHTPy_rv;
// splicer end function.use_impl_worker_internal_ImplWorker1
}

static char PY_UseImplWorker_internal_ImplWorker2__doc__[] =
"documentation"
;

/**
 * \brief Function which uses a templated T in the implemetation.
 *
 */
static PyObject *
PY_UseImplWorker_internal_ImplWorker2(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin function.use_impl_worker_internal_ImplWorker2
    PyObject * SHTPy_rv = NULL;

    int SHCXX_rv = UseImplWorker<internal::ImplWorker2>();
    SHTPy_rv = PyInt_FromLong(SHCXX_rv);
    return (PyObject *) SHTPy_rv;
// splicer end function.use_impl_worker_internal_ImplWorker2
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
        rvobj = PY_FunctionTU_0(self, args, kwds);
        if (!PyErr_Occurred()) {
            return rvobj;
        } else if (! PyErr_ExceptionMatches(PyExc_TypeError)) {
            return rvobj;
        }
        PyErr_Clear();
    }
    if (SHT_nargs == 2) {
        rvobj = PY_FunctionTU_1(self, args, kwds);
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
{"UseImplWorker_internal_ImplWorker1",
    (PyCFunction)PY_UseImplWorker_internal_ImplWorker1, METH_NOARGS,
    PY_UseImplWorker_internal_ImplWorker1__doc__},
{"UseImplWorker_internal_ImplWorker2",
    (PyCFunction)PY_UseImplWorker_internal_ImplWorker2, METH_NOARGS,
    PY_UseImplWorker_internal_ImplWorker2__doc__},
{"FunctionTU", (PyCFunction)PY_FunctionTU, METH_VARARGS|METH_KEYWORDS,
    PY_FunctionTU__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

/*
 * inittemplates - Initialization function for the module
 * *must* be called inittemplates
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
static int templates_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int templates_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "templates", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
    templates_traverse, /* m_traverse */
    templates_clear, /* m_clear */
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
PyInit_templates(void)
#else
inittemplates(void)
#endif
{
    PyObject *m = NULL;
    const char * error_name = "templates.Error";

    // splicer begin C_init_locals
    // splicer end C_init_locals


    /* Create the module and add the functions */
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("templates", PY_methods,
        PY__doc__,
        (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL)
        return RETVAL;
    struct module_state *st = GETSTATE(m);

    import_array();

    {
        PyObject *submodule = PY_init_templates_std();
        if (submodule == NULL)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "std", submodule);
    }

    {
        PyObject *submodule = PY_init_templates_internal();
        if (submodule == NULL)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "internal", submodule);
    }

    // Worker
    PY_Worker_Type.tp_new   = PyType_GenericNew;
    PY_Worker_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Worker_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Worker_Type);
    PyModule_AddObject(m, "Worker", (PyObject *)&PY_Worker_Type);

    // user_int
    PY_user_int_Type.tp_new   = PyType_GenericNew;
    PY_user_int_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_user_int_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_user_int_Type);
    PyModule_AddObject(m, "user_int", (PyObject *)&PY_user_int_Type);

    PY_error_obj = PyErr_NewException((char *) error_name, NULL, NULL);
    if (PY_error_obj == NULL)
        return RETVAL;
    st->error = PY_error_obj;
    PyModule_AddObject(m, "Error", st->error);

    // splicer begin C_init_body
    // splicer end C_init_body

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module templates");
    return RETVAL;
}

