// pytemplatesmodule.cpp
// This is generated code, do not edit
#include "pytemplatesmodule.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "templates.hpp"

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
    FunctionTU(arg1, arg2);
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
    FunctionTU(arg1, arg2);
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
    int SHC_rv = UseImplWorker();
    PyObject * SHTPy_rv = PyInt_FromLong(SHC_rv);
    return (PyObject *) SHTPy_rv;
// splicer end function.use_impl_worker_internal_ImplWorker1
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

    // vector_int
    PY_vector_int_Type.tp_new   = PyType_GenericNew;
    PY_vector_int_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_vector_int_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_vector_int_Type);
    PyModule_AddObject(m, "vector_int", (PyObject *)&PY_vector_int_Type);

    // vector_double
    PY_vector_double_Type.tp_new   = PyType_GenericNew;
    PY_vector_double_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_vector_double_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_vector_double_Type);
    PyModule_AddObject(m, "vector_double", (PyObject *)&PY_vector_double_Type);

    // Worker
    PY_Worker_Type.tp_new   = PyType_GenericNew;
    PY_Worker_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Worker_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Worker_Type);
    PyModule_AddObject(m, "Worker", (PyObject *)&PY_Worker_Type);

    // ImplWorker1
    PY_ImplWorker1_Type.tp_new   = PyType_GenericNew;
    PY_ImplWorker1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_ImplWorker1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_ImplWorker1_Type);
    PyModule_AddObject(m, "ImplWorker1", (PyObject *)&PY_ImplWorker1_Type);

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

