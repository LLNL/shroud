// pytestnames_stdmodule.cpp
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
// splicer begin additional_functions
// splicer end additional_functions
static PyMethodDef PY_methods[] = {
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "testnames.std", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
//    testnames_traverse, /* m_traverse */
//    testnames_clear, /* m_clear */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL  /* m_free */
};
#endif
#define RETVAL NULL

PyObject *PY_init_testnames_std(void)
{
    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3((char *) "testnames.std", PY_methods, NULL);
#endif
    if (m == NULL)
        return NULL;


    // Vvv1
    PY_Vvv1_Type.tp_new   = PyType_GenericNew;
    PY_Vvv1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Vvv1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Vvv1_Type);
    PyModule_AddObject(m, "Vvv1", (PyObject *)&PY_Vvv1_Type);

    // vector_double
    PY_vector_double_Type.tp_new   = PyType_GenericNew;
    PY_vector_double_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_vector_double_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_vector_double_Type);
    PyModule_AddObject(m, "vector_double", (PyObject *)&PY_vector_double_Type);

    // vector_instantiation5
    PY_vector_instantiation5_Type.tp_new   = PyType_GenericNew;
    PY_vector_instantiation5_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_vector_instantiation5_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_vector_instantiation5_Type);
    PyModule_AddObject(m, "vector_instantiation5", (PyObject *)&PY_vector_instantiation5_Type);

    // vector_instantiation3
    PY_vector_instantiation3_Type.tp_new   = PyType_GenericNew;
    PY_vector_instantiation3_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_vector_instantiation3_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_vector_instantiation3_Type);
    PyModule_AddObject(m, "vector_instantiation3", (PyObject *)&PY_vector_instantiation3_Type);

    return m;
}

