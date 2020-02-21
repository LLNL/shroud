// pyns_nsworkmodule.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pynsmodule.hpp"

// splicer begin namespace.nswork.include
// splicer end namespace.nswork.include

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

// splicer begin namespace.nswork.C_definition
// splicer end namespace.nswork.C_definition
// splicer begin namespace.nswork.additional_functions
// splicer end namespace.nswork.additional_functions
static PyMethodDef PY_methods[] = {
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ns.nswork", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
//    ns_traverse, /* m_traverse */
//    ns_clear, /* m_clear */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL  /* m_free */
};
#endif
#define RETVAL NULL

PyObject *PY_init_ns_nswork(void)
{
    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3((char *) "ns.nswork", PY_methods, NULL);
#endif
    if (m == NULL)
        return NULL;


    // ClassWork
    PY_ClassWork_Type.tp_new   = PyType_GenericNew;
    PY_ClassWork_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_ClassWork_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_ClassWork_Type);
    PyModule_AddObject(m, "ClassWork", (PyObject *)&PY_ClassWork_Type);

    return m;
}

