// pytestnames_ns0module.cpp
// This file is generated by Shroud 0.11.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pytestnamesmodule.hpp"

// splicer begin namespace.ns0.include
// splicer end namespace.ns0.include

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

// splicer begin namespace.ns0.C_definition
// splicer end namespace.ns0.C_definition
// splicer begin namespace.ns0.additional_functions
// splicer end namespace.ns0.additional_functions
static PyMethodDef PY_methods[] = {
{nullptr,   (PyCFunction)nullptr, 0, nullptr}            /* sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "testnames.ns0", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    nullptr, /* m_reload */
//    testnames_traverse, /* m_traverse */
//    testnames_clear, /* m_clear */
    nullptr, /* m_traverse */
    nullptr, /* m_clear */
    nullptr  /* m_free */
};
#endif
#define RETVAL nullptr

PyObject *PY_init_testnames_ns0(void)
{
    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3((char *) "testnames.ns0", PY_methods, nullptr);
#endif
    if (m == nullptr)
        return nullptr;


    {
        PyObject *submodule = PY_init_testnames_ns0_inner();
        if (submodule == nullptr)
            INITERROR;
        Py_INCREF(submodule);
        PyModule_AddObject(m, (char *) "inner", submodule);
    }

    // Names
    PY_Names_Type.tp_new   = PyType_GenericNew;
    PY_Names_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Names_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Names_Type);
    PyModule_AddObject(m, "Names", (PyObject *)&PY_Names_Type);

    return m;
}

