// pytestnames_ns1module.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "pytestnamesmodule.hpp"

// splicer begin namespace.ns1.include
// splicer end namespace.ns1.include

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

// splicer begin namespace.ns1.C_definition
// splicer end namespace.ns1.C_definition
// splicer begin namespace.ns1.additional_functions
// splicer end namespace.ns1.additional_functions

static char PY_init_ns1__doc__[] =
"documentation"
;

static PyObject *
PY_init_ns1(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void init_ns1()
// splicer begin namespace.ns1.function.init_ns1
    ns1::init_ns1();
    Py_RETURN_NONE;
// splicer end namespace.ns1.function.init_ns1
}
static PyMethodDef PY_methods[] = {
{"init_ns1", (PyCFunction)PY_init_ns1, METH_NOARGS, PY_init_ns1__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "testnames.ns1", /* m_name */
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

PyObject *PY_init_testnames_ns1(void)
{
    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3((char *) "testnames.ns1", PY_methods, NULL);
#endif
    if (m == NULL)
        return NULL;


    return m;
}

