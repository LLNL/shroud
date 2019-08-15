// pytemplates_stdmodule.cpp
// This is generated code, do not edit
#include "pytemplatesmodule.hpp"
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
// splicer begin additional_functions
// splicer end additional_functions
static PyMethodDef PY_methods[] = {
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "templates", /* m_name */
    PY__doc__, /* m_doc */
    sizeof(struct module_state), /* m_size */
    PY_methods, /* m_methods */
    NULL, /* m_reload */
//    templates_traverse, /* m_traverse */
//    templates_clear, /* m_clear */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL  /* m_free */
};
#endif
#define RETVAL NULL

PyObject *PY_init_templates_std(void)
{
    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3((char *) "templates", PY_methods, NULL);
#endif
    if (m == NULL)
        return NULL;


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

    return m;
}

