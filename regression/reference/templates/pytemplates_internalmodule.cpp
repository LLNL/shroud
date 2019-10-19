// pytemplates_internalmodule.cpp
// This is generated code, do not edit
#include "pytemplatesmodule.hpp"
#include "templates.hpp"

// splicer begin namespace.internal.include
// splicer end namespace.internal.include

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

// splicer begin namespace.internal.C_definition
// splicer end namespace.internal.C_definition
// splicer begin namespace.internal.additional_functions
// splicer end namespace.internal.additional_functions
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

PyObject *PY_init_templates_internal(void)
{
    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3((char *) "templates", PY_methods, NULL);
#endif
    if (m == NULL)
        return NULL;


    // ImplWorker1
    PY_ImplWorker1_Type.tp_new   = PyType_GenericNew;
    PY_ImplWorker1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_ImplWorker1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_ImplWorker1_Type);
    PyModule_AddObject(m, "ImplWorker1", (PyObject *)&PY_ImplWorker1_Type);

    // ImplWorker2
    PY_ImplWorker2_Type.tp_new   = PyType_GenericNew;
    PY_ImplWorker2_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_ImplWorker2_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_ImplWorker2_Type);
    PyModule_AddObject(m, "ImplWorker2", (PyObject *)&PY_ImplWorker2_Type);

    return m;
}

