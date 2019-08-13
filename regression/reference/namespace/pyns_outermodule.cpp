// pyns_outermodule.cpp
// This is generated code, do not edit
#include "pynsmodule.hpp"
#include "namespace.hpp"

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

static char PY_One__doc__[] =
"documentation"
;

static PyObject *
PY_One(
  PyObject *SHROUD_UNUSED(self),
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// void One()
// splicer begin function.one
    outer::One();
    Py_RETURN_NONE;
// splicer end function.one
}
static PyMethodDef PY_methods[] = {
{"One", (PyCFunction)PY_One, METH_NOARGS, PY_One__doc__},
{NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ns.outer", /* m_name */
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

PyObject *PY_init_ns_outer(void)
{
    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3((char *) "ns.outer", PY_methods, NULL);
#endif
    if (m == NULL)
        return NULL;


    // Cstruct1
    PY_Cstruct1_Type.tp_new   = PyType_GenericNew;
    PY_Cstruct1_Type.tp_alloc = PyType_GenericAlloc;
    if (PyType_Ready(&PY_Cstruct1_Type) < 0)
        return RETVAL;
    Py_INCREF(&PY_Cstruct1_Type);
    PyModule_AddObject(m, "Cstruct1", (PyObject *)&PY_Cstruct1_Type);

    return m;
}

