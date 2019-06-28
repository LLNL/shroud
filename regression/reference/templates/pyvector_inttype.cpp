// pyvector_inttype.cpp
// This is generated code, do not edit
#include "pytemplatesmodule.hpp"
#include <vector>
// splicer begin class.vector.impl.include
// splicer end class.vector.impl.include

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
// splicer begin class.vector.impl.C_definition
// splicer end class.vector.impl.C_definition
// splicer begin class.vector.impl.additional_methods
// splicer end class.vector.impl.additional_methods
static void
PY_vector_int_tp_del (PY_vector_int *self)
{
// splicer begin class.vector.type.del
    delete self->obj;
    self->obj = NULL;
// splicer end class.vector.type.del
}

static int
PY_vector_int_tp_init(
  PY_vector_int *self,
  PyObject *SHROUD_UNUSED(args),
  PyObject *SHROUD_UNUSED(kwds))
{
// splicer begin class.vector.method.ctor
    self->obj = new std::vector_int();
    return 0;
// splicer end class.vector.method.ctor
}

static char PY_vector_int_push_back__doc__[] =
"documentation"
;

static PyObject *
PY_vector_int_push_back(
  PY_vector_int *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin class.vector.method.push_back
    int value;
    const char *SHT_kwlist[] = {
        "value",
        NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i:push_back",
        const_cast<char **>(SHT_kwlist), &value))
        return NULL;
    self->obj->push_back(value);
    Py_RETURN_NONE;
// splicer end class.vector.method.push_back
}

static char PY_vector_int_at__doc__[] =
"documentation"
;

static PyObject *
PY_vector_int_at(
  PY_vector_int *self,
  PyObject *args,
  PyObject *kwds)
{
// splicer begin class.vector.method.at
    size_t n;
    const char *SHT_kwlist[] = {
        "n",
        NULL };
    PyObject * SHTPy_rv = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:at",
        const_cast<char **>(SHT_kwlist), &n))
        return NULL;
    int & rv = self->obj->at(n);
    SHTPy_rv = PyArray_SimpleNewFromData(0, NULL, NPY_INT, rv);
    if (SHTPy_rv == NULL) goto fail;
    return (PyObject *) SHTPy_rv;

fail:
    Py_XDECREF(SHTPy_rv);
    return NULL;
// splicer end class.vector.method.at
}
// splicer begin class.vector.impl.after_methods
// splicer end class.vector.impl.after_methods
static PyMethodDef PY_vector_int_methods[] = {
    {"push_back", (PyCFunction)PY_vector_int_push_back,
        METH_VARARGS|METH_KEYWORDS, PY_vector_int_push_back__doc__},
    {"at", (PyCFunction)PY_vector_int_at, METH_VARARGS|METH_KEYWORDS,
        PY_vector_int_at__doc__},
    // splicer begin class.vector.PyMethodDef
    // splicer end class.vector.PyMethodDef
    {NULL,   (PyCFunction)NULL, 0, NULL}            /* sentinel */
};

static char vector_int__doc__[] =
"virtual class"
;

/* static */
PyTypeObject PY_vector_int_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "templates.vector_int",                       /* tp_name */
    sizeof(PY_vector_int),         /* tp_basicsize */
    0,                              /* tp_itemsize */
    /* Methods to implement standard operations */
    (destructor)0,                 /* tp_dealloc */
    (printfunc)0,                   /* tp_print */
    (getattrfunc)0,                 /* tp_getattr */
    (setattrfunc)0,                 /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
    0,                               /* tp_reserved */
#else
    (cmpfunc)0,                     /* tp_compare */
#endif
    (reprfunc)0,                    /* tp_repr */
    /* Method suites for standard classes */
    0,                              /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    /* More standard operations (here for binary compatibility) */
    (hashfunc)0,                    /* tp_hash */
    (ternaryfunc)0,                 /* tp_call */
    (reprfunc)0,                    /* tp_str */
    (getattrofunc)0,                /* tp_getattro */
    (setattrofunc)0,                /* tp_setattro */
    /* Functions to access object as input/output buffer */
    0,                              /* tp_as_buffer */
    /* Flags to define presence of optional/expanded features */
    Py_TPFLAGS_DEFAULT,             /* tp_flags */
    vector_int__doc__,         /* tp_doc */
    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    (traverseproc)0,                /* tp_traverse */
    /* delete references to contained objects */
    (inquiry)0,                     /* tp_clear */
    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    (richcmpfunc)0,                 /* tp_richcompare */
    /* weak reference enabler */
    0,                              /* tp_weaklistoffset */
    /* Added in release 2.2 */
    /* Iterators */
    (getiterfunc)0,                 /* tp_iter */
    (iternextfunc)0,                /* tp_iternext */
    /* Attribute descriptor and subclassing stuff */
    PY_vector_int_methods,                             /* tp_methods */
    0,                              /* tp_members */
    0,                             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    (descrgetfunc)0,                /* tp_descr_get */
    (descrsetfunc)0,                /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)PY_vector_int_tp_init,                   /* tp_init */
    (allocfunc)0,                  /* tp_alloc */
    (newfunc)0,                    /* tp_new */
    (freefunc)0,                   /* tp_free */
    (inquiry)0,                     /* tp_is_gc */
    0,                              /* tp_bases */
    0,                              /* tp_mro */
    0,                              /* tp_cache */
    0,                              /* tp_subclasses */
    0,                              /* tp_weaklist */
    (destructor)PY_vector_int_tp_del,                 /* tp_del */
    0,                              /* tp_version_tag */
#if PY_MAJOR_VERSION >= 3
    (destructor)0,                  /* tp_finalize */
#endif
};
