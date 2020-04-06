/* This file was generated by PyBindGen 0.0.0.0 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>


#if PY_VERSION_HEX < 0x020400F0

#define PyEval_ThreadsInitialized() 1

#define Py_CLEAR(op)				\
        do {                            	\
                if (op) {			\
                        PyObject *tmp = (PyObject *)(op);	\
                        (op) = NULL;		\
                        Py_DECREF(tmp);		\
                }				\
        } while (0)


#define Py_VISIT(op)							\
        do { 								\
                if (op) {						\
                        int vret = visit((PyObject *)(op), arg);	\
                        if (vret)					\
                                return vret;				\
                }							\
        } while (0)

#endif



#if PY_VERSION_HEX < 0x020500F0

typedef int Py_ssize_t;
# define PY_SSIZE_T_MAX INT_MAX
# define PY_SSIZE_T_MIN INT_MIN
typedef inquiry lenfunc;
typedef intargfunc ssizeargfunc;
typedef intobjargproc ssizeobjargproc;

#endif


#ifndef PyVarObject_HEAD_INIT
#define PyVarObject_HEAD_INIT(type, size) \
        PyObject_HEAD_INIT(type) size,
#endif


#if PY_VERSION_HEX >= 0x03000000
#if PY_VERSION_HEX >= 0x03050000
typedef PyAsyncMethods* cmpfunc;
#else
typedef void* cmpfunc;
#endif
#define PyCObject_FromVoidPtr(a, b) PyCapsule_New(a, NULL, b)
#define PyCObject_AsVoidPtr(a) PyCapsule_GetPointer(a, NULL)
#define PyString_FromString(a) PyBytes_FromString(a)
#define Py_TPFLAGS_CHECKTYPES 0 /* this flag doesn't exist in python 3 */
#endif


#if     __GNUC__ > 2
# define PYBINDGEN_UNUSED(param) param __attribute__((__unused__))
#elif     __GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ > 4)
# define PYBINDGEN_UNUSED(param) __attribute__((__unused__)) param
#else
# define PYBINDGEN_UNUSED(param) param
#endif  /* !__GNUC__ */

#ifndef _PyBindGenWrapperFlags_defined_
#define _PyBindGenWrapperFlags_defined_
typedef enum _PyBindGenWrapperFlags {
   PYBINDGEN_WRAPPER_FLAG_NONE = 0,
   PYBINDGEN_WRAPPER_FLAG_OBJECT_NOT_OWNED = (1<<0),
} PyBindGenWrapperFlags;
#endif


#include "classes.hpp"
/* --- forward declarations --- */


typedef struct {
    PyObject_HEAD
    classes::Class1 *obj;
    PyBindGenWrapperFlags flags:8;
} PyClassesClass1;


extern PyTypeObject PyClassesClass1_Type;


typedef struct {
    PyObject_HEAD
    classes::Singleton *obj;
    PyBindGenWrapperFlags flags:8;
} PyClassesSingleton;


extern PyTypeObject PyClassesSingleton_Type;

static PyMethodDef classes_classes_functions[] = {
    {NULL, NULL, 0, NULL}
};
/* --- classes --- */


static PyObject* _wrap_PyClassesClass1__get_m_flag(PyClassesClass1 *self, void * PYBINDGEN_UNUSED(closure))
{
    PyObject *py_retval;

    py_retval = Py_BuildValue((char *) "i", self->obj->m_flag);
    return py_retval;
}
static int _wrap_PyClassesClass1__set_m_flag(PyClassesClass1 *self, PyObject *value, void * PYBINDGEN_UNUSED(closure))
{
    PyObject *py_retval;

    py_retval = Py_BuildValue((char *) "(O)", value);
    if (!PyArg_ParseTuple(py_retval, (char *) "i", &self->obj->m_flag)) {
        Py_DECREF(py_retval);
        return -1;
    }
    Py_DECREF(py_retval);
    return 0;
}
static PyGetSetDef PyClassesClass1__getsets[] = {
    {
        (char*) "m_flag", /* attribute name */
        (getter) _wrap_PyClassesClass1__get_m_flag, /* C function to get the attribute */
        (setter) _wrap_PyClassesClass1__set_m_flag, /* C function to set the attribute */
        NULL, /* optional doc string */
        NULL /* optional additional data for getter and setter */
    },
    { NULL, NULL, NULL, NULL, NULL }
};


static int
_wrap_PyClassesClass1__tp_init__0(PyClassesClass1 *self, PyObject *args, PyObject *kwargs, PyObject **return_exception)
{
    int flag;
    const char *keywords[] = {"flag", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, (char *) "i", (char **) keywords, &flag)) {
        {
            PyObject *exc_type, *traceback;
            PyErr_Fetch(&exc_type, return_exception, &traceback);
            Py_XDECREF(exc_type);
            Py_XDECREF(traceback);
        }
        return -1;
    }
    self->obj = new classes::Class1(flag);
    self->flags = PYBINDGEN_WRAPPER_FLAG_NONE;
    return 0;
}

static int
_wrap_PyClassesClass1__tp_init__1(PyClassesClass1 *self, PyObject *args, PyObject *kwargs, PyObject **return_exception)
{
    const char *keywords[] = {NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, (char *) "", (char **) keywords)) {
        {
            PyObject *exc_type, *traceback;
            PyErr_Fetch(&exc_type, return_exception, &traceback);
            Py_XDECREF(exc_type);
            Py_XDECREF(traceback);
        }
        return -1;
    }
    self->obj = new classes::Class1();
    self->flags = PYBINDGEN_WRAPPER_FLAG_NONE;
    return 0;
}

int _wrap_PyClassesClass1__tp_init(PyClassesClass1 *self, PyObject *args, PyObject *kwargs)
{
    int retval;
    PyObject *error_list;
    PyObject *exceptions[2] = {0,};
    retval = _wrap_PyClassesClass1__tp_init__0(self, args, kwargs, &exceptions[0]);
    if (!exceptions[0]) {
        return retval;
    }
    retval = _wrap_PyClassesClass1__tp_init__1(self, args, kwargs, &exceptions[1]);
    if (!exceptions[1]) {
        Py_DECREF(exceptions[0]);
        return retval;
    }
    error_list = PyList_New(2);
    PyList_SET_ITEM(error_list, 0, PyObject_Str(exceptions[0]));
    Py_DECREF(exceptions[0]);
    PyList_SET_ITEM(error_list, 1, PyObject_Str(exceptions[1]));
    Py_DECREF(exceptions[1]);
    PyErr_SetObject(PyExc_TypeError, error_list);
    Py_DECREF(error_list);
    return -1;
}


PyObject *
_wrap_PyClassesClass1_Method1(PyClassesClass1 *self)
{
    PyObject *py_retval;

    self->obj->Method1();
    Py_INCREF(Py_None);
    py_retval = Py_None;
    return py_retval;
}

static PyMethodDef PyClassesClass1_methods[] = {
    {(char *) "Method1", (PyCFunction) _wrap_PyClassesClass1_Method1, METH_NOARGS, "Method1()\n\n" },
    {NULL, NULL, 0, NULL}
};

static void
_wrap_PyClassesClass1__tp_dealloc(PyClassesClass1 *self)
{
        classes::Class1 *tmp = self->obj;
        self->obj = NULL;
        if (!(self->flags&PYBINDGEN_WRAPPER_FLAG_OBJECT_NOT_OWNED)) {
            delete tmp;
        }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyTypeObject PyClassesClass1_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    (char *) "classes.classes.Class1",            /* tp_name */
    sizeof(PyClassesClass1),                  /* tp_basicsize */
    0,                                 /* tp_itemsize */
    /* methods */
    (destructor)_wrap_PyClassesClass1__tp_dealloc,        /* tp_dealloc */
    (printfunc)0,                      /* tp_print */
    (getattrfunc)NULL,       /* tp_getattr */
    (setattrfunc)NULL,       /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
    NULL,
#else
    (cmpfunc)NULL,           /* tp_compare */
#endif
    (reprfunc)NULL,             /* tp_repr */
    (PyNumberMethods*)NULL,     /* tp_as_number */
    (PySequenceMethods*)NULL, /* tp_as_sequence */
    (PyMappingMethods*)NULL,   /* tp_as_mapping */
    (hashfunc)NULL,             /* tp_hash */
    (ternaryfunc)NULL,          /* tp_call */
    (reprfunc)NULL,              /* tp_str */
    (getattrofunc)NULL,     /* tp_getattro */
    (setattrofunc)NULL,     /* tp_setattro */
    (PyBufferProcs*)NULL,  /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                      /* tp_flags */
    "Class1(flag)\nClass1()",                        /* Documentation string */
    (traverseproc)NULL,     /* tp_traverse */
    (inquiry)NULL,             /* tp_clear */
    (richcmpfunc)NULL,   /* tp_richcompare */
    0,             /* tp_weaklistoffset */
    (getiterfunc)NULL,          /* tp_iter */
    (iternextfunc)NULL,     /* tp_iternext */
    (struct PyMethodDef*)PyClassesClass1_methods, /* tp_methods */
    (struct PyMemberDef*)0,              /* tp_members */
    PyClassesClass1__getsets,                     /* tp_getset */
    NULL,                              /* tp_base */
    NULL,                              /* tp_dict */
    (descrgetfunc)NULL,    /* tp_descr_get */
    (descrsetfunc)NULL,    /* tp_descr_set */
    0,                 /* tp_dictoffset */
    (initproc)_wrap_PyClassesClass1__tp_init,             /* tp_init */
    (allocfunc)PyType_GenericAlloc,           /* tp_alloc */
    (newfunc)PyType_GenericNew,               /* tp_new */
    (freefunc)0,             /* tp_free */
    (inquiry)NULL,             /* tp_is_gc */
    NULL,                              /* tp_bases */
    NULL,                              /* tp_mro */
    NULL,                              /* tp_cache */
    NULL,                              /* tp_subclasses */
    NULL,                              /* tp_weaklist */
    (destructor) NULL                  /* tp_del */
};




static int
_wrap_PyClassesSingleton__tp_init(void)
{
    PyErr_SetString(PyExc_TypeError, "class 'Singleton' cannot be constructed ()");
    return -1;
}


PyObject *
_wrap_PyClassesSingleton_getReference(void)
{
    PyObject *py_retval;
    PyClassesSingleton *py_Singleton;

    classes::Singleton & retval = classes::Singleton::getReference();
    py_Singleton = PyObject_New(PyClassesSingleton, &PyClassesSingleton_Type);
    py_Singleton->obj = (&retval);
    py_Singleton->flags = PYBINDGEN_WRAPPER_FLAG_NONE;
    py_retval = Py_BuildValue((char *) "N", py_Singleton);
    return py_retval;
}

static PyMethodDef PyClassesSingleton_methods[] = {
    {(char *) "getReference", (PyCFunction) _wrap_PyClassesSingleton_getReference, METH_NOARGS|METH_STATIC, "getReference()\n\n" },
    {NULL, NULL, 0, NULL}
};

static void
_wrap_PyClassesSingleton__tp_dealloc(PyClassesSingleton *self)
{

    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyTypeObject PyClassesSingleton_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    (char *) "classes.classes.Singleton",            /* tp_name */
    sizeof(PyClassesSingleton),                  /* tp_basicsize */
    0,                                 /* tp_itemsize */
    /* methods */
    (destructor)_wrap_PyClassesSingleton__tp_dealloc,        /* tp_dealloc */
    (printfunc)0,                      /* tp_print */
    (getattrfunc)NULL,       /* tp_getattr */
    (setattrfunc)NULL,       /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
    NULL,
#else
    (cmpfunc)NULL,           /* tp_compare */
#endif
    (reprfunc)NULL,             /* tp_repr */
    (PyNumberMethods*)NULL,     /* tp_as_number */
    (PySequenceMethods*)NULL, /* tp_as_sequence */
    (PyMappingMethods*)NULL,   /* tp_as_mapping */
    (hashfunc)NULL,             /* tp_hash */
    (ternaryfunc)NULL,          /* tp_call */
    (reprfunc)NULL,              /* tp_str */
    (getattrofunc)NULL,     /* tp_getattro */
    (setattrofunc)NULL,     /* tp_setattro */
    (PyBufferProcs*)NULL,  /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                      /* tp_flags */
    "",                        /* Documentation string */
    (traverseproc)NULL,     /* tp_traverse */
    (inquiry)NULL,             /* tp_clear */
    (richcmpfunc)NULL,   /* tp_richcompare */
    0,             /* tp_weaklistoffset */
    (getiterfunc)NULL,          /* tp_iter */
    (iternextfunc)NULL,     /* tp_iternext */
    (struct PyMethodDef*)PyClassesSingleton_methods, /* tp_methods */
    (struct PyMemberDef*)0,              /* tp_members */
    0,                     /* tp_getset */
    NULL,                              /* tp_base */
    NULL,                              /* tp_dict */
    (descrgetfunc)NULL,    /* tp_descr_get */
    (descrsetfunc)NULL,    /* tp_descr_set */
    0,                 /* tp_dictoffset */
    (initproc)_wrap_PyClassesSingleton__tp_init,             /* tp_init */
    (allocfunc)PyType_GenericAlloc,           /* tp_alloc */
    (newfunc)PyType_GenericNew,               /* tp_new */
    (freefunc)0,             /* tp_free */
    (inquiry)NULL,             /* tp_is_gc */
    NULL,                              /* tp_bases */
    NULL,                              /* tp_mro */
    NULL,                              /* tp_cache */
    NULL,                              /* tp_subclasses */
    NULL,                              /* tp_weaklist */
    (destructor) NULL                  /* tp_del */
};


/* --- enumerations --- */



#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef classes_classes_moduledef = {
    PyModuleDef_HEAD_INIT,
    "classes.classes",
    NULL,
    -1,
    classes_classes_functions,
};
#endif

static PyObject *
initclasses_classes(void)
{
    PyObject *m;
    #if PY_VERSION_HEX >= 0x03000000
    m = PyModule_Create(&classes_classes_moduledef);
    #else
    m = Py_InitModule3((char *) "classes.classes", classes_classes_functions, NULL);
    #endif
    if (m == NULL) {
        return NULL;
    }
    /* Register the 'classes::Class1' class */
    if (PyType_Ready(&PyClassesClass1_Type)) {
        return NULL;
    }
    PyModule_AddObject(m, (char *) "Class1", (PyObject *) &PyClassesClass1_Type);
    /* Register the 'classes::Singleton' class */
    if (PyType_Ready(&PyClassesSingleton_Type)) {
        return NULL;
    }
    PyModule_AddObject(m, (char *) "Singleton", (PyObject *) &PyClassesSingleton_Type);
    {
        PyObject *tmp_value;
         // classes::Class1::UP
        tmp_value = PyLong_FromLong(classes::Class1::UP);
        PyDict_SetItemString((PyObject*) PyClassesClass1_Type.tp_dict, "UP", tmp_value);
        Py_DECREF(tmp_value);
         // classes::Class1::DOWN
        tmp_value = PyLong_FromLong(classes::Class1::DOWN);
        PyDict_SetItemString((PyObject*) PyClassesClass1_Type.tp_dict, "DOWN", tmp_value);
        Py_DECREF(tmp_value);
         // classes::Class1::LEFT
        tmp_value = PyLong_FromLong(classes::Class1::LEFT);
        PyDict_SetItemString((PyObject*) PyClassesClass1_Type.tp_dict, "LEFT", tmp_value);
        Py_DECREF(tmp_value);
         // classes::Class1::RIGHT
        tmp_value = PyLong_FromLong(classes::Class1::RIGHT);
        PyDict_SetItemString((PyObject*) PyClassesClass1_Type.tp_dict, "RIGHT", tmp_value);
        Py_DECREF(tmp_value);
    }
    return m;
}
static PyMethodDef classes_functions[] = {
    {NULL, NULL, 0, NULL}
};
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef classes_moduledef = {
    PyModuleDef_HEAD_INIT,
    "classes",
    NULL,
    -1,
    classes_functions,
};
#endif


#if PY_VERSION_HEX >= 0x03000000
    #define MOD_ERROR NULL
    #define MOD_INIT(name) PyObject* PyInit_##name(void)
    #define MOD_RETURN(val) val
#else
    #define MOD_ERROR
    #define MOD_INIT(name) void init##name(void)
    #define MOD_RETURN(val)
#endif
#if defined(__cplusplus)
extern "C"
#endif
#if defined(__GNUC__) && __GNUC__ >= 4
__attribute__ ((visibility("default")))
#endif


MOD_INIT(classes)
{
    PyObject *m;
    PyObject *submodule;
    #if PY_VERSION_HEX >= 0x03000000
    m = PyModule_Create(&classes_moduledef);
    #else
    m = Py_InitModule3((char *) "classes", classes_functions, NULL);
    #endif
    if (m == NULL) {
        return MOD_ERROR;
    }
    submodule = initclasses_classes();
    if (submodule == NULL) {
        return MOD_ERROR;
    }
    Py_INCREF(submodule);
    PyModule_AddObject(m, (char *) "classes", submodule);
    return MOD_RETURN(m);
}