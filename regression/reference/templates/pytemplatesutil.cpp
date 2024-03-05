// pytemplatesutil.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// shroud
#include "pytemplatesmodule.hpp"

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

const char *PY_vector_int_capsule_name = "vector_int";
const char *PY_vector_double_capsule_name = "vector_double";
const char *PY_ImplWorker1_capsule_name = "ImplWorker1";
const char *PY_ImplWorker2_capsule_name = "ImplWorker2";
const char *PY_Worker_capsule_name = "Worker";
const char *PY_user_int_capsule_name = "user_int";


// Wrap pointer to struct/class.
PyObject *PP_vector_int_to_Object_idtor(std::vector<int> *addr,
    int idtor)
{
    // splicer begin namespace.std.class.vector.utility.to_object
    PY_vector_int *obj =
        PyObject_New(PY_vector_int, &PY_vector_int_Type);
    if (obj == nullptr)
        return nullptr;
    obj->obj = addr;
    obj->idtor = idtor;
    return reinterpret_cast<PyObject *>(obj);
    // splicer end namespace.std.class.vector.utility.to_object
}

// converter which may be used with PyBuild.
PyObject *PP_vector_int_to_Object(std::vector<int> *addr)
{
    // splicer begin namespace.std.class.vector.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_vector_int_capsule_name, nullptr);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_vector_int_Type, args, nullptr);
    Py_DECREF(args);
    return rv;
    // splicer end namespace.std.class.vector.utility.to_object
}

// converter which may be used with PyArg_Parse.
int PP_vector_int_from_Object(PyObject *obj, void **addr)
{
    // splicer begin namespace.std.class.vector.utility.from_object
    if (obj->ob_type != &PY_vector_int_Type) {
        // raise exception
        return 0;
    }
    PY_vector_int * self = (PY_vector_int *) obj;
    *addr = self->obj;
    return 1;
    // splicer end namespace.std.class.vector.utility.from_object
}

// Wrap pointer to struct/class.
PyObject *PP_vector_double_to_Object_idtor(std::vector<double> *addr,
    int idtor)
{
    // splicer begin namespace.std.class.vector.utility.to_object
    PY_vector_double *obj =
        PyObject_New(PY_vector_double, &PY_vector_double_Type);
    if (obj == nullptr)
        return nullptr;
    obj->obj = addr;
    obj->idtor = idtor;
    return reinterpret_cast<PyObject *>(obj);
    // splicer end namespace.std.class.vector.utility.to_object
}

// converter which may be used with PyBuild.
PyObject *PP_vector_double_to_Object(std::vector<double> *addr)
{
    // splicer begin namespace.std.class.vector.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_vector_double_capsule_name, nullptr);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_vector_double_Type, args, nullptr);
    Py_DECREF(args);
    return rv;
    // splicer end namespace.std.class.vector.utility.to_object
}

// converter which may be used with PyArg_Parse.
int PP_vector_double_from_Object(PyObject *obj, void **addr)
{
    // splicer begin namespace.std.class.vector.utility.from_object
    if (obj->ob_type != &PY_vector_double_Type) {
        // raise exception
        return 0;
    }
    PY_vector_double * self = (PY_vector_double *) obj;
    *addr = self->obj;
    return 1;
    // splicer end namespace.std.class.vector.utility.from_object
}

// Wrap pointer to struct/class.
PyObject *PP_ImplWorker1_to_Object_idtor(internal::ImplWorker1 *addr,
    int idtor)
{
    // splicer begin namespace.internal.class.ImplWorker1.utility.to_object
    PY_ImplWorker1 *obj =
        PyObject_New(PY_ImplWorker1, &PY_ImplWorker1_Type);
    if (obj == nullptr)
        return nullptr;
    obj->obj = addr;
    obj->idtor = idtor;
    return reinterpret_cast<PyObject *>(obj);
    // splicer end namespace.internal.class.ImplWorker1.utility.to_object
}

// converter which may be used with PyBuild.
PyObject *PP_ImplWorker1_to_Object(internal::ImplWorker1 *addr)
{
    // splicer begin namespace.internal.class.ImplWorker1.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_ImplWorker1_capsule_name, nullptr);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_ImplWorker1_Type, args, nullptr);
    Py_DECREF(args);
    return rv;
    // splicer end namespace.internal.class.ImplWorker1.utility.to_object
}

// converter which may be used with PyArg_Parse.
int PP_ImplWorker1_from_Object(PyObject *obj, void **addr)
{
    // splicer begin namespace.internal.class.ImplWorker1.utility.from_object
    if (obj->ob_type != &PY_ImplWorker1_Type) {
        // raise exception
        return 0;
    }
    PY_ImplWorker1 * self = (PY_ImplWorker1 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end namespace.internal.class.ImplWorker1.utility.from_object
}

// Wrap pointer to struct/class.
PyObject *PP_ImplWorker2_to_Object_idtor(internal::ImplWorker2 *addr,
    int idtor)
{
    // splicer begin namespace.internal.class.ImplWorker2.utility.to_object
    PY_ImplWorker2 *obj =
        PyObject_New(PY_ImplWorker2, &PY_ImplWorker2_Type);
    if (obj == nullptr)
        return nullptr;
    obj->obj = addr;
    obj->idtor = idtor;
    return reinterpret_cast<PyObject *>(obj);
    // splicer end namespace.internal.class.ImplWorker2.utility.to_object
}

// converter which may be used with PyBuild.
PyObject *PP_ImplWorker2_to_Object(internal::ImplWorker2 *addr)
{
    // splicer begin namespace.internal.class.ImplWorker2.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_ImplWorker2_capsule_name, nullptr);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_ImplWorker2_Type, args, nullptr);
    Py_DECREF(args);
    return rv;
    // splicer end namespace.internal.class.ImplWorker2.utility.to_object
}

// converter which may be used with PyArg_Parse.
int PP_ImplWorker2_from_Object(PyObject *obj, void **addr)
{
    // splicer begin namespace.internal.class.ImplWorker2.utility.from_object
    if (obj->ob_type != &PY_ImplWorker2_Type) {
        // raise exception
        return 0;
    }
    PY_ImplWorker2 * self = (PY_ImplWorker2 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end namespace.internal.class.ImplWorker2.utility.from_object
}

// Wrap pointer to struct/class.
PyObject *PP_Worker_to_Object_idtor(Worker *addr, int idtor)
{
    // splicer begin class.Worker.utility.to_object
    PY_Worker *obj = PyObject_New(PY_Worker, &PY_Worker_Type);
    if (obj == nullptr)
        return nullptr;
    obj->obj = addr;
    obj->idtor = idtor;
    return reinterpret_cast<PyObject *>(obj);
    // splicer end class.Worker.utility.to_object
}

// converter which may be used with PyBuild.
PyObject *PP_Worker_to_Object(Worker *addr)
{
    // splicer begin class.Worker.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Worker_capsule_name, nullptr);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Worker_Type, args, nullptr);
    Py_DECREF(args);
    return rv;
    // splicer end class.Worker.utility.to_object
}

// converter which may be used with PyArg_Parse.
int PP_Worker_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Worker.utility.from_object
    if (obj->ob_type != &PY_Worker_Type) {
        // raise exception
        return 0;
    }
    PY_Worker * self = (PY_Worker *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Worker.utility.from_object
}

// Wrap pointer to struct/class.
PyObject *PP_user_int_to_Object_idtor(user<int> *addr, int idtor)
{
    // splicer begin class.user.utility.to_object
    PY_user_int *obj = PyObject_New(PY_user_int, &PY_user_int_Type);
    if (obj == nullptr)
        return nullptr;
    obj->obj = addr;
    obj->idtor = idtor;
    return reinterpret_cast<PyObject *>(obj);
    // splicer end class.user.utility.to_object
}

// converter which may be used with PyBuild.
PyObject *PP_user_int_to_Object(user<int> *addr)
{
    // splicer begin class.user.utility.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_user_int_capsule_name, nullptr);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_user_int_Type, args, nullptr);
    Py_DECREF(args);
    return rv;
    // splicer end class.user.utility.to_object
}

// converter which may be used with PyArg_Parse.
int PP_user_int_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.user.utility.from_object
    if (obj->ob_type != &PY_user_int_Type) {
        // raise exception
        return 0;
    }
    PY_user_int * self = (PY_user_int *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.user.utility.from_object
}

// ----------------------------------------
typedef struct {
    const char *name;
    void (*dtor)(void *ptr);
} PY_SHROUD_dtor_context;

// 0 - --none--
static void PY_SHROUD_capsule_destructor_0(void *ptr)
{
    // Do not release
}

// 1 - cxx std::vector<int> *
static void PY_SHROUD_capsule_destructor_1(void *ptr)
{
    std::vector<int> * cxx_ptr = static_cast<std::vector<int> *>(ptr);
    delete cxx_ptr;
}

// 2 - cxx std::vector<double> *
static void PY_SHROUD_capsule_destructor_2(void *ptr)
{
    std::vector<double> * cxx_ptr =
        static_cast<std::vector<double> *>(ptr);
    delete cxx_ptr;
}

// Code used to release arrays for NumPy objects
// via a Capsule base object with a destructor.
// Context strings
static PY_SHROUD_dtor_context PY_SHROUD_capsule_context[] = {
    {"--none--", PY_SHROUD_capsule_destructor_0},
    {"cxx std::vector<int> *", PY_SHROUD_capsule_destructor_1},
    {"cxx std::vector<double> *", PY_SHROUD_capsule_destructor_2},
    {nullptr, nullptr},
};

// Release memory based on icontext.
void PY_SHROUD_release_memory(int icontext, void *ptr)
{
    PY_SHROUD_capsule_context[icontext].dtor(ptr);
}

//Fetch garbage collection context.
void *PY_SHROUD_fetch_context(int icontext)
{
    return PY_SHROUD_capsule_context + icontext;
}

// destructor function for PyCapsule
void PY_SHROUD_capsule_destructor(PyObject *cap)
{
    void *ptr = PyCapsule_GetPointer(cap, "PY_array_dtor");
    PY_SHROUD_dtor_context * context = static_cast<PY_SHROUD_dtor_context *>
        (PyCapsule_GetContext(cap));
    context->dtor(ptr);
}
