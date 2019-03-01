// pytemplateshelper.cpp
// This is generated code, do not edit
#include "pytemplatesmodule.hpp"
const char *PY_vector_int_capsule_name = "vector_int";
const char *PY_vector_double_capsule_name = "vector_double";
const char *PY_Worker_capsule_name = "Worker";
const char *PY_ImplWorker1_capsule_name = "ImplWorker1";
const char *PY_user_int_capsule_name = "user_int";


PyObject *PP_vector_int_to_Object(std::vector_int *addr)
{
    // splicer begin class.vector.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_vector_int_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_vector_int_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.vector.helper.to_object
}

int PP_vector_int_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.vector.helper.from_object
    if (obj->ob_type != &PY_vector_int_Type) {
        // raise exception
        return 0;
    }
    PY_vector_int * self = (PY_vector_int *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.vector.helper.from_object
}

PyObject *PP_vector_double_to_Object(std::vector_double *addr)
{
    // splicer begin class.vector.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_vector_double_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_vector_double_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.vector.helper.to_object
}

int PP_vector_double_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.vector.helper.from_object
    if (obj->ob_type != &PY_vector_double_Type) {
        // raise exception
        return 0;
    }
    PY_vector_double * self = (PY_vector_double *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.vector.helper.from_object
}

PyObject *PP_Worker_to_Object(Worker *addr)
{
    // splicer begin class.Worker.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_Worker_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_Worker_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.Worker.helper.to_object
}

int PP_Worker_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.Worker.helper.from_object
    if (obj->ob_type != &PY_Worker_Type) {
        // raise exception
        return 0;
    }
    PY_Worker * self = (PY_Worker *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.Worker.helper.from_object
}

PyObject *PP_ImplWorker1_to_Object(internal::ImplWorker1 *addr)
{
    // splicer begin class.ImplWorker1.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_ImplWorker1_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_ImplWorker1_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.ImplWorker1.helper.to_object
}

int PP_ImplWorker1_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.ImplWorker1.helper.from_object
    if (obj->ob_type != &PY_ImplWorker1_Type) {
        // raise exception
        return 0;
    }
    PY_ImplWorker1 * self = (PY_ImplWorker1 *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.ImplWorker1.helper.from_object
}

PyObject *PP_user_int_to_Object(user_int *addr)
{
    // splicer begin class.user.helper.to_object
    PyObject *voidobj;
    PyObject *args;
    PyObject *rv;

    voidobj = PyCapsule_New(addr, PY_user_int_capsule_name, NULL);
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, voidobj);
    rv = PyObject_Call((PyObject *) &PY_user_int_Type, args, NULL);
    Py_DECREF(args);
    return rv;
    // splicer end class.user.helper.to_object
}

int PP_user_int_from_Object(PyObject *obj, void **addr)
{
    // splicer begin class.user.helper.from_object
    if (obj->ob_type != &PY_user_int_Type) {
        // raise exception
        return 0;
    }
    PY_user_int * self = (PY_user_int *) obj;
    *addr = self->obj;
    return 1;
    // splicer end class.user.helper.from_object
}
