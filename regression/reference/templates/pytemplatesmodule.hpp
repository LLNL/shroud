// pytemplatesmodule.hpp
// This is generated code, do not edit
#ifndef PYTEMPLATESMODULE_HPP
#define PYTEMPLATESMODULE_HPP
#include <Python.h>
// splicer begin header.include
// splicer end header.include

// utility functions
extern void PY_SHROUD_release_memory(int icontext, void *ptr);
extern void *PY_SHROUD_fetch_context(int icontext);
extern void PY_SHROUD_capsule_destructor(PyObject *cap);

// ------------------------------
namespace std {
    class vector;  // forward declare
}
extern PyTypeObject PY_vector_int_Type;
// splicer begin namespace.std.class.vector.C_declaration
// splicer end namespace.std.class.vector.C_declaration

typedef struct {
PyObject_HEAD
    std::vector_int * obj;
    int idtor;
    // splicer begin namespace.std.class.vector.C_object
    // splicer end namespace.std.class.vector.C_object
} PY_vector_int;

extern const char *PY_vector_int_capsule_name;
PyObject *PP_vector_int_to_Object(std::vector_int *addr);
int PP_vector_int_from_Object(PyObject *obj, void **addr);

// ------------------------------
namespace std {
    class vector;  // forward declare
}
extern PyTypeObject PY_vector_double_Type;
// splicer begin namespace.std.class.vector.C_declaration
// splicer end namespace.std.class.vector.C_declaration

typedef struct {
PyObject_HEAD
    std::vector_double * obj;
    int idtor;
    // splicer begin namespace.std.class.vector.C_object
    // splicer end namespace.std.class.vector.C_object
} PY_vector_double;

extern const char *PY_vector_double_capsule_name;
PyObject *PP_vector_double_to_Object(std::vector_double *addr);
int PP_vector_double_from_Object(PyObject *obj, void **addr);

// ------------------------------
namespace internal {
    class ImplWorker1;  // forward declare
}
extern PyTypeObject PY_ImplWorker1_Type;
// splicer begin namespace.internal.class.ImplWorker1.C_declaration
// splicer end namespace.internal.class.ImplWorker1.C_declaration

typedef struct {
PyObject_HEAD
    internal::ImplWorker1 * obj;
    int idtor;
    // splicer begin namespace.internal.class.ImplWorker1.C_object
    // splicer end namespace.internal.class.ImplWorker1.C_object
} PY_ImplWorker1;

extern const char *PY_ImplWorker1_capsule_name;
PyObject *PP_ImplWorker1_to_Object(internal::ImplWorker1 *addr);
int PP_ImplWorker1_from_Object(PyObject *obj, void **addr);

// ------------------------------
class Worker;  // forward declare
extern PyTypeObject PY_Worker_Type;
// splicer begin class.Worker.C_declaration
// splicer end class.Worker.C_declaration

typedef struct {
PyObject_HEAD
    Worker * obj;
    int idtor;
    // splicer begin class.Worker.C_object
    // splicer end class.Worker.C_object
} PY_Worker;

extern const char *PY_Worker_capsule_name;
PyObject *PP_Worker_to_Object(Worker *addr);
int PP_Worker_from_Object(PyObject *obj, void **addr);

// ------------------------------
class user;  // forward declare
extern PyTypeObject PY_user_int_Type;
// splicer begin class.user.C_declaration
// splicer end class.user.C_declaration

typedef struct {
PyObject_HEAD
    user_int * obj;
    int idtor;
    // splicer begin class.user.C_object
    // splicer end class.user.C_object
} PY_user_int;

extern const char *PY_user_int_capsule_name;
PyObject *PP_user_int_to_Object(user_int *addr);
int PP_user_int_from_Object(PyObject *obj, void **addr);
// ------------------------------

// splicer begin header.C_declaration
// splicer end header.C_declaration

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
extern "C" PyMODINIT_FUNC PyInit_templates(void);
#else
extern "C" PyMODINIT_FUNC inittemplates(void);
#endif

#endif  /* PYTEMPLATESMODULE_HPP */
