// pynsmodule.hpp
// This is generated code, do not edit
#ifndef PYNSMODULE_HPP
#define PYNSMODULE_HPP
#include <Python.h>
// splicer begin header.include
// splicer end header.include

// utility functions
extern void PY_SHROUD_release_memory(int icontext, void *ptr);
extern void *PY_SHROUD_fetch_context(int icontext);
extern void PY_SHROUD_capsule_destructor(PyObject *cap);

// ------------------------------
namespace outer {
    class Cstruct1;  // forward declare
}
extern PyTypeObject PY_Cstruct1_Type;
// splicer begin namespace.outer.class.Cstruct1.C_declaration
// splicer end namespace.outer.class.Cstruct1.C_declaration

typedef struct {
PyObject_HEAD
    outer::Cstruct1 * obj;
    int idtor;
    // splicer begin namespace.outer.class.Cstruct1.C_object
    // splicer end namespace.outer.class.Cstruct1.C_object
} PY_Cstruct1;

extern const char *PY_Cstruct1_capsule_name;
PyObject *PP_Cstruct1_to_Object(outer::Cstruct1 *addr);
int PP_Cstruct1_from_Object(PyObject *obj, void **addr);
// ------------------------------

// splicer begin header.C_declaration
// splicer end header.C_declaration

extern PyObject *PY_error_obj;

#if PY_MAJOR_VERSION >= 3
extern "C" PyMODINIT_FUNC PyInit_ns(void);
#else
extern "C" PyMODINIT_FUNC initns(void);
#endif

#endif  /* PYNSMODULE_HPP */
