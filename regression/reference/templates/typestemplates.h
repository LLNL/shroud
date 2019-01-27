// typestemplates.h
// This is generated code, do not edit
// For C users and C++ implementation

#ifndef TYPESTEMPLATES_H
#define TYPESTEMPLATES_H


#ifdef __cplusplus
extern "C" {
#endif

struct s_TEM_implworker1 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TEM_implworker1 TEM_implworker1;

struct s_TEM_user_int {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TEM_user_int TEM_user_int;

struct s_TEM_vector_double {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TEM_vector_double TEM_vector_double;

struct s_TEM_vector_int {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TEM_vector_int TEM_vector_int;

struct s_TEM_worker {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TEM_worker TEM_worker;

struct s_TEM_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TEM_SHROUD_capsule_data TEM_SHROUD_capsule_data;

void TEM_SHROUD_memory_destructor(TEM_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESTEMPLATES_H
