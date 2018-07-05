// typestemplates.h
// This is generated code, do not edit
// For C users and C++ implementation

#ifndef TYPESTEMPLATES_H
#define TYPESTEMPLATES_H


#ifdef __cplusplus
extern "C" {
#endif

struct s_TEM_user_0 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TEM_user_0 TEM_user_0;

struct s_TEM_vector_0 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TEM_vector_0 TEM_vector_0;

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
