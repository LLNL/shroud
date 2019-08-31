// typesns.h
// This is generated code, do not edit
// For C users and C++ implementation

#ifndef TYPESNS_H
#define TYPESNS_H

#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

struct s_NS_classwork {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_NS_classwork NS_classwork;

struct s_NS_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_NS_SHROUD_capsule_data NS_SHROUD_capsule_data;

struct s_NS_SHROUD_array {
    NS_SHROUD_capsule_data cxx;      /* address of C++ memory */
    union {
        const void * cvoidp;
        const char * ccharp;
    } addr;
    size_t len;     /* bytes-per-item or character len of data in cxx */
    size_t size;    /* size of data in cxx */
};
typedef struct s_NS_SHROUD_array NS_SHROUD_array;

void NS_SHROUD_memory_destructor(NS_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESNS_H
