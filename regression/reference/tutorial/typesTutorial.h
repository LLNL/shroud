// typesTutorial.h
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// For C users and C++ implementation

#ifndef TYPESTUTORIAL_H
#define TYPESTUTORIAL_H

#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

// start struct TUT_class1
struct s_TUT_class1 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TUT_class1 TUT_class1;
// end struct TUT_class1

struct s_TUT_singleton {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TUT_singleton TUT_singleton;

struct s_TUT_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TUT_SHROUD_capsule_data TUT_SHROUD_capsule_data;

struct s_TUT_SHROUD_array {
    TUT_SHROUD_capsule_data cxx;      /* address of C++ memory */
    union {
        const void * cvoidp;
        const char * ccharp;
    } addr;
    size_t len;     /* bytes-per-item or character len of data in cxx */
    size_t size;    /* size of data in cxx */
};
typedef struct s_TUT_SHROUD_array TUT_SHROUD_array;

void TUT_SHROUD_memory_destructor(TUT_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESTUTORIAL_H
