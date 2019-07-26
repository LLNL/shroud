// typesstrings.h
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
//
// All rights reserved.
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
// For C users and C++ implementation

#ifndef TYPESSTRINGS_H
#define TYPESSTRINGS_H

#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

struct s_STR_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_STR_SHROUD_capsule_data STR_SHROUD_capsule_data;

// start array_context
struct s_STR_SHROUD_array {
    STR_SHROUD_capsule_data cxx;      /* address of C++ memory */
    union {
        const void * cvoidp;
        const char * ccharp;
    } addr;
    size_t len;     /* bytes-per-item or character len of data in cxx */
    size_t size;    /* size of data in cxx */
};
typedef struct s_STR_SHROUD_array STR_SHROUD_array;
// end array_context

void STR_SHROUD_memory_destructor(STR_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESSTRINGS_H
