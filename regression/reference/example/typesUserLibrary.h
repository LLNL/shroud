// typesUserLibrary.h
// This is generated code, do not edit
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

#ifndef TYPESUSERLIBRARY_H
#define TYPESUSERLIBRARY_H

#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

struct s_AA_exclass1 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_AA_exclass1 AA_exclass1;

struct s_AA_exclass2 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_AA_exclass2 AA_exclass2;

struct s_AA_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_AA_SHROUD_capsule_data AA_SHROUD_capsule_data;

struct s_AA_SHROUD_array {
    AA_SHROUD_capsule_data cxx;      /* address of C++ memory */
    union {
        const void * cvoidp;
        const char * ccharp;
    } addr;
    size_t len;     /* bytes-per-item or character len of data in cxx */
    size_t size;    /* size of data in cxx */
};
typedef struct s_AA_SHROUD_array AA_SHROUD_array;

void AA_SHROUD_memory_destructor(AA_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESUSERLIBRARY_H
