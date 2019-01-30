// typesdefault_library.h
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

#ifndef TYPESDEFAULT_LIBRARY_H
#define TYPESDEFAULT_LIBRARY_H


#ifdef __cplusplus
extern "C" {
#endif

struct s_DEF_class1 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_DEF_class1 DEF_class1;

struct s_DEF_class2 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_DEF_class2 DEF_class2;

struct s_DEF_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_DEF_SHROUD_capsule_data DEF_SHROUD_capsule_data;

void DEF_SHROUD_memory_destructor(DEF_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESDEFAULT_LIBRARY_H
