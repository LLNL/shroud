// typesscope.h
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

#ifndef TYPESSCOPE_H
#define TYPESSCOPE_H


#ifdef __cplusplus
extern "C" {
#endif

struct s_SCO_Class1 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_SCO_Class1 SCO_Class1;

struct s_SCO_Class2 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_SCO_Class2 SCO_Class2;

struct s_SCO_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_SCO_SHROUD_capsule_data SCO_SHROUD_capsule_data;

void SCO_SHROUD_memory_destructor(SCO_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESSCOPE_H
