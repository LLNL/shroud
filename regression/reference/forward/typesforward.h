// typesforward.h
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

#ifndef TYPESFORWARD_H
#define TYPESFORWARD_H


#ifdef __cplusplus
extern "C" {
#endif

struct s_FOR_Class2 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_FOR_Class2 FOR_Class2;

struct s_FOR_Class3 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_FOR_Class3 FOR_Class3;

struct s_FOR_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_FOR_SHROUD_capsule_data FOR_SHROUD_capsule_data;

void FOR_SHROUD_memory_destructor(FOR_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESFORWARD_H
