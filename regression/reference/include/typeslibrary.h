// typeslibrary.h
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

#ifndef TYPESLIBRARY_H
#define TYPESLIBRARY_H


#ifdef __cplusplus
extern "C" {
#endif

struct s_LIB_Class2 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_LIB_Class2 LIB_Class2;

struct s_LIB_three_Class1 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_LIB_three_Class1 LIB_three_Class1;

struct s_LIB_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_LIB_SHROUD_capsule_data LIB_SHROUD_capsule_data;

void LIB_SHROUD_memory_destructor(LIB_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESLIBRARY_H
