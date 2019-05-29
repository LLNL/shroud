// typespreprocess.h
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

#ifndef TYPESPREPROCESS_H
#define TYPESPREPROCESS_H


#ifdef __cplusplus
extern "C" {
#endif

struct s_PRE_user1 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_PRE_user1 PRE_user1;

struct s_PRE_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_PRE_SHROUD_capsule_data PRE_SHROUD_capsule_data;

void PRE_SHROUD_memory_destructor(PRE_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESPREPROCESS_H
