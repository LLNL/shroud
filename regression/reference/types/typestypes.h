// typestypes.h
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

#ifndef TYPESTYPES_H
#define TYPESTYPES_H


#ifdef __cplusplus
extern "C" {
#endif

struct s_TYP_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TYP_SHROUD_capsule_data TYP_SHROUD_capsule_data;

void TYP_SHROUD_memory_destructor(TYP_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESTYPES_H
