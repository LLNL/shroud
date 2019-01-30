// typespointers.h
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

#ifndef TYPESPOINTERS_H
#define TYPESPOINTERS_H


#ifdef __cplusplus
extern "C" {
#endif

struct s_POI_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_POI_SHROUD_capsule_data POI_SHROUD_capsule_data;

void POI_SHROUD_memory_destructor(POI_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESPOINTERS_H
