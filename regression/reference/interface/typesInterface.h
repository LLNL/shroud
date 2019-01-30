// typesInterface.h
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
// For C users and C implementation

#ifndef TYPESINTERFACE_H
#define TYPESINTERFACE_H

struct s_INT_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_INT_SHROUD_capsule_data INT_SHROUD_capsule_data;

void INT_SHROUD_memory_destructor(INT_SHROUD_capsule_data *cap);

#endif  // TYPESINTERFACE_H
