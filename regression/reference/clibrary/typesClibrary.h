// typesClibrary.h
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

#ifndef TYPESCLIBRARY_H
#define TYPESCLIBRARY_H

struct s_CLI_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_CLI_SHROUD_capsule_data CLI_SHROUD_capsule_data;

void CLI_SHROUD_memory_destructor(CLI_SHROUD_capsule_data *cap);

#endif  // TYPESCLIBRARY_H
