// typesInterface.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// For C users and C implementation

#ifndef TYPESINTERFACE_H
#define TYPESINTERFACE_H

// splicer begin types.CXX_declarations
// splicer end types.CXX_declarations

// helper capsule_data_helper
struct s_INT_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_INT_SHROUD_capsule_data INT_SHROUD_capsule_data;

void INT_SHROUD_memory_destructor(INT_SHROUD_capsule_data *cap);

#endif  // TYPESINTERFACE_H
