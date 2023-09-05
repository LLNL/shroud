// typeslibrary.h
// This file is generated by Shroud 0.13.0. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// For C users and C++ implementation

#ifndef TYPESLIBRARY_H
#define TYPESLIBRARY_H

// splicer begin types.CXX_declarations
// splicer end types.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin types.C_declarations
// splicer end types.C_declarations

// helper capsule_data_helper
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
