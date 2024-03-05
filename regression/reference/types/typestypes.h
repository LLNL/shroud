// typestypes.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// For C users and C++ implementation

#ifndef TYPESTYPES_H
#define TYPESTYPES_H

// splicer begin types.CXX_declarations
// splicer end types.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin types.C_declarations
// splicer end types.C_declarations

// helper capsule_data_helper
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
