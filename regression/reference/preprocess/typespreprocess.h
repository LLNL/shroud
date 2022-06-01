// typespreprocess.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// For C users and C++ implementation

#ifndef TYPESPREPROCESS_H
#define TYPESPREPROCESS_H

// splicer begin types.CXX_declarations
// splicer end types.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin types.C_declarations
// splicer end types.C_declarations

// helper capsule_PRE_User1
struct s_PRE_User1 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_PRE_User1 PRE_User1;

// helper capsule_PRE_User2
#ifdef USE_USER2
struct s_PRE_User2 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_PRE_User2 PRE_User2;
#endif  // ifdef USE_USER2

// helper capsule_data_helper
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
