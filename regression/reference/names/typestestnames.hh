// typestestnames.hh
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

#ifndef TYPESTESTNAMES_HH
#define TYPESTESTNAMES_HH


#ifdef __cplusplus
extern "C" {
#endif

struct s_TES_implworker1 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TES_implworker1 TES_implworker1;

struct s_TES_names {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TES_names TES_names;

struct s_TES_names2 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TES_names2 TES_names2;

struct s_TES_vvv1 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TES_vvv1 TES_vvv1;

struct s_TES_vvv2 {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TES_vvv2 TES_vvv2;

struct s_TES_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_TES_SHROUD_capsule_data TES_SHROUD_capsule_data;

void TES_SHROUD_memory_destructor(TES_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESTESTNAMES_HH
