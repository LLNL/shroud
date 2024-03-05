// typesstruct.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// For C users and C++ implementation

#ifndef TYPESSTRUCT_H
#define TYPESSTRUCT_H

// shroud
#include <stddef.h>

// splicer begin types.CXX_declarations
// splicer end types.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin types.C_declarations
// splicer end types.C_declarations

/* helper type_defines */
/* Shroud type defines */
#define SH_TYPE_SIGNED_CHAR 1
#define SH_TYPE_SHORT       2
#define SH_TYPE_INT         3
#define SH_TYPE_LONG        4
#define SH_TYPE_LONG_LONG   5
#define SH_TYPE_SIZE_T      6

#define SH_TYPE_UNSIGNED_SHORT       SH_TYPE_SHORT + 100
#define SH_TYPE_UNSIGNED_INT         SH_TYPE_INT + 100
#define SH_TYPE_UNSIGNED_LONG        SH_TYPE_LONG + 100
#define SH_TYPE_UNSIGNED_LONG_LONG   SH_TYPE_LONG_LONG + 100

#define SH_TYPE_INT8_T      7
#define SH_TYPE_INT16_T     8
#define SH_TYPE_INT32_T     9
#define SH_TYPE_INT64_T    10

#define SH_TYPE_UINT8_T    SH_TYPE_INT8_T + 100
#define SH_TYPE_UINT16_T   SH_TYPE_INT16_T + 100
#define SH_TYPE_UINT32_T   SH_TYPE_INT32_T + 100
#define SH_TYPE_UINT64_T   SH_TYPE_INT64_T + 100

/* least8 least16 least32 least64 */
/* fast8 fast16 fast32 fast64 */
/* intmax_t intptr_t ptrdiff_t */

#define SH_TYPE_FLOAT        22
#define SH_TYPE_DOUBLE       23
#define SH_TYPE_LONG_DOUBLE  24
#define SH_TYPE_FLOAT_COMPLEX       25
#define SH_TYPE_DOUBLE_COMPLEX      26
#define SH_TYPE_LONG_DOUBLE_COMPLEX 27

#define SH_TYPE_BOOL       28
#define SH_TYPE_CHAR       29
#define SH_TYPE_CPTR       30
#define SH_TYPE_STRUCT     31
#define SH_TYPE_OTHER      32

// helper array_context
struct s_STR_SHROUD_array {
    void * base_addr;
    int type;        /* type of element */
    size_t elem_len; /* bytes-per-item or character len in c++ */
    size_t size;     /* size of data in c++ */
    int rank;        /* number of dimensions, 0=scalar */
    long shape[7];
};
typedef struct s_STR_SHROUD_array STR_SHROUD_array;

// helper capsule_data_helper
struct s_STR_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_STR_SHROUD_capsule_data STR_SHROUD_capsule_data;
#if 0

// start C++ capsule STR_Cstruct_as_class
// C++ capsule STR_Cstruct_as_class
struct s_STR_Cstruct_as_class {
    Cstruct_as_class *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_STR_Cstruct_as_class STR_Cstruct_as_class;
// end C++ capsule STR_Cstruct_as_class

// start C++ capsule STR_Cstruct_as_subclass
// C++ capsule STR_Cstruct_as_subclass
struct s_STR_Cstruct_as_subclass {
    Cstruct_as_subclass *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_STR_Cstruct_as_subclass STR_Cstruct_as_subclass;
// end C++ capsule STR_Cstruct_as_subclass
#endif

// start C capsule STR_Cstruct_as_class
// C capsule STR_Cstruct_as_class
struct s_STR_Cstruct_as_class {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_STR_Cstruct_as_class STR_Cstruct_as_class;
// end C capsule STR_Cstruct_as_class

// start C capsule STR_Cstruct_as_subclass
// C capsule STR_Cstruct_as_subclass
struct s_STR_Cstruct_as_subclass {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_STR_Cstruct_as_subclass STR_Cstruct_as_subclass;
// end C capsule STR_Cstruct_as_subclass

void STR_SHROUD_memory_destructor(STR_SHROUD_capsule_data *cap);

#ifdef __cplusplus
}
#endif

#endif  // TYPESSTRUCT_H
