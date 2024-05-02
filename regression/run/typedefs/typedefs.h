// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// typdefs.h - wrapped routines
//

#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

typedef int TypeID;
TypeID typefunc(TypeID arg);
TypeID typefunc_wrap(TypeID arg);

struct s_Struct1 {
    int i;
    double d;
};
typedef struct s_Struct1 Struct1Rename;

void typestruct(Struct1Rename *arg1);

//----------------------------------------------------------------------

#include <stdint.h>

#if defined(USE_64BIT_INDEXTYPE)
typedef int64_t IndexType;
#else
typedef int32_t IndexType;
#endif

int returnBytesForIndexType(IndexType arg);

//----------------------------------------------------------------------

#if defined(USE_64BIT_INDEXTYPE)
typedef int64_t IndexType2;
#else
typedef int32_t IndexType2;
#endif

int returnBytesForIndexType2(IndexType arg);

#endif // TYPEDEFS_HPP
