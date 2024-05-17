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

#include "shared.h"

typedef int Alias;
Alias typefunc(Alias arg);
Alias typefunc_wrap(Alias arg);

//----------------------------------------------------------------------

enum Color {
  RED = 10,
  BLUE,
  WHITE
};

typedef enum Color iColor;

iColor returnEnum(iColor in);

//-----

enum DataTypeID
{
  NO_TYPE_ID = SHARED_NO_TYPE_ID,
  INT_ID = SHARED_INT_ID,
  LONG_ID = SHARED_LONG_ID,
  FLOAT_ID = SHARED_FLOAT_ID,
  DOUBLE_ID = SHARED_DOUBLE_ID
};

typedef enum DataTypeID TypeID;

TypeID returnTypeID(TypeID in);

//----------------------------------------------------------------------

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
IndexType returnShapeSize(int ndims, const IndexType *shape);

//----------------------------------------------------------------------

#if defined(USE_64BIT_INDEXTYPE)
typedef int64_t IndexType2;
#else
typedef int32_t IndexType2;
#endif

int returnBytesForIndexType2(IndexType arg);
IndexType2 returnShapeSize2(int ndims, const IndexType2 *shape);

#endif // TYPEDEFS_HPP
