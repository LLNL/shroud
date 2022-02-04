// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// typemap.cpp - wrapped routines
//
#include <math.h>
#include "typemap.hpp"

// Make sure a 64-bit int is passed with USE_64BIT_INDEXTYPE
// Argument values of i1 must match Fortran.
bool passIndex(IndexType i1, IndexType *i2)
{
    *i2 = i1;
#if defined(USE_64BIT_INDEXTYPE)
    return i1 == pow(2,34);
#else
    return i1 == 2;
#endif
}

void passFloat(FloatType f1)
{
}
