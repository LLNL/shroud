// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// typdefs.h - wrapped routines
//

#include "typedefs.h"

#ifdef __cplusplus
s_Struct1 tmp1;
#endif

TypeID typefunc(TypeID arg)
{
    return arg + 1;
}

TypeID typefunc_wrap(TypeID arg)
{
    return arg + 1;
}

void typestruct(Struct1Rename *arg1)
{
    arg1->d = arg1->i;
}

//----------------------------------------------------------------------

int returnBytesForIndexType(IndexType arg)
{
    return sizeof(arg);
}
