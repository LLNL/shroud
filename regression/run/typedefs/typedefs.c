// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// typdefs.h - wrapped routines
//

#include "typedefs.h"

#ifdef __cplusplus
s_Struct1 tmp1;
#endif

Alias typefunc(Alias arg)
{
    return arg + 1;
}

Alias typefunc_wrap(Alias arg)
{
    return arg + 1;
}

//----------------------------------------------------------------------

iColor returnEnum(iColor in)
{
    return in;
}

TypeID returnTypeID(TypeID in)
{
    return in;
}

//----------------------------------------------------------------------

void typestruct(Struct1Rename *arg1)
{
    arg1->d = arg1->i;
}

//----------------------------------------------------------------------

int returnBytesForIndexType(IndexType arg)
{
    return sizeof(arg);
}

IndexType returnShapeSize(int ndims, const IndexType *shape)
{
    IndexType size = 1;
    
    for (int i=0; i<ndims; ++i) {
        size = size * shape[i];
    }
    return size;
}

int returnBytesForIndexType2(IndexType2 arg)
{
    return sizeof(arg);
}

IndexType returnShapeSize2(int ndims, const IndexType2 *shape)
{
    IndexType2 size = 1;
    
    for (int i=0; i<ndims; ++i) {
        size = size * shape[i];
    }
    return size;
}

