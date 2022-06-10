// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// tutorial.hpp - wrapped routines
//

#ifndef OVERLOAD_HPP
#define OVERLOAD_HPP

// C/C++ includes
#include <cstdint>  // for c++11 fixed with types

enum DataTypeId
{
    NO_TYPE_ID,
    INT8_ID,
    INT16_ID,
    INT32_ID,
    INT64_ID,
};
using TypeID = DataTypeId;

#if defined(USE_64BIT_INDEXTYPE)
using IndexType = std::int64_t;
#else
using IndexType = std::int32_t;
#endif

void apply(IndexType num_elems, IndexType offset = 0, IndexType stride = 1);
void apply(TypeID type, IndexType num_elems, IndexType offset = 0, IndexType stride = 1);


#endif // OVERLOAD_HPP

