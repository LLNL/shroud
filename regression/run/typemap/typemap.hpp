// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// typemap.hpp - wrapped routines
//
#include <cstdint>

#if defined(USE_64BIT_INDEXTYPE)
using IndexType = std::int64_t;
#else
using IndexType = std::int32_t;
#endif

bool passIndex(IndexType i1, IndexType *i2);
void passIndex2(IndexType i1);

#if defined(USE_64BIT_FLOAT)
using FloatType = double;
#else
using FloatType = float;
#endif

void passFloat(FloatType f1);
