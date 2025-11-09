// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// defaultargl.hpp - wrapped routines
//

#ifndef DEFAULTARG_HPP
#define DEFAULTARG_HPP

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

#if INDEXTYPE_SIZE == 64
using IndexType = std::int64_t;
#elif INDEXTYPE_SIZE == 32
using IndexType = std::int32_t;
#else
#error INDEXTYPE_SIZE must be 32 or 64
#endif

void apply_generic(IndexType num_elems, IndexType offset = 0, IndexType stride = 1);
void apply_generic(TypeID type, IndexType num_elems, IndexType offset = 0, IndexType stride = 1);

void apply_require(IndexType num_elems, IndexType offset = 0, IndexType stride = 1);
void apply_require(TypeID type, IndexType num_elems, IndexType offset = 0, IndexType stride = 1);

void apply_optional(IndexType num_elems, IndexType offset = 0, IndexType stride = 1);
void apply_optional(TypeID type, IndexType num_elems, IndexType offset = 0, IndexType stride = 1);

//----------------------------------------------------------------------
class Class1
{
public:
    int m_field1;
    int m_field2;
    int m_field3;
    Class1(int arg1, int arg2 = 1, int arg3 = 2) :
        m_field1(arg1), m_field2(arg2), m_field3(arg3)
    {};
    //    ~Class1();
    void DefaultArguments(int arg1, int arg2 = 1, int arg3 = 2);
};

#endif // DEFAULTARG_HPP

