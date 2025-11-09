// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// defaultarg.cpp - wrapped routines
//

#include "defaultarg.hpp"


void apply_generic(IndexType num_elems, IndexType offset, IndexType stride)
{
}

void apply_generic(TypeID type, IndexType num_elems, IndexType offset, IndexType stride)
{
}

//----------------------------------------------------------------------

void apply_require(IndexType num_elems, IndexType offset, IndexType stride)
{
}

void apply_require(TypeID type, IndexType num_elems, IndexType offset, IndexType stride)
{
}

//----------------------------------------------------------------------

void apply_optional(IndexType num_elems, IndexType offset, IndexType stride)
{
}

void apply_optional(TypeID type, IndexType num_elems, IndexType offset, IndexType stride)
{
}

//----------------------------------------------------------------------

void Class1::DefaultArguments(int arg1, int arg2, int arg3)
{
    this->m_field1 = arg1;
    this->m_field2 = arg2;
    this->m_field3 = arg3;
}
