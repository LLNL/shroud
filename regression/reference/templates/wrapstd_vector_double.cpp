// wrapstd_vector_double.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapstd_vector_double.h"

// cxx_header
#include <vector>
// shroud
#include <cstddef>

// splicer begin namespace.std.class.vector.CXX_definitions
// splicer end namespace.std.class.vector.CXX_definitions

extern "C" {

// splicer begin namespace.std.class.vector.C_definitions
// splicer end namespace.std.class.vector.C_definitions

// ----------------------------------------
// Function:  vector
// Attrs:     +intent(ctor)
// Requested: c_ctor_shadow_scalar
// Match:     c_ctor
TEM_vector_double * TEM_vector_double_ctor(
    TEM_vector_double * SHadow_rv)
{
    // splicer begin namespace.std.class.vector.method.ctor
    std::vector<double> *SHCXX_rv = new std::vector<double>();
    SHadow_rv->addr = static_cast<void *>(SHCXX_rv);
    SHadow_rv->idtor = 2;
    return SHadow_rv;
    // splicer end namespace.std.class.vector.method.ctor
}

// ----------------------------------------
// Function:  ~vector
// Attrs:     +intent(dtor)
// Exact:     c_dtor
void TEM_vector_double_dtor(TEM_vector_double * self)
{
    std::vector<double> *SH_this = static_cast<std::vector<double> *>
        (self->addr);
    // splicer begin namespace.std.class.vector.method.dtor
    delete SH_this;
    self->addr = nullptr;
    // splicer end namespace.std.class.vector.method.dtor
}

// ----------------------------------------
// Function:  void push_back
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const double & value +intent(in)
// Attrs:     +intent(in)
// Requested: c_in_native_&
// Match:     c_default
void TEM_vector_double_push_back(TEM_vector_double * self,
    const double * value)
{
    std::vector<double> *SH_this = static_cast<std::vector<double> *>
        (self->addr);
    // splicer begin namespace.std.class.vector.method.push_back
    SH_this->push_back(*value);
    // splicer end namespace.std.class.vector.method.push_back
}

// ----------------------------------------
// Function:  double & at
// Attrs:     +deref(pointer)+intent(function)
// Requested: c_function_native_&_pointer
// Match:     c_function_native_&
// ----------------------------------------
// Argument:  size_type n +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
double * TEM_vector_double_at(TEM_vector_double * self, size_t n)
{
    std::vector<double> *SH_this = static_cast<std::vector<double> *>
        (self->addr);
    // splicer begin namespace.std.class.vector.method.at
    double & SHC_rv = SH_this->at(n);
    return &SHC_rv;
    // splicer end namespace.std.class.vector.method.at
}

// ----------------------------------------
// Function:  double & at
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_native_&_buf_pointer
// Match:     c_function_native_&
// ----------------------------------------
// Argument:  size_type n +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
double * TEM_vector_double_at_bufferify(TEM_vector_double * self,
    size_t n)
{
    std::vector<double> *SH_this = static_cast<std::vector<double> *>
        (self->addr);
    // splicer begin namespace.std.class.vector.method.at_bufferify
    double & SHC_rv = SH_this->at(n);
    return &SHC_rv;
    // splicer end namespace.std.class.vector.method.at_bufferify
}

}  // extern "C"
