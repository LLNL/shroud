// wrapouter2_class0.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "classA.hpp"
#include "classAb.hpp"
// shroud
#include "wrapouter2_class0.h"


extern "C" {


// ----------------------------------------
// Function:  void method
// Statement: c_subroutine
void LIB_outer2_class0_method(LIB_outer2_class0 * self)
{
    outer2::class0 *SH_this = static_cast<outer2::class0 *>(self->addr);
    SH_this->method();
}

}  // extern "C"
