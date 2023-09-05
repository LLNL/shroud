// wrapthree_Class1.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "class_header.hpp"
// shroud
#include "wrapthree_Class1.h"


extern "C" {


// ----------------------------------------
// Function:  void method1
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  CustomType arg1 +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
void LIB_three_Class1_method1(LIB_three_Class1 * self,
    LIB_CustomType arg1)
{
    three::Class1 *SH_this = static_cast<three::Class1 *>(self->addr);
    SH_this->method1(arg1);
}

}  // extern "C"
