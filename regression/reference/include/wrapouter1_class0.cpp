// wrapouter1_class0.cpp
// This file is generated by Shroud 0.11.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapouter1_class0.h"
#include "outer1.hpp"


extern "C" {


void LIB_outer1_class0_method(LIB_outer1_class0 * self)
{
    outer1::class0 *SH_this = static_cast<outer1::class0 *>(self->addr);
    SH_this->method();
}

}  // extern "C"
