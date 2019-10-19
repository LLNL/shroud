// wrapouter1_class0.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
//
// All rights reserved.
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
#include "wrapouter1_class0.h"
#include "outer1.hpp"


extern "C" {


void LIB_outer1_class0_method(LIB_outer1_class0 * self)
{
    outer1::class0 *SH_this = static_cast<outer1::class0 *>(self->addr);
    SH_this->method();
    return;
}

}  // extern "C"
