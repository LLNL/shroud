// wrapclassA.cpp
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
#include "wrapclassA.h"
#include "classA.hpp"
#include "classAb.hpp"


extern "C" {


void LIB_outer2_classA_method(LIB_outer2_classA * self)
{
    outer2::classA *SH_this = static_cast<outer2::classA *>(self->addr);
    SH_this->method();
    return;
}

}  // extern "C"
