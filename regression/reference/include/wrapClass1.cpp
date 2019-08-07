// wrapClass1.cpp
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
#include "wrapClass1.h"
#include "class_header.hpp"
#include "type_header.hpp"


extern "C" {


void DEF_class1_method1(DEF_class1 * self, int arg1)
{
    three::Class1 *SH_this = static_cast<three::Class1 *>(self->addr);
    SH_this->method1(arg1);
    return;
}

}  // extern "C"
