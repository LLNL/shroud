// wrapClass1.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#include "wrap.hpp"
#include "wrapClass1.h"

// splicer begin class.Class1.CXX_definitions
// splicer end class.Class1.CXX_definitions

extern "C" {

// splicer begin class.Class1.C_definitions
// splicer end class.Class1.C_definitions

int WRA_Class1_FuncInClass_bufferify(WRA_Class1 * self)
{
    Class1 *SH_this = static_cast<Class1 *>(self->addr);
    // splicer begin class.Class1.method.FuncInClass_bufferify
    int SHC_rv = SH_this->FuncInClass();
    return SHC_rv;
    // splicer end class.Class1.method.FuncInClass_bufferify
}

}  // extern "C"
