// wrapSingleton.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapSingleton.h"
#include "tutorial.hpp"

// splicer begin class.Singleton.CXX_definitions
// splicer end class.Singleton.CXX_definitions

extern "C" {

// splicer begin class.Singleton.C_definitions
// splicer end class.Singleton.C_definitions

// static Singleton & getReference()
TUT_Singleton * TUT_Singleton_get_reference(TUT_Singleton * SHC_rv)
{
    // splicer begin class.Singleton.method.get_reference
    tutorial::Singleton & SHCXX_rv = tutorial::Singleton::getReference(
        );
    SHC_rv->addr = static_cast<void *>(&SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end class.Singleton.method.get_reference
}

}  // extern "C"
