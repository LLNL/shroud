// wrapwrapped_inner1.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapwrapped_inner1.h"

// splicer begin namespace.inner1.CXX_definitions
// splicer end namespace.inner1.CXX_definitions

extern "C" {

// splicer begin namespace.inner1.C_definitions
// splicer end namespace.inner1.C_definitions

// void worker()
void WWW_inner1_worker()
{
    // splicer begin namespace.inner1.function.worker
    outer::inner1::worker();
    return;
    // splicer end namespace.inner1.function.worker
}

}  // extern "C"
