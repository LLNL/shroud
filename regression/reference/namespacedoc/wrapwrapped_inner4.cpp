// wrapwrapped_inner4.cpp
// This file is generated by Shroud 0.11.0. Do not edit.
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapwrapped_inner4.h"

// splicer begin namespace.inner4.CXX_definitions
// splicer end namespace.inner4.CXX_definitions

extern "C" {

// splicer begin namespace.inner4.C_definitions
// splicer end namespace.inner4.C_definitions

// void worker4()
// ----------------------------------------
// Result
// Requested: c
// Match:     c_default
void WWW_inner4_worker4()
{
    // splicer begin namespace.inner4.function.worker4
    outer::inner4::worker4();
    // splicer end namespace.inner4.function.worker4
}

}  // extern "C"
