// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// tr29113.cpp
//

#include "tr29113.hpp"

// These variables exist to avoid warning errors
static std::string static_str = std::string("dog");

//----------------------------------------


const std::string * getConstStringPtrAlloc()
{
    // +owner(library)
    return &static_str;
}

const std::string * getConstStringPtrOwnsAlloc()
{
    // +owner(caller)
    std::string * rv = new std::string("getConstStringPtrOwnsAlloc");
    return rv;
}

//----------------------------------------
