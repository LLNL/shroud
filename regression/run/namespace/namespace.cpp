// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "namespace.hpp"

static std::string last_function_called;

const std::string& LastFunctionCalled()
{
    return last_function_called;
}

void outer::One()
{
  last_function_called = "outer::One";
}

void One()
{
  last_function_called = "One";
}

