// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "namespace.hpp"

static std::string last_function_called;

const std::string& LastFunctionCalled()
{
    return last_function_called;
}

void PassLevelEnum(upper::Level value)
{
  last_function_called = "PassLevelEnum";
}

void outer::One()
{
  last_function_called = "outer::One";
}

void One()
{
  last_function_called = "One";
}

