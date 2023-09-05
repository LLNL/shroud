// helloworld.hpp
// Copyright (c) 2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <iostream>
#include <string>

namespace helloworld
{
    class Person
    {
    public:
        static int SayHello();
        static int NamedHello(std::string name);
    };    
}
