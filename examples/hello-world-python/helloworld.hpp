// helloworld.hpp
// Copyright Shroud Project Developers. See LICENSE file for details.
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
