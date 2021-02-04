// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// forward.hpp - wrapped routines
//

#ifndef FORWARD_HPP
#define FORWARD_HPP

// Headers from other library.
#include "struct.hpp"
#include "tutorial.hpp"  // Class1

namespace forward
{
class Class3;

class Class2
{
public:
    void func1(tutorial::Class1 *arg);
    void acceptClass3(Class3 *arg);
};

class Class3
{
};

int passStruct1(const Cstruct1 *s1);

} /* end namespace tutorial */

#endif // FORWARD_HPP
