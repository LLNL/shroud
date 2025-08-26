// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// Tests for defaultarg.cpp
//

#include "defaultarg.hpp"


int main(int argc, char *argv[])
{

    // g++: call of overloaded 'apply(int, int)' is ambiguous
    // icpc: more than one instance of overloaded function "apply" matches the argument list:
    apply_generic(INT32_ID, 1,2);
    
    return 0;
}
