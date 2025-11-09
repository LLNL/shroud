// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#include "statement.hpp"

const std::string name = "the-name";

//----------------------------------------------------------------------
bool isNameValid(const std::string& name)
{
    return true;
}

int GetNameLength()
{
    return 40;
}

const std::string& getNameErrorPattern()
{
    return name;
}

const std::string InvalidName;

// The C and Fortran wrappers provide different implemenations via a splicer.

bool nameIsValid(const std::string& name)
{
    return name != InvalidName;
}


//----------------------------------------------------------------------
