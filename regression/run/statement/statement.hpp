// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// statement.hpp
//

#ifndef STATEMENT_HPP
#define STATEMENT_HPP

#include <string>

bool isNameValid(const std::string& name);
int GetNameLength();
const std::string& getNameErrorPattern();

bool nameIsValid(const std::string& name);

#endif // STATEMENT_HPP
