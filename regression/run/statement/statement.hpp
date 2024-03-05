// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
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

#endif // STATEMENT_HPP
