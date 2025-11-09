// Copyright Shroud Project Developers. See LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// cdesc.hpp

#ifndef CDESC_HPP
#define CDESC_HPP

#include <string>

void Rank2In(int *arg);

void GetScalar1(std::string & name, void *value);

template<typename DataType>
DataType getData();

#endif // CDESC_HPP
