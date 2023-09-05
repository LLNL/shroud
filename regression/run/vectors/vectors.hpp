// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// #######################################################################
//
// vectors.hpp - wrapped routines
//

#ifndef VECTORS_HPP
#define VECTORS_HPP

#include <string>
#include <vector>

int vector_sum(const std::vector<int> &arg);
void vector_iota_out(std::vector<int> &arg);
void vector_iota_out_with_num(std::vector<int> &arg);
void vector_iota_out_with_num2(std::vector<int> &arg);
void vector_iota_out_alloc(std::vector<int> &arg);
void vector_iota_inout_alloc(std::vector<int> &arg);
void vector_increment(std::vector<int> &arg);

void vector_iota_out_d(std::vector<double> &arg);

int vector_of_pointers(std::vector<const double *> &arg1, int num);

int vector_string_count(const std::vector< std::string > &arg);
void vector_string_fill(std::vector< std::string > &arg);
void vector_string_fill_allocatable(std::vector< std::string > &arg);
void vector_string_fill_allocatable_len(std::vector< std::string > &arg);
void vector_string_append(std::vector< std::string > &arg);

std::vector<int> ReturnVectorAlloc(int i);

int returnDim2(int *arg, int len);

#endif // VECTORS_HPP
