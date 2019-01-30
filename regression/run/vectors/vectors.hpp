// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
//
// Produced at the Lawrence Livermore National Laboratory 
//
// LLNL-CODE-738041.
//
// All rights reserved. 
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
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
void vector_iota_out_alloc(std::vector<int> &arg);
void vector_iota_inout_alloc(std::vector<int> &arg);
void vector_increment(std::vector<int> &arg);

int vector_string_count(const std::vector< std::string > &arg);
void vector_string_fill(std::vector< std::string > &arg);
void vector_string_append(std::vector< std::string > &arg);

std::vector<int> ReturnVectorAlloc(int i);


#endif // VECTORS_HPP
