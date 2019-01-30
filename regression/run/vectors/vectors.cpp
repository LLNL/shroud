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

#include "vectors.hpp"

//----------------------------------------------------------------------
// vector reference as argument.
// arg+intent(in)

int vector_sum(const std::vector<int> &arg)
{
  int sum = 0;
  for(std::vector<int>::const_iterator it = arg.begin(); it != arg.end(); ++it) {
    sum += *it;
  }
  return sum;
}

//----------------------------------------------------------------------
// vector reference as argument.
// arg+intent(out)
// Start with empty vector and fill in values

void vector_iota_out(std::vector<int> &arg)
{
  for(unsigned int i=0; i < 5; i++) {
    arg.push_back(i + 1);
  }
  return;
}

//----------------------------------------------------------------------
// vector reference as argument.
// arg+intent(out)+deref(allocatable)
// Start with empty vector and fill in values

void vector_iota_out_alloc(std::vector<int> &arg)
{
  for(unsigned int i=0; i < 5; i++) {
    arg.push_back(i + 1);
  }
  return;
}

//----------------------------------------------------------------------
// vector reference as argument.
// arg+intent(out)+deref(allocatable)
// Start with empty vector and fill in values

void vector_iota_inout_alloc(std::vector<int> &arg)
{
  for(unsigned int i=0; i < 5; i++) {
    arg.push_back(i + 11);
  }
  return;
}

//----------------------------------------------------------------------
// vector reference as argument.
// arg+intent(inout)

void vector_increment(std::vector<int> &arg)
{
  for(unsigned int i=0; i < arg.size(); i++) {
    arg[i] += 1;
  }
  return;
}

//----------------------------------------------------------------------
// count underscore in strings
// arg+intent(in)

int vector_string_count(const std::vector< std::string > &arg)
{
  int count = 0;
  for(unsigned int i=0; i < arg.size(); i++) {
    for (unsigned int j = 0; j < arg[i].size(); j++) {
      if (arg[i][j] == '_') {
        count++;
      }
    }
  }
  return count;
}

//----------------------------------------------------------------------
// Add strings to arg.
// arg+intent(out)

void vector_string_fill(std::vector< std::string > &arg)
{
  arg.push_back("dog");
  arg.push_back("bird");
  return;
}

//----------------------------------------------------------------------
// Append to strings in arg.
// arg+intent(inout)

void vector_string_append(std::vector< std::string > &arg)
{
  for(unsigned int i=0; i < arg.size(); i++) {
    arg[i] += "-like";
  }
  return;
}


//----------------------------------------------------------------------

std::vector<int> ReturnVectorAlloc(int i)
{
  std::vector<int> rv;
  for (int i=0; i < 5; i++) {
    rv.push_back(i+1);
  }
  return rv;
}

