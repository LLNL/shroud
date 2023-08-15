// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// vectors.cpp

#include "vectors.hpp"

//----------------------------------------------------------------------
// vector reference as argument.
// arg+intent(in)

// start vector_sum
int vector_sum(const std::vector<int> &arg)
{
  int sum = 0;
  for(std::vector<int>::const_iterator it = arg.begin(); it != arg.end(); ++it) {
    sum += *it;
  }
  return sum;
}
// end vector_sum

//----------------------------------------------------------------------
// vector reference as argument.
// arg+intent(out)
// Start with empty vector and fill in values.
// Three copies of the same function but wrapped differently.

// start vector_iota_out
void vector_iota_out(std::vector<int> &arg)
{
  for(unsigned int i=0; i < 5; i++) {
    arg.push_back(i + 1);
  }
  return;
}
// end vector_iota_out

// C and Fortran wrapper will be a function which returns arg.size()
void vector_iota_out_with_num(std::vector<int> &arg)
{
  for(unsigned int i=0; i < 5; i++) {
    arg.push_back(i + 1);
  }
  return;
}

// Fortran wrapper will be a function which Darg%size
void vector_iota_out_with_num2(std::vector<int> &arg)
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

// start vector_iota_out_alloc
void vector_iota_out_alloc(std::vector<int> &arg)
{
  for(unsigned int i=0; i < 5; i++) {
    arg.push_back(i + 1);
  }
  return;
}
// end vector_iota_out_alloc

//----------------------------------------------------------------------
// vector reference as argument.
// arg+intent(out)+deref(allocatable)
// Start with empty vector and fill in values

// start vector_iota_inout_alloc
void vector_iota_inout_alloc(std::vector<int> &arg)
{
  for(unsigned int i=0; i < 5; i++) {
    arg.push_back(i + 11);
  }
  return;
}
// end vector_iota_inout_alloc

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
// vector reference as argument.
// arg+intent(out)
// Start with empty vector and fill in values

void vector_iota_out_d(std::vector<double> &arg)
{
  for(unsigned int i=0; i < 5; i++) {
    arg.push_back(i + 1.0);
  }
  return;
}

//----------------------------------------------------------------------
int vector_of_pointers(std::vector<const double *> &arg1, int num)
{
    int sum = 0;
    for (size_t i=0; i < arg1.size(); i++) {
        const double *array = arg1[i];
        for (int j=0; j < num; ++j) {
            sum += array[j];
        }
    }
    return sum;
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
// Add strings to arg.
// arg+intent(out)+deref(allocatable)

void vector_string_fill_allocatable(std::vector< std::string > &arg)
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

std::vector<int> ReturnVectorAlloc(int n)
{
  std::vector<int> rv;
  for (int i=1; i <= n; i++) {
    rv.push_back(i);
  }
  return rv;
}

//----------------------------------------------------------------------
// Test implied(size(arg,2))

int returnDim2(int *arg, int len)
{
    return len;
}

