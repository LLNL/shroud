// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
// Produced at the Lawrence Livermore National Laboratory 
//
// LLNL-CODE-738041.
// All rights reserved. 
//
// This file is part of Shroud.  For details, see
// https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the disclaimer (as noted below)
//   in the documentation and/or other materials provided with the
//   distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
// LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

