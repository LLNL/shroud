/* Copyright (c) 2017, Lawrence Livermore National Security, LLC. 
 * Produced at the Lawrence Livermore National Laboratory 
 *
 * LLNL-CODE-738041.
 * All rights reserved. 
 *
 * This file is part of Shroud.  For details, see
 * https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 * 
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the disclaimer (as noted below)
 *   in the documentation and/or other materials provided with the
 *   distribution.
 *
 * * Neither the name of the LLNS/LLNL nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
 * LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * #######################################################################
 *
 * tutorial.hpp - wrapped routines
 */

#include "clibrary.h"

#include <string.h>

#define MAXLAST 50
static char last_function_called[MAXLAST];

// These variables exist to avoid warning errors
//static std::string global_str;
//static int global_int;
//static double global_double;



void Function1()
{
    strncpy(last_function_called, "Function1", MAXLAST);
    return;
}

double Function2(double arg1, int arg2)
{
    strncpy(last_function_called, "Function2", MAXLAST);
    return arg1 + arg2;
}

#if 0
bool Function3(bool arg)
{
    strncpy(last_function_called, "Function3", MAXLAST);
    return ! arg;
}

void Function3b(const bool arg1, bool *arg2, bool *arg3)
{
    strncpy(last_function_called, "Function3b", MAXLAST);
    *arg2 = ! arg1;
    *arg3 = ! *arg3;
    return;
}

const std::string Function4a(const std::string& arg1, const std::string& arg2)
{
    strncpy(last_function_called, "Function4a", MAXLAST);
    return arg1 + arg2;
}
const std::string& Function4b(const std::string& arg1, const std::string& arg2)
{
    strncpy(last_function_called, "Function4b", MAXLAST);
    return global_str = arg1 + arg2;
    return global_str;
}

double Function5(double arg1, bool arg2)
{
    strncpy(last_function_called, "Function5", MAXLAST);
    if (arg2) {
	return arg1 + 10.0;
    } else {
	return arg1;
    }
}

void Function6(const std::string& name)
{
    strncpy(last_function_called, "Function6(string)", MAXLAST);
    global_str = name;
    return;
}
void Function6(int indx)
{
    strncpy(last_function_called, "Function6(int)", MAXLAST);
    global_int = indx;
    return;
}

void Function9(double arg)
{
    strncpy(last_function_called, "Function9", MAXLAST);
    global_double = arg;
    return;
}

void Function10()
{
    strncpy(last_function_called, "Function10_0", MAXLAST);
}

void Function10(const std::string &name, double arg2)
{
    strncpy(last_function_called, "Function10_1", MAXLAST);
    global_str = name;
    global_double = arg2;
}

void Sum(int len, int *values, int *result)
{
    strncpy(last_function_called, "Sum", MAXLAST);

    int sum = 0;
    for (int i=0; i < len; i++) {
	sum += values[i];
    }
    *result = sum;
    return;
}

TypeID typefunc(TypeID arg)
{
    strncpy(last_function_called, "typefunc", MAXLAST);
    return static_cast<int>(arg);
}

EnumTypeID enumfunc(EnumTypeID arg)
{
    strncpy(last_function_called, "enumfunc", MAXLAST);
    switch (arg) {
    default:
	return ENUM2;
    }
}

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

void vector_iota(std::vector<int> &arg)
{
  for(unsigned int i=0; i < arg.size(); i++) {
    arg[i] = i + 1;
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

#endif
//----------------------------------------------------------------------

const char *LastFunctionCalled()
{
    return last_function_called;
}
