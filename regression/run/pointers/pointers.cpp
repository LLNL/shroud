/* Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
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

#include "pointers.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAXLAST 50
static char last_function_called[MAXLAST];

// These variables exist to avoid warning errors
//static std::string global_str;
//static int global_int;
//static double global_double;

//----------------------------------------------------------------------

void intargs(const int argin, int * arginout, int * argout)
{
  *argout = *arginout;
  *arginout = argin;
}

//----------------------------------------------------------------------

//#include <math.h>
/*  Compute the cosine of each element in in_array, storing the result in
 *  out_array. */
// replace cos with simpler function
void cos_doubles(double *in, double *out, int size)
{
    int i;
    for(i = 0; i < size; i++) {
      out[i] = cos(in[i]);
    }
}

//----------------------------------------------------------------------
// convert from double to int.

void truncate_to_int(double *in, int *out, int size)
{
    int i;
    for(i = 0; i < size; i++) {
        out[i] = in[i];
    }
}

//----------------------------------------------------------------------
// array +intent(inout)

void increment(int *array, int size)
{
    int i;
    for(i = 0; i < size; i++) {
       array[i] += 1;
    }
}

//----------------------------------------------------------------------
// values +intent(out)
// Note that we must assume that values is long enough.
// Otherwise, memory will be overwritten.

const int num_fill_values = 3;

void get_values(int *nvalues, int *values)
{
    int i;
    for(i = 0; i < num_fill_values; i++) {
       values[i] = i + 1;
    }
    *nvalues = num_fill_values;
    return;
}

//----------------------------------------------------------------------
const char *LastFunctionCalled(void)
{
    return last_function_called;
}
