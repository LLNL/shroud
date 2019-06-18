/* Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
 *
 * Produced at the Lawrence Livermore National Laboratory 
 *
 * LLNL-CODE-738041.
 *
 * All rights reserved. 
 *
 * This file is part of Shroud.
 *
 * For details about use and distribution, please read LICENSE.
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

void get_values2(int *arg1, int *arg2)
{
    int i;
    for(i = 0; i < num_fill_values; i++) {
       arg1[i] = i + 1;
       arg2[i] = i + 11;
    }
    return;
}

//----------------------------------------------------------------------

void Sum(int len, int *values, int *result)
{
    strncpy(last_function_called, "Sum", MAXLAST);

    int i;
    int sum = 0;
    for (i=0; i < len; i++) {
	sum += values[i];
    }
    *result = sum;
    return;
}

// out is assumed to be at least 3 long
void fillIntArray(int *out)
{
  out[0] = 1;
  out[1] = 2;
  out[2] = 3;
}

void incrementIntArray(int *values, int len)
{
    for(int i=0; i < len; i++) {
        values[i] += 1;
    }
    return;
}

//----------------------------------------------------------------------
const char *LastFunctionCalled(void)
{
    return last_function_called;
}
