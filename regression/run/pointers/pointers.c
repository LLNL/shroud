// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// pointers.c

#include "pointers.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAXLAST 50
static char last_function_called[MAXLAST];

// These variables exist to avoid warning errors
//static std::string global_str;
static int global_int = 0;
static int global_fixed_array[10];
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

// values is assumed to be nvar long.

void iota_allocatable(int nvar, int *values)
{
    int i;
    for(i = 0; i < nvar; i++) {
        values[i] = i + 1;
    }
}

void iota_dimension(int nvar, int *values)
{
    int i;
    for(i = 0; i < nvar; i++) {
        values[i] = i + 1;
    }
}


//----------------------------------------------------------------------

// start Sum
void Sum(int len, int *values, int *result)
{
    int sum = 0;
    for (int i=0; i < len; i++) {
	sum += values[i];
    }
    *result = sum;
    return;
}
// end Sum

// out is assumed to be at least 3 long
void fillIntArray(int *out)
{
  out[0] = 1;
  out[1] = 2;
  out[2] = 3;
}

// array +intent(inout)
void incrementIntArray(int *array, int size)
{
    for(int i=0; i < size; i++) {
        array[i] += 1;
    }
    return;
}

//----------------------------------------------------------------------

void Rank2Input(int *arg)
{
}

//----------------------------------------------------------------------

void acceptCharArrayIn(char **names)
{
}

//----------------------------------------------------------------------

// Set global_int to use with testing.
void setGlobalInt(int value)
{
    global_int = value;
}

// Used with testing.
int sumFixedArray(void)
{
    int sum = 0;
    int nitems = sizeof(global_fixed_array)/sizeof(int);
    for (int i=0; i < nitems; ++i)
    {
        sum += global_fixed_array[i];
    }
    return sum;
}

/**** Return a Fortran pointer */
/* Return pointer to a scalar in the argument. */
void getPtrToScalar(int **nitems)
{
    *nitems = &global_int;
}

void getPtrToFixedArray(int **count)
{
    *count = (int *) &global_fixed_array;
}

void getPtrToDynamicArray(int **count, int *len)
{
    *count = (int *) &global_fixed_array;
    *len = sizeof(global_fixed_array)/sizeof(int);
}

// Return length of global_fixed_array.
int getLen(void)
{
    return sizeof(global_fixed_array)/sizeof(int);
}

// length is computed by function getlen.
void getPtrToFuncArray(int **count)
{
    *count = (int *) &global_fixed_array;
}

/**** Return a Fortran pointer for const argument*/
void getPtrToConstScalar(const int **nitems)
{
    *nitems = &global_int;
}

void getPtrToFixedConstArray(const int **count)
{
    *count = (int *) &global_fixed_array;
}

void getPtrToDynamicConstArray(const int **count, int *len)
{
    *count = (int *) &global_fixed_array;
    *len = sizeof(global_fixed_array)/sizeof(int);
}

/**** Return a type(C_PTR) pointer */
/* Return pointer to a scalar in the argument. */
void getRawPtrToScalar(int **nitems)
{
    *nitems = &global_int;
}

void getRawPtrToFixedArray(int **count)
{
    *count = (int *) &global_fixed_array;
}

//----------------------------------------------------------------------

// Return a raw pointer to global_int.
void *returnAddress1(int flag)
{
    return (void *) &global_int;
}
void *returnAddress2(int flag)
{
    return (void *) &global_int;
}

int *returnIntPtrToScalar(void)
{
    return &global_int;
}

int *returnIntPtrToFixedArray(void)
{
    return (int *) &global_fixed_array;
}

const int *returnIntPtrToConstScalar(void)
{
    return &global_int;
}

const int *returnIntPtrToFixedConstArray(void)
{
    return (int *) &global_fixed_array;
}

// Fortran wrapper has +deref(scalar)
int *returnIntScalar(void)
{
    return &global_int;
}

//----------------------------------------------------------------------
const char *LastFunctionCalled(void)
{
    return last_function_called;
}
