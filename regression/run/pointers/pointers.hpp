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
 * pointers.hpp - wrapped routines
 */

#ifndef POINTERS_HPP
#define POINTERS_HPP

void intargs(const int argin, int * argout, int * arginout);

void cos_doubles(double * in, double * out, int sizein);

void truncate_to_int(double *in, int *out, int size);

void increment(int *array, int size);

void get_values(int *nvalues, int *values);

void get_values2(int *arg1, int *arg2);

void Sum(int len, int * values, int *result);
void fillIntArray(int *out);
void incrementIntArray(int *values, int len);


#endif // POINTERS_HPP
