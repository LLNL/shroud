// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// pointers.hpp - wrapped routines

#ifndef POINTERS_HPP
#define POINTERS_HPP

void intargs(const int argin, int * argout, int * arginout);

void cos_doubles(double * in, double * out, int sizein);

void truncate_to_int(double *in, int *out, int size);

void get_values(int *nvalues, int *values);

void get_values2(int *arg1, int *arg2);

void iota_allocatable(int nvar, int *values);
void iota_dimension(int nvar, int *values);

void Sum(int len, int * values, int *result);
void fillIntArray(int *out);
void incrementIntArray(int *array, int size);

void Rank2Input(int *arg);

void acceptCharArrayIn(char **names);

void setGlobalInt(int value);
int sumFixedArray(void);
void getPtrToScalar(int **nitems);
void getPtrToFixedArray(int **count);
void getPtrToDynamicArray(int **count, int *len);
int getlen(void);
void getPtrToFuncArray(int **count);

void getRawPtrToScalar(int **nitems);
void getRawPtrToFixedArray(int **count);

void *returnAddress1(int flag);
void *returnAddress2(int flag);

int *returnIntPtrToScalar(void);
int *returnIntPtrToFixedArray(void);

//void getPtrToArray(int **count+intent(out));

#endif // POINTERS_HPP
