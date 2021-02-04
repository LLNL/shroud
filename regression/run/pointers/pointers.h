// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// pointers.hpp - wrapped routines

#ifndef POINTERS_HPP
#define POINTERS_HPP

#include <stddef.h>

void  intargs_in(const int *arg);
void  intargs_inout(int *arg);
void  intargs_out(int *arg);
void intargs(const int argin, int * argout, int * arginout);

void cos_doubles(double * in, double * out, int sizein);

void truncate_to_int(double *in, int *out, int size);

void get_values(int *nvalues, int *values);

void get_values2(int *arg1, int *arg2);

void iota_allocatable(int nvar, int *values);
void iota_dimension(int nvar, int *values);

void Sum(int len, const int * values, int *result);
void fillIntArray(int *out);
void incrementIntArray(int *array, int size);

void fill_with_zeros(double* x, int x_length);
int accumulate(const int *arr, size_t len);


void Rank2Input(int *arg);

int acceptCharArrayIn(char **names);

void setGlobalInt(int value);
int sumFixedArray(void);

void getPtrToScalar(int **nitems);
void getPtrToFixedArray(int **count);
void getPtrToDynamicArray(int **count, int *len);
int getLen(void);
void getPtrToFuncArray(int **count);

void getPtrToConstScalar(const int **nitems);
void getPtrToFixedConstArray(const int **count);
void getPtrToDynamicConstArray(const int **count, int *len);

void getRawPtrToScalar(int **nitems);
void getRawPtrToScalarForce(int **nitems);
void getRawPtrToFixedArray(int **count);
void getRawPtrToFixedArrayForce(int **count);
void getRawPtrToInt2d(int ***arg);
int checkInt2d(int **arg);

void DimensionIn(const int *arg);

void *returnAddress1(int flag);
void *returnAddress2(int flag);
void fetchVoidPtr(void **addr);
int VoidPtrArray(void **addr);

int *returnIntPtrToScalar(void);
int *returnIntPtrToFixedArray(void);
const int *returnIntPtrToConstScalar(void);
const int *returnIntPtrToFixedConstArray(void);
int *returnIntScalar(void);
int *returnIntRaw(void);
int *returnIntRawWithArgs(const char *name);
int **returnRawPtrToInt2d(void);

//void getPtrToArray(int **count+intent(out));

#endif // POINTERS_HPP
