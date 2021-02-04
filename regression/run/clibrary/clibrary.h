/*
 * Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * clibrary.h - wrapped routines
 */

#ifndef CLIBRARY_H
#define CLIBRARY_H

#include <stdbool.h>

/* Size of buffer passed from Fortran */
#define LENOUTBUF 40

/* A function macro - pass a constant as the first argument */
#define PassByValueMacro(__arg2) PassByValue(1.0, __arg2)

enum EnumTypeID {
    ENUM0,
    ENUM1,
    ENUM2
};

typedef int TypeID;

typedef struct {
  int tc;
} array_info;

void NoReturnNoArguments(void);

double PassByValue(double arg1, int arg2);
void PassByReference(double *arg1, int *arg2);

void checkBool(const bool arg1, bool *arg2, bool *arg3);

char *Function4a(const char *arg1, const char *arg2);
void acceptName(const char *name);

void passCharPtrInOut(char *s);

#define MAXNAME 20
void returnOneName(char *name1);
void returnTwoNames(char *name1, char *name2);

void ImpliedTextLen(char *text, int ltext);
int ImpliedLen(const char *text, int ltext, bool flag);
int ImpliedLenTrim(const char *text, int ltext, bool flag);
bool ImpliedBoolTrue(bool flag);
bool ImpliedBoolFalse(bool flag);

void bindC1(void);
void bindC2(char * outbuf);

void passVoidStarStar(void *in, void **out);

int passAssumedType(void *arg);
void passAssumedTypeDim(void *arg);
int passAssumedTypeBuf(void *arg, char *outbuf);

void callback1(int type, void (*incr)(void));
void callback1a(int type, void (*incr)(void));
void callback2(int type, void * in, void (*incr)(int *));
void callback3(const char *type, void * in, void (*incr)(int *), char *outbuf);
void callback_set_alloc(int tc, array_info *arr, void (*alloc)(int tc, array_info *arr));

#if 0
const std::string& Function4b(const std::string& arg1, const std::string& arg2);

double Function5(double arg1 = 3.1415, bool arg2 = true);

void Function6(const std::string& name);
void Function6(int indx);

void Function9(double arg);

void Function10(void);
void Function10(const std::string &name, double arg2);
#endif

void Sum(int len, int *values, int *result);

#if 0
TypeID typefunc(TypeID arg);

EnumTypeID enumfunc(EnumTypeID arg);

const char *LastFunctionCalled(void);

int vector_sum(const std::vector<int> &arg);
void vector_iota(std::vector<int> &arg);
void vector_increment(std::vector<int> &arg);

int vector_string_count(const std::vector< std::string > &arg);
void vector_string_fill(std::vector< std::string > &arg);
void vector_string_append(std::vector< std::string > &arg);
#endif

void intargs(const int argin, int * argout, int * arginout);

void cos_doubles(double * in, double * out, int sizein);

void truncate_to_int(double *in, int *out, int size);

void increment(int *array, int size);

void get_values(int *nvalues, int *values);

#endif // CLIBRARY_H
