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
 * clibrary.hpp - wrapped routines
 */

#ifndef CLIBRARY_HPP
#define CLIBRARY_HPP

#include <stdbool.h>

/* Size of buffer passed from Fortran */
#define LENOUTBUF 40

enum EnumTypeID {
    ENUM0,
    ENUM1,
    ENUM2
};

typedef int TypeID;

struct Cstruct1 {
  int ifield;
};
typedef struct Cstruct1 Cstruct1;

void Function1(void);

double Function2(double arg1, int arg2);

bool Function3(bool arg);
void Function3b(const bool arg1, bool *arg2, bool *arg3);

char *Function4a(const char *arg1, const char *arg2);

#define MAXNAME 20
void returnOneName(char *name1);
void returnTwoNames(char *name1, char *name2);

int ImpliedLen(const char *text, int ltext, bool flag);
int ImpliedLenTrim(const char *text, int ltext, bool flag);
bool ImpliedBoolTrue(bool flag);
bool ImpliedBoolFalse(bool flag);

void bindC1(void);
void bindC2(char * outbuf);

void passVoidStarStar(void *in, void **out);

int passAssumedType(void *arg);
int passAssumedTypeBuf(void *arg, char *outbuf);

void callback2(int type, void * in, void (*incr)(int *));
void callback3(const char *type, void * in, void (*incr)(int *), char *outbuf);

int passStruct1(Cstruct1 *s1);
int passStruct2(Cstruct1 *s1, char *outbuf);
Cstruct1 *returnStructPtr1(int ifield);
Cstruct1 *returnStructPtr2(int ifield, char *outbuf);

#if 0
const std::string& Function4b(const std::string& arg1, const std::string& arg2);

double Function5(double arg1 = 3.1415, bool arg2 = true);

void Function6(const std::string& name);
void Function6(int indx);

void Function9(double arg);

void Function10(void);
void Function10(const std::string &name, double arg2);
#endif

void Sum(int len, int * values, int *result);

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

#endif // CLIBRARY_HPP
