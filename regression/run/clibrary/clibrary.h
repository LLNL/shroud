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

int ImpliedLen(const char *text, int ltext);
int ImpliedLenTrim(const char *text, int ltext);

void bindC1(void);
void bindC2(const char * name);

int passStruct1(Cstruct1 *s1);
int passStruct2(Cstruct1 *s1, const char *name);

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
