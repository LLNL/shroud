// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
// clibrary.c

#include "clibrary.h"

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

void Function1(void)
{
    strncpy(last_function_called, "Function1", MAXLAST);
    return;
}

// start PassByValue
double PassByValue(double arg1, int arg2)
{
    strncpy(last_function_called, "PassByValue", MAXLAST);
    return arg1 + arg2;
}
// end PassByValue

// start PassByReference
void PassByReference(double *arg1, int *arg2)
{
    strncpy(last_function_called, "PassByReference", MAXLAST);
    *arg2 = *arg1;
}
// end PassByReference

bool Function3(bool arg)
{
    strncpy(last_function_called, "Function3", MAXLAST);
    return ! arg;
}

// start checkBool
void checkBool(const bool arg1, bool *arg2, bool *arg3)
{
    strncpy(last_function_called, "checkBool", MAXLAST);
    *arg2 = ! arg1;
    *arg3 = ! *arg3;
    return;
}
// end checkBool

/* Note that the caller is responsible to free memory */
char *Function4a(const char *arg1, const char *arg2)
{
    strncpy(last_function_called, "Function4a", MAXLAST);
    size_t narg1 = strlen(arg1);
    size_t narg2 = strlen(arg2);
    char *out = malloc(narg1 + narg2 + 1);
    sprintf(out, "%s%s", arg1, arg2);
    return out;
}

// start acceptName
void acceptName(const char *name)
{
    strncpy(last_function_called, "acceptName", MAXLAST);
}
// end acceptName

//----------------------------------------------------------------------
// Test charlen attribute.
// Each argument is assumed to be MAXNAME long.

// start returnOneName
void returnOneName(char *name1)
{
  strcpy(name1, "bill");
}
// end returnOneName

void returnTwoNames(char *name1, char *name2)
{
  strcpy(name1, "tom");
  strcpy(name2, "frank");
}

//----------------------------------------------------------------------

// start ImpliedTextLen
void ImpliedTextLen(char *text, int ltext)
{
    strncpy(text, "ImpliedTextLen", ltext);
    strncpy(last_function_called, "ImpliedTextLen", MAXLAST);
}
// end ImpliedTextLen

int ImpliedLen(const char *text, int ltext, bool flag)
{
    strncpy(last_function_called, "ImpliedLen", MAXLAST);
    return ltext;
}

int ImpliedLenTrim(const char *text, int ltext, bool flag)
{
    strncpy(last_function_called, "ImpliedLenTrim", MAXLAST);
    return ltext;
}

bool ImpliedBoolTrue(bool flag)
{
    strncpy(last_function_called, "ImpliedBoolTrue", MAXLAST);
    return flag;
}

bool ImpliedBoolFalse(bool flag)
{
    strncpy(last_function_called, "ImpliedBoolFalse", MAXLAST);
    return flag;
}

void bindC1()
{
    strncpy(last_function_called, "bindC1", MAXLAST);
}

void bindC2(char * outbuf)
{
    strncpy(outbuf, "bindC2", LENOUTBUF);
    strncpy(last_function_called, "bindC2", MAXLAST);
}

// start passVoidStarStar
void passVoidStarStar(void *in, void **out)
{
    strncpy(last_function_called, "passVoidStarStar", MAXLAST);
    *out = in;
}
// end passVoidStarStar

/* arg is assumed to be an int. */

// start passAssumedType
int passAssumedType(void *arg)
{
    strncpy(last_function_called, "passAssumedType", MAXLAST);
    return *(int *) arg;
}
// end passAssumedType

// start passAssumedTypeDim
void passAssumedTypeDim(void *arg)
{
    strncpy(last_function_called, "passAssumedTypeDim", MAXLAST);
}
// end passAssumedTypeDim

/* arg is assumed to be an int. */

int passAssumedTypeBuf(void *arg, char *outbuf)
{
    strncpy(outbuf, "passAssumedTypeBuf", LENOUTBUF);
    strncpy(last_function_called, "passAssumedTypeBuf", MAXLAST);
    return *(int *) arg;
}

//----------------------------------------------------------------------

// start callback1
void callback1(int type, void (*incr)(void))
{
  // Use type to decide how to call incr
}
// end callback1

void callback2(int type, void * in, void (*incr)(int *))
{
  if (type == 1) {
    // default function pointer from prototype
    incr(in);
  } else if (type == 2) {
    void (*incr2)(double *) = (void(*)(double *)) incr;
    incr2(in);
  }
}

void callback3(const char *type, void * in, void (*incr)(int *),
               char *outbuf)
{
  if (strcmp(type, "int") == 0) {
    // default function pointer from prototype
    incr(in);
  } else if (strcmp(type, "double") == 0) {
    void (*incr2)(double *) = (void(*)(double *)) incr;
    incr2(in);
  }
  strncpy(outbuf, "callback3", LENOUTBUF);
}

//----------------------------------------------------------------------

#if 0
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

void Function10(void)
{
    strncpy(last_function_called, "Function10_0", MAXLAST);
}

void Function10(const std::string &name, double arg2)
{
    strncpy(last_function_called, "Function10_1", MAXLAST);
    global_str = name;
    global_double = arg2;
}
#endif

// start Sum
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
// end Sum

#if 0
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
