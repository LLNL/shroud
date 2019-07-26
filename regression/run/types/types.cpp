// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
//
// All rights reserved.
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
//
// types.hpp - wrapped routines
//

#include "types.hpp"

// #######################################################################

short short_func(short arg1)
{
  return arg1;
}

int int_func(int arg1)
{
  return arg1;
}

long long_func(long arg1)
{
  return arg1;
}

long long long_long_func(long long arg1)
{
  return arg1;
}

short int short_int_func(short int arg1)
{
  return arg1;
}

long int long_int_func(long int arg1)
{
  return arg1;
}

long long int long_long_int_func(long long int arg1)
{
  return arg1;
}

// #######################################################################

unsigned unsigned_func(unsigned arg1)
{
  return arg1;
}

unsigned short ushort_func(unsigned short arg1)
{
  return arg1;
}

unsigned int uint_func(unsigned int arg1)
{
  return arg1;
}

unsigned long ulong_func(unsigned long arg1)
{
  return arg1;
}

unsigned long long ulong_long_func(unsigned long long arg1)
{
  return arg1;
}

unsigned long int ulong_int_func(unsigned long int arg1)
{
  return arg1;
}

// #######################################################################

int8_t int8_func(int8_t arg1)
{
  return arg1;
}

int16_t int16_func(int16_t arg1)
{
  return arg1;
}

int32_t int32_func(int32_t arg1)
{
  return arg1;
}

int64_t int64_func(int64_t arg1)
{
  return arg1;
}

uint8_t uint8_func(uint8_t arg1)
{
  return arg1;
}

uint16_t uint16_func(uint16_t arg1)
{
  return arg1;
}

uint32_t uint32_func(uint32_t arg1)
{
  return arg1;
}

uint64_t uint64_func(uint64_t arg1)
{
  return arg1;
}

size_t size_func(size_t arg1)
{
  return arg1;
}

bool bool_func(bool arg1)
{
  return arg1;
}

// #######################################################################

/* Test Python wrapper which returns bool and other intent(out) argument. */

bool returnBoolAndOthers(int *flag)
{
  *flag = 1;
  return true;
}
