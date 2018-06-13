// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
// Produced at the Lawrence Livermore National Laboratory 
//
// LLNL-CODE-738041.
// All rights reserved. 
//
// This file is part of Shroud.  For details, see
// https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the disclaimer (as noted below)
//   in the documentation and/or other materials provided with the
//   distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
// LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// #######################################################################

#include "ownership.hpp"

static const Class1 *global_class1 = NULL;

//----------------------------------------------------------------------
// POD pointers

//#######################################
// return int scalar

int * ReturnIntPtrRaw()
{
  static int buffer = 1;
  return &buffer;
}

int * ReturnIntPtrScalar()
{
  static int buffer = 10;
  return &buffer;
}

int * ReturnIntPtrPointer()
{
  static int buffer = 1;
  return &buffer;
}

//#######################################
//#######################################
// return dimension(len) owner(library)
// Return a pointer to an existing, static array

int * ReturnIntPtrDimRaw(int *len)
{
  static int buffer[] = { 1, 2, 3, 4, 5, 6, 7 };
  *len = sizeof buffer / sizeof buffer[1];
  return buffer;
}

// As Fortran POINTER
int * ReturnIntPtrDimPointer(int *len)
{
  static int buffer[] = { 11, 12, 13, 14, 15, 16, 17 };
  *len = sizeof buffer / sizeof buffer[1];
  return buffer;
}

// As Fortran ALLOCATABLE
int * ReturnIntPtrDimAlloc(int *len)
{
  static int buffer[] = { 21, 22, 23, 24, 25, 26, 27 };
  *len = sizeof buffer / sizeof buffer[1];
  return buffer;
}

// as return_dimension_pointer
int * ReturnIntPtrDimDefault(int *len)
{
  static int buffer[] = { 31, 32, 33, 34, 35, 36, 37 };
  *len = sizeof buffer / sizeof buffer[1];
  return buffer;
}

//#######################################
// return int(len) owner(caller)
// Return a pointer to a new array

int * ReturnIntPtrDimRawNew(int *len)
{
  int *buffer = new int[5];
  for (int i=0; i < 5; i++) {
    buffer[i] = i;
  }
  *len = 5;
  return buffer;
}

// As Fortran POINTER
int * ReturnIntPtrDimPointerNew(int *len)
{
  int *buffer = new int[5];
  for (int i=0; i < 5; i++) {
    buffer[i] = i + 10;
  }
  *len = 5;
  return buffer;
}

// As Fortran ALLOCATABLE
int * ReturnIntPtrDimAllocNew(int *len)
{
  int *buffer = new int[5];
  for (int i=0; i < 5; i++) {
    buffer[i] = i + 20;
  }
  *len = 5;
  return buffer;
}

int * ReturnIntPtrDimDefaultNew(int *len)
{
  int *buffer = new int[5];
  for (int i=0; i < 5; i++) {
    buffer[i] = i + 30;
  }
  *len = 5;
  return buffer;
}

//#######################################
//#######################################
// intent(out) dimension(len) owner(library)
// Return a pointer to an existing, static array

void IntPtrDimRaw(int **array, int *len)
{
  static int buffer[] = { 1, 2, 3, 4, 5, 6, 7 };
  *len = sizeof buffer / sizeof buffer[1];
  *array = buffer;
}

// As Fortran POINTER
void IntPtrDimPointer(int **array, int *len)
{
  static int buffer[] = { 11, 12, 13, 14, 15, 16, 17 };
  *len = sizeof buffer / sizeof buffer[1];
  *array = buffer;
}

// As Fortran ALLOCATABLE
void IntPtrDimAlloc(int **array, int *len)
{
  static int buffer[] = { 21, 22, 23, 24, 25, 26, 27 };
  *len = sizeof buffer / sizeof buffer[1];
  *array = buffer;
}

// as return_dimension_pointer
void IntPtrDimDefault(int **array, int *len)
{
  static int buffer[] = { 31, 32, 33, 34, 35, 36, 37 };
  *len = sizeof buffer / sizeof buffer[1];
  *array = buffer;
}

//#######################################
// return int(len) owner(caller)
// Return a pointer to a new array

void IntPtrDimRawNew(int **array, int *len)
{
  int *buffer = new int[5];
  for (int i=0; i < 5; i++) {
    buffer[i] = i;
  }
  *len = 5;
  *array = buffer;
}

// As Fortran POINTER
void IntPtrDimPointerNew(int **array, int *len)
{
  int *buffer = new int[5];
  for (int i=0; i < 5; i++) {
    buffer[i] = i + 10;
  }
  *len = 5;
  *array = buffer;
}

// As Fortran ALLOCATABLE
void IntPtrDimAllocNew(int **array, int *len)
{
  int *buffer = new int[5];
  for (int i=0; i < 5; i++) {
    buffer[i] = i + 20;
  }
  *len = 5;
  *array = buffer;
}

void IntPtrDimDefaultNew(int **array, int *len)
{
  int *buffer = new int[5];
  for (int i=0; i < 5; i++) {
    buffer[i] = i + 30;
  }
  *len = 5;
  *array = buffer;
}

//----------------------------------------------------------------------
// Instance pointers

// Create a global Class1 which may be fetched by getClassStatic
void createClassStatic(int flag)
{
  if (global_class1 != NULL) {
    delete global_class1;
  }
  global_class1 = new Class1(flag);
}

Class1 * getClassStatic()
{
    return const_cast<Class1 *>(global_class1);
}

/* Return a new instance */
Class1 * getClassNew(int flag)
{
    Class1 *node = new Class1(flag);
    return node;
}

//----------------------------------------------------------------------
// example lifted from pybindgen PyBindGen-0.18.0/tests/foo.cc

static Zbr *g_zbr = NULL;
int Zbr::instance_count = 0;

void store_zbr (Zbr *zbr)
{
    if (g_zbr)
        g_zbr->Unref ();
    // steal the reference
    g_zbr = zbr;
}

int invoke_zbr (int x)
{
    return g_zbr->get_int (x);
}

void delete_stored_zbr (void)
{
    if (g_zbr)
        g_zbr->Unref ();
    g_zbr = NULL;
}
//----------------------------------------------------------------------
