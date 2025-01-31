// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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
