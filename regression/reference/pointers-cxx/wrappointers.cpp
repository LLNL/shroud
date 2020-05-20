// wrappointers.cpp
// This file is generated by Shroud 0.11.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrappointers.h"
#include <cstdlib>
#include <cstring>
#include "pointers.h"
#include "typespointers.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper ShroudLenTrim
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
static int ShroudLenTrim(const char *src, int nsrc) {
    int i;

    for (i = nsrc - 1; i >= 0; i--) {
        if (src[i] != ' ') {
            break;
        }
    }

    return i + 1;
}


// helper ShroudStrArrayAlloc
// Copy src into new memory and null terminate.
// char **src +size(nsrc) +len(len)
// CHARACTER(len) src(nsrc)
static char **ShroudStrArrayAlloc(const char *src, int nsrc, int len)
{
   char **rv = static_cast<char **>(std::malloc(sizeof(char *) * nsrc));
   const char *src0 = src;
   for(int i=0; i < nsrc; ++i) {
      int ntrim = ShroudLenTrim(src0, len);
      char *tgt = static_cast<char *>(std::malloc(ntrim+1));
      std::memcpy(tgt, src0, ntrim);
      tgt[ntrim] = '\0';
      rv[i] = tgt;
      src0 += len;
   }
   return rv;
}

// helper ShroudStrArrayFree
// Release memory allocated by ShroudStrArrayAlloc
static void ShroudStrArrayFree(char **src, int nsrc)
{
   for(int i=0; i < nsrc; ++i) {
       std::free(src[i]);
   }
   std::free(src);
}
// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  void intargs_in
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const int * arg +intent(in)
// Requested: c_native_*_in
// Match:     c_default
// start POI_intargs_in
void POI_intargs_in(const int * arg)
{
    // splicer begin function.intargs_in
    intargs_in(arg);
    // splicer end function.intargs_in
}
// end POI_intargs_in

// ----------------------------------------
// Function:  void intargs_inout
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * arg +intent(inout)
// Requested: c_native_*_inout
// Match:     c_default
// start POI_intargs_inout
void POI_intargs_inout(int * arg)
{
    // splicer begin function.intargs_inout
    intargs_inout(arg);
    // splicer end function.intargs_inout
}
// end POI_intargs_inout

// ----------------------------------------
// Function:  void intargs_out
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * arg +intent(out)
// Requested: c_native_*_out
// Match:     c_default
// start POI_intargs_out
void POI_intargs_out(int * arg)
{
    // splicer begin function.intargs_out
    intargs_out(arg);
    // splicer end function.intargs_out
}
// end POI_intargs_out

// ----------------------------------------
// Function:  void intargs
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const int argin +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// ----------------------------------------
// Argument:  int * arginout +intent(inout)
// Requested: c_native_*_inout
// Match:     c_default
// ----------------------------------------
// Argument:  int * argout +intent(out)
// Requested: c_native_*_out
// Match:     c_default
// start POI_intargs
void POI_intargs(const int argin, int * arginout, int * argout)
{
    // splicer begin function.intargs
    intargs(argin, arginout, argout);
    // splicer end function.intargs
}
// end POI_intargs

/**
 * \brief compute cos of IN and save in OUT
 *
 * allocate OUT same type as IN implied size of array
 */
// ----------------------------------------
// Function:  void cos_doubles
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  double * in +intent(in)+rank(1)
// Requested: c_native_*_in
// Match:     c_default
// ----------------------------------------
// Argument:  double * out +deref(allocatable)+dimension(size(in))+intent(out)
// Requested: c_native_*_out
// Match:     c_default
// ----------------------------------------
// Argument:  int sizein +implied(size(in))+intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// start POI_cos_doubles
void POI_cos_doubles(double * in, double * out, int sizein)
{
    // splicer begin function.cos_doubles
    cos_doubles(in, out, sizein);
    // splicer end function.cos_doubles
}
// end POI_cos_doubles

/**
 * \brief truncate IN argument and save in OUT
 *
 * allocate OUT different type as IN
 * implied size of array
 */
// ----------------------------------------
// Function:  void truncate_to_int
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  double * in +intent(in)+rank(1)
// Requested: c_native_*_in
// Match:     c_default
// ----------------------------------------
// Argument:  int * out +deref(allocatable)+dimension(size(in))+intent(out)
// Requested: c_native_*_out
// Match:     c_default
// ----------------------------------------
// Argument:  int sizein +implied(size(in))+intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// start POI_truncate_to_int
void POI_truncate_to_int(double * in, int * out, int sizein)
{
    // splicer begin function.truncate_to_int
    truncate_to_int(in, out, sizein);
    // splicer end function.truncate_to_int
}
// end POI_truncate_to_int

/**
 * \brief fill values into array
 *
 * The function knows how long the array must be.
 * Fortran will treat the dimension as assumed-length.
 * The Python wrapper will create a NumPy array or list so it must
 * have an explicit dimension (not assumed-length).
 */
// ----------------------------------------
// Function:  void get_values
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * nvalues +intent(out)
// Requested: c_native_*_out
// Match:     c_default
// ----------------------------------------
// Argument:  int * values +dimension(3)+intent(out)
// Requested: c_native_*_out
// Match:     c_default
// start POI_get_values
void POI_get_values(int * nvalues, int * values)
{
    // splicer begin function.get_values
    get_values(nvalues, values);
    // splicer end function.get_values
}
// end POI_get_values

/**
 * \brief fill values into two arrays
 *
 * Test two intent(out) arguments.
 * Make sure error handling works with C++.
 */
// ----------------------------------------
// Function:  void get_values2
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * arg1 +dimension(3)+intent(out)
// Requested: c_native_*_out
// Match:     c_default
// ----------------------------------------
// Argument:  int * arg2 +dimension(3)+intent(out)
// Requested: c_native_*_out
// Match:     c_default
// start POI_get_values2
void POI_get_values2(int * arg1, int * arg2)
{
    // splicer begin function.get_values2
    get_values2(arg1, arg2);
    // splicer end function.get_values2
}
// end POI_get_values2

// ----------------------------------------
// Function:  void iota_allocatable
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int nvar +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// ----------------------------------------
// Argument:  int * values +deref(allocatable)+dimension(nvar)+intent(out)
// Requested: c_native_*_out
// Match:     c_default
// start POI_iota_allocatable
void POI_iota_allocatable(int nvar, int * values)
{
    // splicer begin function.iota_allocatable
    iota_allocatable(nvar, values);
    // splicer end function.iota_allocatable
}
// end POI_iota_allocatable

// ----------------------------------------
// Function:  void iota_dimension
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int nvar +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// ----------------------------------------
// Argument:  int * values +dimension(nvar)+intent(out)
// Requested: c_native_*_out
// Match:     c_default
// start POI_iota_dimension
void POI_iota_dimension(int nvar, int * values)
{
    // splicer begin function.iota_dimension
    iota_dimension(nvar, values);
    // splicer end function.iota_dimension
}
// end POI_iota_dimension

// ----------------------------------------
// Function:  void Sum
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int len +implied(size(values))+intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// ----------------------------------------
// Argument:  int * values +intent(in)+rank(1)
// Requested: c_native_*_in
// Match:     c_default
// ----------------------------------------
// Argument:  int * result +intent(out)
// Requested: c_native_*_out
// Match:     c_default
// start POI_sum
void POI_sum(int len, int * values, int * result)
{
    // splicer begin function.sum
    Sum(len, values, result);
    // splicer end function.sum
}
// end POI_sum

/**
 * Return three values into memory the user provides.
 */
// ----------------------------------------
// Function:  void fillIntArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * out +dimension(3)+intent(out)
// Requested: c_native_*_out
// Match:     c_default
// start POI_fill_int_array
void POI_fill_int_array(int * out)
{
    // splicer begin function.fill_int_array
    fillIntArray(out);
    // splicer end function.fill_int_array
}
// end POI_fill_int_array

/**
 * Increment array in place using intent(INOUT).
 */
// ----------------------------------------
// Function:  void incrementIntArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * array +intent(inout)+rank(1)
// Requested: c_native_*_inout
// Match:     c_default
// ----------------------------------------
// Argument:  int sizein +implied(size(array))+intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// start POI_increment_int_array
void POI_increment_int_array(int * array, int sizein)
{
    // splicer begin function.increment_int_array
    incrementIntArray(array, sizein);
    // splicer end function.increment_int_array
}
// end POI_increment_int_array

// ----------------------------------------
// Function:  void acceptCharArrayIn
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  char * * names +intent(in)+rank(1)
// Exact:     c_char_**_in
// start POI_accept_char_array_in
void POI_accept_char_array_in(char **names)
{
    // splicer begin function.accept_char_array_in
    acceptCharArrayIn(names);
    // splicer end function.accept_char_array_in
}
// end POI_accept_char_array_in

// ----------------------------------------
// Function:  void acceptCharArrayIn
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  char * * names +intent(in)+len(Nnames)+rank(1)+size(Snames)
// Exact:     c_char_**_in_buf
// start POI_accept_char_array_in_bufferify
void POI_accept_char_array_in_bufferify(char *names, long Snames,
    int Nnames)
{
    // splicer begin function.accept_char_array_in_bufferify
    char **SHCXX_names = ShroudStrArrayAlloc(names, Snames, Nnames);
    acceptCharArrayIn(SHCXX_names);
    ShroudStrArrayFree(SHCXX_names, Snames);
    // splicer end function.accept_char_array_in_bufferify
}
// end POI_accept_char_array_in_bufferify

// ----------------------------------------
// Function:  void setGlobalInt
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int value +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// start POI_set_global_int
void POI_set_global_int(int value)
{
    // splicer begin function.set_global_int
    setGlobalInt(value);
    // splicer end function.set_global_int
}
// end POI_set_global_int

/**
 * Used to test values global_array.
 */
// ----------------------------------------
// Function:  int sumFixedArray
// Requested: c_native_scalar_result
// Match:     c_default
// start POI_sum_fixed_array
int POI_sum_fixed_array(void)
{
    // splicer begin function.sum_fixed_array
    int SHC_rv = sumFixedArray();
    return SHC_rv;
    // splicer end function.sum_fixed_array
}
// end POI_sum_fixed_array

// ----------------------------------------
// Function:  void getPtrToScalar
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * nitems +deref(pointer)+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// start POI_get_ptr_to_scalar
void POI_get_ptr_to_scalar(int * * nitems)
{
    // splicer begin function.get_ptr_to_scalar
    getPtrToScalar(nitems);
    // splicer end function.get_ptr_to_scalar
}
// end POI_get_ptr_to_scalar

// ----------------------------------------
// Function:  void getPtrToScalar
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * nitems +context(Dnitems)+deref(pointer)+intent(out)
// Exact:     c_native_**_out_buf
// start POI_get_ptr_to_scalar_bufferify
void POI_get_ptr_to_scalar_bufferify(POI_SHROUD_array *Dnitems)
{
    // splicer begin function.get_ptr_to_scalar_bufferify
    int *nitems;
    getPtrToScalar(&nitems);
    Dnitems->cxx.addr  = nitems;
    Dnitems->cxx.idtor = 0;
    Dnitems->addr.base = nitems;
    Dnitems->type = SH_TYPE_INT;
    Dnitems->elem_len = sizeof(int);
    Dnitems->rank = 0;
    Dnitems->size = 1;
    // splicer end function.get_ptr_to_scalar_bufferify
}
// end POI_get_ptr_to_scalar_bufferify

/**
 * Return a Fortran pointer to an array which is always the same length.
 */
// ----------------------------------------
// Function:  void getPtrToFixedArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * count +deref(pointer)+dimension(10)+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// start POI_get_ptr_to_fixed_array
void POI_get_ptr_to_fixed_array(int * * count)
{
    // splicer begin function.get_ptr_to_fixed_array
    getPtrToFixedArray(count);
    // splicer end function.get_ptr_to_fixed_array
}
// end POI_get_ptr_to_fixed_array

/**
 * Return a Fortran pointer to an array which is always the same length.
 */
// ----------------------------------------
// Function:  void getPtrToFixedArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * count +context(Dcount)+deref(pointer)+dimension(10)+intent(out)
// Exact:     c_native_**_out_buf
// start POI_get_ptr_to_fixed_array_bufferify
void POI_get_ptr_to_fixed_array_bufferify(POI_SHROUD_array *Dcount)
{
    // splicer begin function.get_ptr_to_fixed_array_bufferify
    int *count;
    getPtrToFixedArray(&count);
    Dcount->cxx.addr  = count;
    Dcount->cxx.idtor = 0;
    Dcount->addr.base = count;
    Dcount->type = SH_TYPE_INT;
    Dcount->elem_len = sizeof(int);
    Dcount->rank = 1;
    Dcount->shape[0] = 10;
    Dcount->size = Dcount->shape[0];
    // splicer end function.get_ptr_to_fixed_array_bufferify
}
// end POI_get_ptr_to_fixed_array_bufferify

/**
 * Return a Fortran pointer to an array which is the length of
 * the argument ncount.
 */
// ----------------------------------------
// Function:  void getPtrToDynamicArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * count +deref(pointer)+dimension(ncount)+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// ----------------------------------------
// Argument:  int * ncount +hidden+intent(out)
// Requested: c_native_*_out
// Match:     c_default
// start POI_get_ptr_to_dynamic_array
void POI_get_ptr_to_dynamic_array(int * * count, int * ncount)
{
    // splicer begin function.get_ptr_to_dynamic_array
    getPtrToDynamicArray(count, ncount);
    // splicer end function.get_ptr_to_dynamic_array
}
// end POI_get_ptr_to_dynamic_array

/**
 * Return a Fortran pointer to an array which is the length of
 * the argument ncount.
 */
// ----------------------------------------
// Function:  void getPtrToDynamicArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * count +context(Dcount)+deref(pointer)+dimension(ncount)+intent(out)
// Exact:     c_native_**_out_buf
// ----------------------------------------
// Argument:  int * ncount +hidden+intent(out)
// Requested: c_native_*_out_buf
// Match:     c_default
// start POI_get_ptr_to_dynamic_array_bufferify
void POI_get_ptr_to_dynamic_array_bufferify(POI_SHROUD_array *Dcount,
    int * ncount)
{
    // splicer begin function.get_ptr_to_dynamic_array_bufferify
    int *count;
    getPtrToDynamicArray(&count, ncount);
    Dcount->cxx.addr  = count;
    Dcount->cxx.idtor = 0;
    Dcount->addr.base = count;
    Dcount->type = SH_TYPE_INT;
    Dcount->elem_len = sizeof(int);
    Dcount->rank = 1;
    Dcount->shape[0] = *ncount;
    Dcount->size = Dcount->shape[0];
    // splicer end function.get_ptr_to_dynamic_array_bufferify
}
// end POI_get_ptr_to_dynamic_array_bufferify

/**
 * Return a Fortran pointer to an array which is the length
 * is computed by C++ function getLen.
 * getLen will be called from C/C++ to compute the shape.
 */
// ----------------------------------------
// Function:  void getPtrToFuncArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * count +deref(pointer)+dimension(getLen())+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// start POI_get_ptr_to_func_array
void POI_get_ptr_to_func_array(int * * count)
{
    // splicer begin function.get_ptr_to_func_array
    getPtrToFuncArray(count);
    // splicer end function.get_ptr_to_func_array
}
// end POI_get_ptr_to_func_array

/**
 * Return a Fortran pointer to an array which is the length
 * is computed by C++ function getLen.
 * getLen will be called from C/C++ to compute the shape.
 */
// ----------------------------------------
// Function:  void getPtrToFuncArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * count +context(Dcount)+deref(pointer)+dimension(getLen())+intent(out)
// Exact:     c_native_**_out_buf
// start POI_get_ptr_to_func_array_bufferify
void POI_get_ptr_to_func_array_bufferify(POI_SHROUD_array *Dcount)
{
    // splicer begin function.get_ptr_to_func_array_bufferify
    int *count;
    getPtrToFuncArray(&count);
    Dcount->cxx.addr  = count;
    Dcount->cxx.idtor = 0;
    Dcount->addr.base = count;
    Dcount->type = SH_TYPE_INT;
    Dcount->elem_len = sizeof(int);
    Dcount->rank = 1;
    Dcount->shape[0] = getLen();
    Dcount->size = Dcount->shape[0];
    // splicer end function.get_ptr_to_func_array_bufferify
}
// end POI_get_ptr_to_func_array_bufferify

// ----------------------------------------
// Function:  void getPtrToConstScalar
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const int * * nitems +deref(pointer)+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// start POI_get_ptr_to_const_scalar
void POI_get_ptr_to_const_scalar(const int * * nitems)
{
    // splicer begin function.get_ptr_to_const_scalar
    getPtrToConstScalar(nitems);
    // splicer end function.get_ptr_to_const_scalar
}
// end POI_get_ptr_to_const_scalar

// ----------------------------------------
// Function:  void getPtrToConstScalar
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const int * * nitems +context(Dnitems)+deref(pointer)+intent(out)
// Exact:     c_native_**_out_buf
// start POI_get_ptr_to_const_scalar_bufferify
void POI_get_ptr_to_const_scalar_bufferify(POI_SHROUD_array *Dnitems)
{
    // splicer begin function.get_ptr_to_const_scalar_bufferify
    const int *nitems;
    getPtrToConstScalar(&nitems);
    Dnitems->cxx.addr  = const_cast<int *>(nitems);
    Dnitems->cxx.idtor = 0;
    Dnitems->addr.base = nitems;
    Dnitems->type = SH_TYPE_INT;
    Dnitems->elem_len = sizeof(int);
    Dnitems->rank = 0;
    Dnitems->size = 1;
    // splicer end function.get_ptr_to_const_scalar_bufferify
}
// end POI_get_ptr_to_const_scalar_bufferify

// ----------------------------------------
// Function:  void getPtrToFixedConstArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const int * * count +deref(pointer)+dimension(10)+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// start POI_get_ptr_to_fixed_const_array
void POI_get_ptr_to_fixed_const_array(const int * * count)
{
    // splicer begin function.get_ptr_to_fixed_const_array
    getPtrToFixedConstArray(count);
    // splicer end function.get_ptr_to_fixed_const_array
}
// end POI_get_ptr_to_fixed_const_array

// ----------------------------------------
// Function:  void getPtrToFixedConstArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const int * * count +context(Dcount)+deref(pointer)+dimension(10)+intent(out)
// Exact:     c_native_**_out_buf
// start POI_get_ptr_to_fixed_const_array_bufferify
void POI_get_ptr_to_fixed_const_array_bufferify(
    POI_SHROUD_array *Dcount)
{
    // splicer begin function.get_ptr_to_fixed_const_array_bufferify
    const int *count;
    getPtrToFixedConstArray(&count);
    Dcount->cxx.addr  = const_cast<int *>(count);
    Dcount->cxx.idtor = 0;
    Dcount->addr.base = count;
    Dcount->type = SH_TYPE_INT;
    Dcount->elem_len = sizeof(int);
    Dcount->rank = 1;
    Dcount->shape[0] = 10;
    Dcount->size = Dcount->shape[0];
    // splicer end function.get_ptr_to_fixed_const_array_bufferify
}
// end POI_get_ptr_to_fixed_const_array_bufferify

// ----------------------------------------
// Function:  void getPtrToDynamicConstArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const int * * count +deref(pointer)+dimension(ncount)+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// ----------------------------------------
// Argument:  int * ncount +hidden+intent(out)
// Requested: c_native_*_out
// Match:     c_default
// start POI_get_ptr_to_dynamic_const_array
void POI_get_ptr_to_dynamic_const_array(const int * * count,
    int * ncount)
{
    // splicer begin function.get_ptr_to_dynamic_const_array
    getPtrToDynamicConstArray(count, ncount);
    // splicer end function.get_ptr_to_dynamic_const_array
}
// end POI_get_ptr_to_dynamic_const_array

// ----------------------------------------
// Function:  void getPtrToDynamicConstArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const int * * count +context(Dcount)+deref(pointer)+dimension(ncount)+intent(out)
// Exact:     c_native_**_out_buf
// ----------------------------------------
// Argument:  int * ncount +hidden+intent(out)
// Requested: c_native_*_out_buf
// Match:     c_default
// start POI_get_ptr_to_dynamic_const_array_bufferify
void POI_get_ptr_to_dynamic_const_array_bufferify(
    POI_SHROUD_array *Dcount, int * ncount)
{
    // splicer begin function.get_ptr_to_dynamic_const_array_bufferify
    const int *count;
    getPtrToDynamicConstArray(&count, ncount);
    Dcount->cxx.addr  = const_cast<int *>(count);
    Dcount->cxx.idtor = 0;
    Dcount->addr.base = count;
    Dcount->type = SH_TYPE_INT;
    Dcount->elem_len = sizeof(int);
    Dcount->rank = 1;
    Dcount->shape[0] = *ncount;
    Dcount->size = Dcount->shape[0];
    // splicer end function.get_ptr_to_dynamic_const_array_bufferify
}
// end POI_get_ptr_to_dynamic_const_array_bufferify

/**
 * Called directly via an interface in Fortran.
 */
// ----------------------------------------
// Function:  void getRawPtrToScalar
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * nitems +deref(raw)+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// start POI_get_raw_ptr_to_scalar
void POI_get_raw_ptr_to_scalar(int * * nitems)
{
    // splicer begin function.get_raw_ptr_to_scalar
    getRawPtrToScalar(nitems);
    // splicer end function.get_raw_ptr_to_scalar
}
// end POI_get_raw_ptr_to_scalar

/**
 * Called directly via an interface in Fortran.
 */
// ----------------------------------------
// Function:  void getRawPtrToScalar
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * nitems +context(Dnitems)+deref(raw)+intent(out)
// Exact:     c_native_**_out_buf
// start POI_get_raw_ptr_to_scalar_bufferify
void POI_get_raw_ptr_to_scalar_bufferify(POI_SHROUD_array *Dnitems)
{
    // splicer begin function.get_raw_ptr_to_scalar_bufferify
    int *nitems;
    getRawPtrToScalar(&nitems);
    Dnitems->cxx.addr  = nitems;
    Dnitems->cxx.idtor = 0;
    Dnitems->addr.base = nitems;
    Dnitems->type = SH_TYPE_INT;
    Dnitems->elem_len = sizeof(int);
    Dnitems->rank = 0;
    Dnitems->size = 1;
    // splicer end function.get_raw_ptr_to_scalar_bufferify
}
// end POI_get_raw_ptr_to_scalar_bufferify

/**
 * Return a type(C_PTR) to an array which is always the same length.
 * Called directly via an interface in Fortran.
 * # Uses +deref(raw) instead of +dimension(10) like getPtrToFixedArray.
 */
// ----------------------------------------
// Function:  void getRawPtrToFixedArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * count +deref(raw)+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// start POI_get_raw_ptr_to_fixed_array
void POI_get_raw_ptr_to_fixed_array(int * * count)
{
    // splicer begin function.get_raw_ptr_to_fixed_array
    getRawPtrToFixedArray(count);
    // splicer end function.get_raw_ptr_to_fixed_array
}
// end POI_get_raw_ptr_to_fixed_array

/**
 * Return a type(C_PTR) to an array which is always the same length.
 * Called directly via an interface in Fortran.
 * # Uses +deref(raw) instead of +dimension(10) like getPtrToFixedArray.
 */
// ----------------------------------------
// Function:  void getRawPtrToFixedArray
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int * * count +context(Dcount)+deref(raw)+intent(out)
// Exact:     c_native_**_out_buf
// start POI_get_raw_ptr_to_fixed_array_bufferify
void POI_get_raw_ptr_to_fixed_array_bufferify(POI_SHROUD_array *Dcount)
{
    // splicer begin function.get_raw_ptr_to_fixed_array_bufferify
    int *count;
    getRawPtrToFixedArray(&count);
    Dcount->cxx.addr  = count;
    Dcount->cxx.idtor = 0;
    Dcount->addr.base = count;
    Dcount->type = SH_TYPE_INT;
    Dcount->elem_len = sizeof(int);
    Dcount->rank = 0;
    Dcount->size = 1;
    // splicer end function.get_raw_ptr_to_fixed_array_bufferify
}
// end POI_get_raw_ptr_to_fixed_array_bufferify

// ----------------------------------------
// Function:  void * returnAddress1
// Requested: c_unknown_*_result
// Match:     c_default
// ----------------------------------------
// Argument:  int flag +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// start POI_return_address1
void * POI_return_address1(int flag)
{
    // splicer begin function.return_address1
    void * SHC_rv = returnAddress1(flag);
    return SHC_rv;
    // splicer end function.return_address1
}
// end POI_return_address1

// ----------------------------------------
// Function:  void * returnAddress2
// Requested: c_unknown_*_result
// Match:     c_default
// ----------------------------------------
// Argument:  int flag +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// start POI_return_address2
void * POI_return_address2(int flag)
{
    // splicer begin function.return_address2
    void * SHC_rv = returnAddress2(flag);
    return SHC_rv;
    // splicer end function.return_address2
}
// end POI_return_address2

// ----------------------------------------
// Function:  void fetchVoidPtr
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  void * * addr +intent(out)
// Requested: c_unknown_**_out
// Match:     c_default
// start POI_fetch_void_ptr
void POI_fetch_void_ptr(void * * addr)
{
    // splicer begin function.fetch_void_ptr
    fetchVoidPtr(addr);
    // splicer end function.fetch_void_ptr
}
// end POI_fetch_void_ptr

// ----------------------------------------
// Function:  int * returnIntPtrToScalar +deref(pointer)
// Requested: c_native_*_result
// Match:     c_default
// start POI_return_int_ptr_to_scalar
int * POI_return_int_ptr_to_scalar(void)
{
    // splicer begin function.return_int_ptr_to_scalar
    int * SHC_rv = returnIntPtrToScalar();
    return SHC_rv;
    // splicer end function.return_int_ptr_to_scalar
}
// end POI_return_int_ptr_to_scalar

// ----------------------------------------
// Function:  int * returnIntPtrToFixedArray +deref(pointer)+dimension(10)
// Requested: c_native_*_result
// Match:     c_default
// start POI_return_int_ptr_to_fixed_array
int * POI_return_int_ptr_to_fixed_array(void)
{
    // splicer begin function.return_int_ptr_to_fixed_array
    int * SHC_rv = returnIntPtrToFixedArray();
    return SHC_rv;
    // splicer end function.return_int_ptr_to_fixed_array
}
// end POI_return_int_ptr_to_fixed_array

// ----------------------------------------
// Function:  int * returnIntPtrToFixedArray +context(DSHC_rv)+deref(pointer)+dimension(10)
// Exact:     c_native_*_result_buf
// start POI_return_int_ptr_to_fixed_array_bufferify
int * POI_return_int_ptr_to_fixed_array_bufferify(
    POI_SHROUD_array *DSHC_rv)
{
    // splicer begin function.return_int_ptr_to_fixed_array_bufferify
    int * SHC_rv = returnIntPtrToFixedArray();
    DSHC_rv->cxx.addr  = SHC_rv;
    DSHC_rv->cxx.idtor = 0;
    DSHC_rv->addr.base = SHC_rv;
    DSHC_rv->type = SH_TYPE_INT;
    DSHC_rv->elem_len = sizeof(int);
    DSHC_rv->rank = 1;
    DSHC_rv->shape[0] = 10;
    DSHC_rv->size = DSHC_rv->shape[0];
    return SHC_rv;
    // splicer end function.return_int_ptr_to_fixed_array_bufferify
}
// end POI_return_int_ptr_to_fixed_array_bufferify

// ----------------------------------------
// Function:  const int * returnIntPtrToConstScalar +deref(pointer)
// Requested: c_native_*_result
// Match:     c_default
// start POI_return_int_ptr_to_const_scalar
const int * POI_return_int_ptr_to_const_scalar(void)
{
    // splicer begin function.return_int_ptr_to_const_scalar
    const int * SHC_rv = returnIntPtrToConstScalar();
    return SHC_rv;
    // splicer end function.return_int_ptr_to_const_scalar
}
// end POI_return_int_ptr_to_const_scalar

// ----------------------------------------
// Function:  const int * returnIntPtrToFixedConstArray +deref(pointer)+dimension(10)
// Requested: c_native_*_result
// Match:     c_default
// start POI_return_int_ptr_to_fixed_const_array
const int * POI_return_int_ptr_to_fixed_const_array(void)
{
    // splicer begin function.return_int_ptr_to_fixed_const_array
    const int * SHC_rv = returnIntPtrToFixedConstArray();
    return SHC_rv;
    // splicer end function.return_int_ptr_to_fixed_const_array
}
// end POI_return_int_ptr_to_fixed_const_array

// ----------------------------------------
// Function:  const int * returnIntPtrToFixedConstArray +context(DSHC_rv)+deref(pointer)+dimension(10)
// Exact:     c_native_*_result_buf
// start POI_return_int_ptr_to_fixed_const_array_bufferify
const int * POI_return_int_ptr_to_fixed_const_array_bufferify(
    POI_SHROUD_array *DSHC_rv)
{
    // splicer begin function.return_int_ptr_to_fixed_const_array_bufferify
    const int * SHC_rv = returnIntPtrToFixedConstArray();
    DSHC_rv->cxx.addr  = const_cast<int *>(SHC_rv);
    DSHC_rv->cxx.idtor = 0;
    DSHC_rv->addr.base = SHC_rv;
    DSHC_rv->type = SH_TYPE_INT;
    DSHC_rv->elem_len = sizeof(int);
    DSHC_rv->rank = 1;
    DSHC_rv->shape[0] = 10;
    DSHC_rv->size = DSHC_rv->shape[0];
    return SHC_rv;
    // splicer end function.return_int_ptr_to_fixed_const_array_bufferify
}
// end POI_return_int_ptr_to_fixed_const_array_bufferify

// ----------------------------------------
// Function:  int * returnIntScalar +deref(scalar)
// Requested: c_native_*_result
// Match:     c_default
// start POI_return_int_scalar
int POI_return_int_scalar(void)
{
    // splicer begin function.return_int_scalar
    int * SHC_rv = returnIntScalar();
    return *SHC_rv;
    // splicer end function.return_int_scalar
}
// end POI_return_int_scalar

// start release allocated memory
// Release library allocated memory.
void POI_SHROUD_memory_destructor(POI_SHROUD_capsule_data *cap)
{
    cap->addr = nullptr;
    cap->idtor = 0;  // avoid deleting again
}
// end release allocated memory

}  // extern "C"
