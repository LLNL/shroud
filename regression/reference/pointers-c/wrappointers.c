// wrappointers.c
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrappointers.h"

// cxx_header
#include "pointers.h"
// shroud
#include <stdlib.h>
#include <string.h>


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
static char **ShroudStrArrayAlloc(const char *src, int nsrc, int len)
{
   char **rv = malloc(sizeof(char *) * nsrc);
   const char *src0 = src;
   for(int i=0; i < nsrc; ++i) {
      int ntrim = ShroudLenTrim(src0, len);
      char *tgt = malloc(ntrim+1);
      memcpy(tgt, src0, ntrim);
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
       free(src[i]);
   }
   free(src);
}
// splicer begin C_definitions
// splicer end C_definitions

/**
 * Return strlen of the first index as a check.
 */
// ----------------------------------------
// Function:  int acceptCharArrayIn
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  char * * names +intent(in)+rank(1)
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_char_**_buf
// start POI_accept_char_array_in_bufferify
int POI_accept_char_array_in_bufferify(const char *names,
    size_t SHT_names_size, int SHT_names_len)
{
    // splicer begin function.accept_char_array_in_bufferify
    char **SHCXX_names = ShroudStrArrayAlloc(names, SHT_names_size,
        SHT_names_len);
    int SHC_rv = acceptCharArrayIn(SHCXX_names);
    ShroudStrArrayFree(SHCXX_names, SHT_names_size);
    return SHC_rv;
    // splicer end function.accept_char_array_in_bufferify
}
// end POI_accept_char_array_in_bufferify

// ----------------------------------------
// Function:  void getPtrToScalar
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int * * nitems +intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// start POI_get_ptr_to_scalar_bufferify
void POI_get_ptr_to_scalar_bufferify(POI_SHROUD_array *SHT_nitems_cdesc)
{
    // splicer begin function.get_ptr_to_scalar_bufferify
    int *nitems;
    getPtrToScalar(&nitems);
    SHT_nitems_cdesc->cxx.addr  = nitems;
    SHT_nitems_cdesc->cxx.idtor = 0;
    SHT_nitems_cdesc->addr.base = nitems;
    SHT_nitems_cdesc->type = SH_TYPE_INT;
    SHT_nitems_cdesc->elem_len = sizeof(int);
    SHT_nitems_cdesc->rank = 0;
    SHT_nitems_cdesc->size = 1;
    // splicer end function.get_ptr_to_scalar_bufferify
}
// end POI_get_ptr_to_scalar_bufferify

/**
 * Return a Fortran pointer to an array which is always the same length.
 */
// ----------------------------------------
// Function:  void getPtrToFixedArray
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int * * count +dimension(10)+intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// start POI_get_ptr_to_fixed_array_bufferify
void POI_get_ptr_to_fixed_array_bufferify(
    POI_SHROUD_array *SHT_count_cdesc)
{
    // splicer begin function.get_ptr_to_fixed_array_bufferify
    int *count;
    getPtrToFixedArray(&count);
    SHT_count_cdesc->cxx.addr  = count;
    SHT_count_cdesc->cxx.idtor = 0;
    SHT_count_cdesc->addr.base = count;
    SHT_count_cdesc->type = SH_TYPE_INT;
    SHT_count_cdesc->elem_len = sizeof(int);
    SHT_count_cdesc->rank = 1;
    SHT_count_cdesc->shape[0] = 10;
    SHT_count_cdesc->size = SHT_count_cdesc->shape[0];
    // splicer end function.get_ptr_to_fixed_array_bufferify
}
// end POI_get_ptr_to_fixed_array_bufferify

/**
 * Return a Fortran pointer to an array which is the length of
 * the argument ncount.
 */
// ----------------------------------------
// Function:  void getPtrToDynamicArray
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int * * count +dimension(ncount)+intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// ----------------------------------------
// Argument:  int * ncount +hidden+intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_*
// Match:     c_default
// start POI_get_ptr_to_dynamic_array_bufferify
void POI_get_ptr_to_dynamic_array_bufferify(
    POI_SHROUD_array *SHT_count_cdesc, int * ncount)
{
    // splicer begin function.get_ptr_to_dynamic_array_bufferify
    int *count;
    getPtrToDynamicArray(&count, ncount);
    SHT_count_cdesc->cxx.addr  = count;
    SHT_count_cdesc->cxx.idtor = 0;
    SHT_count_cdesc->addr.base = count;
    SHT_count_cdesc->type = SH_TYPE_INT;
    SHT_count_cdesc->elem_len = sizeof(int);
    SHT_count_cdesc->rank = 1;
    SHT_count_cdesc->shape[0] = *ncount;
    SHT_count_cdesc->size = SHT_count_cdesc->shape[0];
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
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int * * count +dimension(getLen())+intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// start POI_get_ptr_to_func_array_bufferify
void POI_get_ptr_to_func_array_bufferify(
    POI_SHROUD_array *SHT_count_cdesc)
{
    // splicer begin function.get_ptr_to_func_array_bufferify
    int *count;
    getPtrToFuncArray(&count);
    SHT_count_cdesc->cxx.addr  = count;
    SHT_count_cdesc->cxx.idtor = 0;
    SHT_count_cdesc->addr.base = count;
    SHT_count_cdesc->type = SH_TYPE_INT;
    SHT_count_cdesc->elem_len = sizeof(int);
    SHT_count_cdesc->rank = 1;
    SHT_count_cdesc->shape[0] = getLen();
    SHT_count_cdesc->size = SHT_count_cdesc->shape[0];
    // splicer end function.get_ptr_to_func_array_bufferify
}
// end POI_get_ptr_to_func_array_bufferify

// ----------------------------------------
// Function:  void getPtrToConstScalar
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const int * * nitems +intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// start POI_get_ptr_to_const_scalar_bufferify
void POI_get_ptr_to_const_scalar_bufferify(
    POI_SHROUD_array *SHT_nitems_cdesc)
{
    // splicer begin function.get_ptr_to_const_scalar_bufferify
    const int *nitems;
    getPtrToConstScalar(&nitems);
    SHT_nitems_cdesc->cxx.addr  = (int *) nitems;
    SHT_nitems_cdesc->cxx.idtor = 0;
    SHT_nitems_cdesc->addr.base = nitems;
    SHT_nitems_cdesc->type = SH_TYPE_INT;
    SHT_nitems_cdesc->elem_len = sizeof(int);
    SHT_nitems_cdesc->rank = 0;
    SHT_nitems_cdesc->size = 1;
    // splicer end function.get_ptr_to_const_scalar_bufferify
}
// end POI_get_ptr_to_const_scalar_bufferify

// ----------------------------------------
// Function:  void getPtrToFixedConstArray
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const int * * count +dimension(10)+intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// start POI_get_ptr_to_fixed_const_array_bufferify
void POI_get_ptr_to_fixed_const_array_bufferify(
    POI_SHROUD_array *SHT_count_cdesc)
{
    // splicer begin function.get_ptr_to_fixed_const_array_bufferify
    const int *count;
    getPtrToFixedConstArray(&count);
    SHT_count_cdesc->cxx.addr  = (int *) count;
    SHT_count_cdesc->cxx.idtor = 0;
    SHT_count_cdesc->addr.base = count;
    SHT_count_cdesc->type = SH_TYPE_INT;
    SHT_count_cdesc->elem_len = sizeof(int);
    SHT_count_cdesc->rank = 1;
    SHT_count_cdesc->shape[0] = 10;
    SHT_count_cdesc->size = SHT_count_cdesc->shape[0];
    // splicer end function.get_ptr_to_fixed_const_array_bufferify
}
// end POI_get_ptr_to_fixed_const_array_bufferify

// ----------------------------------------
// Function:  void getPtrToDynamicConstArray
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const int * * count +dimension(ncount)+intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// ----------------------------------------
// Argument:  int * ncount +hidden+intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_*
// Match:     c_default
// start POI_get_ptr_to_dynamic_const_array_bufferify
void POI_get_ptr_to_dynamic_const_array_bufferify(
    POI_SHROUD_array *SHT_count_cdesc, int * ncount)
{
    // splicer begin function.get_ptr_to_dynamic_const_array_bufferify
    const int *count;
    getPtrToDynamicConstArray(&count, ncount);
    SHT_count_cdesc->cxx.addr  = (int *) count;
    SHT_count_cdesc->cxx.idtor = 0;
    SHT_count_cdesc->addr.base = count;
    SHT_count_cdesc->type = SH_TYPE_INT;
    SHT_count_cdesc->elem_len = sizeof(int);
    SHT_count_cdesc->rank = 1;
    SHT_count_cdesc->shape[0] = *ncount;
    SHT_count_cdesc->size = SHT_count_cdesc->shape[0];
    // splicer end function.get_ptr_to_dynamic_const_array_bufferify
}
// end POI_get_ptr_to_dynamic_const_array_bufferify

// ----------------------------------------
// Function:  int * returnIntPtrToScalar
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_native_*_buf_pointer
// Match:     c_function_native_*_buf
// start POI_return_int_ptr_to_scalar_bufferify
void POI_return_int_ptr_to_scalar_bufferify(
    POI_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.return_int_ptr_to_scalar_bufferify
    int * SHC_rv = returnIntPtrToScalar();
    SHT_rv_cdesc->cxx.addr  = SHC_rv;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 0;
    SHT_rv_cdesc->size = 1;
    // splicer end function.return_int_ptr_to_scalar_bufferify
}
// end POI_return_int_ptr_to_scalar_bufferify

// ----------------------------------------
// Function:  int * returnIntPtrToFixedArray +dimension(10)
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_native_*_buf_pointer
// Match:     c_function_native_*_buf
// start POI_return_int_ptr_to_fixed_array_bufferify
void POI_return_int_ptr_to_fixed_array_bufferify(
    POI_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.return_int_ptr_to_fixed_array_bufferify
    int * SHC_rv = returnIntPtrToFixedArray();
    SHT_rv_cdesc->cxx.addr  = SHC_rv;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = 10;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.return_int_ptr_to_fixed_array_bufferify
}
// end POI_return_int_ptr_to_fixed_array_bufferify

// ----------------------------------------
// Function:  const int * returnIntPtrToConstScalar
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_native_*_buf_pointer
// Match:     c_function_native_*_buf
// start POI_return_int_ptr_to_const_scalar_bufferify
void POI_return_int_ptr_to_const_scalar_bufferify(
    POI_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.return_int_ptr_to_const_scalar_bufferify
    const int * SHC_rv = returnIntPtrToConstScalar();
    SHT_rv_cdesc->cxx.addr  = (int *) SHC_rv;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 0;
    SHT_rv_cdesc->size = 1;
    // splicer end function.return_int_ptr_to_const_scalar_bufferify
}
// end POI_return_int_ptr_to_const_scalar_bufferify

// ----------------------------------------
// Function:  const int * returnIntPtrToFixedConstArray +dimension(10)
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_native_*_buf_pointer
// Match:     c_function_native_*_buf
// start POI_return_int_ptr_to_fixed_const_array_bufferify
void POI_return_int_ptr_to_fixed_const_array_bufferify(
    POI_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.return_int_ptr_to_fixed_const_array_bufferify
    const int * SHC_rv = returnIntPtrToFixedConstArray();
    SHT_rv_cdesc->cxx.addr  = (int *) SHC_rv;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = 10;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.return_int_ptr_to_fixed_const_array_bufferify
}
// end POI_return_int_ptr_to_fixed_const_array_bufferify

// ----------------------------------------
// Function:  int * returnIntScalar +deref(scalar)
// Attrs:     +deref(scalar)+intent(function)
// Exact:     c_function_native_*_scalar
// start POI_return_int_scalar
int POI_return_int_scalar(void)
{
    // splicer begin function.return_int_scalar
    int * SHC_rv = returnIntScalar();
    return *SHC_rv;
    // splicer end function.return_int_scalar
}
// end POI_return_int_scalar
