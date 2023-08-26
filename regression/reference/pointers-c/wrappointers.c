// wrappointers.c
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "pointers.h"
// shroud
#include <stdlib.h>
#include <string.h>
#include "wrappointers.h"


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
// Exact:     c_function_native_scalar
// ----------------------------------------
// Argument:  char * * names +intent(in)+rank(1)
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_char_**_buf
// start POI_acceptCharArrayIn_bufferify
int POI_acceptCharArrayIn_bufferify(const char *names,
    size_t SHT_names_size, int SHT_names_len)
{
    // splicer begin function.acceptCharArrayIn_bufferify
    char **SHCXX_names = ShroudStrArrayAlloc(names, SHT_names_size,
        SHT_names_len);
    int SHC_rv = acceptCharArrayIn(SHCXX_names);
    ShroudStrArrayFree(SHCXX_names, SHT_names_size);
    return SHC_rv;
    // splicer end function.acceptCharArrayIn_bufferify
}
// end POI_acceptCharArrayIn_bufferify

// ----------------------------------------
// Function:  void getPtrToScalar
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int * * nitems +intent(out)
// Attrs:     +api(cdesc)+deref(pointer)+intent(out)
// Requested: c_out_native_**_cdesc_pointer
// Match:     c_out_native_**_cdesc
// start POI_getPtrToScalar_bufferify
void POI_getPtrToScalar_bufferify(POI_SHROUD_array *SHT_nitems_cdesc)
{
    // splicer begin function.getPtrToScalar_bufferify
    int *nitems;
    getPtrToScalar(&nitems);
    SHT_nitems_cdesc->cxx.addr  = nitems;
    SHT_nitems_cdesc->cxx.idtor = 0;
    SHT_nitems_cdesc->addr.base = nitems;
    SHT_nitems_cdesc->type = SH_TYPE_INT;
    SHT_nitems_cdesc->elem_len = sizeof(int);
    SHT_nitems_cdesc->rank = 0;
    SHT_nitems_cdesc->size = 1;
    // splicer end function.getPtrToScalar_bufferify
}
// end POI_getPtrToScalar_bufferify

/**
 * Return a Fortran pointer to an array which is always the same length.
 */
// ----------------------------------------
// Function:  void getPtrToFixedArray
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int * * count +dimension(10)+intent(out)
// Attrs:     +api(cdesc)+deref(pointer)+intent(out)
// Requested: c_out_native_**_cdesc_pointer
// Match:     c_out_native_**_cdesc
// start POI_getPtrToFixedArray_bufferify
void POI_getPtrToFixedArray_bufferify(POI_SHROUD_array *SHT_count_cdesc)
{
    // splicer begin function.getPtrToFixedArray_bufferify
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
    // splicer end function.getPtrToFixedArray_bufferify
}
// end POI_getPtrToFixedArray_bufferify

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
// Attrs:     +api(cdesc)+deref(pointer)+intent(out)
// Requested: c_out_native_**_cdesc_pointer
// Match:     c_out_native_**_cdesc
// ----------------------------------------
// Argument:  int * ncount +hidden+intent(out)
// Attrs:     +intent(out)
// Exact:     c_out_native_*_hidden
// start POI_getPtrToDynamicArray_bufferify
void POI_getPtrToDynamicArray_bufferify(
    POI_SHROUD_array *SHT_count_cdesc)
{
    // splicer begin function.getPtrToDynamicArray_bufferify
    int *count;
    int ncount;
    getPtrToDynamicArray(&count, &ncount);
    SHT_count_cdesc->cxx.addr  = count;
    SHT_count_cdesc->cxx.idtor = 0;
    SHT_count_cdesc->addr.base = count;
    SHT_count_cdesc->type = SH_TYPE_INT;
    SHT_count_cdesc->elem_len = sizeof(int);
    SHT_count_cdesc->rank = 1;
    SHT_count_cdesc->shape[0] = ncount;
    SHT_count_cdesc->size = SHT_count_cdesc->shape[0];
    // splicer end function.getPtrToDynamicArray_bufferify
}
// end POI_getPtrToDynamicArray_bufferify

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
// Attrs:     +api(cdesc)+deref(pointer)+intent(out)
// Requested: c_out_native_**_cdesc_pointer
// Match:     c_out_native_**_cdesc
// start POI_getPtrToFuncArray_bufferify
void POI_getPtrToFuncArray_bufferify(POI_SHROUD_array *SHT_count_cdesc)
{
    // splicer begin function.getPtrToFuncArray_bufferify
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
    // splicer end function.getPtrToFuncArray_bufferify
}
// end POI_getPtrToFuncArray_bufferify

// ----------------------------------------
// Function:  void getPtrToConstScalar
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const int * * nitems +intent(out)
// Attrs:     +api(cdesc)+deref(pointer)+intent(out)
// Requested: c_out_native_**_cdesc_pointer
// Match:     c_out_native_**_cdesc
// start POI_getPtrToConstScalar_bufferify
void POI_getPtrToConstScalar_bufferify(
    POI_SHROUD_array *SHT_nitems_cdesc)
{
    // splicer begin function.getPtrToConstScalar_bufferify
    const int *nitems;
    getPtrToConstScalar(&nitems);
    SHT_nitems_cdesc->cxx.addr  = (int *) nitems;
    SHT_nitems_cdesc->cxx.idtor = 0;
    SHT_nitems_cdesc->addr.base = nitems;
    SHT_nitems_cdesc->type = SH_TYPE_INT;
    SHT_nitems_cdesc->elem_len = sizeof(int);
    SHT_nitems_cdesc->rank = 0;
    SHT_nitems_cdesc->size = 1;
    // splicer end function.getPtrToConstScalar_bufferify
}
// end POI_getPtrToConstScalar_bufferify

// ----------------------------------------
// Function:  void getPtrToFixedConstArray
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const int * * count +dimension(10)+intent(out)
// Attrs:     +api(cdesc)+deref(pointer)+intent(out)
// Requested: c_out_native_**_cdesc_pointer
// Match:     c_out_native_**_cdesc
// start POI_getPtrToFixedConstArray_bufferify
void POI_getPtrToFixedConstArray_bufferify(
    POI_SHROUD_array *SHT_count_cdesc)
{
    // splicer begin function.getPtrToFixedConstArray_bufferify
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
    // splicer end function.getPtrToFixedConstArray_bufferify
}
// end POI_getPtrToFixedConstArray_bufferify

// ----------------------------------------
// Function:  void getPtrToDynamicConstArray
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const int * * count +dimension(ncount)+intent(out)
// Attrs:     +api(cdesc)+deref(pointer)+intent(out)
// Requested: c_out_native_**_cdesc_pointer
// Match:     c_out_native_**_cdesc
// ----------------------------------------
// Argument:  int * ncount +hidden+intent(out)
// Attrs:     +intent(out)
// Exact:     c_out_native_*_hidden
// start POI_getPtrToDynamicConstArray_bufferify
void POI_getPtrToDynamicConstArray_bufferify(
    POI_SHROUD_array *SHT_count_cdesc)
{
    // splicer begin function.getPtrToDynamicConstArray_bufferify
    const int *count;
    int ncount;
    getPtrToDynamicConstArray(&count, &ncount);
    SHT_count_cdesc->cxx.addr  = (int *) count;
    SHT_count_cdesc->cxx.idtor = 0;
    SHT_count_cdesc->addr.base = count;
    SHT_count_cdesc->type = SH_TYPE_INT;
    SHT_count_cdesc->elem_len = sizeof(int);
    SHT_count_cdesc->rank = 1;
    SHT_count_cdesc->shape[0] = ncount;
    SHT_count_cdesc->size = SHT_count_cdesc->shape[0];
    // splicer end function.getPtrToDynamicConstArray_bufferify
}
// end POI_getPtrToDynamicConstArray_bufferify

/**
 * Return a Fortran pointer to an array which is always the same length.
 */
// ----------------------------------------
// Function:  void getAllocToFixedArray
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int * * count +deref(allocatable)+dimension(10)+intent(out)
// Attrs:     +api(cdesc)+deref(allocatable)+intent(out)
// Requested: c_out_native_**_cdesc_allocatable
// Match:     c_out_native_**_cdesc
// start POI_getAllocToFixedArray_bufferify
void POI_getAllocToFixedArray_bufferify(
    POI_SHROUD_array *SHT_count_cdesc)
{
    // splicer begin function.getAllocToFixedArray_bufferify
    int *count;
    getAllocToFixedArray(&count);
    SHT_count_cdesc->cxx.addr  = count;
    SHT_count_cdesc->cxx.idtor = 0;
    SHT_count_cdesc->addr.base = count;
    SHT_count_cdesc->type = SH_TYPE_INT;
    SHT_count_cdesc->elem_len = sizeof(int);
    SHT_count_cdesc->rank = 1;
    SHT_count_cdesc->shape[0] = 10;
    SHT_count_cdesc->size = SHT_count_cdesc->shape[0];
    // splicer end function.getAllocToFixedArray_bufferify
}
// end POI_getAllocToFixedArray_bufferify

// ----------------------------------------
// Function:  int * returnIntPtrToFixedArray +dimension(10)
// Attrs:     +api(cdesc)+deref(pointer)+intent(function)
// Exact:     c_function_native_*_cdesc_pointer
// start POI_returnIntPtrToFixedArray_bufferify
void POI_returnIntPtrToFixedArray_bufferify(
    POI_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.returnIntPtrToFixedArray_bufferify
    int * SHC_rv = returnIntPtrToFixedArray();
    SHT_rv_cdesc->cxx.addr  = SHC_rv;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = 10;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.returnIntPtrToFixedArray_bufferify
}
// end POI_returnIntPtrToFixedArray_bufferify

// ----------------------------------------
// Function:  const int * returnIntPtrToFixedConstArray +dimension(10)
// Attrs:     +api(cdesc)+deref(pointer)+intent(function)
// Exact:     c_function_native_*_cdesc_pointer
// start POI_returnIntPtrToFixedConstArray_bufferify
void POI_returnIntPtrToFixedConstArray_bufferify(
    POI_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.returnIntPtrToFixedConstArray_bufferify
    const int * SHC_rv = returnIntPtrToFixedConstArray();
    SHT_rv_cdesc->cxx.addr  = (int *) SHC_rv;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = 10;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.returnIntPtrToFixedConstArray_bufferify
}
// end POI_returnIntPtrToFixedConstArray_bufferify

// ----------------------------------------
// Function:  int * returnIntScalar +deref(scalar)
// Attrs:     +deref(scalar)+intent(function)
// Exact:     c_function_native_*_scalar
// start POI_returnIntScalar
int POI_returnIntScalar(void)
{
    // splicer begin function.returnIntScalar
    int * SHC_rv = returnIntScalar();
    return *SHC_rv;
    // splicer end function.returnIntScalar
}
// end POI_returnIntScalar

// ----------------------------------------
// Function:  int * returnIntAllocToFixedArray +deref(allocatable)+dimension(10)
// Attrs:     +api(cdesc)+deref(allocatable)+intent(function)
// Exact:     c_function_native_*_cdesc_allocatable
// start POI_returnIntAllocToFixedArray_bufferify
void POI_returnIntAllocToFixedArray_bufferify(
    POI_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.returnIntAllocToFixedArray_bufferify
    int * SHC_rv = returnIntAllocToFixedArray();
    SHT_rv_cdesc->cxx.addr  = SHC_rv;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = 10;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.returnIntAllocToFixedArray_bufferify
}
// end POI_returnIntAllocToFixedArray_bufferify
