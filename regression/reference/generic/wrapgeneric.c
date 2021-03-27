// wrapgeneric.c
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapgeneric.h"

// cxx_header
#include "generic.h"
#include "helper.h"

// splicer begin C_definitions
// splicer end C_definitions

/**
 * \brief scalar or array argument using assumed rank
 *
 */
// ----------------------------------------
// Function:  int SumValues
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const int * values +rank(0)
// Attrs:     +assumed-rank+intent(in)
// Requested: c_in_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  int nvalues +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int GEN_sum_values_0d(const int * values, int nvalues)
{
    // splicer begin function.sum_values_0d
    int SHC_rv = SumValues(values, nvalues);
    return SHC_rv;
    // splicer end function.sum_values_0d
}

/**
 * \brief scalar or array argument using assumed rank
 *
 */
// ----------------------------------------
// Function:  int SumValues
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const int * values +rank(1)
// Attrs:     +assumed-rank+intent(in)
// Requested: c_in_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  int nvalues +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int GEN_sum_values_1d(const int * values, int nvalues)
{
    // splicer begin function.sum_values_1d
    int SHC_rv = SumValues(values, nvalues);
    return SHC_rv;
    // splicer end function.sum_values_1d
}

/**
 * \brief scalar or array argument using assumed rank
 *
 */
// ----------------------------------------
// Function:  int SumValues
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const int * values +rank(2)
// Attrs:     +assumed-rank+intent(in)
// Requested: c_in_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  int nvalues +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int GEN_sum_values_2d(const int * values, int nvalues)
{
    // splicer begin function.sum_values_2d
    int SHC_rv = SumValues(values, nvalues);
    return SHC_rv;
    // splicer end function.sum_values_2d
}

/**
 * Broadcast if nfrom == 1
 * Copy if nfrom == nto
 */
// ----------------------------------------
// Function:  void AssignValues
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const int * from
// Attrs:     +intent(in)
// Requested: c_in_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  int nfrom +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int * to
// Attrs:     +intent(inout)
// Requested: c_inout_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  int nto +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void GEN_assign_values_scalar(const int * from, int nfrom, int * to,
    int nto)
{
    // splicer begin function.assign_values_scalar
    AssignValues(from, nfrom, to, nto);
    // splicer end function.assign_values_scalar
}

/**
 * Broadcast if nfrom == 1
 * Copy if nfrom == nto
 */
// ----------------------------------------
// Function:  void AssignValues
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const int * from
// Attrs:     +intent(in)
// Requested: c_in_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  int nfrom +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int * to +rank(1)
// Attrs:     +intent(inout)
// Requested: c_inout_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  int nto +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void GEN_assign_values_broadcast(const int * from, int nfrom, int * to,
    int nto)
{
    // splicer begin function.assign_values_broadcast
    AssignValues(from, nfrom, to, nto);
    // splicer end function.assign_values_broadcast
}

/**
 * Broadcast if nfrom == 1
 * Copy if nfrom == nto
 */
// ----------------------------------------
// Function:  void AssignValues
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const int * from +rank(1)
// Attrs:     +intent(in)
// Requested: c_in_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  int nfrom +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int * to +rank(1)
// Attrs:     +intent(inout)
// Requested: c_inout_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  int nto +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void GEN_assign_values_copy(const int * from, int nfrom, int * to,
    int nto)
{
    // splicer begin function.assign_values_copy
    AssignValues(from, nfrom, to, nto);
    // splicer end function.assign_values_copy
}

#if 1
// ----------------------------------------
// Function:  void SavePointer
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  float * addr +deref(raw)+intent(in)+rank(1)
// Attrs:     +deref(raw)+intent(in)
// Requested: c_in_native_*_raw
// Match:     c_default
// ----------------------------------------
// Argument:  int type +implied(T_FLOAT)+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  size_t size +implied(size(addr))+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void GEN_save_pointer_float1d(float * addr, int type, size_t size)
{
    // splicer begin function.save_pointer_float1d
    SavePointer(addr, type, size);
    // splicer end function.save_pointer_float1d
}
#endif  // if 1

#if 1
// ----------------------------------------
// Function:  void SavePointer
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  float * addr +deref(raw)+intent(in)+rank(2)
// Attrs:     +deref(raw)+intent(in)
// Requested: c_in_native_*_raw
// Match:     c_default
// ----------------------------------------
// Argument:  int type +implied(T_FLOAT)+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  size_t size +implied(size(addr))+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void GEN_save_pointer_float2d(float * addr, int type, size_t size)
{
    // splicer begin function.save_pointer_float2d
    SavePointer(addr, type, size);
    // splicer end function.save_pointer_float2d
}
#endif  // if 1

// ----------------------------------------
// Function:  void SavePointer2
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  void * addr +value
// Attrs:     +intent(in)
// Requested: c_in_void_*
// Match:     c_default
// ----------------------------------------
// Argument:  int type +implied(type(addr))+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  size_t size +implied(size(addr))+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void GEN_save_pointer2(void * addr, int type, size_t size)
{
    // splicer begin function.save_pointer2
    // Test adding a blank line below.

    type = convert_type(type);
    SavePointer2(addr, type, size);
    // splicer end function.save_pointer2
}

// ----------------------------------------
// Function:  void SavePointer2
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  float * addr +deref(raw)+intent(in)+rank(1)
// Attrs:     +deref(raw)+intent(in)
// Requested: c_in_native_*_raw
// Match:     c_default
// ----------------------------------------
// Argument:  int type +implied(type(addr))+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  size_t size +implied(size(addr))+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void GEN_save_pointer2_float1d(float * addr, int type, size_t size)
{
    // splicer begin function.save_pointer2_float1d
    // Test adding a blank line below.

    type = convert_type(type);
    SavePointer2(addr, type, size);
    // splicer end function.save_pointer2_float1d
}

// ----------------------------------------
// Function:  void SavePointer2
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  float * addr +deref(raw)+intent(in)+rank(2)
// Attrs:     +deref(raw)+intent(in)
// Requested: c_in_native_*_raw
// Match:     c_default
// ----------------------------------------
// Argument:  int type +implied(type(addr))+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  size_t size +implied(size(addr))+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void GEN_save_pointer2_float2d(float * addr, int type, size_t size)
{
    // splicer begin function.save_pointer2_float2d
    // Test adding a blank line below.

    type = convert_type(type);
    SavePointer2(addr, type, size);
    // splicer end function.save_pointer2_float2d
}

#if 0
// ----------------------------------------
// Function:  void GetPointerAsPointer
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  float * * addr +deref(pointer)+intent(out)+rank(1)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// ----------------------------------------
// Argument:  int * type +hidden+intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  size_t * size +hidden+intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_*
// Match:     c_default
void GEN_get_pointer_as_pointer_float1d_bufferify(
    GEN_SHROUD_array *SHT_addr_cdesc, int * type, size_t * size)
{
    // splicer begin function.get_pointer_as_pointer_float1d_bufferify
    float *addr;
    GetPointerAsPointer(&addr, type, size);
    SHT_addr_cdesc->cxx.addr  = addr;
    SHT_addr_cdesc->cxx.idtor = 0;
    SHT_addr_cdesc->addr.base = addr;
    SHT_addr_cdesc->type = SH_TYPE_FLOAT;
    SHT_addr_cdesc->elem_len = sizeof(float);
    SHT_addr_cdesc->rank = 0;
    SHT_addr_cdesc->size = 1;
    // splicer end function.get_pointer_as_pointer_float1d_bufferify
}
#endif  // if 0

#if 0
// ----------------------------------------
// Function:  void GetPointerAsPointer
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  float * * addr +deref(pointer)+intent(out)+rank(2)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// ----------------------------------------
// Argument:  int * type +hidden+intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_*
// Match:     c_default
// ----------------------------------------
// Argument:  size_t * size +hidden+intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_*
// Match:     c_default
void GEN_get_pointer_as_pointer_float2d_bufferify(
    GEN_SHROUD_array *SHT_addr_cdesc, int * type, size_t * size)
{
    // splicer begin function.get_pointer_as_pointer_float2d_bufferify
    float *addr;
    GetPointerAsPointer(&addr, type, size);
    SHT_addr_cdesc->cxx.addr  = addr;
    SHT_addr_cdesc->cxx.idtor = 0;
    SHT_addr_cdesc->addr.base = addr;
    SHT_addr_cdesc->type = SH_TYPE_FLOAT;
    SHT_addr_cdesc->elem_len = sizeof(float);
    SHT_addr_cdesc->rank = 0;
    SHT_addr_cdesc->size = 1;
    // splicer end function.get_pointer_as_pointer_float2d_bufferify
}
#endif  // if 0
