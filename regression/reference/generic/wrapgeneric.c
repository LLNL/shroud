// wrapgeneric.c
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "generic.h"
#include "helper.h"
// shroud
#include "wrapgeneric.h"

// splicer begin C_definitions
// splicer end C_definitions

/**
 * \brief scalar or array argument using assumed rank
 *
 */
// Generated by fortran_generic
// ----------------------------------------
// Function:  int SumValues
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
// ----------------------------------------
// Argument:  const int * values +rank(0)
// Attrs:     +intent(in)
// Statement: f_in_native_*
// ----------------------------------------
// Argument:  int nvalues +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
int GEN_SumValues_0d(const int * values, int nvalues)
{
    // splicer begin function.SumValues_0d
    int SHC_rv = SumValues(values, nvalues);
    return SHC_rv;
    // splicer end function.SumValues_0d
}

/**
 * \brief scalar or array argument using assumed rank
 *
 */
// Generated by fortran_generic
// ----------------------------------------
// Function:  int SumValues
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
// ----------------------------------------
// Argument:  const int * values +rank(1)
// Attrs:     +intent(in)
// Statement: f_in_native_*
// ----------------------------------------
// Argument:  int nvalues +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
int GEN_SumValues_1d(const int * values, int nvalues)
{
    // splicer begin function.SumValues_1d
    int SHC_rv = SumValues(values, nvalues);
    return SHC_rv;
    // splicer end function.SumValues_1d
}

/**
 * \brief scalar or array argument using assumed rank
 *
 */
// Generated by fortran_generic
// ----------------------------------------
// Function:  int SumValues
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
// ----------------------------------------
// Argument:  const int * values +rank(2)
// Attrs:     +intent(in)
// Statement: f_in_native_*
// ----------------------------------------
// Argument:  int nvalues +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
int GEN_SumValues_2d(const int * values, int nvalues)
{
    // splicer begin function.SumValues_2d
    int SHC_rv = SumValues(values, nvalues);
    return SHC_rv;
    // splicer end function.SumValues_2d
}

/**
 * Broadcast if nfrom == 1
 * Copy if nfrom == nto
 */
// Generated by fortran_generic
// ----------------------------------------
// Function:  void AssignValues
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  const int * from
// Attrs:     +intent(in)
// Statement: f_in_native_*
// ----------------------------------------
// Argument:  int nfrom +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  int * to
// Attrs:     +intent(inout)
// Statement: f_inout_native_*
// ----------------------------------------
// Argument:  int nto +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void GEN_AssignValues_scalar(const int * from, int nfrom, int * to,
    int nto)
{
    // splicer begin function.AssignValues_scalar
    AssignValues(from, nfrom, to, nto);
    // splicer end function.AssignValues_scalar
}

/**
 * Broadcast if nfrom == 1
 * Copy if nfrom == nto
 */
// Generated by fortran_generic
// ----------------------------------------
// Function:  void AssignValues
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  const int * from
// Attrs:     +intent(in)
// Statement: f_in_native_*
// ----------------------------------------
// Argument:  int nfrom +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  int * to +rank(1)
// Attrs:     +intent(inout)
// Statement: f_inout_native_*
// ----------------------------------------
// Argument:  int nto +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void GEN_AssignValues_broadcast(const int * from, int nfrom, int * to,
    int nto)
{
    // splicer begin function.AssignValues_broadcast
    AssignValues(from, nfrom, to, nto);
    // splicer end function.AssignValues_broadcast
}

/**
 * Broadcast if nfrom == 1
 * Copy if nfrom == nto
 */
// Generated by fortran_generic
// ----------------------------------------
// Function:  void AssignValues
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  const int * from +rank(1)
// Attrs:     +intent(in)
// Statement: f_in_native_*
// ----------------------------------------
// Argument:  int nfrom +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  int * to +rank(1)
// Attrs:     +intent(inout)
// Statement: f_inout_native_*
// ----------------------------------------
// Argument:  int nto +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void GEN_AssignValues_copy(const int * from, int nfrom, int * to,
    int nto)
{
    // splicer begin function.AssignValues_copy
    AssignValues(from, nfrom, to, nto);
    // splicer end function.AssignValues_copy
}

#if 1
// Generated by fortran_generic
// ----------------------------------------
// Function:  void SavePointer
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  float * addr +intent(in)+rank(1)
// Attrs:     +intent(in)
// Statement: f_in_native_*
// ----------------------------------------
// Argument:  int type +implied(T_FLOAT)+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  size_t size +implied(size(addr))+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void GEN_SavePointer_float1d(float * addr, int type, size_t size)
{
    // splicer begin function.SavePointer_float1d
    SavePointer(addr, type, size);
    // splicer end function.SavePointer_float1d
}
#endif  // if 1

#if 1
// Generated by fortran_generic
// ----------------------------------------
// Function:  void SavePointer
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  float * addr +intent(in)+rank(2)
// Attrs:     +intent(in)
// Statement: f_in_native_*
// ----------------------------------------
// Argument:  int type +implied(T_FLOAT)+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  size_t size +implied(size(addr))+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void GEN_SavePointer_float2d(float * addr, int type, size_t size)
{
    // splicer begin function.SavePointer_float2d
    SavePointer(addr, type, size);
    // splicer end function.SavePointer_float2d
}
#endif  // if 1

// ----------------------------------------
// Function:  void SavePointer2
// Attrs:     +intent(subroutine)
// Statement: c_subroutine
// ----------------------------------------
// Argument:  void * addr +value
// Attrs:     +intent(in)
// Statement: f_in_void_*
// ----------------------------------------
// Argument:  int type +implied(type(addr))+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  size_t size +implied(size(addr))+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void GEN_SavePointer2(void * addr, int type, size_t size)
{
    // splicer begin function.SavePointer2
    // Test adding a blank line below.

    type = convert_type(type);
    SavePointer2(addr, type, size);
    // splicer end function.SavePointer2
}

// Generated by fortran_generic
// ----------------------------------------
// Function:  void SavePointer2
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  float * addr +intent(in)+rank(1)
// Attrs:     +intent(in)
// Statement: f_in_native_*
// ----------------------------------------
// Argument:  int type +implied(type(addr))+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  size_t size +implied(size(addr))+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void GEN_SavePointer2_float1d(float * addr, int type, size_t size)
{
    // splicer begin function.SavePointer2_float1d
    // Test adding a blank line below.

    type = convert_type(type);
    SavePointer2(addr, type, size);
    // splicer end function.SavePointer2_float1d
}

// Generated by fortran_generic
// ----------------------------------------
// Function:  void SavePointer2
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  float * addr +intent(in)+rank(2)
// Attrs:     +intent(in)
// Statement: f_in_native_*
// ----------------------------------------
// Argument:  int type +implied(type(addr))+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  size_t size +implied(size(addr))+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void GEN_SavePointer2_float2d(float * addr, int type, size_t size)
{
    // splicer begin function.SavePointer2_float2d
    // Test adding a blank line below.

    type = convert_type(type);
    SavePointer2(addr, type, size);
    // splicer end function.SavePointer2_float2d
}

#if 0
// Generated by fortran_generic - arg_to_buffer
// ----------------------------------------
// Function:  void GetPointerAsPointer
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  float * * addr +deref(pointer)+intent(out)+rank(1)
// Attrs:     +api(cdesc)+deref(pointer)+intent(out)
// Statement: f_out_native_**_cdesc_pointer
// ----------------------------------------
// Argument:  int * type +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*_hidden
// ----------------------------------------
// Argument:  size_t * size +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*_hidden
void GEN_GetPointerAsPointer_float1d_bufferify(
    GEN_SHROUD_array *SHT_addr_cdesc)
{
    // splicer begin function.GetPointerAsPointer_float1d_bufferify
    float *addr;
    int type;
    size_t size;
    GetPointerAsPointer(&addr, &type, &size);
    SHT_addr_cdesc->base_addr = addr;
    SHT_addr_cdesc->type = SH_TYPE_FLOAT;
    SHT_addr_cdesc->elem_len = sizeof(float);
    SHT_addr_cdesc->rank = 0;
    SHT_addr_cdesc->size = 1;
    // splicer end function.GetPointerAsPointer_float1d_bufferify
}
#endif  // if 0

#if 0
// Generated by fortran_generic - arg_to_buffer
// ----------------------------------------
// Function:  void GetPointerAsPointer
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  float * * addr +deref(pointer)+intent(out)+rank(2)
// Attrs:     +api(cdesc)+deref(pointer)+intent(out)
// Statement: f_out_native_**_cdesc_pointer
// ----------------------------------------
// Argument:  int * type +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*_hidden
// ----------------------------------------
// Argument:  size_t * size +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*_hidden
void GEN_GetPointerAsPointer_float2d_bufferify(
    GEN_SHROUD_array *SHT_addr_cdesc)
{
    // splicer begin function.GetPointerAsPointer_float2d_bufferify
    float *addr;
    int type;
    size_t size;
    GetPointerAsPointer(&addr, &type, &size);
    SHT_addr_cdesc->base_addr = addr;
    SHT_addr_cdesc->type = SH_TYPE_FLOAT;
    SHT_addr_cdesc->elem_len = sizeof(float);
    SHT_addr_cdesc->rank = 0;
    SHT_addr_cdesc->size = 1;
    // splicer end function.GetPointerAsPointer_float2d_bufferify
}
#endif  // if 0

// ----------------------------------------
// Function:  StructAsClass * CreateStructAsClass
// Attrs:     +api(capptr)+intent(function)
// Statement: c_function_shadow_*_capptr
GEN_StructAsClass * GEN_CreateStructAsClass(GEN_StructAsClass * SHC_rv)
{
    // splicer begin function.CreateStructAsClass
    StructAsClass * SHCXX_rv = CreateStructAsClass();
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end function.CreateStructAsClass
}

// ----------------------------------------
// Function:  long UpdateStructAsClass
// Attrs:     +intent(function)
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  StructAsClass * arg
// Attrs:     +intent(inout)
// Statement: f_inout_shadow_*
// ----------------------------------------
// Argument:  long inew +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
long GEN_UpdateStructAsClass(GEN_StructAsClass * arg, long inew)
{
    // splicer begin function.UpdateStructAsClass
    StructAsClass * SHCXX_arg = (StructAsClass *) arg->addr;
    long SHC_rv = UpdateStructAsClass(SHCXX_arg, inew);
    return SHC_rv;
    // splicer end function.UpdateStructAsClass
}

// Generated by fortran_generic
// ----------------------------------------
// Function:  long UpdateStructAsClass
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
// ----------------------------------------
// Argument:  StructAsClass * arg
// Attrs:     +intent(inout)
// Statement: f_inout_shadow_*
// ----------------------------------------
// Argument:  int inew +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
long GEN_UpdateStructAsClass_int(GEN_StructAsClass * arg, int inew)
{
    // splicer begin function.UpdateStructAsClass_int
    StructAsClass * SHCXX_arg = (StructAsClass *) arg->addr;
    long SHC_rv = UpdateStructAsClass(SHCXX_arg, inew);
    return SHC_rv;
    // splicer end function.UpdateStructAsClass_int
}

// Generated by fortran_generic
// ----------------------------------------
// Function:  long UpdateStructAsClass
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
// ----------------------------------------
// Argument:  StructAsClass * arg
// Attrs:     +intent(inout)
// Statement: f_inout_shadow_*
// ----------------------------------------
// Argument:  long inew +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
long GEN_UpdateStructAsClass_long(GEN_StructAsClass * arg, long inew)
{
    // splicer begin function.UpdateStructAsClass_long
    StructAsClass * SHCXX_arg = (StructAsClass *) arg->addr;
    long SHC_rv = UpdateStructAsClass(SHCXX_arg, inew);
    return SHC_rv;
    // splicer end function.UpdateStructAsClass_long
}
