// wrapownership.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "ownership.hpp"
// shroud
#include "wrapownership.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  int * ReturnIntPtrRaw +deref(raw)
// Attrs:     +deref(raw)+intent(function)
// Statement: f_function_native_*_raw
int * OWN_ReturnIntPtrRaw(void)
{
    // splicer begin function.ReturnIntPtrRaw
    int * SHC_rv = ReturnIntPtrRaw();
    return SHC_rv;
    // splicer end function.ReturnIntPtrRaw
}

// ----------------------------------------
// Function:  int * ReturnIntPtrScalar +deref(scalar)
// Attrs:     +deref(scalar)+intent(function)
// Statement: f_function_native_*_scalar
int OWN_ReturnIntPtrScalar(void)
{
    // splicer begin function.ReturnIntPtrScalar
    int * SHC_rv = ReturnIntPtrScalar();
    return *SHC_rv;
    // splicer end function.ReturnIntPtrScalar
}

// ----------------------------------------
// Function:  int * ReturnIntPtrPointer +deref(pointer)
// Attrs:     +deref(pointer)+intent(function)
// Statement: f_function_native_*_pointer
int * OWN_ReturnIntPtrPointer(void)
{
    // splicer begin function.ReturnIntPtrPointer
    int * SHC_rv = ReturnIntPtrPointer();
    return SHC_rv;
    // splicer end function.ReturnIntPtrPointer
}

// ----------------------------------------
// Function:  int * ReturnIntPtrDimRaw +deref(raw)
// Attrs:     +deref(raw)+intent(function)
// Statement: f_function_native_*_raw
// ----------------------------------------
// Argument:  int * len +intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*
int * OWN_ReturnIntPtrDimRaw(int * len)
{
    // splicer begin function.ReturnIntPtrDimRaw
    int * SHC_rv = ReturnIntPtrDimRaw(len);
    return SHC_rv;
    // splicer end function.ReturnIntPtrDimRaw
}

// ----------------------------------------
// Function:  int * ReturnIntPtrDimPointer +deref(pointer)+dimension(len)
// Attrs:     +deref(pointer)+intent(function)
// Statement: f_function_native_*_pointer
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*
// start OWN_ReturnIntPtrDimPointer
int * OWN_ReturnIntPtrDimPointer(int * len)
{
    // splicer begin function.ReturnIntPtrDimPointer
    int * SHC_rv = ReturnIntPtrDimPointer(len);
    return SHC_rv;
    // splicer end function.ReturnIntPtrDimPointer
}
// end OWN_ReturnIntPtrDimPointer

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  int * ReturnIntPtrDimPointer +deref(pointer)+dimension(len)
// Attrs:     +api(cdesc)+deref(pointer)+intent(function)
// Statement: f_function_native_*_cdesc_pointer
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*_hidden
// start OWN_ReturnIntPtrDimPointer_bufferify
void OWN_ReturnIntPtrDimPointer_bufferify(
    OWN_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.ReturnIntPtrDimPointer_bufferify
    int len;
    int * SHC_rv = ReturnIntPtrDimPointer(&len);
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = len;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.ReturnIntPtrDimPointer_bufferify
}
// end OWN_ReturnIntPtrDimPointer_bufferify

// ----------------------------------------
// Function:  int * ReturnIntPtrDimAlloc +deref(allocatable)+dimension(len)
// Attrs:     +deref(allocatable)+intent(function)
// Statement: f_function_native_*_allocatable
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*
// start OWN_ReturnIntPtrDimAlloc
int * OWN_ReturnIntPtrDimAlloc(int * len)
{
    // splicer begin function.ReturnIntPtrDimAlloc
    int * SHC_rv = ReturnIntPtrDimAlloc(len);
    return SHC_rv;
    // splicer end function.ReturnIntPtrDimAlloc
}
// end OWN_ReturnIntPtrDimAlloc

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  int * ReturnIntPtrDimAlloc +deref(allocatable)+dimension(len)
// Attrs:     +api(cdesc)+deref(allocatable)+intent(function)
// Statement: f_function_native_*_cdesc_allocatable
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*_hidden
// start OWN_ReturnIntPtrDimAlloc_bufferify
void OWN_ReturnIntPtrDimAlloc_bufferify(OWN_SHROUD_array *SHT_rv_cdesc,
    OWN_SHROUD_capsule_data *SHT_rv_capsule)
{
    // splicer begin function.ReturnIntPtrDimAlloc_bufferify
    int len;
    int * SHC_rv = ReturnIntPtrDimAlloc(&len);
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = len;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    SHT_rv_capsule->addr  = SHC_rv;
    SHT_rv_capsule->idtor = 0;
    // splicer end function.ReturnIntPtrDimAlloc_bufferify
}
// end OWN_ReturnIntPtrDimAlloc_bufferify

// ----------------------------------------
// Function:  int * ReturnIntPtrDimDefault +dimension(len)
// Attrs:     +deref(pointer)+intent(function)
// Statement: f_function_native_*_pointer
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*
int * OWN_ReturnIntPtrDimDefault(int * len)
{
    // splicer begin function.ReturnIntPtrDimDefault
    int * SHC_rv = ReturnIntPtrDimDefault(len);
    return SHC_rv;
    // splicer end function.ReturnIntPtrDimDefault
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  int * ReturnIntPtrDimDefault +dimension(len)
// Attrs:     +api(cdesc)+deref(pointer)+intent(function)
// Statement: f_function_native_*_cdesc_pointer
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*_hidden
void OWN_ReturnIntPtrDimDefault_bufferify(
    OWN_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.ReturnIntPtrDimDefault_bufferify
    int len;
    int * SHC_rv = ReturnIntPtrDimDefault(&len);
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = len;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.ReturnIntPtrDimDefault_bufferify
}

// ----------------------------------------
// Function:  int * ReturnIntPtrDimRawNew +dimension(len)+owner(caller)
// Attrs:     +deref(pointer)+intent(function)
// Statement: f_function_native_*_pointer_caller
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*
int * OWN_ReturnIntPtrDimRawNew(int * len)
{
    // splicer begin function.ReturnIntPtrDimRawNew
    int * SHC_rv = ReturnIntPtrDimRawNew(len);
    return SHC_rv;
    // splicer end function.ReturnIntPtrDimRawNew
}

// ----------------------------------------
// Function:  int * ReturnIntPtrDimPointerNew +deref(pointer)+dimension(len)+owner(caller)
// Attrs:     +deref(pointer)+intent(function)
// Statement: f_function_native_*_pointer_caller
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*
int * OWN_ReturnIntPtrDimPointerNew(int * len)
{
    // splicer begin function.ReturnIntPtrDimPointerNew
    int * SHC_rv = ReturnIntPtrDimPointerNew(len);
    return SHC_rv;
    // splicer end function.ReturnIntPtrDimPointerNew
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  int * ReturnIntPtrDimPointerNew +deref(pointer)+dimension(len)+owner(caller)
// Attrs:     +api(cdesc)+deref(pointer)+intent(function)
// Statement: f_function_native_*_cdesc_pointer_caller
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*_hidden
void OWN_ReturnIntPtrDimPointerNew_bufferify(
    OWN_SHROUD_array *SHT_rv_cdesc,
    OWN_SHROUD_capsule_data *SHT_rv_capsule)
{
    // splicer begin function.ReturnIntPtrDimPointerNew_bufferify
    int len;
    int * SHC_rv = ReturnIntPtrDimPointerNew(&len);
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = len;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    SHT_rv_capsule->addr  = SHC_rv;
    SHT_rv_capsule->idtor = 2;
    // splicer end function.ReturnIntPtrDimPointerNew_bufferify
}

// ----------------------------------------
// Function:  int * ReturnIntPtrDimAllocNew +deref(allocatable)+dimension(len)+owner(caller)
// Attrs:     +deref(allocatable)+intent(function)
// Statement: f_function_native_*_allocatable_caller
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*
int * OWN_ReturnIntPtrDimAllocNew(int * len)
{
    // splicer begin function.ReturnIntPtrDimAllocNew
    int * SHC_rv = ReturnIntPtrDimAllocNew(len);
    return SHC_rv;
    // splicer end function.ReturnIntPtrDimAllocNew
}

// ----------------------------------------
// Function:  int * ReturnIntPtrDimDefaultNew +dimension(len)+owner(caller)
// Attrs:     +deref(pointer)+intent(function)
// Statement: f_function_native_*_pointer_caller
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*
int * OWN_ReturnIntPtrDimDefaultNew(int * len)
{
    // splicer begin function.ReturnIntPtrDimDefaultNew
    int * SHC_rv = ReturnIntPtrDimDefaultNew(len);
    return SHC_rv;
    // splicer end function.ReturnIntPtrDimDefaultNew
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  int * ReturnIntPtrDimDefaultNew +dimension(len)+owner(caller)
// Attrs:     +api(cdesc)+deref(pointer)+intent(function)
// Statement: f_function_native_*_cdesc_pointer_caller
// ----------------------------------------
// Argument:  int * len +hidden+intent(out)
// Attrs:     +intent(out)
// Statement: f_out_native_*_hidden
void OWN_ReturnIntPtrDimDefaultNew_bufferify(
    OWN_SHROUD_array *SHT_rv_cdesc,
    OWN_SHROUD_capsule_data *SHT_rv_capsule)
{
    // splicer begin function.ReturnIntPtrDimDefaultNew_bufferify
    int len;
    int * SHC_rv = ReturnIntPtrDimDefaultNew(&len);
    SHT_rv_cdesc->addr.base = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = len;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    SHT_rv_capsule->addr  = SHC_rv;
    SHT_rv_capsule->idtor = 2;
    // splicer end function.ReturnIntPtrDimDefaultNew_bufferify
}

// ----------------------------------------
// Function:  void createClassStatic
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  int flag +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void OWN_createClassStatic(int flag)
{
    // splicer begin function.createClassStatic
    createClassStatic(flag);
    // splicer end function.createClassStatic
}

// ----------------------------------------
// Function:  Class1 * getClassStatic +owner(library)
// Attrs:     +api(capptr)+intent(function)
// Statement: f_function_shadow_*_capptr_library
OWN_Class1 * OWN_getClassStatic(OWN_Class1 * SHC_rv)
{
    // splicer begin function.getClassStatic
    Class1 * SHCXX_rv = getClassStatic();
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end function.getClassStatic
}

/**
 * \brief Return pointer to new Class1 instance.
 *
 */
// ----------------------------------------
// Function:  Class1 * getClassNew +owner(caller)
// Attrs:     +api(capptr)+intent(function)
// Statement: f_function_shadow_*_capptr_caller
// ----------------------------------------
// Argument:  int flag +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
OWN_Class1 * OWN_getClassNew(int flag, OWN_Class1 * SHC_rv)
{
    // splicer begin function.getClassNew
    Class1 * SHCXX_rv = getClassNew(flag);
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 1;
    return SHC_rv;
    // splicer end function.getClassNew
}

}  // extern "C"
