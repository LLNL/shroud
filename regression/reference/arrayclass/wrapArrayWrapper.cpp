// wrapArrayWrapper.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapArrayWrapper.h"

// cxx_header
#include "arrayclass.hpp"

// splicer begin class.ArrayWrapper.CXX_definitions
// splicer end class.ArrayWrapper.CXX_definitions

extern "C" {

// splicer begin class.ArrayWrapper.C_definitions
// splicer end class.ArrayWrapper.C_definitions

// ----------------------------------------
// Function:  ArrayWrapper
// Attrs:     +intent(ctor)
// Requested: c_ctor_shadow_scalar
// Match:     c_ctor
ARR_ArrayWrapper * ARR_ArrayWrapper_ctor(ARR_ArrayWrapper * SHadow_rv)
{
    // splicer begin class.ArrayWrapper.method.ctor
    ArrayWrapper *SHCXX_rv = new ArrayWrapper();
    SHadow_rv->addr = static_cast<void *>(SHCXX_rv);
    SHadow_rv->idtor = 1;
    return SHadow_rv;
    // splicer end class.ArrayWrapper.method.ctor
}

// ----------------------------------------
// Function:  void setSize
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int size +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void ARR_ArrayWrapper_set_size(ARR_ArrayWrapper * self, int size)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.set_size
    SH_this->setSize(size);
    // splicer end class.ArrayWrapper.method.set_size
}

// ----------------------------------------
// Function:  int getSize
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_default
int ARR_ArrayWrapper_get_size(const ARR_ArrayWrapper * self)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.get_size
    int SHC_rv = SH_this->getSize();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_size
}

// ----------------------------------------
// Function:  void fillSize
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int & size +intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_&
// Match:     c_default
void ARR_ArrayWrapper_fill_size(ARR_ArrayWrapper * self, int * size)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fill_size
    SH_this->fillSize(*size);
    // splicer end class.ArrayWrapper.method.fill_size
}

// ----------------------------------------
// Function:  void allocate
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
void ARR_ArrayWrapper_allocate(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.allocate
    SH_this->allocate();
    // splicer end class.ArrayWrapper.method.allocate
}

// ----------------------------------------
// Function:  double * getArray +dimension(getSize())
// Attrs:     +deref(pointer)+intent(function)
// Requested: c_function_native_*_pointer
// Match:     c_function_native_*
double * ARR_ArrayWrapper_get_array(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.get_array
    double * SHC_rv = SH_this->getArray();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_array
}

// ----------------------------------------
// Function:  double * getArray +dimension(getSize())
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_native_*_buf_pointer
// Match:     c_function_native_*_buf
void ARR_ArrayWrapper_get_array_bufferify(ARR_ArrayWrapper * self,
    ARR_SHROUD_array *SHC_rv_temp0)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_bufferify
    double * SHC_rv = SH_this->getArray();
    SHC_rv_temp0->cxx.addr  = SHC_rv;
    SHC_rv_temp0->cxx.idtor = 0;
    SHC_rv_temp0->addr.base = SHC_rv;
    SHC_rv_temp0->type = SH_TYPE_DOUBLE;
    SHC_rv_temp0->elem_len = sizeof(double);
    SHC_rv_temp0->rank = 1;
    SHC_rv_temp0->shape[0] = SH_this->getSize();
    SHC_rv_temp0->size = SHC_rv_temp0->shape[0];
    // splicer end class.ArrayWrapper.method.get_array_bufferify
}

// ----------------------------------------
// Function:  double * getArrayConst +dimension(getSize())
// Attrs:     +deref(pointer)+intent(function)
// Requested: c_function_native_*_pointer
// Match:     c_function_native_*
double * ARR_ArrayWrapper_get_array_const(const ARR_ArrayWrapper * self)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_const
    double * SHC_rv = SH_this->getArrayConst();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_array_const
}

// ----------------------------------------
// Function:  double * getArrayConst +dimension(getSize())
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_native_*_buf_pointer
// Match:     c_function_native_*_buf
void ARR_ArrayWrapper_get_array_const_bufferify(
    const ARR_ArrayWrapper * self, ARR_SHROUD_array *SHC_rv_temp0)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_const_bufferify
    double * SHC_rv = SH_this->getArrayConst();
    SHC_rv_temp0->cxx.addr  = SHC_rv;
    SHC_rv_temp0->cxx.idtor = 0;
    SHC_rv_temp0->addr.base = SHC_rv;
    SHC_rv_temp0->type = SH_TYPE_DOUBLE;
    SHC_rv_temp0->elem_len = sizeof(double);
    SHC_rv_temp0->rank = 1;
    SHC_rv_temp0->shape[0] = SH_this->getSize();
    SHC_rv_temp0->size = SHC_rv_temp0->shape[0];
    // splicer end class.ArrayWrapper.method.get_array_const_bufferify
}

// ----------------------------------------
// Function:  const double * getArrayC +dimension(getSize())
// Attrs:     +deref(pointer)+intent(function)
// Requested: c_function_native_*_pointer
// Match:     c_function_native_*
const double * ARR_ArrayWrapper_get_array_c(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_c
    const double * SHC_rv = SH_this->getArrayC();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_array_c
}

// ----------------------------------------
// Function:  const double * getArrayC +dimension(getSize())
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_native_*_buf_pointer
// Match:     c_function_native_*_buf
void ARR_ArrayWrapper_get_array_c_bufferify(ARR_ArrayWrapper * self,
    ARR_SHROUD_array *SHC_rv_temp0)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_c_bufferify
    const double * SHC_rv = SH_this->getArrayC();
    SHC_rv_temp0->cxx.addr  = const_cast<double *>(SHC_rv);
    SHC_rv_temp0->cxx.idtor = 0;
    SHC_rv_temp0->addr.base = SHC_rv;
    SHC_rv_temp0->type = SH_TYPE_DOUBLE;
    SHC_rv_temp0->elem_len = sizeof(double);
    SHC_rv_temp0->rank = 1;
    SHC_rv_temp0->shape[0] = SH_this->getSize();
    SHC_rv_temp0->size = SHC_rv_temp0->shape[0];
    // splicer end class.ArrayWrapper.method.get_array_c_bufferify
}

// ----------------------------------------
// Function:  const double * getArrayConstC +dimension(getSize())
// Attrs:     +deref(pointer)+intent(function)
// Requested: c_function_native_*_pointer
// Match:     c_function_native_*
const double * ARR_ArrayWrapper_get_array_const_c(
    const ARR_ArrayWrapper * self)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_const_c
    const double * SHC_rv = SH_this->getArrayConstC();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_array_const_c
}

// ----------------------------------------
// Function:  const double * getArrayConstC +dimension(getSize())
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_native_*_buf_pointer
// Match:     c_function_native_*_buf
void ARR_ArrayWrapper_get_array_const_c_bufferify(
    const ARR_ArrayWrapper * self, ARR_SHROUD_array *SHC_rv_temp0)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_const_c_bufferify
    const double * SHC_rv = SH_this->getArrayConstC();
    SHC_rv_temp0->cxx.addr  = const_cast<double *>(SHC_rv);
    SHC_rv_temp0->cxx.idtor = 0;
    SHC_rv_temp0->addr.base = SHC_rv;
    SHC_rv_temp0->type = SH_TYPE_DOUBLE;
    SHC_rv_temp0->elem_len = sizeof(double);
    SHC_rv_temp0->rank = 1;
    SHC_rv_temp0->shape[0] = SH_this->getSize();
    SHC_rv_temp0->size = SHC_rv_temp0->shape[0];
    // splicer end class.ArrayWrapper.method.get_array_const_c_bufferify
}

// ----------------------------------------
// Function:  void fetchArrayPtr
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  double * * array +dimension(isize)+intent(out)
// Attrs:     +deref(pointer)+intent(out)
// Requested: c_out_native_**_pointer
// Match:     c_default
// ----------------------------------------
// Argument:  int * isize +hidden
// Attrs:     +intent(inout)
// Requested: c_inout_native_*
// Match:     c_default
void ARR_ArrayWrapper_fetch_array_ptr(ARR_ArrayWrapper * self,
    double * * array, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetch_array_ptr
    SH_this->fetchArrayPtr(array, isize);
    // splicer end class.ArrayWrapper.method.fetch_array_ptr
}

// ----------------------------------------
// Function:  void fetchArrayPtr
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  double * * array +context(Darray)+dimension(isize)+intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// ----------------------------------------
// Argument:  int * isize +hidden
// Attrs:     +intent(inout)
// Requested: c_inout_native_*
// Match:     c_default
void ARR_ArrayWrapper_fetch_array_ptr_bufferify(ARR_ArrayWrapper * self,
    ARR_SHROUD_array *Darray, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetch_array_ptr_bufferify
    double *array;
    SH_this->fetchArrayPtr(&array, isize);
    Darray->cxx.addr  = array;
    Darray->cxx.idtor = 0;
    Darray->addr.base = array;
    Darray->type = SH_TYPE_DOUBLE;
    Darray->elem_len = sizeof(double);
    Darray->rank = 1;
    Darray->shape[0] = *isize;
    Darray->size = Darray->shape[0];
    // splicer end class.ArrayWrapper.method.fetch_array_ptr_bufferify
}

// ----------------------------------------
// Function:  void fetchArrayRef
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  double * & array +dimension(isize)+intent(out)
// Attrs:     +deref(pointer)+intent(out)
// Requested: c_out_native_*&_pointer
// Match:     c_default
// ----------------------------------------
// Argument:  int & isize +hidden
// Attrs:     +intent(inout)
// Requested: c_inout_native_&
// Match:     c_default
void ARR_ArrayWrapper_fetch_array_ref(ARR_ArrayWrapper * self,
    double * * array, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetch_array_ref
    SH_this->fetchArrayRef(*array, *isize);
    // splicer end class.ArrayWrapper.method.fetch_array_ref
}

// ----------------------------------------
// Function:  void fetchArrayRef
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  double * & array +context(Darray)+dimension(isize)+intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_*&_buf_pointer
// Match:     c_out_native_*&_buf
// ----------------------------------------
// Argument:  int & isize +hidden
// Attrs:     +intent(inout)
// Requested: c_inout_native_&
// Match:     c_default
void ARR_ArrayWrapper_fetch_array_ref_bufferify(ARR_ArrayWrapper * self,
    ARR_SHROUD_array *Darray, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetch_array_ref_bufferify
    double *array;
    SH_this->fetchArrayRef(array, *isize);
    Darray->cxx.addr  = array;
    Darray->cxx.idtor = 0;
    Darray->addr.base = array;
    Darray->type = SH_TYPE_DOUBLE;
    Darray->elem_len = sizeof(double);
    Darray->rank = 1;
    Darray->shape[0] = *isize;
    Darray->size = Darray->shape[0];
    // splicer end class.ArrayWrapper.method.fetch_array_ref_bufferify
}

// ----------------------------------------
// Function:  void fetchArrayPtrConst
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const double * * array +dimension(isize)+intent(out)
// Attrs:     +deref(pointer)+intent(out)
// Requested: c_out_native_**_pointer
// Match:     c_default
// ----------------------------------------
// Argument:  int * isize +hidden
// Attrs:     +intent(inout)
// Requested: c_inout_native_*
// Match:     c_default
void ARR_ArrayWrapper_fetch_array_ptr_const(ARR_ArrayWrapper * self,
    const double * * array, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetch_array_ptr_const
    SH_this->fetchArrayPtrConst(array, isize);
    // splicer end class.ArrayWrapper.method.fetch_array_ptr_const
}

// ----------------------------------------
// Function:  void fetchArrayPtrConst
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const double * * array +context(Darray)+dimension(isize)+intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_**_buf_pointer
// Match:     c_out_native_**_buf
// ----------------------------------------
// Argument:  int * isize +hidden
// Attrs:     +intent(inout)
// Requested: c_inout_native_*
// Match:     c_default
void ARR_ArrayWrapper_fetch_array_ptr_const_bufferify(
    ARR_ArrayWrapper * self, ARR_SHROUD_array *Darray, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetch_array_ptr_const_bufferify
    const double *array;
    SH_this->fetchArrayPtrConst(&array, isize);
    Darray->cxx.addr  = const_cast<double *>(array);
    Darray->cxx.idtor = 0;
    Darray->addr.base = array;
    Darray->type = SH_TYPE_DOUBLE;
    Darray->elem_len = sizeof(double);
    Darray->rank = 1;
    Darray->shape[0] = *isize;
    Darray->size = Darray->shape[0];
    // splicer end class.ArrayWrapper.method.fetch_array_ptr_const_bufferify
}

// ----------------------------------------
// Function:  void fetchArrayRefConst
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const double * & array +dimension(isize)+intent(out)
// Attrs:     +deref(pointer)+intent(out)
// Requested: c_out_native_*&_pointer
// Match:     c_default
// ----------------------------------------
// Argument:  int & isize +hidden
// Attrs:     +intent(inout)
// Requested: c_inout_native_&
// Match:     c_default
void ARR_ArrayWrapper_fetch_array_ref_const(ARR_ArrayWrapper * self,
    const double * * array, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetch_array_ref_const
    SH_this->fetchArrayRefConst(*array, *isize);
    // splicer end class.ArrayWrapper.method.fetch_array_ref_const
}

// ----------------------------------------
// Function:  void fetchArrayRefConst
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const double * & array +context(Darray)+dimension(isize)+intent(out)
// Attrs:     +api(buf)+deref(pointer)+intent(out)
// Requested: c_out_native_*&_buf_pointer
// Match:     c_out_native_*&_buf
// ----------------------------------------
// Argument:  int & isize +hidden
// Attrs:     +intent(inout)
// Requested: c_inout_native_&
// Match:     c_default
void ARR_ArrayWrapper_fetch_array_ref_const_bufferify(
    ARR_ArrayWrapper * self, ARR_SHROUD_array *Darray, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetch_array_ref_const_bufferify
    const double *array;
    SH_this->fetchArrayRefConst(array, *isize);
    Darray->cxx.addr  = const_cast<double *>(array);
    Darray->cxx.idtor = 0;
    Darray->addr.base = array;
    Darray->type = SH_TYPE_DOUBLE;
    Darray->elem_len = sizeof(double);
    Darray->rank = 1;
    Darray->shape[0] = *isize;
    Darray->size = Darray->shape[0];
    // splicer end class.ArrayWrapper.method.fetch_array_ref_const_bufferify
}

// ----------------------------------------
// Function:  void fetchVoidPtr
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  void * * array +intent(out)
// Attrs:     +intent(out)
// Requested: c_out_void_**
// Match:     c_default
void ARR_ArrayWrapper_fetch_void_ptr(ARR_ArrayWrapper * self,
    void * * array)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetch_void_ptr
    SH_this->fetchVoidPtr(array);
    // splicer end class.ArrayWrapper.method.fetch_void_ptr
}

// ----------------------------------------
// Function:  void fetchVoidRef
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  void * & array +intent(out)
// Attrs:     +intent(out)
// Requested: c_out_void_*&
// Match:     c_default
void ARR_ArrayWrapper_fetch_void_ref(ARR_ArrayWrapper * self,
    void * * array)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetch_void_ref
    SH_this->fetchVoidRef(*array);
    // splicer end class.ArrayWrapper.method.fetch_void_ref
}

// ----------------------------------------
// Function:  bool checkPtr
// Attrs:     +intent(function)
// Requested: c_function_bool_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  void * array +value
// Attrs:     +intent(in)
// Requested: c_in_void_*
// Match:     c_default
bool ARR_ArrayWrapper_check_ptr(ARR_ArrayWrapper * self, void * array)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.check_ptr
    bool SHC_rv = SH_this->checkPtr(array);
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.check_ptr
}

// ----------------------------------------
// Function:  double sumArray
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_default
double ARR_ArrayWrapper_sum_array(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.sum_array
    double SHC_rv = SH_this->sumArray();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.sum_array
}

}  // extern "C"
