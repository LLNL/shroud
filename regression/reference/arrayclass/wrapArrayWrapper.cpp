// wrapArrayWrapper.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapArrayWrapper.h"
#include "arrayclass.hpp"

// splicer begin class.ArrayWrapper.CXX_definitions
// splicer end class.ArrayWrapper.CXX_definitions

extern "C" {

// splicer begin class.ArrayWrapper.C_definitions
// splicer end class.ArrayWrapper.C_definitions

// ----------------------------------------
// Function:  ArrayWrapper
// Exact:     c_shadow_scalar_ctor
ARR_ArrayWrapper * ARR_ArrayWrapper_ctor(ARR_ArrayWrapper * SHC_rv)
{
    // splicer begin class.ArrayWrapper.method.ctor
    ArrayWrapper *SHCXX_rv = new ArrayWrapper();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.ctor
}

// ----------------------------------------
// Function:  void setSize
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int size +intent(in)+value
// Requested: c_native_scalar_in
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
// Requested: c_native_scalar_result
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int & size +intent(out)
// Requested: c_native_&_out
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
// Requested: c
// Match:     c_default
void ARR_ArrayWrapper_allocate(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.allocate
    SH_this->allocate();
    // splicer end class.ArrayWrapper.method.allocate
}

// ----------------------------------------
// Function:  double * getArray +deref(pointer)+dimension(getSize())
// Requested: c_native_*_result
// Match:     c_default
double * ARR_ArrayWrapper_get_array(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.get_array
    double * SHC_rv = SH_this->getArray();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_array
}

// ----------------------------------------
// Function:  double * getArray +context(DSHC_rv)+deref(pointer)+dimension(getSize())
// Exact:     c_native_*_result_buf
double * ARR_ArrayWrapper_get_array_bufferify(ARR_ArrayWrapper * self,
    ARR_SHROUD_array *DSHC_rv)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_bufferify
    double * SHC_rv = SH_this->getArray();
    DSHC_rv->cxx.addr  = SHC_rv;
    DSHC_rv->cxx.idtor = 0;
    DSHC_rv->addr.base = SHC_rv;
    DSHC_rv->type = SH_TYPE_DOUBLE;
    DSHC_rv->elem_len = sizeof(double);
    DSHC_rv->rank = 1;
    DSHC_rv->shape[0] = SH_this->getSize();
    DSHC_rv->size = DSHC_rv->shape[0];
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_array_bufferify
}

// ----------------------------------------
// Function:  double * getArrayConst +deref(pointer)+dimension(getSize())
// Requested: c_native_*_result
// Match:     c_default
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
// Function:  double * getArrayConst +context(DSHC_rv)+deref(pointer)+dimension(getSize())
// Exact:     c_native_*_result_buf
double * ARR_ArrayWrapper_get_array_const_bufferify(
    const ARR_ArrayWrapper * self, ARR_SHROUD_array *DSHC_rv)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_const_bufferify
    double * SHC_rv = SH_this->getArrayConst();
    DSHC_rv->cxx.addr  = SHC_rv;
    DSHC_rv->cxx.idtor = 0;
    DSHC_rv->addr.base = SHC_rv;
    DSHC_rv->type = SH_TYPE_DOUBLE;
    DSHC_rv->elem_len = sizeof(double);
    DSHC_rv->rank = 1;
    DSHC_rv->shape[0] = SH_this->getSize();
    DSHC_rv->size = DSHC_rv->shape[0];
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_array_const_bufferify
}

// ----------------------------------------
// Function:  const double * getArrayC +deref(pointer)+dimension(getSize())
// Requested: c_native_*_result
// Match:     c_default
const double * ARR_ArrayWrapper_get_array_c(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_c
    const double * SHC_rv = SH_this->getArrayC();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_array_c
}

// ----------------------------------------
// Function:  const double * getArrayC +context(DSHC_rv)+deref(pointer)+dimension(getSize())
// Exact:     c_native_*_result_buf
const double * ARR_ArrayWrapper_get_array_c_bufferify(
    ARR_ArrayWrapper * self, ARR_SHROUD_array *DSHC_rv)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_c_bufferify
    const double * SHC_rv = SH_this->getArrayC();
    DSHC_rv->cxx.addr  = const_cast<double *>(SHC_rv);
    DSHC_rv->cxx.idtor = 0;
    DSHC_rv->addr.base = SHC_rv;
    DSHC_rv->type = SH_TYPE_DOUBLE;
    DSHC_rv->elem_len = sizeof(double);
    DSHC_rv->rank = 1;
    DSHC_rv->shape[0] = SH_this->getSize();
    DSHC_rv->size = DSHC_rv->shape[0];
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_array_c_bufferify
}

// ----------------------------------------
// Function:  const double * getArrayConstC +deref(pointer)+dimension(getSize())
// Requested: c_native_*_result
// Match:     c_default
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
// Function:  const double * getArrayConstC +context(DSHC_rv)+deref(pointer)+dimension(getSize())
// Exact:     c_native_*_result_buf
const double * ARR_ArrayWrapper_get_array_const_c_bufferify(
    const ARR_ArrayWrapper * self, ARR_SHROUD_array *DSHC_rv)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.get_array_const_c_bufferify
    const double * SHC_rv = SH_this->getArrayConstC();
    DSHC_rv->cxx.addr  = const_cast<double *>(SHC_rv);
    DSHC_rv->cxx.idtor = 0;
    DSHC_rv->addr.base = SHC_rv;
    DSHC_rv->type = SH_TYPE_DOUBLE;
    DSHC_rv->elem_len = sizeof(double);
    DSHC_rv->rank = 1;
    DSHC_rv->shape[0] = SH_this->getSize();
    DSHC_rv->size = DSHC_rv->shape[0];
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.get_array_const_c_bufferify
}

// ----------------------------------------
// Function:  void fetchArrayPtr
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  double * * array +deref(pointer)+dimension(isize)+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// ----------------------------------------
// Argument:  int * isize +hidden+intent(inout)
// Requested: c_native_*_inout
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  double * * array +context(Darray)+deref(pointer)+dimension(isize)+intent(out)
// Exact:     c_native_**_out_buf
// ----------------------------------------
// Argument:  int * isize +hidden+intent(inout)
// Requested: c_native_*_inout_buf
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  double * & array +deref(pointer)+dimension(isize)+intent(out)
// Requested: c_native_*&_out
// Match:     c_default
// ----------------------------------------
// Argument:  int & isize +hidden+intent(inout)
// Requested: c_native_&_inout
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  double * & array +context(Darray)+deref(pointer)+dimension(isize)+intent(out)
// Exact:     c_native_*&_out_buf
// ----------------------------------------
// Argument:  int & isize +hidden+intent(inout)
// Requested: c_native_&_inout_buf
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const double * * array +deref(pointer)+dimension(isize)+intent(out)
// Requested: c_native_**_out
// Match:     c_default
// ----------------------------------------
// Argument:  int * isize +hidden+intent(inout)
// Requested: c_native_*_inout
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const double * * array +context(Darray)+deref(pointer)+dimension(isize)+intent(out)
// Exact:     c_native_**_out_buf
// ----------------------------------------
// Argument:  int * isize +hidden+intent(inout)
// Requested: c_native_*_inout_buf
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const double * & array +deref(pointer)+dimension(isize)+intent(out)
// Requested: c_native_*&_out
// Match:     c_default
// ----------------------------------------
// Argument:  int & isize +hidden+intent(inout)
// Requested: c_native_&_inout
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  const double * & array +context(Darray)+deref(pointer)+dimension(isize)+intent(out)
// Exact:     c_native_*&_out_buf
// ----------------------------------------
// Argument:  int & isize +hidden+intent(inout)
// Requested: c_native_&_inout_buf
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  void * * array +intent(out)
// Requested: c_void_**_out
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  void * & array +intent(out)
// Requested: c_void_*&_out
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
// Requested: c_bool_scalar_result
// Match:     c_default
// ----------------------------------------
// Argument:  void * array +intent(in)+value
// Requested: c_void_*_in
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
// Requested: c_native_scalar_result
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
