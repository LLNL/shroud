// wrapArrayWrapper.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "arrayclass.hpp"
// shroud
#include "wrapArrayWrapper.h"

// splicer begin class.ArrayWrapper.CXX_definitions
// splicer end class.ArrayWrapper.CXX_definitions

extern "C" {

// splicer begin class.ArrayWrapper.C_definitions
// splicer end class.ArrayWrapper.C_definitions

// ----------------------------------------
// Function:  ArrayWrapper
// Statement: c_ctor_shadow_scalar_capptr
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
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int size
// Statement: c_in_native_scalar
void ARR_ArrayWrapper_setSize(ARR_ArrayWrapper * self, int size)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.setSize
    SH_this->setSize(size);
    // splicer end class.ArrayWrapper.method.setSize
}

// ----------------------------------------
// Function:  int getSize
// Statement: c_function_native_scalar
int ARR_ArrayWrapper_getSize(const ARR_ArrayWrapper * self)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.getSize
    int SHC_rv = SH_this->getSize();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.getSize
}

// ----------------------------------------
// Function:  void fillSize
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int & size +intent(out)
// Statement: c_out_native_&
void ARR_ArrayWrapper_fillSize(ARR_ArrayWrapper * self, int * size)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fillSize
    SH_this->fillSize(*size);
    // splicer end class.ArrayWrapper.method.fillSize
}

// ----------------------------------------
// Function:  void allocate
// Statement: c_subroutine
void ARR_ArrayWrapper_allocate(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.allocate
    SH_this->allocate();
    // splicer end class.ArrayWrapper.method.allocate
}

// ----------------------------------------
// Function:  double * getArray +dimension(getSize())
// Statement: c_function_native_*
double * ARR_ArrayWrapper_getArray(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.getArray
    double * SHC_rv = SH_this->getArray();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.getArray
}

// ----------------------------------------
// Function:  double * getArray +dimension(getSize())
// Statement: f_function_native_*_cdesc_pointer
void ARR_ArrayWrapper_getArray_bufferify(ARR_ArrayWrapper * self,
    ARR_SHROUD_array *SHT_rv_cdesc)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.getArray_bufferify
    double * SHC_rv = SH_this->getArray();
    SHT_rv_cdesc->base_addr = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_DOUBLE;
    SHT_rv_cdesc->elem_len = sizeof(double);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->getSize();
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end class.ArrayWrapper.method.getArray_bufferify
}

// ----------------------------------------
// Function:  double * getArrayConst +dimension(getSize())
// Statement: c_function_native_*
double * ARR_ArrayWrapper_getArrayConst(const ARR_ArrayWrapper * self)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.getArrayConst
    double * SHC_rv = SH_this->getArrayConst();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.getArrayConst
}

// ----------------------------------------
// Function:  double * getArrayConst +dimension(getSize())
// Statement: f_function_native_*_cdesc_pointer
void ARR_ArrayWrapper_getArrayConst_bufferify(
    const ARR_ArrayWrapper * self, ARR_SHROUD_array *SHT_rv_cdesc)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.getArrayConst_bufferify
    double * SHC_rv = SH_this->getArrayConst();
    SHT_rv_cdesc->base_addr = SHC_rv;
    SHT_rv_cdesc->type = SH_TYPE_DOUBLE;
    SHT_rv_cdesc->elem_len = sizeof(double);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->getSize();
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end class.ArrayWrapper.method.getArrayConst_bufferify
}

// ----------------------------------------
// Function:  const double * getArrayC +dimension(getSize())
// Statement: c_function_native_*
const double * ARR_ArrayWrapper_getArrayC(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.getArrayC
    const double * SHC_rv = SH_this->getArrayC();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.getArrayC
}

// ----------------------------------------
// Function:  const double * getArrayC +dimension(getSize())
// Statement: f_function_native_*_cdesc_pointer
void ARR_ArrayWrapper_getArrayC_bufferify(ARR_ArrayWrapper * self,
    ARR_SHROUD_array *SHT_rv_cdesc)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.getArrayC_bufferify
    const double * SHC_rv = SH_this->getArrayC();
    SHT_rv_cdesc->base_addr = const_cast<double *>(SHC_rv);
    SHT_rv_cdesc->type = SH_TYPE_DOUBLE;
    SHT_rv_cdesc->elem_len = sizeof(double);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->getSize();
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end class.ArrayWrapper.method.getArrayC_bufferify
}

// ----------------------------------------
// Function:  const double * getArrayConstC +dimension(getSize())
// Statement: c_function_native_*
const double * ARR_ArrayWrapper_getArrayConstC(
    const ARR_ArrayWrapper * self)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.getArrayConstC
    const double * SHC_rv = SH_this->getArrayConstC();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.getArrayConstC
}

// ----------------------------------------
// Function:  const double * getArrayConstC +dimension(getSize())
// Statement: f_function_native_*_cdesc_pointer
void ARR_ArrayWrapper_getArrayConstC_bufferify(
    const ARR_ArrayWrapper * self, ARR_SHROUD_array *SHT_rv_cdesc)
{
    const ArrayWrapper *SH_this = static_cast<const ArrayWrapper *>
        (self->addr);
    // splicer begin class.ArrayWrapper.method.getArrayConstC_bufferify
    const double * SHC_rv = SH_this->getArrayConstC();
    SHT_rv_cdesc->base_addr = const_cast<double *>(SHC_rv);
    SHT_rv_cdesc->type = SH_TYPE_DOUBLE;
    SHT_rv_cdesc->elem_len = sizeof(double);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->getSize();
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end class.ArrayWrapper.method.getArrayConstC_bufferify
}

// ----------------------------------------
// Function:  void fetchArrayPtr
// Statement: c_subroutine
// ----------------------------------------
// Argument:  double * * array +dimension(isize)+intent(out)
// Statement: c_out_native_**
// ----------------------------------------
// Argument:  int * isize +hidden
// Statement: c_inout_native_*
void ARR_ArrayWrapper_fetchArrayPtr(ARR_ArrayWrapper * self,
    double * * array, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetchArrayPtr
    SH_this->fetchArrayPtr(array, isize);
    // splicer end class.ArrayWrapper.method.fetchArrayPtr
}

// ----------------------------------------
// Function:  void fetchArrayPtr
// Statement: f_subroutine
// ----------------------------------------
// Argument:  double * * array +dimension(isize)+intent(out)
// Statement: f_out_native_**_cdesc_pointer
// ----------------------------------------
// Argument:  int * isize +hidden
// Statement: f_inout_native_*_hidden
void ARR_ArrayWrapper_fetchArrayPtr_bufferify(ARR_ArrayWrapper * self,
    ARR_SHROUD_array *SHT_array_cdesc)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetchArrayPtr_bufferify
    double *array;
    int isize;
    SH_this->fetchArrayPtr(&array, &isize);
    SHT_array_cdesc->base_addr = array;
    SHT_array_cdesc->type = SH_TYPE_DOUBLE;
    SHT_array_cdesc->elem_len = sizeof(double);
    SHT_array_cdesc->rank = 1;
    SHT_array_cdesc->shape[0] = isize;
    SHT_array_cdesc->size = SHT_array_cdesc->shape[0];
    // splicer end class.ArrayWrapper.method.fetchArrayPtr_bufferify
}

// ----------------------------------------
// Function:  void fetchArrayRef
// Statement: c_subroutine
// ----------------------------------------
// Argument:  double * & array +dimension(isize)+intent(out)
// Statement: c_out_native_*&
// ----------------------------------------
// Argument:  int & isize +hidden
// Statement: c_inout_native_&
void ARR_ArrayWrapper_fetchArrayRef(ARR_ArrayWrapper * self,
    double * * array, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetchArrayRef
    SH_this->fetchArrayRef(*array, *isize);
    // splicer end class.ArrayWrapper.method.fetchArrayRef
}

// ----------------------------------------
// Function:  void fetchArrayRef
// Statement: f_subroutine
// ----------------------------------------
// Argument:  double * & array +dimension(isize)+intent(out)
// Statement: f_out_native_*&_cdesc_pointer
// ----------------------------------------
// Argument:  int & isize +hidden
// Statement: f_inout_native_&_hidden
void ARR_ArrayWrapper_fetchArrayRef_bufferify(ARR_ArrayWrapper * self,
    ARR_SHROUD_array *SHT_array_cdesc)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetchArrayRef_bufferify
    double *array;
    int isize;
    SH_this->fetchArrayRef(array, isize);
    SHT_array_cdesc->base_addr = array;
    SHT_array_cdesc->type = SH_TYPE_DOUBLE;
    SHT_array_cdesc->elem_len = sizeof(double);
    SHT_array_cdesc->rank = 1;
    SHT_array_cdesc->shape[0] = isize;
    SHT_array_cdesc->size = SHT_array_cdesc->shape[0];
    // splicer end class.ArrayWrapper.method.fetchArrayRef_bufferify
}

// ----------------------------------------
// Function:  void fetchArrayPtrConst
// Statement: c_subroutine
// ----------------------------------------
// Argument:  const double * * array +dimension(isize)+intent(out)
// Statement: c_out_native_**
// ----------------------------------------
// Argument:  int * isize +hidden
// Statement: c_inout_native_*
void ARR_ArrayWrapper_fetchArrayPtrConst(ARR_ArrayWrapper * self,
    const double * * array, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetchArrayPtrConst
    SH_this->fetchArrayPtrConst(array, isize);
    // splicer end class.ArrayWrapper.method.fetchArrayPtrConst
}

// ----------------------------------------
// Function:  void fetchArrayPtrConst
// Statement: f_subroutine
// ----------------------------------------
// Argument:  const double * * array +dimension(isize)+intent(out)
// Statement: f_out_native_**_cdesc_pointer
// ----------------------------------------
// Argument:  int * isize +hidden
// Statement: f_inout_native_*_hidden
void ARR_ArrayWrapper_fetchArrayPtrConst_bufferify(
    ARR_ArrayWrapper * self, ARR_SHROUD_array *SHT_array_cdesc)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetchArrayPtrConst_bufferify
    const double *array;
    int isize;
    SH_this->fetchArrayPtrConst(&array, &isize);
    SHT_array_cdesc->base_addr = const_cast<double *>(array);
    SHT_array_cdesc->type = SH_TYPE_DOUBLE;
    SHT_array_cdesc->elem_len = sizeof(double);
    SHT_array_cdesc->rank = 1;
    SHT_array_cdesc->shape[0] = isize;
    SHT_array_cdesc->size = SHT_array_cdesc->shape[0];
    // splicer end class.ArrayWrapper.method.fetchArrayPtrConst_bufferify
}

// ----------------------------------------
// Function:  void fetchArrayRefConst
// Statement: c_subroutine
// ----------------------------------------
// Argument:  const double * & array +dimension(isize)+intent(out)
// Statement: c_out_native_*&
// ----------------------------------------
// Argument:  int & isize +hidden
// Statement: c_inout_native_&
void ARR_ArrayWrapper_fetchArrayRefConst(ARR_ArrayWrapper * self,
    const double * * array, int * isize)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetchArrayRefConst
    SH_this->fetchArrayRefConst(*array, *isize);
    // splicer end class.ArrayWrapper.method.fetchArrayRefConst
}

// ----------------------------------------
// Function:  void fetchArrayRefConst
// Statement: f_subroutine
// ----------------------------------------
// Argument:  const double * & array +dimension(isize)+intent(out)
// Statement: f_out_native_*&_cdesc_pointer
// ----------------------------------------
// Argument:  int & isize +hidden
// Statement: f_inout_native_&_hidden
void ARR_ArrayWrapper_fetchArrayRefConst_bufferify(
    ARR_ArrayWrapper * self, ARR_SHROUD_array *SHT_array_cdesc)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetchArrayRefConst_bufferify
    const double *array;
    int isize;
    SH_this->fetchArrayRefConst(array, isize);
    SHT_array_cdesc->base_addr = const_cast<double *>(array);
    SHT_array_cdesc->type = SH_TYPE_DOUBLE;
    SHT_array_cdesc->elem_len = sizeof(double);
    SHT_array_cdesc->rank = 1;
    SHT_array_cdesc->shape[0] = isize;
    SHT_array_cdesc->size = SHT_array_cdesc->shape[0];
    // splicer end class.ArrayWrapper.method.fetchArrayRefConst_bufferify
}

// ----------------------------------------
// Function:  void fetchVoidPtr
// Statement: c_subroutine
// ----------------------------------------
// Argument:  void * * array +intent(out)
// Statement: c_out_void_**
void ARR_ArrayWrapper_fetchVoidPtr(ARR_ArrayWrapper * self,
    void **array)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetchVoidPtr
    SH_this->fetchVoidPtr(array);
    // splicer end class.ArrayWrapper.method.fetchVoidPtr
}

// ----------------------------------------
// Function:  void fetchVoidRef
// Statement: c_subroutine
// ----------------------------------------
// Argument:  void * & array +intent(out)
// Statement: c_out_void_*&
void ARR_ArrayWrapper_fetchVoidRef(ARR_ArrayWrapper * self,
    void * * array)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.fetchVoidRef
    SH_this->fetchVoidRef(*array);
    // splicer end class.ArrayWrapper.method.fetchVoidRef
}

// ----------------------------------------
// Function:  bool checkPtr
// Statement: c_function_bool_scalar
// ----------------------------------------
// Argument:  void * array
// Statement: c_in_void_*
bool ARR_ArrayWrapper_checkPtr(ARR_ArrayWrapper * self, void * array)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.checkPtr
    bool SHC_rv = SH_this->checkPtr(array);
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.checkPtr
}

// ----------------------------------------
// Function:  double sumArray
// Statement: c_function_native_scalar
double ARR_ArrayWrapper_sumArray(ARR_ArrayWrapper * self)
{
    ArrayWrapper *SH_this = static_cast<ArrayWrapper *>(self->addr);
    // splicer begin class.ArrayWrapper.method.sumArray
    double SHC_rv = SH_this->sumArray();
    return SHC_rv;
    // splicer end class.ArrayWrapper.method.sumArray
}

}  // extern "C"
