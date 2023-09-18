// wrapData.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "classes.hpp"
// shroud
#include <cstddef>
#include "wrapData.h"

// splicer begin class.Data.CXX_definitions
// splicer end class.Data.CXX_definitions

extern "C" {

// splicer begin class.Data.C_definitions
// splicer end class.Data.C_definitions

// ----------------------------------------
// Function:  void allocate
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int n +value
// Attrs:     +intent(in)
// Exact:     f_in_native_scalar
// start CLA_Data_allocate
void CLA_Data_allocate(CLA_Data * self, int n)
{
    classes::Data *SH_this = static_cast<classes::Data *>(self->addr);
    // splicer begin class.Data.method.allocate
    SH_this->allocate(n);
    // splicer end class.Data.method.allocate
}
// end CLA_Data_allocate

// ----------------------------------------
// Function:  void free
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// start CLA_Data_free
void CLA_Data_free(CLA_Data * self)
{
    classes::Data *SH_this = static_cast<classes::Data *>(self->addr);
    // splicer begin class.Data.method.free
    SH_this->free();
    // splicer end class.Data.method.free
}
// end CLA_Data_free

// ----------------------------------------
// Function:  Data
// Attrs:     +api(capptr)+intent(ctor)
// Exact:     f_ctor_shadow_scalar_capptr
// start CLA_Data_ctor
CLA_Data * CLA_Data_ctor(CLA_Data * SHC_rv)
{
    // splicer begin class.Data.method.ctor
    classes::Data *SHCXX_rv = new classes::Data();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 4;
    return SHC_rv;
    // splicer end class.Data.method.ctor
}
// end CLA_Data_ctor

// ----------------------------------------
// Function:  ~Data
// Attrs:     +intent(dtor)
// Exact:     c_dtor
// start CLA_Data_dtor
void CLA_Data_dtor(CLA_Data * self)
{
    classes::Data *SH_this = static_cast<classes::Data *>(self->addr);
    // splicer begin class.Data.method.dtor
    delete SH_this;
    self->addr = nullptr;
    // splicer end class.Data.method.dtor
}
// end CLA_Data_dtor

// Generated by getter/setter
// ----------------------------------------
// Function:  int get_nitems
// Attrs:     +intent(getter)
// Exact:     f_getter_native_scalar
// start CLA_Data_get_nitems
int CLA_Data_get_nitems(CLA_Data * self)
{
    classes::Data *SH_this = static_cast<classes::Data *>(self->addr);
    // splicer begin class.Data.method.get_nitems
    // skip call c_getter
    return SH_this->nitems;
    // splicer end class.Data.method.get_nitems
}
// end CLA_Data_get_nitems

// Generated by getter/setter
// ----------------------------------------
// Function:  void set_nitems
// Attrs:     +intent(setter)
// Exact:     c_setter
// ----------------------------------------
// Argument:  int val +intent(in)+value
// Attrs:     +intent(setter)
// Exact:     f_setter_native_scalar
// start CLA_Data_set_nitems
void CLA_Data_set_nitems(CLA_Data * self, int val)
{
    classes::Data *SH_this = static_cast<classes::Data *>(self->addr);
    // splicer begin class.Data.method.set_nitems
    // skip call c_setter
    SH_this->nitems = val;
    // splicer end class.Data.method.set_nitems
}
// end CLA_Data_set_nitems

// Generated by arg_to_buffer - getter/setter
// ----------------------------------------
// Function:  int * get_items
// Attrs:     +api(cdesc)+deref(pointer)+intent(getter)
// Exact:     f_getter_native_*_cdesc_pointer
// start CLA_Data_get_items_bufferify
void CLA_Data_get_items_bufferify(CLA_Data * self,
    CLA_SHROUD_array *SHT_rv_cdesc)
{
    classes::Data *SH_this = static_cast<classes::Data *>(self->addr);
    // splicer begin class.Data.method.get_items_bufferify
    // skip call c_getter
    SHT_rv_cdesc->cxx.addr  = SH_this->items;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SH_this->items;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->nitems;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end class.Data.method.get_items_bufferify
}
// end CLA_Data_get_items_bufferify

// Generated by getter/setter
// ----------------------------------------
// Function:  void set_items
// Attrs:     +intent(setter)
// Exact:     c_setter
// ----------------------------------------
// Argument:  int * val +intent(in)+rank(1)
// Attrs:     +intent(setter)
// Exact:     f_setter_native_*
// start CLA_Data_set_items
void CLA_Data_set_items(CLA_Data * self, int * val)
{
    classes::Data *SH_this = static_cast<classes::Data *>(self->addr);
    // splicer begin class.Data.method.set_items
    // skip call c_setter
    SH_this->items = val;
    // splicer end class.Data.method.set_items
}
// end CLA_Data_set_items

}  // extern "C"
