// wrapscope_ns2.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// shroud
#include "wrapscope_ns2.h"

// splicer begin namespace.ns2.CXX_definitions
// splicer end namespace.ns2.CXX_definitions

extern "C" {

// splicer begin namespace.ns2.C_definitions
// splicer end namespace.ns2.C_definitions

// Generated by getter/setter
// ----------------------------------------
// Function:  int * DataPointer_get_items +dimension(nitems)+intent(getter)
// Statement: f_getter_native_*_cdesc_pointer
// ----------------------------------------
// Argument:  ns2::DataPointer * SH_this +intent(in)
// Statement: f_in_struct_*
void SCO_ns2_DataPointer_get_items(SCO_datapointer * SH_this,
    SCO_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin namespace.ns2.function.DataPointer_get_items
    SHT_rv_cdesc->base_addr = SH_this->items;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->nitems;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end namespace.ns2.function.DataPointer_get_items
}

// Generated by getter/setter
// ----------------------------------------
// Function:  void DataPointer_set_items +intent(setter)
// Statement: f_setter
// ----------------------------------------
// Argument:  ns2::DataPointer * SH_this
// Statement: f_inout_struct_*
// ----------------------------------------
// Argument:  int * val +intent(setter)+rank(1)
// Statement: f_setter_native_*
void SCO_ns2_DataPointer_set_items(SCO_datapointer * SH_this, int * val)
{
    // splicer begin namespace.ns2.function.DataPointer_set_items
    // skip call c_setter
    SH_this->items = val;
    // splicer end namespace.ns2.function.DataPointer_set_items
}

}  // extern "C"
