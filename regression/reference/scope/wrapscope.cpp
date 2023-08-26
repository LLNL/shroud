// wrapscope.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// shroud
#include "wrapscope.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  int * DataPointer_get_items
// Attrs:     +api(cdesc)+deref(pointer)+intent(getter)+struct(ns3_DataPointer)
// Exact:     c_getter_native_*_cdesc_pointer
// ----------------------------------------
// Argument:  ns3::DataPointer * SH_this
// Attrs:     +intent(in)+struct(ns3_DataPointer)
// Requested: c_in_struct_*
// Match:     c_in_struct
void SCO_DataPointer_get_items_bufferify(SCO_datapointer * SH_this,
    SCO_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.DataPointer_get_items_bufferify
    ns3::DataPointer * SHCXX_SH_this = static_cast<ns3::DataPointer *>
        (static_cast<void *>(SH_this));
    // skip call c_getter
    SHT_rv_cdesc->cxx.addr  = SHCXX_SH_this->items;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SHCXX_SH_this->items;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->nitems;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.DataPointer_get_items_bufferify
}

// ----------------------------------------
// Function:  void DataPointer_set_items
// Attrs:     +intent(setter)
// Exact:     c_setter
// ----------------------------------------
// Argument:  ns3::DataPointer * SH_this
// Attrs:     +intent(inout)+struct(ns3_DataPointer)
// Requested: c_inout_struct_*
// Match:     c_inout_struct
// ----------------------------------------
// Argument:  int * val +intent(in)+rank(1)
// Attrs:     +intent(setter)
// Exact:     c_setter_native_*
void SCO_DataPointer_set_items(SCO_datapointer * SH_this, int * val)
{
    // splicer begin function.DataPointer_set_items
    ns3::DataPointer * SHCXX_SH_this = static_cast<ns3::DataPointer *>
        (static_cast<void *>(SH_this));
    // skip call c_setter
    SHCXX_SH_this->items = val;
    // splicer end function.DataPointer_set_items
}

}  // extern "C"
