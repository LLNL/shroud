// wrapscope_ns2.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapscope_ns2.h
 * \brief Shroud generated wrapper for ns2 namespace
 */
// For C users and C++ implementation

#ifndef WRAPSCOPE_NS2_H
#define WRAPSCOPE_NS2_H

// shroud
#include "typesscope.h"

// splicer begin namespace.ns2.CXX_declarations
// splicer end namespace.ns2.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

//  ns2::Color
enum SCO_ns2_Color {
    SCO_ns2_RED = 30,
    SCO_ns2_BLUE,
    SCO_ns2_WHITE
};


struct s_SCO_datapointer {
    int nitems;
    int * items;
};
typedef struct s_SCO_datapointer SCO_datapointer;

// splicer begin namespace.ns2.C_declarations
// splicer end namespace.ns2.C_declarations

void SCO_ns2_data_pointer_get_items_bufferify(SCO_datapointer * SH_this,
    SCO_SHROUD_array *SHT_rv_cdesc);

void SCO_ns2_data_pointer_set_items(SCO_datapointer * SH_this,
    int * val);

#ifdef __cplusplus
}
#endif

#endif  // WRAPSCOPE_NS2_H
