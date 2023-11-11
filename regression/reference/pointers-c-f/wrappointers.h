// wrappointers.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrappointers.h
 * \brief Shroud generated wrapper for pointers library
 */
// For C users and C implementation

#ifndef WRAPPOINTERS_H
#define WRAPPOINTERS_H

// shroud
#include "typespointers.h"

// splicer begin C_declarations
// splicer end C_declarations

int POI_acceptCharArrayIn_bufferify(const char *names,
    size_t SHT_names_size, int SHT_names_len);

void POI_getPtrToScalar_bufferify(POI_SHROUD_array *SHT_nitems_cdesc);

void POI_getPtrToFixedArray_bufferify(
    POI_SHROUD_array *SHT_count_cdesc);

void POI_getPtrToDynamicArray_bufferify(
    POI_SHROUD_array *SHT_count_cdesc);

void POI_getPtrToFuncArray_bufferify(POI_SHROUD_array *SHT_count_cdesc);

void POI_getPtrToConstScalar_bufferify(
    POI_SHROUD_array *SHT_nitems_cdesc);

void POI_getPtrToFixedConstArray_bufferify(
    POI_SHROUD_array *SHT_count_cdesc);

void POI_getPtrToDynamicConstArray_bufferify(
    POI_SHROUD_array *SHT_count_cdesc);

void POI_getAllocToFixedArray_bufferify(
    POI_SHROUD_array *SHT_count_cdesc,
    POI_SHROUD_capsule_data *SHT_count_capsule);

void POI_returnIntPtrToFixedArray_bufferify(
    POI_SHROUD_array *SHT_rv_cdesc);

void POI_returnIntPtrToFixedConstArray_bufferify(
    POI_SHROUD_array *SHT_rv_cdesc);

int POI_returnIntScalar_bufferify(void);

void POI_returnIntAllocToFixedArray_bufferify(
    POI_SHROUD_array *SHT_rv_cdesc,
    POI_SHROUD_capsule_data *SHT_rv_capsule);

#endif  // WRAPPOINTERS_H
