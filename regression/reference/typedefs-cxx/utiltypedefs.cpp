// utiltypedefs.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// shroud
#include "typestypedefs.h"


#ifdef __cplusplus
extern "C" {
#endif

// start release allocated memory
// Release library allocated memory.
void TYP_SHROUD_memory_destructor(TYP_SHROUD_capsule_data *cap)
{
    cap->addr = nullptr;
    cap->idtor = 0;  // avoid deleting again
}
// end release allocated memory

#ifdef __cplusplus
}
#endif
