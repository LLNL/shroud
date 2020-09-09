// wrapCstruct_as_class.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapCstruct_as_class.h"
#include "struct.h"

// splicer begin class.Cstruct_as_class.CXX_definitions
// splicer end class.Cstruct_as_class.CXX_definitions

extern "C" {

// splicer begin class.Cstruct_as_class.C_definitions
// splicer end class.Cstruct_as_class.C_definitions

// ----------------------------------------
// Function:  int getX1
// Requested: c_native_scalar_result
// Match:     c_default
int STR_Cstruct_as_class_get_x1(STR_Cstruct_as_class * self)
{
    Cstruct_as_class *SH_this = static_cast<Cstruct_as_class *>
        (self->addr);
    // splicer begin class.Cstruct_as_class.method.get_x1
    return SH_this->x1;
    // splicer end class.Cstruct_as_class.method.get_x1
}

// ----------------------------------------
// Function:  void setX1
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int val +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
void STR_Cstruct_as_class_set_x1(STR_Cstruct_as_class * self, int val)
{
    Cstruct_as_class *SH_this = static_cast<Cstruct_as_class *>
        (self->addr);
    // splicer begin class.Cstruct_as_class.method.set_x1
    SH_this->x1 = val;
    return;
    // splicer end class.Cstruct_as_class.method.set_x1
}

// ----------------------------------------
// Function:  int getY1
// Requested: c_native_scalar_result
// Match:     c_default
int STR_Cstruct_as_class_get_y1(STR_Cstruct_as_class * self)
{
    Cstruct_as_class *SH_this = static_cast<Cstruct_as_class *>
        (self->addr);
    // splicer begin class.Cstruct_as_class.method.get_y1
    return SH_this->y1;
    // splicer end class.Cstruct_as_class.method.get_y1
}

// ----------------------------------------
// Function:  void setY1
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int val +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
void STR_Cstruct_as_class_set_y1(STR_Cstruct_as_class * self, int val)
{
    Cstruct_as_class *SH_this = static_cast<Cstruct_as_class *>
        (self->addr);
    // splicer begin class.Cstruct_as_class.method.set_y1
    SH_this->y1 = val;
    return;
    // splicer end class.Cstruct_as_class.method.set_y1
}

}  // extern "C"
