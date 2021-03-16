// wrapCstruct_as_subclass.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapCstruct_as_subclass.h"

// cxx_header
#include "struct.h"

// splicer begin class.Cstruct_as_subclass.CXX_definitions
// splicer end class.Cstruct_as_subclass.CXX_definitions

extern "C" {

// splicer begin class.Cstruct_as_subclass.C_definitions
// splicer end class.Cstruct_as_subclass.C_definitions

// ----------------------------------------
// Function:  int getX1
// Attrs:     +intent(subroutine)
// Requested: c_subroutine_native_scalar
// Match:     c_subroutine
// start STR_Cstruct_as_subclass_get_x1
int STR_Cstruct_as_subclass_get_x1(STR_Cstruct_as_subclass * self)
{
    Cstruct_as_subclass *SH_this = static_cast<Cstruct_as_subclass *>
        (self->addr);
    // splicer begin class.Cstruct_as_subclass.method.get_x1
    return SH_this->x1;
    // splicer end class.Cstruct_as_subclass.method.get_x1
}
// end STR_Cstruct_as_subclass_get_x1

// ----------------------------------------
// Function:  void setX1
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int val +intent(in)+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// start STR_Cstruct_as_subclass_set_x1
void STR_Cstruct_as_subclass_set_x1(STR_Cstruct_as_subclass * self,
    int val)
{
    Cstruct_as_subclass *SH_this = static_cast<Cstruct_as_subclass *>
        (self->addr);
    // splicer begin class.Cstruct_as_subclass.method.set_x1
    SH_this->x1 = val;
    return;
    // splicer end class.Cstruct_as_subclass.method.set_x1
}
// end STR_Cstruct_as_subclass_set_x1

// ----------------------------------------
// Function:  int getY1
// Attrs:     +intent(subroutine)
// Requested: c_subroutine_native_scalar
// Match:     c_subroutine
// start STR_Cstruct_as_subclass_get_y1
int STR_Cstruct_as_subclass_get_y1(STR_Cstruct_as_subclass * self)
{
    Cstruct_as_subclass *SH_this = static_cast<Cstruct_as_subclass *>
        (self->addr);
    // splicer begin class.Cstruct_as_subclass.method.get_y1
    return SH_this->y1;
    // splicer end class.Cstruct_as_subclass.method.get_y1
}
// end STR_Cstruct_as_subclass_get_y1

// ----------------------------------------
// Function:  void setY1
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int val +intent(in)+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// start STR_Cstruct_as_subclass_set_y1
void STR_Cstruct_as_subclass_set_y1(STR_Cstruct_as_subclass * self,
    int val)
{
    Cstruct_as_subclass *SH_this = static_cast<Cstruct_as_subclass *>
        (self->addr);
    // splicer begin class.Cstruct_as_subclass.method.set_y1
    SH_this->y1 = val;
    return;
    // splicer end class.Cstruct_as_subclass.method.set_y1
}
// end STR_Cstruct_as_subclass_set_y1

// ----------------------------------------
// Function:  int getZ1
// Attrs:     +intent(subroutine)
// Requested: c_subroutine_native_scalar
// Match:     c_subroutine
// start STR_Cstruct_as_subclass_get_z1
int STR_Cstruct_as_subclass_get_z1(STR_Cstruct_as_subclass * self)
{
    Cstruct_as_subclass *SH_this = static_cast<Cstruct_as_subclass *>
        (self->addr);
    // splicer begin class.Cstruct_as_subclass.method.get_z1
    return SH_this->z1;
    // splicer end class.Cstruct_as_subclass.method.get_z1
}
// end STR_Cstruct_as_subclass_get_z1

// ----------------------------------------
// Function:  void setZ1
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int val +intent(in)+value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// start STR_Cstruct_as_subclass_set_z1
void STR_Cstruct_as_subclass_set_z1(STR_Cstruct_as_subclass * self,
    int val)
{
    Cstruct_as_subclass *SH_this = static_cast<Cstruct_as_subclass *>
        (self->addr);
    // splicer begin class.Cstruct_as_subclass.method.set_z1
    SH_this->z1 = val;
    return;
    // splicer end class.Cstruct_as_subclass.method.set_z1
}
// end STR_Cstruct_as_subclass_set_z1

}  // extern "C"
