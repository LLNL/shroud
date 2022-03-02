// wrapCstruct_as_subclass.c
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "struct.h"
// shroud
#include "wrapCstruct_as_subclass.h"

// splicer begin class.Cstruct_as_subclass.C_definitions
// splicer end class.Cstruct_as_subclass.C_definitions

// ----------------------------------------
// Function:  int getX1
// Attrs:     +intent(getter)
// Exact:     c_getter_native_scalar
// start STR_Cstruct_as_subclass_get_x1
int STR_Cstruct_as_subclass_get_x1(STR_Cstruct_as_subclass * self)
{
    Cstruct_as_subclass *SH_this = (Cstruct_as_subclass *) self->addr;
    // splicer begin class.Cstruct_as_subclass.method.get_x1
    // skip call c_getter
    return SH_this->x1;
    // splicer end class.Cstruct_as_subclass.method.get_x1
}
// end STR_Cstruct_as_subclass_get_x1

// ----------------------------------------
// Function:  void setX1
// Attrs:     +intent(setter)
// Exact:     c_setter
// ----------------------------------------
// Argument:  int val +intent(in)+value
// Attrs:     +intent(setter)
// Exact:     c_setter_native_scalar
// start STR_Cstruct_as_subclass_set_x1
void STR_Cstruct_as_subclass_set_x1(STR_Cstruct_as_subclass * self,
    int val)
{
    Cstruct_as_subclass *SH_this = (Cstruct_as_subclass *) self->addr;
    // splicer begin class.Cstruct_as_subclass.method.set_x1
    // skip call c_setter
    SH_this->x1 = val;
    // splicer end class.Cstruct_as_subclass.method.set_x1
}
// end STR_Cstruct_as_subclass_set_x1

// ----------------------------------------
// Function:  int getY1
// Attrs:     +intent(getter)
// Exact:     c_getter_native_scalar
// start STR_Cstruct_as_subclass_get_y1
int STR_Cstruct_as_subclass_get_y1(STR_Cstruct_as_subclass * self)
{
    Cstruct_as_subclass *SH_this = (Cstruct_as_subclass *) self->addr;
    // splicer begin class.Cstruct_as_subclass.method.get_y1
    // skip call c_getter
    return SH_this->y1;
    // splicer end class.Cstruct_as_subclass.method.get_y1
}
// end STR_Cstruct_as_subclass_get_y1

// ----------------------------------------
// Function:  void setY1
// Attrs:     +intent(setter)
// Exact:     c_setter
// ----------------------------------------
// Argument:  int val +intent(in)+value
// Attrs:     +intent(setter)
// Exact:     c_setter_native_scalar
// start STR_Cstruct_as_subclass_set_y1
void STR_Cstruct_as_subclass_set_y1(STR_Cstruct_as_subclass * self,
    int val)
{
    Cstruct_as_subclass *SH_this = (Cstruct_as_subclass *) self->addr;
    // splicer begin class.Cstruct_as_subclass.method.set_y1
    // skip call c_setter
    SH_this->y1 = val;
    // splicer end class.Cstruct_as_subclass.method.set_y1
}
// end STR_Cstruct_as_subclass_set_y1

// ----------------------------------------
// Function:  int getZ1
// Attrs:     +intent(getter)
// Exact:     c_getter_native_scalar
// start STR_Cstruct_as_subclass_get_z1
int STR_Cstruct_as_subclass_get_z1(STR_Cstruct_as_subclass * self)
{
    Cstruct_as_subclass *SH_this = (Cstruct_as_subclass *) self->addr;
    // splicer begin class.Cstruct_as_subclass.method.get_z1
    // skip call c_getter
    return SH_this->z1;
    // splicer end class.Cstruct_as_subclass.method.get_z1
}
// end STR_Cstruct_as_subclass_get_z1

// ----------------------------------------
// Function:  void setZ1
// Attrs:     +intent(setter)
// Exact:     c_setter
// ----------------------------------------
// Argument:  int val +intent(in)+value
// Attrs:     +intent(setter)
// Exact:     c_setter_native_scalar
// start STR_Cstruct_as_subclass_set_z1
void STR_Cstruct_as_subclass_set_z1(STR_Cstruct_as_subclass * self,
    int val)
{
    Cstruct_as_subclass *SH_this = (Cstruct_as_subclass *) self->addr;
    // splicer begin class.Cstruct_as_subclass.method.set_z1
    // skip call c_setter
    SH_this->z1 = val;
    // splicer end class.Cstruct_as_subclass.method.set_z1
}
// end STR_Cstruct_as_subclass_set_z1
