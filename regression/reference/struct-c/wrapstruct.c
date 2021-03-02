// wrapstruct.c
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapstruct.h"

// cxx_header
#include "struct.h"
// shroud
#include "typesstruct.h"
#include <stdlib.h>
#include <string.h>


// helper ShroudStrBlankFill
// blank fill dest starting at trailing NULL.
static void ShroudStrBlankFill(char *dest, int ndest)
{
   int nm = strlen(dest);
   if(ndest > nm) memset(dest+nm,' ',ndest-nm);
}
// splicer begin C_definitions
// splicer end C_definitions

/**
 * Pass name argument which will build a bufferify function.
 */
// ----------------------------------------
// Function:  int passStruct2
// Attrs:     +intent(result)
// Requested: c_native_scalar_result_buf
// Match:     c_default
// ----------------------------------------
// Argument:  const Cstruct1 * s1
// Attrs:     +intent(in)
// Requested: c_struct_*_in_buf
// Match:     c_struct
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)+len(Noutbuf)
// Attrs:     +intent(out)
// Exact:     c_char_*_out_buf
int STR_pass_struct2_bufferify(const Cstruct1 * s1, char * outbuf,
    int Noutbuf)
{
    // splicer begin function.pass_struct2_bufferify
    int SHC_rv = passStruct2(s1, outbuf);
    ShroudStrBlankFill(outbuf, Noutbuf);
    return SHC_rv;
    // splicer end function.pass_struct2_bufferify
}

/**
 * \brief Return a pointer to a struct
 *
 * Generates a bufferify C wrapper function.
 */
// ----------------------------------------
// Function:  Cstruct1 * returnStructPtr2 +deref(pointer)
// Attrs:     +deref(pointer)+intent(result)
// Requested: c_struct_*_result_buf
// Match:     c_struct_result
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Requested: c_native_scalar_in_buf
// Match:     c_default
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Requested: c_native_scalar_in_buf
// Match:     c_default
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)+len(Noutbuf)
// Attrs:     +intent(out)
// Exact:     c_char_*_out_buf
Cstruct1 * STR_return_struct_ptr2_bufferify(int i, double d,
    char * outbuf, int Noutbuf)
{
    // splicer begin function.return_struct_ptr2_bufferify
    Cstruct1 * SHC_rv = returnStructPtr2(i, d, outbuf);
    ShroudStrBlankFill(outbuf, Noutbuf);
    return SHC_rv;
    // splicer end function.return_struct_ptr2_bufferify
}

// ----------------------------------------
// Function:  Cstruct_as_class * Create_Cstruct_as_class
// Attrs:     +intent(result)
// Requested: c_shadow_*_result
// Match:     c_shadow_result
// start STR_create__cstruct_as_class
STR_Cstruct_as_class * STR_create__cstruct_as_class(
    STR_Cstruct_as_class * SHadow_rv)
{
    // splicer begin function.create__cstruct_as_class
    Cstruct_as_class * SHC_rv = Create_Cstruct_as_class();
    SHadow_rv->addr = SHC_rv;
    SHadow_rv->idtor = 0;
    return SHadow_rv;
    // splicer end function.create__cstruct_as_class
}
// end STR_create__cstruct_as_class

// ----------------------------------------
// Function:  Cstruct_as_class * Create_Cstruct_as_class_args
// Attrs:     +intent(result)
// Requested: c_shadow_*_result
// Match:     c_shadow_result
// ----------------------------------------
// Argument:  int x +value
// Attrs:     +intent(in)
// Requested: c_native_scalar_in
// Match:     c_default
// ----------------------------------------
// Argument:  int y +value
// Attrs:     +intent(in)
// Requested: c_native_scalar_in
// Match:     c_default
STR_Cstruct_as_class * STR_create__cstruct_as_class_args(int x, int y,
    STR_Cstruct_as_class * SHadow_rv)
{
    // splicer begin function.create__cstruct_as_class_args
    Cstruct_as_class * SHC_rv = Create_Cstruct_as_class_args(x, y);
    SHadow_rv->addr = SHC_rv;
    SHadow_rv->idtor = 0;
    return SHadow_rv;
    // splicer end function.create__cstruct_as_class_args
}

// ----------------------------------------
// Function:  int Cstruct_as_class_sum
// Attrs:     +intent(result)
// Requested: c_native_scalar_result
// Match:     c_default
// ----------------------------------------
// Argument:  const Cstruct_as_class * point +pass
// Attrs:     +intent(in)
// Requested: c_shadow_*_in
// Match:     c_shadow_in
int STR_cstruct_as_class_sum(STR_Cstruct_as_class * point)
{
    // splicer begin function.cstruct_as_class_sum
    const Cstruct_as_class * SHCXX_point =
        (const Cstruct_as_class *) point->addr;
    int SHC_rv = Cstruct_as_class_sum(SHCXX_point);
    return SHC_rv;
    // splicer end function.cstruct_as_class_sum
}

// ----------------------------------------
// Function:  Cstruct_as_subclass * Create_Cstruct_as_subclass_args
// Attrs:     +intent(result)
// Requested: c_shadow_*_result
// Match:     c_shadow_result
// ----------------------------------------
// Argument:  int x +value
// Attrs:     +intent(in)
// Requested: c_native_scalar_in
// Match:     c_default
// ----------------------------------------
// Argument:  int y +value
// Attrs:     +intent(in)
// Requested: c_native_scalar_in
// Match:     c_default
// ----------------------------------------
// Argument:  int z +value
// Attrs:     +intent(in)
// Requested: c_native_scalar_in
// Match:     c_default
STR_Cstruct_as_subclass * STR_create__cstruct_as_subclass_args(int x,
    int y, int z, STR_Cstruct_as_subclass * SHadow_rv)
{
    // splicer begin function.create__cstruct_as_subclass_args
    Cstruct_as_subclass * SHC_rv = Create_Cstruct_as_subclass_args(x, y,
        z);
    SHadow_rv->addr = SHC_rv;
    SHadow_rv->idtor = 0;
    return SHadow_rv;
    // splicer end function.create__cstruct_as_subclass_args
}

// start release allocated memory
// Release library allocated memory.
void STR_SHROUD_memory_destructor(STR_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}
// end release allocated memory
