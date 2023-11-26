// wrapstruct.c
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "struct.h"
// shroud
#include <string.h>
#include "wrapstruct.h"


// helper char_blank_fill
// blank fill dest starting at trailing NULL.
static void ShroudCharBlankFill(char *dest, int ndest)
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
// Statement: f_function_native_scalar
// ----------------------------------------
// Argument:  const Cstruct1 * s1
// Statement: f_in_struct_*
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Statement: f_out_char_*_buf
int STR_passStruct2_bufferify(const Cstruct1 * s1, char *outbuf,
    int SHT_outbuf_len)
{
    // splicer begin function.passStruct2_bufferify
    int SHC_rv = passStruct2(s1, outbuf);
    ShroudCharBlankFill(outbuf, SHT_outbuf_len);
    return SHC_rv;
    // splicer end function.passStruct2_bufferify
}

// ----------------------------------------
// Function:  Cstruct1 returnStructByValue
// Statement: c_function_struct_scalar
// ----------------------------------------
// Argument:  int i
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  double d
// Statement: c_in_native_scalar
void STR_returnStructByValue(int i, double d, Cstruct1 *SHC_rv)
{
    // splicer begin function.returnStructByValue
    *SHC_rv = returnStructByValue(i, d);
    // splicer end function.returnStructByValue
}

/**
 * \brief Return a pointer to a struct
 *
 * Does not generate a bufferify C wrapper.
 */
// ----------------------------------------
// Function:  Cstruct1 * returnStructPtr1
// Statement: c_function_struct_*
// ----------------------------------------
// Argument:  int i
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  double d
// Statement: c_in_native_scalar
Cstruct1 * STR_returnStructPtr1(int i, double d)
{
    // splicer begin function.returnStructPtr1
    Cstruct1 * SHC_rv = returnStructPtr1(i, d);
    return SHC_rv;
    // splicer end function.returnStructPtr1
}

/**
 * \brief Return a pointer to a struct
 *
 * Generates a bufferify C wrapper function.
 */
// ----------------------------------------
// Function:  Cstruct1 * returnStructPtr2
// Statement: c_function_struct_*
// ----------------------------------------
// Argument:  int i
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  double d
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Statement: c_out_char_*
Cstruct1 * STR_returnStructPtr2(int i, double d, char * outbuf)
{
    // splicer begin function.returnStructPtr2
    Cstruct1 * SHC_rv = returnStructPtr2(i, d, outbuf);
    return SHC_rv;
    // splicer end function.returnStructPtr2
}

/**
 * \brief Return a pointer to a struct
 *
 * Generates a bufferify C wrapper function.
 */
// ----------------------------------------
// Function:  Cstruct1 * returnStructPtr2
// Statement: f_function_struct_*_pointer
// ----------------------------------------
// Argument:  int i
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  double d
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Statement: f_out_char_*_buf
Cstruct1 * STR_returnStructPtr2_bufferify(int i, double d, char *outbuf,
    int SHT_outbuf_len)
{
    // splicer begin function.returnStructPtr2_bufferify
    Cstruct1 * SHC_rv = returnStructPtr2(i, d, outbuf);
    ShroudCharBlankFill(outbuf, SHT_outbuf_len);
    return SHC_rv;
    // splicer end function.returnStructPtr2_bufferify
}

// ----------------------------------------
// Function:  Cstruct_list * get_global_struct_list
// Statement: c_function_struct_*
Cstruct_list * STR_get_global_struct_list(void)
{
    // splicer begin function.get_global_struct_list
    Cstruct_list * SHC_rv = get_global_struct_list();
    return SHC_rv;
    // splicer end function.get_global_struct_list
}

// ----------------------------------------
// Function:  Cstruct_as_class * Create_Cstruct_as_class
// Statement: c_function_shadow_*_capptr
// start STR_Create_Cstruct_as_class
STR_Cstruct_as_class * STR_Create_Cstruct_as_class(
    STR_Cstruct_as_class * SHC_rv)
{
    // splicer begin function.Create_Cstruct_as_class
    Cstruct_as_class * SHCXX_rv = Create_Cstruct_as_class();
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end function.Create_Cstruct_as_class
}
// end STR_Create_Cstruct_as_class

// ----------------------------------------
// Function:  Cstruct_as_class * Create_Cstruct_as_class_args
// Statement: c_function_shadow_*_capptr
// ----------------------------------------
// Argument:  int x
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int y
// Statement: c_in_native_scalar
STR_Cstruct_as_class * STR_Create_Cstruct_as_class_args(int x, int y,
    STR_Cstruct_as_class * SHC_rv)
{
    // splicer begin function.Create_Cstruct_as_class_args
    Cstruct_as_class * SHCXX_rv = Create_Cstruct_as_class_args(x, y);
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end function.Create_Cstruct_as_class_args
}

// ----------------------------------------
// Function:  int Cstruct_as_class_sum
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  const Cstruct_as_class * point +pass
// Statement: c_in_shadow_*
int STR_Cstruct_as_class_sum(STR_Cstruct_as_class * point)
{
    // splicer begin function.Cstruct_as_class_sum
    const Cstruct_as_class * SHCXX_point =
        (const Cstruct_as_class *) point->addr;
    int SHC_rv = Cstruct_as_class_sum(SHCXX_point);
    return SHC_rv;
    // splicer end function.Cstruct_as_class_sum
}

// ----------------------------------------
// Function:  Cstruct_as_subclass * Create_Cstruct_as_subclass_args
// Statement: c_function_shadow_*_capptr
// ----------------------------------------
// Argument:  int x
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int y
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int z
// Statement: c_in_native_scalar
STR_Cstruct_as_subclass * STR_Create_Cstruct_as_subclass_args(int x,
    int y, int z, STR_Cstruct_as_subclass * SHC_rv)
{
    // splicer begin function.Create_Cstruct_as_subclass_args
    Cstruct_as_subclass * SHCXX_rv = Create_Cstruct_as_subclass_args(x,
        y, z);
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end function.Create_Cstruct_as_subclass_args
}

// Generated by getter/setter
// ----------------------------------------
// Function:  const double * Cstruct_ptr_get_const_dvalue
// Statement: f_getter_native_*_pointer
// ----------------------------------------
// Argument:  Cstruct_ptr * SH_this
// Statement: f_in_struct_*
const double * STR_Cstruct_ptr_get_const_dvalue(Cstruct_ptr * SH_this)
{
    // splicer begin function.Cstruct_ptr_get_const_dvalue
    // skip call c_getter
    return SH_this->const_dvalue;
    // splicer end function.Cstruct_ptr_get_const_dvalue
}

// Generated by getter/setter
// ----------------------------------------
// Function:  void Cstruct_ptr_set_const_dvalue
// Statement: f_setter
// ----------------------------------------
// Argument:  Cstruct_ptr * SH_this
// Statement: f_inout_struct_*
// ----------------------------------------
// Argument:  const double * val +intent(in)
// Statement: f_setter_native_*
void STR_Cstruct_ptr_set_const_dvalue(Cstruct_ptr * SH_this,
    const double * val)
{
    // splicer begin function.Cstruct_ptr_set_const_dvalue
    // skip call c_setter
    SH_this->const_dvalue = val;
    // splicer end function.Cstruct_ptr_set_const_dvalue
}

// Generated by getter/setter
// ----------------------------------------
// Function:  int * Cstruct_list_get_ivalue +dimension(nitems+nitems)
// Statement: f_getter_native_*_cdesc_pointer
// ----------------------------------------
// Argument:  Cstruct_list * SH_this
// Statement: f_in_struct_*
void STR_Cstruct_list_get_ivalue(Cstruct_list * SH_this,
    STR_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.Cstruct_list_get_ivalue
    SHT_rv_cdesc->base_addr = SH_this->ivalue;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->nitems+SH_this->nitems;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.Cstruct_list_get_ivalue
}

// Generated by getter/setter
// ----------------------------------------
// Function:  void Cstruct_list_set_ivalue
// Statement: f_setter
// ----------------------------------------
// Argument:  Cstruct_list * SH_this
// Statement: f_inout_struct_*
// ----------------------------------------
// Argument:  int * val +intent(in)+rank(1)
// Statement: f_setter_native_*
void STR_Cstruct_list_set_ivalue(Cstruct_list * SH_this, int * val)
{
    // splicer begin function.Cstruct_list_set_ivalue
    // skip call c_setter
    SH_this->ivalue = val;
    // splicer end function.Cstruct_list_set_ivalue
}

// Generated by getter/setter
// ----------------------------------------
// Function:  double * Cstruct_list_get_dvalue +dimension(nitems*TWO)
// Statement: f_getter_native_*_cdesc_pointer
// ----------------------------------------
// Argument:  Cstruct_list * SH_this
// Statement: f_in_struct_*
void STR_Cstruct_list_get_dvalue(Cstruct_list * SH_this,
    STR_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.Cstruct_list_get_dvalue
    SHT_rv_cdesc->base_addr = SH_this->dvalue;
    SHT_rv_cdesc->type = SH_TYPE_DOUBLE;
    SHT_rv_cdesc->elem_len = sizeof(double);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->nitems*TWO;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.Cstruct_list_get_dvalue
}

// Generated by getter/setter
// ----------------------------------------
// Function:  void Cstruct_list_set_dvalue
// Statement: f_setter
// ----------------------------------------
// Argument:  Cstruct_list * SH_this
// Statement: f_inout_struct_*
// ----------------------------------------
// Argument:  double * val +intent(in)+rank(1)
// Statement: f_setter_native_*
void STR_Cstruct_list_set_dvalue(Cstruct_list * SH_this, double * val)
{
    // splicer begin function.Cstruct_list_set_dvalue
    // skip call c_setter
    SH_this->dvalue = val;
    // splicer end function.Cstruct_list_set_dvalue
}
