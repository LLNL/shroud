// wrapstruct.cpp
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
#include <cstring>
#include "wrapstruct.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper ShroudStrBlankFill
// blank fill dest starting at trailing NULL.
static void ShroudStrBlankFill(char *dest, int ndest)
{
   int nm = std::strlen(dest);
   if(ndest > nm) std::memset(dest+nm,' ',ndest-nm);
}
// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  int passStructByValue
// Attrs:     +intent(function)
// Exact:     c_function_native_scalar
// ----------------------------------------
// Argument:  Cstruct1 arg +value
// Attrs:     +intent(in)
// Exact:     c_in_struct_scalar
// start STR_passStructByValue
int STR_passStructByValue(STR_cstruct1 arg)
{
    // splicer begin function.passStructByValue
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        &arg));
    int SHC_rv = passStructByValue(*SHCXX_arg);
    return SHC_rv;
    // splicer end function.passStructByValue
}
// end STR_passStructByValue

// ----------------------------------------
// Function:  int passStruct1
// Attrs:     +intent(function)
// Exact:     c_function_native_scalar
// ----------------------------------------
// Argument:  const Cstruct1 * arg
// Attrs:     +intent(in)
// Exact:     c_in_struct_*
// start STR_passStruct1
int STR_passStruct1(const STR_cstruct1 * arg)
{
    // splicer begin function.passStruct1
    const Cstruct1 * SHCXX_arg = static_cast<const Cstruct1 *>
        (static_cast<const void *>(arg));
    int SHC_rv = passStruct1(SHCXX_arg);
    return SHC_rv;
    // splicer end function.passStruct1
}
// end STR_passStruct1

/**
 * Pass name argument which will build a bufferify function.
 */
// ----------------------------------------
// Function:  int passStruct2
// Attrs:     +intent(function)
// Exact:     c_function_native_scalar
// ----------------------------------------
// Argument:  const Cstruct1 * s1
// Attrs:     +intent(in)
// Exact:     c_in_struct_*
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Attrs:     +intent(out)
// Requested: c_out_char_*
// Match:     c_default
int STR_passStruct2(const STR_cstruct1 * s1, char * outbuf)
{
    // splicer begin function.passStruct2
    const Cstruct1 * SHCXX_s1 = static_cast<const Cstruct1 *>
        (static_cast<const void *>(s1));
    int SHC_rv = passStruct2(SHCXX_s1, outbuf);
    return SHC_rv;
    // splicer end function.passStruct2
}

/**
 * Pass name argument which will build a bufferify function.
 */
// ----------------------------------------
// Function:  int passStruct2
// Attrs:     +intent(function)
// Exact:     c_function_native_scalar
// ----------------------------------------
// Argument:  const Cstruct1 * s1
// Attrs:     +intent(in)
// Exact:     c_in_struct_*
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Attrs:     +api(buf)+intent(out)
// Exact:     c_out_char_*_buf
int STR_passStruct2_bufferify(const STR_cstruct1 * s1, char *outbuf,
    int SHT_outbuf_len)
{
    // splicer begin function.passStruct2_bufferify
    const Cstruct1 * SHCXX_s1 = static_cast<const Cstruct1 *>
        (static_cast<const void *>(s1));
    int SHC_rv = passStruct2(SHCXX_s1, outbuf);
    ShroudStrBlankFill(outbuf, SHT_outbuf_len);
    return SHC_rv;
    // splicer end function.passStruct2_bufferify
}

// ----------------------------------------
// Function:  int acceptStructInPtr
// Attrs:     +intent(function)
// Exact:     c_function_native_scalar
// ----------------------------------------
// Argument:  Cstruct1 * arg +intent(in)
// Attrs:     +intent(in)
// Exact:     c_in_struct_*
int STR_acceptStructInPtr(STR_cstruct1 * arg)
{
    // splicer begin function.acceptStructInPtr
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        arg));
    int SHC_rv = acceptStructInPtr(SHCXX_arg);
    return SHC_rv;
    // splicer end function.acceptStructInPtr
}

/**
 * Pass name argument which will build a bufferify function.
 */
// ----------------------------------------
// Function:  void acceptStructOutPtr
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  Cstruct1 * arg +intent(out)
// Attrs:     +intent(out)
// Exact:     c_out_struct_*
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
void STR_acceptStructOutPtr(STR_cstruct1 * arg, int i, double d)
{
    // splicer begin function.acceptStructOutPtr
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        arg));
    acceptStructOutPtr(SHCXX_arg, i, d);
    // splicer end function.acceptStructOutPtr
}

// ----------------------------------------
// Function:  void acceptStructInOutPtr
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  Cstruct1 * arg +intent(inout)
// Attrs:     +intent(inout)
// Exact:     c_inout_struct_*
void STR_acceptStructInOutPtr(STR_cstruct1 * arg)
{
    // splicer begin function.acceptStructInOutPtr
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        arg));
    acceptStructInOutPtr(SHCXX_arg);
    // splicer end function.acceptStructInOutPtr
}

// ----------------------------------------
// Function:  Cstruct1 returnStructByValue
// Attrs:     +intent(function)
// Exact:     c_function_struct_scalar
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
void STR_returnStructByValue(int i, double d, STR_cstruct1 *SHC_rv)
{
    // splicer begin function.returnStructByValue
    Cstruct1 SHCXX_rv = returnStructByValue(i, d);
    memcpy((void *) SHC_rv, (void *) &SHCXX_rv, sizeof(SHCXX_rv));
    // splicer end function.returnStructByValue
}

/**
 * \brief Return a pointer to a struct
 *
 * Does not generate a bufferify C wrapper.
 */
// ----------------------------------------
// Function:  Cstruct1 * returnStructPtr1
// Attrs:     +deref(pointer)+intent(function)
// Exact:     c_function_struct_*_pointer
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
STR_cstruct1 * STR_returnStructPtr1(int i, double d)
{
    // splicer begin function.returnStructPtr1
    Cstruct1 * SHCXX_rv = returnStructPtr1(i, d);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(SHCXX_rv));
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
// Attrs:     +deref(pointer)+intent(function)
// Exact:     c_function_struct_*_pointer
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Attrs:     +intent(out)
// Requested: c_out_char_*
// Match:     c_default
STR_cstruct1 * STR_returnStructPtr2(int i, double d, char * outbuf)
{
    // splicer begin function.returnStructPtr2
    Cstruct1 * SHCXX_rv = returnStructPtr2(i, d, outbuf);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(SHCXX_rv));
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
// Attrs:     +deref(pointer)+intent(function)
// Exact:     c_function_struct_*_pointer
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Attrs:     +api(buf)+intent(out)
// Exact:     c_out_char_*_buf
STR_cstruct1 * STR_returnStructPtr2_bufferify(int i, double d,
    char *outbuf, int SHT_outbuf_len)
{
    // splicer begin function.returnStructPtr2_bufferify
    Cstruct1 * SHCXX_rv = returnStructPtr2(i, d, outbuf);
    ShroudStrBlankFill(outbuf, SHT_outbuf_len);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
    // splicer end function.returnStructPtr2_bufferify
}

// ----------------------------------------
// Function:  Cstruct_list * get_global_struct_list
// Attrs:     +deref(pointer)+intent(function)
// Exact:     c_function_struct_*_pointer
STR_cstruct_list * STR_get_global_struct_list(void)
{
    // splicer begin function.get_global_struct_list
    Cstruct_list * SHCXX_rv = get_global_struct_list();
    STR_cstruct_list * SHC_rv = static_cast<STR_cstruct_list *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
    // splicer end function.get_global_struct_list
}

// ----------------------------------------
// Function:  Cstruct_as_class * Create_Cstruct_as_class
// Attrs:     +api(capptr)+intent(function)
// Exact:     c_function_shadow_*_capptr
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
// Attrs:     +api(capptr)+intent(function)
// Exact:     c_function_shadow_*_capptr
// ----------------------------------------
// Argument:  int x +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
// ----------------------------------------
// Argument:  int y +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
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
// Attrs:     +intent(function)
// Exact:     c_function_native_scalar
// ----------------------------------------
// Argument:  const Cstruct_as_class * point +pass
// Attrs:     +intent(in)
// Exact:     c_in_shadow_*
int STR_Cstruct_as_class_sum(STR_Cstruct_as_class * point)
{
    // splicer begin function.Cstruct_as_class_sum
    const Cstruct_as_class * SHCXX_point =
        static_cast<const Cstruct_as_class *>(point->addr);
    int SHC_rv = Cstruct_as_class_sum(SHCXX_point);
    return SHC_rv;
    // splicer end function.Cstruct_as_class_sum
}

// ----------------------------------------
// Function:  Cstruct_as_subclass * Create_Cstruct_as_subclass_args
// Attrs:     +api(capptr)+intent(function)
// Exact:     c_function_shadow_*_capptr
// ----------------------------------------
// Argument:  int x +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
// ----------------------------------------
// Argument:  int y +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
// ----------------------------------------
// Argument:  int z +value
// Attrs:     +intent(in)
// Exact:     c_in_native_scalar
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

// ----------------------------------------
// Function:  const double * Cstruct_ptr_get_const_dvalue
// Attrs:     +deref(pointer)+intent(getter)+struct(Cstruct_ptr)
// Exact:     c_getter_native_*_pointer
// ----------------------------------------
// Argument:  Cstruct_ptr * SH_this
// Attrs:     +intent(in)+struct(Cstruct_ptr)
// Exact:     c_in_struct_*
const double * STR_Cstruct_ptr_get_const_dvalue(
    STR_cstruct_ptr * SH_this)
{
    // splicer begin function.Cstruct_ptr_get_const_dvalue
    Cstruct_ptr * SHCXX_SH_this = static_cast<Cstruct_ptr *>
        (static_cast<void *>(SH_this));
    // skip call c_getter
    return SHCXX_SH_this->const_dvalue;
    // splicer end function.Cstruct_ptr_get_const_dvalue
}

// ----------------------------------------
// Function:  void Cstruct_ptr_set_const_dvalue
// Attrs:     +intent(setter)
// Exact:     c_setter
// ----------------------------------------
// Argument:  Cstruct_ptr * SH_this
// Attrs:     +intent(inout)+struct(Cstruct_ptr)
// Exact:     c_inout_struct_*
// ----------------------------------------
// Argument:  const double * val +intent(in)
// Attrs:     +intent(setter)
// Exact:     c_setter_native_*
void STR_Cstruct_ptr_set_const_dvalue(STR_cstruct_ptr * SH_this,
    const double * val)
{
    // splicer begin function.Cstruct_ptr_set_const_dvalue
    Cstruct_ptr * SHCXX_SH_this = static_cast<Cstruct_ptr *>
        (static_cast<void *>(SH_this));
    // skip call c_setter
    SHCXX_SH_this->const_dvalue = val;
    // splicer end function.Cstruct_ptr_set_const_dvalue
}

// ----------------------------------------
// Function:  int * Cstruct_list_get_ivalue
// Attrs:     +api(cdesc)+deref(pointer)+intent(getter)+struct(Cstruct_list)
// Exact:     c_getter_native_*_cdesc_pointer
// ----------------------------------------
// Argument:  Cstruct_list * SH_this
// Attrs:     +intent(in)+struct(Cstruct_list)
// Exact:     c_in_struct_*
void STR_Cstruct_list_get_ivalue_bufferify(STR_cstruct_list * SH_this,
    STR_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.Cstruct_list_get_ivalue_bufferify
    Cstruct_list * SHCXX_SH_this = static_cast<Cstruct_list *>
        (static_cast<void *>(SH_this));
    // skip call c_getter
    SHT_rv_cdesc->cxx.addr  = SHCXX_SH_this->ivalue;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SHCXX_SH_this->ivalue;
    SHT_rv_cdesc->type = SH_TYPE_INT;
    SHT_rv_cdesc->elem_len = sizeof(int);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->nitems+SH_this->nitems;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.Cstruct_list_get_ivalue_bufferify
}

// ----------------------------------------
// Function:  void Cstruct_list_set_ivalue
// Attrs:     +intent(setter)
// Exact:     c_setter
// ----------------------------------------
// Argument:  Cstruct_list * SH_this
// Attrs:     +intent(inout)+struct(Cstruct_list)
// Exact:     c_inout_struct_*
// ----------------------------------------
// Argument:  int * val +intent(in)+rank(1)
// Attrs:     +intent(setter)
// Exact:     c_setter_native_*
void STR_Cstruct_list_set_ivalue(STR_cstruct_list * SH_this, int * val)
{
    // splicer begin function.Cstruct_list_set_ivalue
    Cstruct_list * SHCXX_SH_this = static_cast<Cstruct_list *>
        (static_cast<void *>(SH_this));
    // skip call c_setter
    SHCXX_SH_this->ivalue = val;
    // splicer end function.Cstruct_list_set_ivalue
}

// ----------------------------------------
// Function:  double * Cstruct_list_get_dvalue
// Attrs:     +api(cdesc)+deref(pointer)+intent(getter)+struct(Cstruct_list)
// Exact:     c_getter_native_*_cdesc_pointer
// ----------------------------------------
// Argument:  Cstruct_list * SH_this
// Attrs:     +intent(in)+struct(Cstruct_list)
// Exact:     c_in_struct_*
void STR_Cstruct_list_get_dvalue_bufferify(STR_cstruct_list * SH_this,
    STR_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.Cstruct_list_get_dvalue_bufferify
    Cstruct_list * SHCXX_SH_this = static_cast<Cstruct_list *>
        (static_cast<void *>(SH_this));
    // skip call c_getter
    SHT_rv_cdesc->cxx.addr  = SHCXX_SH_this->dvalue;
    SHT_rv_cdesc->cxx.idtor = 0;
    SHT_rv_cdesc->addr.base = SHCXX_SH_this->dvalue;
    SHT_rv_cdesc->type = SH_TYPE_DOUBLE;
    SHT_rv_cdesc->elem_len = sizeof(double);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->nitems*TWO;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.Cstruct_list_get_dvalue_bufferify
}

// ----------------------------------------
// Function:  void Cstruct_list_set_dvalue
// Attrs:     +intent(setter)
// Exact:     c_setter
// ----------------------------------------
// Argument:  Cstruct_list * SH_this
// Attrs:     +intent(inout)+struct(Cstruct_list)
// Exact:     c_inout_struct_*
// ----------------------------------------
// Argument:  double * val +intent(in)+rank(1)
// Attrs:     +intent(setter)
// Exact:     c_setter_native_*
void STR_Cstruct_list_set_dvalue(STR_cstruct_list * SH_this,
    double * val)
{
    // splicer begin function.Cstruct_list_set_dvalue
    Cstruct_list * SHCXX_SH_this = static_cast<Cstruct_list *>
        (static_cast<void *>(SH_this));
    // skip call c_setter
    SHCXX_SH_this->dvalue = val;
    // splicer end function.Cstruct_list_set_dvalue
}

}  // extern "C"
