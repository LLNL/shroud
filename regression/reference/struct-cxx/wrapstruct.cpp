// wrapstruct.cpp
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
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  Cstruct1 arg +value
// Attrs:     +intent(in)
// Requested: c_in_struct_scalar
// Match:     c_in_struct
// start STR_pass_struct_by_value
int STR_pass_struct_by_value(STR_cstruct1 arg)
{
    // splicer begin function.pass_struct_by_value
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        &arg));
    int SHC_rv = passStructByValue(*SHCXX_arg);
    return SHC_rv;
    // splicer end function.pass_struct_by_value
}
// end STR_pass_struct_by_value

// ----------------------------------------
// Function:  int passStruct1
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const Cstruct1 * arg
// Attrs:     +intent(in)
// Requested: c_in_struct_*
// Match:     c_in_struct
// start STR_pass_struct1
int STR_pass_struct1(const STR_cstruct1 * arg)
{
    // splicer begin function.pass_struct1
    const Cstruct1 * SHCXX_arg = static_cast<const Cstruct1 *>
        (static_cast<const void *>(arg));
    int SHC_rv = passStruct1(SHCXX_arg);
    return SHC_rv;
    // splicer end function.pass_struct1
}
// end STR_pass_struct1

/**
 * Pass name argument which will build a bufferify function.
 */
// ----------------------------------------
// Function:  int passStruct2
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const Cstruct1 * s1
// Attrs:     +intent(in)
// Requested: c_in_struct_*
// Match:     c_in_struct
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Attrs:     +intent(out)
// Requested: c_out_char_*
// Match:     c_default
int STR_pass_struct2(const STR_cstruct1 * s1, char * outbuf)
{
    // splicer begin function.pass_struct2
    const Cstruct1 * SHCXX_s1 = static_cast<const Cstruct1 *>
        (static_cast<const void *>(s1));
    int SHC_rv = passStruct2(SHCXX_s1, outbuf);
    return SHC_rv;
    // splicer end function.pass_struct2
}

/**
 * Pass name argument which will build a bufferify function.
 */
// ----------------------------------------
// Function:  int passStruct2
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const Cstruct1 * s1
// Attrs:     +intent(in)
// Requested: c_in_struct_*
// Match:     c_in_struct
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Attrs:     +api(buf)+intent(out)
// Exact:     c_out_char_*_buf
int STR_pass_struct2_bufferify(const STR_cstruct1 * s1, char *outbuf,
    int SHT_outbuf_len)
{
    // splicer begin function.pass_struct2_bufferify
    const Cstruct1 * SHCXX_s1 = static_cast<const Cstruct1 *>
        (static_cast<const void *>(s1));
    int SHC_rv = passStruct2(SHCXX_s1, outbuf);
    ShroudStrBlankFill(outbuf, SHT_outbuf_len);
    return SHC_rv;
    // splicer end function.pass_struct2_bufferify
}

// ----------------------------------------
// Function:  int acceptStructInPtr
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  Cstruct1 * arg +intent(in)
// Attrs:     +intent(in)
// Requested: c_in_struct_*
// Match:     c_in_struct
int STR_accept_struct_in_ptr(STR_cstruct1 * arg)
{
    // splicer begin function.accept_struct_in_ptr
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        arg));
    int SHC_rv = acceptStructInPtr(SHCXX_arg);
    return SHC_rv;
    // splicer end function.accept_struct_in_ptr
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
// Requested: c_out_struct_*
// Match:     c_out_struct
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void STR_accept_struct_out_ptr(STR_cstruct1 * arg, int i, double d)
{
    // splicer begin function.accept_struct_out_ptr
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        arg));
    acceptStructOutPtr(SHCXX_arg, i, d);
    // splicer end function.accept_struct_out_ptr
}

// ----------------------------------------
// Function:  void acceptStructInOutPtr
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  Cstruct1 * arg +intent(inout)
// Attrs:     +intent(inout)
// Requested: c_inout_struct_*
// Match:     c_inout_struct
void STR_accept_struct_in_out_ptr(STR_cstruct1 * arg)
{
    // splicer begin function.accept_struct_in_out_ptr
    Cstruct1 * SHCXX_arg = static_cast<Cstruct1 *>(static_cast<void *>(
        arg));
    acceptStructInOutPtr(SHCXX_arg);
    // splicer end function.accept_struct_in_out_ptr
}

// ----------------------------------------
// Function:  Cstruct1 returnStructByValue
// Attrs:     +intent(function)
// Requested: c_function_struct_scalar
// Match:     c_function_struct
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
STR_cstruct1 STR_return_struct_by_value(int i, double d)
{
    // splicer begin function.return_struct_by_value
    Cstruct1 SHCXX_rv = returnStructByValue(i, d);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(&SHCXX_rv));
    return *SHC_rv;
    // splicer end function.return_struct_by_value
}

/**
 * \brief Return a pointer to a struct
 *
 * Does not generate a bufferify C wrapper.
 */
// ----------------------------------------
// Function:  Cstruct1 * returnStructPtr1
// Attrs:     +deref(pointer)+intent(function)
// Requested: c_function_struct_*_pointer
// Match:     c_function_struct_*
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
STR_cstruct1 * STR_return_struct_ptr1(int i, double d)
{
    // splicer begin function.return_struct_ptr1
    Cstruct1 * SHCXX_rv = returnStructPtr1(i, d);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
    // splicer end function.return_struct_ptr1
}

/**
 * \brief Return a pointer to a struct
 *
 * Does not generate a bufferify C wrapper.
 */
// ----------------------------------------
// Function:  Cstruct1 * returnStructPtr1
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_struct_*_buf_pointer
// Match:     c_function_struct_*
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
STR_cstruct1 * STR_return_struct_ptr1_bufferify(int i, double d)
{
    // splicer begin function.return_struct_ptr1_bufferify
    Cstruct1 * SHCXX_rv = returnStructPtr1(i, d);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
    // splicer end function.return_struct_ptr1_bufferify
}

/**
 * \brief Return a pointer to a struct
 *
 * Generates a bufferify C wrapper function.
 */
// ----------------------------------------
// Function:  Cstruct1 * returnStructPtr2
// Attrs:     +deref(pointer)+intent(function)
// Requested: c_function_struct_*_pointer
// Match:     c_function_struct_*
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Attrs:     +intent(out)
// Requested: c_out_char_*
// Match:     c_default
STR_cstruct1 * STR_return_struct_ptr2(int i, double d, char * outbuf)
{
    // splicer begin function.return_struct_ptr2
    Cstruct1 * SHCXX_rv = returnStructPtr2(i, d, outbuf);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
    // splicer end function.return_struct_ptr2
}

/**
 * \brief Return a pointer to a struct
 *
 * Generates a bufferify C wrapper function.
 */
// ----------------------------------------
// Function:  Cstruct1 * returnStructPtr2
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_struct_*_buf_pointer
// Match:     c_function_struct_*
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  double d +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  char * outbuf +charlen(LENOUTBUF)+intent(out)
// Attrs:     +api(buf)+intent(out)
// Exact:     c_out_char_*_buf
STR_cstruct1 * STR_return_struct_ptr2_bufferify(int i, double d,
    char *outbuf, int SHT_outbuf_len)
{
    // splicer begin function.return_struct_ptr2_bufferify
    Cstruct1 * SHCXX_rv = returnStructPtr2(i, d, outbuf);
    ShroudStrBlankFill(outbuf, SHT_outbuf_len);
    STR_cstruct1 * SHC_rv = static_cast<STR_cstruct1 *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
    // splicer end function.return_struct_ptr2_bufferify
}

// ----------------------------------------
// Function:  Cstruct_list * get_global_struct_list
// Attrs:     +deref(pointer)+intent(function)
// Requested: c_function_struct_*_pointer
// Match:     c_function_struct_*
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
// Function:  Cstruct_list * get_global_struct_list
// Attrs:     +api(buf)+deref(pointer)+intent(function)
// Requested: c_function_struct_*_buf_pointer
// Match:     c_function_struct_*
STR_cstruct_list * STR_get_global_struct_list_bufferify(void)
{
    // splicer begin function.get_global_struct_list_bufferify
    Cstruct_list * SHCXX_rv = get_global_struct_list();
    STR_cstruct_list * SHC_rv = static_cast<STR_cstruct_list *>(
        static_cast<void *>(SHCXX_rv));
    return SHC_rv;
    // splicer end function.get_global_struct_list_bufferify
}

// ----------------------------------------
// Function:  Cstruct_as_class * Create_Cstruct_as_class
// Attrs:     +intent(function)
// Requested: c_function_shadow_*
// Match:     c_function_shadow
// start STR_create__cstruct_as_class
void STR_create__cstruct_as_class(STR_Cstruct_as_class * SHC_rv)
{
    // splicer begin function.create__cstruct_as_class
    Cstruct_as_class * SHCXX_rv = Create_Cstruct_as_class();
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    // splicer end function.create__cstruct_as_class
}
// end STR_create__cstruct_as_class

// ----------------------------------------
// Function:  Cstruct_as_class * Create_Cstruct_as_class_args
// Attrs:     +intent(function)
// Requested: c_function_shadow_*
// Match:     c_function_shadow
// ----------------------------------------
// Argument:  int x +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int y +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void STR_create__cstruct_as_class_args(int x, int y,
    STR_Cstruct_as_class * SHC_rv)
{
    // splicer begin function.create__cstruct_as_class_args
    Cstruct_as_class * SHCXX_rv = Create_Cstruct_as_class_args(x, y);
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    // splicer end function.create__cstruct_as_class_args
}

// ----------------------------------------
// Function:  int Cstruct_as_class_sum
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const Cstruct_as_class * point +pass
// Attrs:     +intent(in)
// Requested: c_in_shadow_*
// Match:     c_in_shadow
int STR_cstruct_as_class_sum(STR_Cstruct_as_class * point)
{
    // splicer begin function.cstruct_as_class_sum
    const Cstruct_as_class * SHCXX_point =
        static_cast<const Cstruct_as_class *>(point->addr);
    int SHC_rv = Cstruct_as_class_sum(SHCXX_point);
    return SHC_rv;
    // splicer end function.cstruct_as_class_sum
}

// ----------------------------------------
// Function:  Cstruct_as_subclass * Create_Cstruct_as_subclass_args
// Attrs:     +intent(function)
// Requested: c_function_shadow_*
// Match:     c_function_shadow
// ----------------------------------------
// Argument:  int x +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int y +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int z +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void STR_create__cstruct_as_subclass_args(int x, int y, int z,
    STR_Cstruct_as_subclass * SHC_rv)
{
    // splicer begin function.create__cstruct_as_subclass_args
    Cstruct_as_subclass * SHCXX_rv = Create_Cstruct_as_subclass_args(x,
        y, z);
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    // splicer end function.create__cstruct_as_subclass_args
}

}  // extern "C"
