// wrapcxxlibrary.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "cxxlibrary.hpp"
// typemap
#include <string>
// shroud
#include <cstring>
#include "wrapcxxlibrary.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper ShroudCharCopy
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudCharCopy(char *dest, int ndest, const char *src, int nsrc)
{
   if (src == NULL) {
     std::memset(dest,' ',ndest); // convert NULL pointer to blank filled string
   } else {
     if (nsrc < 0) nsrc = std::strlen(src);
     int nm = nsrc < ndest ? nsrc : ndest;
     std::memcpy(dest,src,nm);
     if(ndest > nm) std::memset(dest+nm,' ',ndest-nm); // blank fill
   }
}
// splicer begin C_definitions
// splicer end C_definitions

// Generated by has_default_arg
// ----------------------------------------
// Function:  bool defaultPtrIsNULL
// Statement: c_function_bool_scalar
bool CXX_defaultPtrIsNULL_0(void)
{
    // splicer begin function.defaultPtrIsNULL_0
    bool SHC_rv = defaultPtrIsNULL();
    return SHC_rv;
    // splicer end function.defaultPtrIsNULL_0
}

// ----------------------------------------
// Function:  bool defaultPtrIsNULL
// Statement: c_function_bool_scalar
// ----------------------------------------
// Argument:  double * data=nullptr +intent(IN)+rank(1)
// Statement: c_in_native_*
bool CXX_defaultPtrIsNULL_1(double * data)
{
    // splicer begin function.defaultPtrIsNULL_1
    bool SHC_rv = defaultPtrIsNULL(data);
    return SHC_rv;
    // splicer end function.defaultPtrIsNULL_1
}

// Generated by has_default_arg
// ----------------------------------------
// Function:  void defaultArgsInOut
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int in1 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int * out1 +intent(out)
// Statement: c_out_native_*
// ----------------------------------------
// Argument:  int * out2 +intent(out)
// Statement: c_out_native_*
void CXX_defaultArgsInOut_0(int in1, int * out1, int * out2)
{
    // splicer begin function.defaultArgsInOut_0
    defaultArgsInOut(in1, out1, out2);
    // splicer end function.defaultArgsInOut_0
}

// ----------------------------------------
// Function:  void defaultArgsInOut
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int in1 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int * out1 +intent(out)
// Statement: c_out_native_*
// ----------------------------------------
// Argument:  int * out2 +intent(out)
// Statement: c_out_native_*
// ----------------------------------------
// Argument:  bool flag=false +value
// Statement: c_in_bool_scalar
void CXX_defaultArgsInOut_1(int in1, int * out1, int * out2, bool flag)
{
    // splicer begin function.defaultArgsInOut_1
    defaultArgsInOut(in1, out1, out2, flag);
    // splicer end function.defaultArgsInOut_1
}

/**
 * \brief String reference function with scalar generic args
 *
 */
// ----------------------------------------
// Function:  const std::string & getGroupName +len(30)
// Statement: c_function_string_&
// ----------------------------------------
// Argument:  long idx +value
// Statement: c_in_native_scalar
const char * CXX_getGroupName(long idx)
{
    // splicer begin function.getGroupName
    const std::string & SHCXX_rv = getGroupName(idx);
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end function.getGroupName
}

/**
 * \brief String reference function with scalar generic args
 *
 */
// Generated by fortran_generic
// ----------------------------------------
// Function:  const std::string & getGroupName +len(30)
// Statement: f_function_string_&_buf_copy
// ----------------------------------------
// Argument:  int32_t idx +value
// Statement: f_in_native_scalar
void CXX_getGroupName_int32_t_bufferify(int32_t idx, char *SHC_rv,
    int SHT_rv_len)
{
    // splicer begin function.getGroupName_int32_t_bufferify
    const std::string & SHCXX_rv = getGroupName(idx);
    if (SHCXX_rv.empty()) {
        ShroudCharCopy(SHC_rv, SHT_rv_len, nullptr, 0);
    } else {
        ShroudCharCopy(SHC_rv, SHT_rv_len, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    // splicer end function.getGroupName_int32_t_bufferify
}

/**
 * \brief String reference function with scalar generic args
 *
 */
// Generated by fortran_generic
// ----------------------------------------
// Function:  const std::string & getGroupName +len(30)
// Statement: f_function_string_&_buf_copy
// ----------------------------------------
// Argument:  int64_t idx +value
// Statement: f_in_native_scalar
void CXX_getGroupName_int64_t_bufferify(int64_t idx, char *SHC_rv,
    int SHT_rv_len)
{
    // splicer begin function.getGroupName_int64_t_bufferify
    const std::string & SHCXX_rv = getGroupName(idx);
    if (SHCXX_rv.empty()) {
        ShroudCharCopy(SHC_rv, SHT_rv_len, nullptr, 0);
    } else {
        ShroudCharCopy(SHC_rv, SHT_rv_len, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    // splicer end function.getGroupName_int64_t_bufferify
}

// Generated by getter/setter
// ----------------------------------------
// Function:  nested * nested_get_parent
// Statement: f_getter_struct_*_cdesc_pointer
// ----------------------------------------
// Argument:  nested * SH_this
// Statement: f_in_struct_*
void CXX_nested_get_parent(CXX_nested * SH_this,
    CXX_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.nested_get_parent
    SHT_rv_cdesc->base_addr = SH_this->parent;
    SHT_rv_cdesc->type = SH_TYPE_STRUCT;
    SHT_rv_cdesc->elem_len = sizeof(nested);
    SHT_rv_cdesc->rank = 0;
    SHT_rv_cdesc->size = 1;
    // splicer end function.nested_get_parent
}

// Generated by getter/setter
// ----------------------------------------
// Function:  void nested_set_parent
// Statement: f_setter
// ----------------------------------------
// Argument:  nested * SH_this
// Statement: f_inout_struct_*
// ----------------------------------------
// Argument:  nested * val +intent(in)
// Statement: f_setter_struct_*
void CXX_nested_set_parent(CXX_nested * SH_this, CXX_nested * val)
{
    // splicer begin function.nested_set_parent
    // skip call c_setter
    SH_this->parent = val;
    // splicer end function.nested_set_parent
}

// Generated by getter/setter
// ----------------------------------------
// Function:  nested * * nested_get_child +dimension(sublevels)
// Statement: f_getter_struct_**_cdesc_pointer
// ----------------------------------------
// Argument:  nested * SH_this
// Statement: f_in_struct_*
void CXX_nested_get_child(CXX_nested * SH_this,
    CXX_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.nested_get_child
    SHT_rv_cdesc->base_addr = SH_this->child;
    SHT_rv_cdesc->type = SH_TYPE_STRUCT;
    SHT_rv_cdesc->elem_len = sizeof(nested);
    SHT_rv_cdesc->rank = 1;
    SHT_rv_cdesc->shape[0] = SH_this->sublevels;
    SHT_rv_cdesc->size = SHT_rv_cdesc->shape[0];
    // splicer end function.nested_get_child
}

// Generated by getter/setter
// ----------------------------------------
// Function:  void nested_set_child
// Statement: f_setter
// ----------------------------------------
// Argument:  nested * SH_this
// Statement: f_inout_struct_*
// ----------------------------------------
// Argument:  nested * * val +intent(in)+rank(1)
// Statement: f_setter_struct_**
void CXX_nested_set_child(CXX_nested * SH_this, CXX_nested * * val)
{
    // splicer begin function.nested_set_child
    // skip call c_setter
    SH_this->child = val;
    // splicer end function.nested_set_child
}

}  // extern "C"
