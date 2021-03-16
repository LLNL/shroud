// wrapclasses.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapclasses.h"

// cxx_header
#include "classes.hpp"
// typemap
#include <string>
// shroud
#include <cstring>

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper ShroudStrCopy
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudStrCopy(char *dest, int ndest, const char *src, int nsrc)
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

// ----------------------------------------
// Function:  Class1::DIRECTION directionFunc
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  Class1::DIRECTION arg +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int CLA_direction_func(int arg)
{
    // splicer begin function.direction_func
    classes::Class1::DIRECTION SHCXX_arg =
        static_cast<classes::Class1::DIRECTION>(arg);
    classes::Class1::DIRECTION SHCXX_rv = classes::directionFunc(
        SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
    // splicer end function.direction_func
}

/**
 * \brief Pass arguments to a function.
 *
 */
// ----------------------------------------
// Function:  void passClassByValue
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  Class1 arg +value
// Attrs:     +intent(in)
// Exact:     c_in_shadow_scalar
void CLA_pass_class_by_value(CLA_Class1 arg)
{
    // splicer begin function.pass_class_by_value
    classes::Class1 * SHCXX_arg = static_cast<classes::Class1 *>
        (arg.addr);
    classes::passClassByValue(*SHCXX_arg);
    // splicer end function.pass_class_by_value
}

// ----------------------------------------
// Function:  int useclass
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  const Class1 * arg
// Attrs:     +intent(in)
// Requested: c_in_shadow_*
// Match:     c_in_shadow
int CLA_useclass(CLA_Class1 * arg)
{
    // splicer begin function.useclass
    const classes::Class1 * SHCXX_arg =
        static_cast<const classes::Class1 *>(arg->addr);
    int SHC_rv = classes::useclass(SHCXX_arg);
    return SHC_rv;
    // splicer end function.useclass
}

// ----------------------------------------
// Function:  const Class1 * getclass2
// Attrs:     +intent(function)
// Requested: c_function_shadow_*
// Match:     c_function_shadow
CLA_Class1 * CLA_getclass2(CLA_Class1 * SHadow_rv)
{
    // splicer begin function.getclass2
    const classes::Class1 * SHCXX_rv = classes::getclass2();
    SHadow_rv->addr = const_cast<classes::Class1 *>(SHCXX_rv);
    SHadow_rv->idtor = 0;
    return SHadow_rv;
    // splicer end function.getclass2
}

// ----------------------------------------
// Function:  Class1 * getclass3
// Attrs:     +intent(function)
// Requested: c_function_shadow_*
// Match:     c_function_shadow
CLA_Class1 * CLA_getclass3(CLA_Class1 * SHadow_rv)
{
    // splicer begin function.getclass3
    classes::Class1 * SHCXX_rv = classes::getclass3();
    SHadow_rv->addr = SHCXX_rv;
    SHadow_rv->idtor = 0;
    return SHadow_rv;
    // splicer end function.getclass3
}

// ----------------------------------------
// Function:  const Class1 & getConstClassReference
// Attrs:     +intent(function)
// Requested: c_function_shadow_&
// Match:     c_function_shadow
CLA_Class1 * CLA_get_const_class_reference(CLA_Class1 * SHadow_rv)
{
    // splicer begin function.get_const_class_reference
    const classes::Class1 & SHCXX_rv = classes::getConstClassReference(
        );
    SHadow_rv->addr = const_cast<classes::Class1 *>(&SHCXX_rv);
    SHadow_rv->idtor = 0;
    return SHadow_rv;
    // splicer end function.get_const_class_reference
}

// ----------------------------------------
// Function:  Class1 & getClassReference
// Attrs:     +intent(function)
// Requested: c_function_shadow_&
// Match:     c_function_shadow
CLA_Class1 * CLA_get_class_reference(CLA_Class1 * SHadow_rv)
{
    // splicer begin function.get_class_reference
    classes::Class1 & SHCXX_rv = classes::getClassReference();
    SHadow_rv->addr = &SHCXX_rv;
    SHadow_rv->idtor = 0;
    return SHadow_rv;
    // splicer end function.get_class_reference
}

/**
 * \brief Return Class1 instance by value, uses copy constructor
 *
 */
// ----------------------------------------
// Function:  Class1 getClassCopy
// Attrs:     +intent(function)
// Exact:     c_function_shadow_scalar
// ----------------------------------------
// Argument:  int flag +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
CLA_Class1 * CLA_get_class_copy(int flag, CLA_Class1 * SHadow_rv)
{
    // splicer begin function.get_class_copy
    classes::Class1 * SHCXX_rv = new classes::Class1;
    *SHCXX_rv = classes::getClassCopy(flag);
    SHadow_rv->addr = SHCXX_rv;
    SHadow_rv->idtor = 1;
    return SHadow_rv;
    // splicer end function.get_class_copy
}

// ----------------------------------------
// Function:  void set_global_flag
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int arg +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void CLA_set_global_flag(int arg)
{
    // splicer begin function.set_global_flag
    classes::set_global_flag(arg);
    // splicer end function.set_global_flag
}

// ----------------------------------------
// Function:  int get_global_flag
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_default
int CLA_get_global_flag(void)
{
    // splicer begin function.get_global_flag
    int SHC_rv = classes::get_global_flag();
    return SHC_rv;
    // splicer end function.get_global_flag
}

// ----------------------------------------
// Function:  const std::string & LastFunctionCalled +len(30)
// Attrs:     +deref(result-as-arg)+intent(function)
// Requested: c_function_string_&_result-as-arg
// Match:     c_function_string_&
const char * CLA_last_function_called(void)
{
    // splicer begin function.last_function_called
    const std::string & SHCXX_rv = classes::LastFunctionCalled();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end function.last_function_called
}

// ----------------------------------------
// Function:  void LastFunctionCalled +len(30)
// Attrs:     +intent(subroutine)
// Requested: c_subroutine_void_scalar_buf
// Match:     c_subroutine
// ----------------------------------------
// Argument:  std::string & SHF_rv +len(NSHF_rv)
// Attrs:     +intent(out)+is_result
// Exact:     c_function_string_&_buf
void CLA_last_function_called_bufferify(char * SHF_rv, int NSHF_rv)
{
    // splicer begin function.last_function_called_bufferify
    const std::string & SHCXX_rv = classes::LastFunctionCalled();
    if (SHCXX_rv.empty()) {
        ShroudStrCopy(SHF_rv, NSHF_rv, nullptr, 0);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    // splicer end function.last_function_called_bufferify
}

}  // extern "C"
