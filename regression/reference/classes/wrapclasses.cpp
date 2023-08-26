// wrapclasses.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "classes.hpp"
// typemap
#include <string>
// shroud
#include <cstring>
#include "wrapclasses.h"

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
// Exact:     c_function_native_scalar
// ----------------------------------------
// Argument:  Class1::DIRECTION arg +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int CLA_directionFunc(int arg)
{
    // splicer begin function.directionFunc
    classes::Class1::DIRECTION SHCXX_arg =
        static_cast<classes::Class1::DIRECTION>(arg);
    classes::Class1::DIRECTION SHCXX_rv = classes::directionFunc(
        SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
    // splicer end function.directionFunc
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
void CLA_passClassByValue(CLA_Class1 arg)
{
    // splicer begin function.passClassByValue
    classes::Class1 * SHCXX_arg = static_cast<classes::Class1 *>
        (arg.addr);
    classes::passClassByValue(*SHCXX_arg);
    // splicer end function.passClassByValue
}

// ----------------------------------------
// Function:  int useclass
// Attrs:     +intent(function)
// Exact:     c_function_native_scalar
// ----------------------------------------
// Argument:  const Class1 * arg
// Attrs:     +intent(in)
// Exact:     c_in_shadow_*
int CLA_useclass(CLA_Class1 * arg)
{
    // splicer begin function.useclass
    const classes::Class1 * SHCXX_arg =
        static_cast<const classes::Class1 *>(arg->addr);
    int SHC_rv = classes::useclass(SHCXX_arg);
    return SHC_rv;
    // splicer end function.useclass
}

/**
 * \brief Return const class pointer
 *
 */
// ----------------------------------------
// Function:  const Class1 * getclass2
// Attrs:     +api(capptr)+intent(function)
// Exact:     c_function_shadow_*_capptr
CLA_Class1 * CLA_getclass2(CLA_Class1 * SHC_rv)
{
    // splicer begin function.getclass2
    const classes::Class1 * SHCXX_rv = classes::getclass2();
    SHC_rv->addr = const_cast<classes::Class1 *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end function.getclass2
}

/**
 * \brief Return class pointer
 *
 */
// ----------------------------------------
// Function:  Class1 * getclass3
// Attrs:     +api(capptr)+intent(function)
// Exact:     c_function_shadow_*_capptr
CLA_Class1 * CLA_getclass3(CLA_Class1 * SHC_rv)
{
    // splicer begin function.getclass3
    classes::Class1 * SHCXX_rv = classes::getclass3();
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end function.getclass3
}

/**
 * \brief C wrapper will return void
 *
 */
// ----------------------------------------
// Function:  const Class1 * getclass2_void
// Attrs:     +api(capsule)+intent(function)
// Exact:     c_function_shadow_*_capsule
void CLA_getclass2_void(CLA_Class1 * SHC_rv)
{
    // splicer begin function.getclass2_void
    const classes::Class1 * SHCXX_rv = classes::getclass2_void();
    SHC_rv->addr = const_cast<classes::Class1 *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    // splicer end function.getclass2_void
}

/**
 * \brief C wrapper will return void
 *
 */
// ----------------------------------------
// Function:  Class1 * getclass3_void
// Attrs:     +api(capsule)+intent(function)
// Exact:     c_function_shadow_*_capsule
void CLA_getclass3_void(CLA_Class1 * SHC_rv)
{
    // splicer begin function.getclass3_void
    classes::Class1 * SHCXX_rv = classes::getclass3_void();
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    // splicer end function.getclass3_void
}

// ----------------------------------------
// Function:  const Class1 & getConstClassReference
// Attrs:     +api(capptr)+intent(function)
// Exact:     c_function_shadow_&_capptr
CLA_Class1 * CLA_getConstClassReference(CLA_Class1 * SHC_rv)
{
    // splicer begin function.getConstClassReference
    const classes::Class1 & SHCXX_rv = classes::getConstClassReference(
        );
    SHC_rv->addr = const_cast<classes::Class1 *>(&SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end function.getConstClassReference
}

// ----------------------------------------
// Function:  Class1 & getClassReference
// Attrs:     +api(capptr)+intent(function)
// Exact:     c_function_shadow_&_capptr
CLA_Class1 * CLA_getClassReference(CLA_Class1 * SHC_rv)
{
    // splicer begin function.getClassReference
    classes::Class1 & SHCXX_rv = classes::getClassReference();
    SHC_rv->addr = &SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end function.getClassReference
}

/**
 * \brief Return Class1 instance by value, uses copy constructor
 *
 */
// ----------------------------------------
// Function:  Class1 getClassCopy
// Attrs:     +api(capptr)+intent(function)
// Exact:     c_function_shadow_scalar_capptr
// ----------------------------------------
// Argument:  int flag +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
CLA_Class1 * CLA_getClassCopy(int flag, CLA_Class1 * SHC_rv)
{
    // splicer begin function.getClassCopy
    classes::Class1 * SHCXX_rv = new classes::Class1;
    *SHCXX_rv = classes::getClassCopy(flag);
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 1;
    return SHC_rv;
    // splicer end function.getClassCopy
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
// Exact:     c_function_native_scalar
int CLA_get_global_flag(void)
{
    // splicer begin function.get_global_flag
    int SHC_rv = classes::get_global_flag();
    return SHC_rv;
    // splicer end function.get_global_flag
}

// ----------------------------------------
// Function:  const std::string & LastFunctionCalled +len(30)
// Attrs:     +deref(copy)+intent(function)
// Requested: c_function_string_&_copy
// Match:     c_function_string_&
const char * CLA_LastFunctionCalled(void)
{
    // splicer begin function.LastFunctionCalled
    const std::string & SHCXX_rv = classes::LastFunctionCalled();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end function.LastFunctionCalled
}

// ----------------------------------------
// Function:  const std::string & LastFunctionCalled +len(30)
// Attrs:     +api(buf)+deref(copy)+intent(function)
// Exact:     c_function_string_&_buf_copy
void CLA_LastFunctionCalled_bufferify(char *SHC_rv, int SHT_rv_len)
{
    // splicer begin function.LastFunctionCalled_bufferify
    const std::string & SHCXX_rv = classes::LastFunctionCalled();
    if (SHCXX_rv.empty()) {
        ShroudStrCopy(SHC_rv, SHT_rv_len, nullptr, 0);
    } else {
        ShroudStrCopy(SHC_rv, SHT_rv_len, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    // splicer end function.LastFunctionCalled_bufferify
}

}  // extern "C"
