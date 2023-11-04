// wrapexample_nested_ExClass1.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "ExClass1.hpp"
// typemap
#include <string>
// shroud
#include <cstddef>
#include <cstring>
#include "wrapexample_nested_ExClass1.h"

// splicer begin namespace.example::nested.class.ExClass1.CXX_definitions
//   namespace.example::nested.class.ExClass1.CXX_definitions
// splicer end namespace.example::nested.class.ExClass1.CXX_definitions

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

// helper char_len_trim
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
static int ShroudCharLenTrim(const char *src, int nsrc) {
    int i;

    for (i = nsrc - 1; i >= 0; i--) {
        if (src[i] != ' ') {
            break;
        }
    }

    return i + 1;
}


// helper string_to_cdesc
// Save std::string metadata into array to allow Fortran to access values.
// CHARACTER(len=elem_size) src
static void ShroudStringToCdesc(AA_SHROUD_array *cdesc,
    const std::string * src)
{
    if (src->empty()) {
        cdesc->base_addr = NULL;
        cdesc->elem_len = 0;
    } else {
        cdesc->base_addr = const_cast<char *>(src->data());
        cdesc->elem_len = src->length();
    }
    cdesc->size = 1;
    cdesc->rank = 0;  // scalar
}
// splicer begin namespace.example::nested.class.ExClass1.C_definitions
// splicer end namespace.example::nested.class.ExClass1.C_definitions

// ----------------------------------------
// Function:  ExClass1
// Statement: c_ctor_shadow_scalar_capptr
AA_example_nested_ExClass1 * AA_example_nested_ExClass1_ctor_0(
    AA_example_nested_ExClass1 * SHC_rv)
{
    // splicer begin namespace.example::nested.class.ExClass1.method.ctor_0
    example::nested::ExClass1 *SHCXX_rv =
        new example::nested::ExClass1();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.ctor_0
}

/**
 * \brief constructor
 *
 * longer description
 * usually multiple lines
 *
 * \return return new instance
 */
// ----------------------------------------
// Function:  ExClass1
// Statement: c_ctor_shadow_scalar_capptr
// ----------------------------------------
// Argument:  const string * name
// Statement: c_in_string_*
AA_example_nested_ExClass1 * AA_example_nested_ExClass1_ctor_1(
    const char * name, AA_example_nested_ExClass1 * SHC_rv)
{
    // splicer begin namespace.example::nested.class.ExClass1.method.ctor_1
    const std::string SHCXX_name(name);
    example::nested::ExClass1 *SHCXX_rv =
        new example::nested::ExClass1(&SHCXX_name);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.ctor_1
}

/**
 * \brief constructor
 *
 * longer description
 * usually multiple lines
 *
 * \return return new instance
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  ExClass1
// Statement: f_ctor_shadow_scalar_capptr
// ----------------------------------------
// Argument:  const string * name
// Statement: f_in_string_*_buf
AA_example_nested_ExClass1 * AA_example_nested_ExClass1_ctor_1_bufferify(
    char *name, int SHT_name_len, AA_example_nested_ExClass1 * SHC_rv)
{
    // splicer begin namespace.example::nested.class.ExClass1.method.ctor_1_bufferify
    const std::string SHCXX_name(name,
        ShroudCharLenTrim(name, SHT_name_len));
    example::nested::ExClass1 *SHCXX_rv =
        new example::nested::ExClass1(&SHCXX_name);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.ctor_1_bufferify
}

/**
 * \brief destructor
 *
 * longer description joined with previous line
 */
// ----------------------------------------
// Function:  ~ExClass1
// Statement: c_dtor
void AA_example_nested_ExClass1_dtor(AA_example_nested_ExClass1 * self)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.dtor
    delete SH_this;
    self->addr = nullptr;
    // splicer end namespace.example::nested.class.ExClass1.method.dtor
}

// ----------------------------------------
// Function:  int incrementCount
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  int incr +value
// Statement: c_in_native_scalar
int AA_example_nested_ExClass1_incrementCount(
    AA_example_nested_ExClass1 * self, int incr)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.incrementCount
    int SHC_rv = SH_this->incrementCount(incr);
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.incrementCount
}

// ----------------------------------------
// Function:  const string & getNameErrorCheck
// Statement: c_function_string_&
const char * AA_example_nested_ExClass1_getNameErrorCheck(
    const AA_example_nested_ExClass1 * self)
{
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.getNameErrorCheck
    const std::string & SHCXX_rv = SH_this->getNameErrorCheck();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.getNameErrorCheck
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  const string & getNameErrorCheck
// Statement: f_function_string_&_cdesc_allocatable
void AA_example_nested_ExClass1_getNameErrorCheck_bufferify(
    const AA_example_nested_ExClass1 * self,
    AA_SHROUD_array *SHT_rv_cdesc,
    AA_SHROUD_capsule_data *SHT_rv_capsule)
{
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.getNameErrorCheck_bufferify
    const std::string & SHCXX_rv = SH_this->getNameErrorCheck();
    ShroudStringToCdesc(SHT_rv_cdesc, &SHCXX_rv);
    SHT_rv_capsule->addr  = const_cast<std::string *>(&SHCXX_rv);
    SHT_rv_capsule->idtor = 0;
    // splicer end namespace.example::nested.class.ExClass1.method.getNameErrorCheck_bufferify
}

// ----------------------------------------
// Function:  const string & getNameArg
// Statement: c_function_string_&
const char * AA_example_nested_ExClass1_getNameArg(
    const AA_example_nested_ExClass1 * self)
{
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.getNameArg
    const std::string & SHCXX_rv = SH_this->getNameArg();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.getNameArg
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  const string & getNameArg
// Statement: f_function_string_&_buf_arg
void AA_example_nested_ExClass1_getNameArg_bufferify(
    const AA_example_nested_ExClass1 * self, char *name, int nname)
{
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.getNameArg_bufferify
    const std::string & SHCXX_rv = SH_this->getNameArg();
    if (SHCXX_rv.empty()) {
        ShroudCharCopy(name, nname, nullptr, 0);
    } else {
        ShroudCharCopy(name, nname, SHCXX_rv.data(), SHCXX_rv.size());
    }
    // splicer end namespace.example::nested.class.ExClass1.method.getNameArg_bufferify
}

// ----------------------------------------
// Function:  int getValue
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  int value +value
// Statement: c_in_native_scalar
int AA_example_nested_ExClass1_getValue_from_int(
    AA_example_nested_ExClass1 * self, int value)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.getValue_from_int
    int SHC_rv = SH_this->getValue(value);
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.getValue_from_int
}

// ----------------------------------------
// Function:  long getValue
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  long value +value
// Statement: c_in_native_scalar
long AA_example_nested_ExClass1_getValue_1(
    AA_example_nested_ExClass1 * self, long value)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.getValue_1
    long SHC_rv = SH_this->getValue(value);
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.getValue_1
}

// ----------------------------------------
// Function:  bool hasAddr
// Statement: c_function_bool_scalar
// ----------------------------------------
// Argument:  bool in +value
// Statement: c_in_bool_scalar
bool AA_example_nested_ExClass1_hasAddr(
    AA_example_nested_ExClass1 * self, bool in)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.hasAddr
    bool SHC_rv = SH_this->hasAddr(in);
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.hasAddr
}

// ----------------------------------------
// Function:  void SplicerSpecial
// Statement: c_subroutine
void AA_example_nested_ExClass1_SplicerSpecial(
    AA_example_nested_ExClass1 * self)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.SplicerSpecial
#//   splicer for SplicerSpecial
    // splicer end namespace.example::nested.class.ExClass1.method.SplicerSpecial
}

}  // extern "C"
