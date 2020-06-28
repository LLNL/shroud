// wrapexample_nested_ExClass1.cpp
// This file is generated by Shroud 0.12.0. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapexample_nested_ExClass1.h"
#include <cstddef>
#include <cstring>
#include <string>
#include "ExClass1.hpp"

// splicer begin namespace.example::nested.class.ExClass1.CXX_definitions
//   namespace.example::nested.class.ExClass1.CXX_definitions
// splicer end namespace.example::nested.class.ExClass1.CXX_definitions

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

// helper ShroudStrToArray
// Save str metadata into array to allow Fortran to access values.
// CHARACTER(len=elem_size) src
static void ShroudStrToArray(AA_SHROUD_array *array, const std::string * src, int idtor)
{
    array->cxx.addr = const_cast<std::string *>(src);
    array->cxx.idtor = idtor;
    if (src->empty()) {
        array->addr.ccharp = NULL;
        array->elem_len = 0;
    } else {
        array->addr.ccharp = src->data();
        array->elem_len = src->length();
    }
    array->size = 1;
    array->rank = 0;  // scalar
}
// splicer begin namespace.example::nested.class.ExClass1.C_definitions
// splicer end namespace.example::nested.class.ExClass1.C_definitions

// ----------------------------------------
// Function:  ExClass1
// Exact:     c_shadow_scalar_ctor
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
// Exact:     c_shadow_scalar_ctor
// ----------------------------------------
// Argument:  const string * name +intent(in)
// Requested: c_string_*_in
// Match:     c_string_in
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
// ----------------------------------------
// Function:  ExClass1
// Requested: c_shadow_scalar_ctor_buf
// Match:     c_shadow_scalar_ctor
// ----------------------------------------
// Argument:  const string * name +intent(in)+len_trim(Lname)
// Requested: c_string_*_in_buf
// Match:     c_string_in_buf
AA_example_nested_ExClass1 * AA_example_nested_ExClass1_ctor_1_bufferify(
    const char * name, int Lname, AA_example_nested_ExClass1 * SHC_rv)
{
    // splicer begin namespace.example::nested.class.ExClass1.method.ctor_1_bufferify
    const std::string SHCXX_name(name, Lname);
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
// Exact:     c_shadow_dtor
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
// Requested: c_native_scalar_result
// Match:     c_default
// ----------------------------------------
// Argument:  int incr +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
int AA_example_nested_ExClass1_increment_count(
    AA_example_nested_ExClass1 * self, int incr)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.increment_count
    int SHC_rv = SH_this->incrementCount(incr);
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.increment_count
}

// ----------------------------------------
// Function:  const string & getNameErrorCheck +deref(allocatable)
// Requested: c_string_&_result
// Match:     c_string_result
const char * AA_example_nested_ExClass1_get_name_error_check(
    const AA_example_nested_ExClass1 * self)
{
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.get_name_error_check
    const std::string & SHCXX_rv = SH_this->getNameErrorCheck();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.get_name_error_check
}

// ----------------------------------------
// Function:  void getNameErrorCheck
// Requested: c_void_scalar_result_buf
// Match:     c_default
// ----------------------------------------
// Argument:  const string & SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out)
// Requested: c_string_&_result_buf_allocatable
// Match:     c_string_result_buf_allocatable
void AA_example_nested_ExClass1_get_name_error_check_bufferify(
    const AA_example_nested_ExClass1 * self, AA_SHROUD_array *DSHF_rv)
{
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.get_name_error_check_bufferify
    const std::string & SHCXX_rv = SH_this->getNameErrorCheck();
    ShroudStrToArray(DSHF_rv, &SHCXX_rv, 0);
    // splicer end namespace.example::nested.class.ExClass1.method.get_name_error_check_bufferify
}

// ----------------------------------------
// Function:  const string & getNameArg +deref(result-as-arg)
// Requested: c_string_&_result
// Match:     c_string_result
const char * AA_example_nested_ExClass1_get_name_arg(
    const AA_example_nested_ExClass1 * self)
{
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.get_name_arg
    const std::string & SHCXX_rv = SH_this->getNameArg();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.get_name_arg
}

// ----------------------------------------
// Function:  void getNameArg
// Requested: c_void_scalar_result_buf
// Match:     c_default
// ----------------------------------------
// Argument:  string & name +intent(out)+len(Nname)
// Requested: c_string_&_result_buf
// Match:     c_string_result_buf
void AA_example_nested_ExClass1_get_name_arg_bufferify(
    const AA_example_nested_ExClass1 * self, char * name, int Nname)
{
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.get_name_arg_bufferify
    const std::string & SHCXX_rv = SH_this->getNameArg();
    if (SHCXX_rv.empty()) {
        ShroudStrCopy(name, Nname, nullptr, 0);
    } else {
        ShroudStrCopy(name, Nname, SHCXX_rv.data(), SHCXX_rv.size());
    }
    // splicer end namespace.example::nested.class.ExClass1.method.get_name_arg_bufferify
}

// ----------------------------------------
// Function:  int getValue
// Requested: c_native_scalar_result
// Match:     c_default
// ----------------------------------------
// Argument:  int value +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
int AA_example_nested_ExClass1_get_value_from_int(
    AA_example_nested_ExClass1 * self, int value)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.get_value_from_int
    int SHC_rv = SH_this->getValue(value);
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.get_value_from_int
}

// ----------------------------------------
// Function:  long getValue
// Requested: c_native_scalar_result
// Match:     c_default
// ----------------------------------------
// Argument:  long value +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
long AA_example_nested_ExClass1_get_value_1(
    AA_example_nested_ExClass1 * self, long value)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.get_value_1
    long SHC_rv = SH_this->getValue(value);
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.get_value_1
}

// ----------------------------------------
// Function:  bool hasAddr
// Requested: c_bool_scalar_result
// Match:     c_default
// ----------------------------------------
// Argument:  bool in +intent(in)+value
// Requested: c_bool_scalar_in
// Match:     c_default
bool AA_example_nested_ExClass1_has_addr(
    AA_example_nested_ExClass1 * self, bool in)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.has_addr
    bool SHC_rv = SH_this->hasAddr(in);
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass1.method.has_addr
}

// ----------------------------------------
// Function:  void SplicerSpecial
// Requested: c
// Match:     c_default
void AA_example_nested_ExClass1_splicer_special(
    AA_example_nested_ExClass1 * self)
{
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass1.method.splicer_special
    //   splicer for SplicerSpecial
    // splicer end namespace.example::nested.class.ExClass1.method.splicer_special
}

}  // extern "C"
