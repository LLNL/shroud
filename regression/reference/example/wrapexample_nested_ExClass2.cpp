// wrapexample_nested_ExClass2.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "ExClass2.hpp"
// typemap
#include <string>
#include "ExClass1.hpp"
// shroud
#include <cstddef>
#include <cstring>
#include "wrapexample_nested_ExClass2.h"

// splicer begin namespace.example::nested.class.ExClass2.CXX_definitions
//   namespace.example::nested.class.ExClass2.CXX_definitions
// splicer end namespace.example::nested.class.ExClass2.CXX_definitions

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
// splicer begin namespace.example::nested.class.ExClass2.C_definitions
// splicer end namespace.example::nested.class.ExClass2.C_definitions

/**
 * \brief constructor
 *
 */
// ----------------------------------------
// Function:  ExClass2
// Attrs:     +api(capptr)+intent(ctor)
// Statement: f_ctor_shadow_scalar_capptr
// ----------------------------------------
// Argument:  const string * name +len_trim(trim_name)
// Attrs:     +intent(in)
// Statement: f_in_string_*
AA_example_nested_ExClass2 * AA_example_nested_ExClass2_ctor(
    const char * name, AA_example_nested_ExClass2 * SHC_rv)
{
    // splicer begin namespace.example::nested.class.ExClass2.method.ctor
    const std::string SHCXX_name(name);
    example::nested::ExClass2 *SHCXX_rv =
        new example::nested::ExClass2(&SHCXX_name);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 2;
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass2.method.ctor
}

/**
 * \brief constructor
 *
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  ExClass2
// Attrs:     +api(capptr)+intent(ctor)
// Statement: f_ctor_shadow_scalar_capptr
// ----------------------------------------
// Argument:  const string * name +len_trim(trim_name)
// Attrs:     +api(buf)+intent(in)
// Statement: f_in_string_*_buf
AA_example_nested_ExClass2 * AA_example_nested_ExClass2_ctor_bufferify(
    char *name, int SHT_name_len, AA_example_nested_ExClass2 * SHC_rv)
{
    // splicer begin namespace.example::nested.class.ExClass2.method.ctor_bufferify
    const std::string SHCXX_name(name,
        ShroudCharLenTrim(name, SHT_name_len));
    example::nested::ExClass2 *SHCXX_rv =
        new example::nested::ExClass2(&SHCXX_name);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 2;
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass2.method.ctor_bufferify
}

/**
 * \brief destructor
 *
 */
// ----------------------------------------
// Function:  ~ExClass2
// Attrs:     +intent(dtor)
// Statement: f_dtor
void AA_example_nested_ExClass2_dtor(AA_example_nested_ExClass2 * self)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.dtor
    delete SH_this;
    self->addr = nullptr;
    // splicer end namespace.example::nested.class.ExClass2.method.dtor
}

// ----------------------------------------
// Function:  const string & getName +len(aa_exclass2_get_name_length({F_this}%{F_derived_member}))
// Attrs:     +deref(copy)+intent(function)
// Statement: f_function_string_&_copy
const char * AA_example_nested_ExClass2_getName(
    const AA_example_nested_ExClass2 * self)
{
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getName
    const std::string & SHCXX_rv = SH_this->getName();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass2.method.getName
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  const string & getName +len(aa_exclass2_get_name_length({F_this}%{F_derived_member}))
// Attrs:     +api(buf)+deref(copy)+intent(function)
// Statement: f_function_string_&_buf_copy
void AA_example_nested_ExClass2_getName_bufferify(
    const AA_example_nested_ExClass2 * self, char *SHC_rv,
    int SHT_rv_len)
{
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getName_bufferify
    const std::string & SHCXX_rv = SH_this->getName();
    if (SHCXX_rv.empty()) {
        ShroudCharCopy(SHC_rv, SHT_rv_len, nullptr, 0);
    } else {
        ShroudCharCopy(SHC_rv, SHT_rv_len, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    // splicer end namespace.example::nested.class.ExClass2.method.getName_bufferify
}

// ----------------------------------------
// Function:  const string & getName2
// Attrs:     +deref(allocatable)+intent(function)
// Statement: f_function_string_&_allocatable
const char * AA_example_nested_ExClass2_getName2(
    AA_example_nested_ExClass2 * self)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getName2
    const std::string & SHCXX_rv = SH_this->getName2();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass2.method.getName2
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  const string & getName2
// Attrs:     +api(cdesc)+deref(allocatable)+intent(function)
// Statement: f_function_string_&_cdesc_allocatable
void AA_example_nested_ExClass2_getName2_bufferify(
    AA_example_nested_ExClass2 * self, AA_SHROUD_array *SHT_rv_cdesc,
    AA_SHROUD_capsule_data *SHT_rv_capsule)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getName2_bufferify
    const std::string & SHCXX_rv = SH_this->getName2();
    ShroudStringToCdesc(SHT_rv_cdesc, &SHCXX_rv);
    SHT_rv_capsule->addr  = const_cast<std::string *>(&SHCXX_rv);
    SHT_rv_capsule->idtor = 0;
    // splicer end namespace.example::nested.class.ExClass2.method.getName2_bufferify
}

// ----------------------------------------
// Function:  string & getName3
// Attrs:     +deref(allocatable)+intent(function)
// Statement: f_function_string_&_allocatable
char * AA_example_nested_ExClass2_getName3(
    const AA_example_nested_ExClass2 * self)
{
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getName3
    std::string & SHCXX_rv = SH_this->getName3();
    char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass2.method.getName3
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  string & getName3
// Attrs:     +api(cdesc)+deref(allocatable)+intent(function)
// Statement: f_function_string_&_cdesc_allocatable
void AA_example_nested_ExClass2_getName3_bufferify(
    const AA_example_nested_ExClass2 * self,
    AA_SHROUD_array *SHT_rv_cdesc,
    AA_SHROUD_capsule_data *SHT_rv_capsule)
{
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getName3_bufferify
    std::string & SHCXX_rv = SH_this->getName3();
    ShroudStringToCdesc(SHT_rv_cdesc, &SHCXX_rv);
    SHT_rv_capsule->addr  = &SHCXX_rv;
    SHT_rv_capsule->idtor = 0;
    // splicer end namespace.example::nested.class.ExClass2.method.getName3_bufferify
}

// ----------------------------------------
// Function:  string & getName4
// Attrs:     +deref(allocatable)+intent(function)
// Statement: f_function_string_&_allocatable
char * AA_example_nested_ExClass2_getName4(
    AA_example_nested_ExClass2 * self)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getName4
    std::string & SHCXX_rv = SH_this->getName4();
    char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass2.method.getName4
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  string & getName4
// Attrs:     +api(cdesc)+deref(allocatable)+intent(function)
// Statement: f_function_string_&_cdesc_allocatable
void AA_example_nested_ExClass2_getName4_bufferify(
    AA_example_nested_ExClass2 * self, AA_SHROUD_array *SHT_rv_cdesc,
    AA_SHROUD_capsule_data *SHT_rv_capsule)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getName4_bufferify
    std::string & SHCXX_rv = SH_this->getName4();
    ShroudStringToCdesc(SHT_rv_cdesc, &SHCXX_rv);
    SHT_rv_capsule->addr  = &SHCXX_rv;
    SHT_rv_capsule->idtor = 0;
    // splicer end namespace.example::nested.class.ExClass2.method.getName4_bufferify
}

/**
 * \brief helper function for Fortran
 *
 */
// ----------------------------------------
// Function:  int GetNameLength
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
int AA_example_nested_ExClass2_GetNameLength(
    const AA_example_nested_ExClass2 * self)
{
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.GetNameLength
    return SH_this->getName().length();
    // splicer end namespace.example::nested.class.ExClass2.method.GetNameLength
}

// ----------------------------------------
// Function:  ExClass1 * get_class1
// Attrs:     +api(capptr)+intent(function)
// Statement: f_function_shadow_*_capptr
// ----------------------------------------
// Argument:  const ExClass1 * in
// Attrs:     +intent(in)
// Statement: f_in_shadow_*
AA_example_nested_ExClass1 * AA_example_nested_ExClass2_get_class1(
    AA_example_nested_ExClass2 * self, AA_example_nested_ExClass1 * in,
    AA_example_nested_ExClass1 * SHC_rv)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.get_class1
    const example::nested::ExClass1 * SHCXX_in =
        static_cast<const example::nested::ExClass1 *>(in->addr);
    example::nested::ExClass1 * SHCXX_rv = SH_this->get_class1(
        SHCXX_in);
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass2.method.get_class1
}

// Generated by has_default_arg
// ----------------------------------------
// Function:  ExClass2 * declare
// Attrs:     +api(this)+intent(function)
// Statement: f_function_shadow_*_this
// ----------------------------------------
// Argument:  TypeID type +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void AA_example_nested_ExClass2_declare_0(
    AA_example_nested_ExClass2 * self, AA_TypeID type)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.declare_0
    TypeID SHCXX_type = getTypeID(type);
    SH_this->declare(SHCXX_type);
    // splicer end namespace.example::nested.class.ExClass2.method.declare_0
}

// ----------------------------------------
// Function:  ExClass2 * declare
// Attrs:     +api(this)+intent(function)
// Statement: f_function_shadow_*_this
// ----------------------------------------
// Argument:  TypeID type +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  SidreLength len=1 +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void AA_example_nested_ExClass2_declare_1(
    AA_example_nested_ExClass2 * self, AA_TypeID type,
    SIDRE_SidreLength len)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.declare_1
    TypeID SHCXX_type = getTypeID(type);
    SH_this->declare(SHCXX_type, len);
    // splicer end namespace.example::nested.class.ExClass2.method.declare_1
}

// ----------------------------------------
// Function:  void destroyall
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
void AA_example_nested_ExClass2_destroyall(
    AA_example_nested_ExClass2 * self)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.destroyall
    SH_this->destroyall();
    // splicer end namespace.example::nested.class.ExClass2.method.destroyall
}

// ----------------------------------------
// Function:  TypeID getTypeID
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
AA_TypeID AA_example_nested_ExClass2_getTypeID(
    const AA_example_nested_ExClass2 * self)
{
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getTypeID
    TypeID SHCXX_rv = SH_this->getTypeID();
    AA_TypeID SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass2.method.getTypeID
}

// Generated by cxx_template
// ----------------------------------------
// Function:  void setValue
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  int value +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void AA_example_nested_ExClass2_setValue_int(
    AA_example_nested_ExClass2 * self, int value)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.setValue_int
    SH_this->setValue<int>(value);
    // splicer end namespace.example::nested.class.ExClass2.method.setValue_int
}

// Generated by cxx_template
// ----------------------------------------
// Function:  void setValue
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  long value +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void AA_example_nested_ExClass2_setValue_long(
    AA_example_nested_ExClass2 * self, long value)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.setValue_long
    SH_this->setValue<long>(value);
    // splicer end namespace.example::nested.class.ExClass2.method.setValue_long
}

// Generated by cxx_template
// ----------------------------------------
// Function:  void setValue
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  float value +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void AA_example_nested_ExClass2_setValue_float(
    AA_example_nested_ExClass2 * self, float value)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.setValue_float
    SH_this->setValue<float>(value);
    // splicer end namespace.example::nested.class.ExClass2.method.setValue_float
}

// Generated by cxx_template
// ----------------------------------------
// Function:  void setValue
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  double value +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
void AA_example_nested_ExClass2_setValue_double(
    AA_example_nested_ExClass2 * self, double value)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.setValue_double
    SH_this->setValue<double>(value);
    // splicer end namespace.example::nested.class.ExClass2.method.setValue_double
}

// Generated by cxx_template
// ----------------------------------------
// Function:  int getValue
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
int AA_example_nested_ExClass2_getValue_int(
    AA_example_nested_ExClass2 * self)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getValue_int
    int SHC_rv = SH_this->getValue<int>();
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass2.method.getValue_int
}

// Generated by cxx_template
// ----------------------------------------
// Function:  double getValue
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
double AA_example_nested_ExClass2_getValue_double(
    AA_example_nested_ExClass2 * self)
{
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    // splicer begin namespace.example::nested.class.ExClass2.method.getValue_double
    double SHC_rv = SH_this->getValue<double>();
    return SHC_rv;
    // splicer end namespace.example::nested.class.ExClass2.method.getValue_double
}

}  // extern "C"
