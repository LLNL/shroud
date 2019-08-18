// wrapExClass1.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
//
// All rights reserved.
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
#include "wrapExClass1.h"
#include <cstddef>
#include <cstring>
#include <stdlib.h>
#include <string>
#include "ExClass1.hpp"

// splicer begin class.ExClass1.CXX_definitions
// splicer end class.ExClass1.CXX_definitions

extern "C" {


// helper function
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

// helper function
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void AA_ShroudCopyStringAndFree(AA_SHROUD_array *data, char *c_var, size_t c_var_len) {
    const char *cxx_var = data->addr.ccharp;
    size_t n = c_var_len;
    if (data->len < n) n = data->len;
    std::strncpy(c_var, cxx_var, n);
    AA_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}

// splicer begin class.ExClass1.C_definitions
// splicer end class.ExClass1.C_definitions

// ExClass1()
AA_exclass1 * AA_example_nested_ExClass1_ctor_0(AA_exclass1 * SHC_rv)
{
// splicer begin class.ExClass1.method.ctor_0
    example::nested::ExClass1 *SHCXX_rv =
        new example::nested::ExClass1();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.ExClass1.method.ctor_0
}

// ExClass1(const string * name +intent(in))
/**
 * \brief constructor
 *
 * longer description
 * usually multiple lines
 *
 * \return return new instance
 */
AA_exclass1 * AA_example_nested_ExClass1_ctor_1(const char * name,
    AA_exclass1 * SHC_rv)
{
// splicer begin class.ExClass1.method.ctor_1
    const std::string SH_name(name);
    example::nested::ExClass1 *SHCXX_rv =
        new example::nested::ExClass1(&SH_name);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.ExClass1.method.ctor_1
}

// ExClass1(const string * name +intent(in)+len_trim(Lname))
/**
 * \brief constructor
 *
 * longer description
 * usually multiple lines
 *
 * \return return new instance
 */
AA_exclass1 * AA_example_nested_ExClass1_ctor_1_bufferify(
    const char * name, int Lname, AA_exclass1 * SHC_rv)
{
// splicer begin class.ExClass1.method.ctor_1_bufferify
    const std::string SH_name(name, Lname);
    example::nested::ExClass1 *SHCXX_rv =
        new example::nested::ExClass1(&SH_name);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.ExClass1.method.ctor_1_bufferify
}

// ~ExClass1()
/**
 * \brief destructor
 *
 * longer description joined with previous line
 */
void AA_example_nested_ExClass1_dtor(AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.dtor
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    delete SH_this;
    self->addr = NULL;
    return;
// splicer end class.ExClass1.method.dtor
}

// int incrementCount(int incr +intent(in)+value)
int AA_example_nested_ExClass1_increment_count(AA_exclass1 * self,
    int incr)
{
// splicer begin class.ExClass1.method.increment_count
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    int SHC_rv = SH_this->incrementCount(incr);
    return SHC_rv;
// splicer end class.ExClass1.method.increment_count
}

// const string & getNameErrorPattern() const +deref(result_as_arg)+len(aa_exclass1_get_name_length({F_this}%{F_derived_member}))
const char * AA_example_nested_ExClass1_get_name_error_pattern(
    const AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_name_error_pattern
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameErrorPattern();
    // C_error_pattern
    if (! isNameValid(SHCXX_rv)) {
        return NULL;
    }

    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.ExClass1.method.get_name_error_pattern
}

// void getNameErrorPattern(string & SHF_rv +intent(out)+len(NSHF_rv)) const +len(aa_exclass1_get_name_length({F_this}%{F_derived_member}))
void AA_example_nested_ExClass1_get_name_error_pattern_bufferify(
    const AA_exclass1 * self, char * SHF_rv, int NSHF_rv)
{
// splicer begin class.ExClass1.method.get_name_error_pattern_bufferify
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameErrorPattern();
    if (SHCXX_rv.empty()) {
        ShroudStrCopy(SHF_rv, NSHF_rv, NULL, 0);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    return;
// splicer end class.ExClass1.method.get_name_error_pattern_bufferify
}

// int GetNameLength() const
/**
 * \brief helper function for Fortran to get length of name.
 *
 */
int AA_example_nested_ExClass1_get_name_length(const AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_name_length
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    return SH_this->getName().length();

// splicer end class.ExClass1.method.get_name_length
}

// const string & getNameErrorCheck() const +deref(allocatable)
const char * AA_example_nested_ExClass1_get_name_error_check(
    const AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_name_error_check
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameErrorCheck();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.ExClass1.method.get_name_error_check
}

// void getNameErrorCheck(const std::string * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out)) const
void AA_example_nested_ExClass1_get_name_error_check_bufferify(
    const AA_exclass1 * self, AA_SHROUD_array *DSHF_rv)
{
// splicer begin class.ExClass1.method.get_name_error_check_bufferify
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameErrorCheck();
    DSHF_rv->cxx.addr = static_cast<void *>(const_cast<std::string *>
        (&SHCXX_rv));
    DSHF_rv->cxx.idtor = 0;
    if (SHCXX_rv.empty()) {
        DSHF_rv->addr.ccharp = NULL;
        DSHF_rv->len = 0;
    } else {
        DSHF_rv->addr.ccharp = SHCXX_rv.data();
        DSHF_rv->len = SHCXX_rv.size();
    }
    DSHF_rv->size = 1;
    return;
// splicer end class.ExClass1.method.get_name_error_check_bufferify
}

// const string & getNameArg() const +deref(result_as_arg)
const char * AA_example_nested_ExClass1_get_name_arg(
    const AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_name_arg
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameArg();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.ExClass1.method.get_name_arg
}

// void getNameArg(string & name +intent(out)+len(Nname)) const
void AA_example_nested_ExClass1_get_name_arg_bufferify(
    const AA_exclass1 * self, char * name, int Nname)
{
// splicer begin class.ExClass1.method.get_name_arg_bufferify
    const example::nested::ExClass1 *SH_this =
        static_cast<const example::nested::ExClass1 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getNameArg();
    if (SHCXX_rv.empty()) {
        ShroudStrCopy(name, Nname, NULL, 0);
    } else {
        ShroudStrCopy(name, Nname, SHCXX_rv.data(), SHCXX_rv.size());
    }
    return;
// splicer end class.ExClass1.method.get_name_arg_bufferify
}

// void * getRoot()
void * AA_example_nested_ExClass1_get_root(AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_root
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    void * SHC_rv = SH_this->getRoot();
    return SHC_rv;
// splicer end class.ExClass1.method.get_root
}

// int getValue(int value +intent(in)+value)
int AA_example_nested_ExClass1_get_value_from_int(AA_exclass1 * self,
    int value)
{
// splicer begin class.ExClass1.method.get_value_from_int
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    int SHC_rv = SH_this->getValue(value);
    return SHC_rv;
// splicer end class.ExClass1.method.get_value_from_int
}

// long getValue(long value +intent(in)+value)
long AA_example_nested_ExClass1_get_value_1(AA_exclass1 * self,
    long value)
{
// splicer begin class.ExClass1.method.get_value_1
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    long SHC_rv = SH_this->getValue(value);
    return SHC_rv;
// splicer end class.ExClass1.method.get_value_1
}

// void * getAddr()
void * AA_example_nested_ExClass1_get_addr(AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.get_addr
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    void * SHC_rv = SH_this->getAddr();
    return SHC_rv;
// splicer end class.ExClass1.method.get_addr
}

// bool hasAddr(bool in +intent(in)+value)
bool AA_example_nested_ExClass1_has_addr(AA_exclass1 * self, bool in)
{
// splicer begin class.ExClass1.method.has_addr
    example::nested::ExClass1 *SH_this =
        static_cast<example::nested::ExClass1 *>(self->addr);
    bool SHC_rv = SH_this->hasAddr(in);
    return SHC_rv;
// splicer end class.ExClass1.method.has_addr
}

// void SplicerSpecial()
void AA_example_nested_ExClass1_splicer_special(AA_exclass1 * self)
{
// splicer begin class.ExClass1.method.splicer_special
//   splicer for SplicerSpecial
// splicer end class.ExClass1.method.splicer_special
}

}  // extern "C"
