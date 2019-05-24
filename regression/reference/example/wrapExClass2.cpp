// wrapExClass2.cpp
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
#include "wrapExClass2.h"
#include <cstddef>
#include <cstring>
#include <stdlib.h>
#include <string>
#include "ExClass1.hpp"
#include "ExClass2.hpp"
#include "sidre/SidreWrapperHelpers.hpp"

// splicer begin class.ExClass2.CXX_definitions
// splicer end class.ExClass2.CXX_definitions

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
void AA_ShroudCopyStringAndFree(USE_SHROUD_array *data, char *c_var, size_t c_var_len) {
    const char *cxx_var = data->addr.ccharp;
    size_t n = c_var_len;
    if (data->len < n) n = data->len;
    strncpy(c_var, cxx_var, n);
    AA_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}

// splicer begin class.ExClass2.C_definitions
// splicer end class.ExClass2.C_definitions

// ExClass2(const string * name +intent(in)+len_trim(trim_name))
/**
 * \brief constructor
 *
 */
AA_exclass2 * AA_exclass2_ctor(const char * name, AA_exclass2 * SHC_rv)
{
// splicer begin class.ExClass2.method.ctor
    const std::string SH_name(name);
    example::nested::ExClass2 *SHCXX_rv =
        new example::nested::ExClass2(&SH_name);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.ExClass2.method.ctor
}

// ExClass2(const string * name +intent(in)+len_trim(trim_name))
/**
 * \brief constructor
 *
 */
AA_exclass2 * AA_exclass2_ctor_bufferify(const char * name,
    int trim_name, AA_exclass2 * SHC_rv)
{
// splicer begin class.ExClass2.method.ctor_bufferify
    const std::string SH_name(name, trim_name);
    example::nested::ExClass2 *SHCXX_rv =
        new example::nested::ExClass2(&SH_name);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.ExClass2.method.ctor_bufferify
}

// ~ExClass2()
/**
 * \brief destructor
 *
 */
void AA_exclass2_dtor(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.dtor
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    delete SH_this;
    self->addr = NULL;
    return;
// splicer end class.ExClass2.method.dtor
}

// const string & getName() const +deref(result_as_arg)+len(aa_exclass2_get_name_length({F_this}%{F_derived_member}))
const char * AA_exclass2_get_name(const AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_name
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getName();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.ExClass2.method.get_name
}

// void getName(string & SHF_rv +intent(out)+len(NSHF_rv)) const +len(aa_exclass2_get_name_length({F_this}%{F_derived_member}))
void AA_exclass2_get_name_bufferify(const AA_exclass2 * self,
    char * SHF_rv, int NSHF_rv)
{
// splicer begin class.ExClass2.method.get_name_bufferify
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getName();
    if (SHCXX_rv.empty()) {
        ShroudStrCopy(SHF_rv, NSHF_rv, NULL, 0);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    return;
// splicer end class.ExClass2.method.get_name_bufferify
}

// const string & getName2() +deref(allocatable)
const char * AA_exclass2_get_name2(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_name2
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getName2();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.ExClass2.method.get_name2
}

// void getName2(const std::string * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out))
void AA_exclass2_get_name2_bufferify(AA_exclass2 * self,
    USE_SHROUD_array *DSHF_rv)
{
// splicer begin class.ExClass2.method.get_name2_bufferify
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    const std::string & SHCXX_rv = SH_this->getName2();
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
// splicer end class.ExClass2.method.get_name2_bufferify
}

// string & getName3() const +deref(allocatable)
char * AA_exclass2_get_name3(const AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_name3
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    std::string & SHCXX_rv = SH_this->getName3();
    char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.ExClass2.method.get_name3
}

// void getName3(std::string * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out)) const
void AA_exclass2_get_name3_bufferify(const AA_exclass2 * self,
    USE_SHROUD_array *DSHF_rv)
{
// splicer begin class.ExClass2.method.get_name3_bufferify
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    std::string & SHCXX_rv = SH_this->getName3();
    DSHF_rv->cxx.addr = static_cast<void *>(&SHCXX_rv);
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
// splicer end class.ExClass2.method.get_name3_bufferify
}

// string & getName4() +deref(allocatable)
char * AA_exclass2_get_name4(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_name4
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    std::string & SHCXX_rv = SH_this->getName4();
    char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end class.ExClass2.method.get_name4
}

// void getName4(std::string * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out))
void AA_exclass2_get_name4_bufferify(AA_exclass2 * self,
    USE_SHROUD_array *DSHF_rv)
{
// splicer begin class.ExClass2.method.get_name4_bufferify
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    std::string & SHCXX_rv = SH_this->getName4();
    DSHF_rv->cxx.addr = static_cast<void *>(&SHCXX_rv);
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
// splicer end class.ExClass2.method.get_name4_bufferify
}

// int GetNameLength() const
/**
 * \brief helper function for Fortran
 *
 */
int AA_exclass2_get_name_length(const AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_name_length
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    return SH_this->getName().length();

// splicer end class.ExClass2.method.get_name_length
}

// ExClass1 * get_class1(const ExClass1 * in +intent(in))
AA_exclass1 * AA_exclass2_get_class1(AA_exclass2 * self,
    const AA_exclass1 * in, AA_exclass1 * SHC_rv)
{
// splicer begin class.ExClass2.method.get_class1
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    const example::nested::ExClass1 * SHCXX_in =
        static_cast<const example::nested::ExClass1 *>(in->addr);
    example::nested::ExClass1 * SHCXX_rv = SH_this->get_class1(
        SHCXX_in);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.ExClass2.method.get_class1
}

// void * declare(TypeID type +intent(in)+value)
void AA_exclass2_declare_0(AA_exclass2 * self, int type)
{
// splicer begin class.ExClass2.method.declare_0
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    TypeID SHCXX_type = getTypeID(type);
    SH_this->declare(SHCXX_type);
    return;
// splicer end class.ExClass2.method.declare_0
}

// void * declare(TypeID type +intent(in)+value, SidreLength len=1 +intent(in)+value)
void AA_exclass2_declare_1(AA_exclass2 * self, int type,
    SIDRE_SidreLength len)
{
// splicer begin class.ExClass2.method.declare_1
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    TypeID SHCXX_type = getTypeID(type);
    SH_this->declare(SHCXX_type, len);
    return;
// splicer end class.ExClass2.method.declare_1
}

// void destroyall()
void AA_exclass2_destroyall(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.destroyall
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    SH_this->destroyall();
    return;
// splicer end class.ExClass2.method.destroyall
}

// TypeID getTypeID() const
int AA_exclass2_get_type_id(const AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_type_id
    const example::nested::ExClass2 *SH_this =
        static_cast<const example::nested::ExClass2 *>(self->addr);
    TypeID SHCXX_rv = SH_this->getTypeID();
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
// splicer end class.ExClass2.method.get_type_id
}

// void setValue(int value +intent(in)+value)
void AA_exclass2_set_value_int(AA_exclass2 * self, int value)
{
// splicer begin class.ExClass2.method.set_value_int
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    SH_this->setValue<int>(value);
    return;
// splicer end class.ExClass2.method.set_value_int
}

// void setValue(long value +intent(in)+value)
void AA_exclass2_set_value_long(AA_exclass2 * self, long value)
{
// splicer begin class.ExClass2.method.set_value_long
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    SH_this->setValue<long>(value);
    return;
// splicer end class.ExClass2.method.set_value_long
}

// void setValue(float value +intent(in)+value)
void AA_exclass2_set_value_float(AA_exclass2 * self, float value)
{
// splicer begin class.ExClass2.method.set_value_float
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    SH_this->setValue<float>(value);
    return;
// splicer end class.ExClass2.method.set_value_float
}

// void setValue(double value +intent(in)+value)
void AA_exclass2_set_value_double(AA_exclass2 * self, double value)
{
// splicer begin class.ExClass2.method.set_value_double
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    SH_this->setValue<double>(value);
    return;
// splicer end class.ExClass2.method.set_value_double
}

// int getValue()
int AA_exclass2_get_value_int(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_value_int
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    int SHC_rv = SH_this->getValue<int>();
    return SHC_rv;
// splicer end class.ExClass2.method.get_value_int
}

// double getValue()
double AA_exclass2_get_value_double(AA_exclass2 * self)
{
// splicer begin class.ExClass2.method.get_value_double
    example::nested::ExClass2 *SH_this =
        static_cast<example::nested::ExClass2 *>(self->addr);
    double SHC_rv = SH_this->getValue<double>();
    return SHC_rv;
// splicer end class.ExClass2.method.get_value_double
}

}  // extern "C"
