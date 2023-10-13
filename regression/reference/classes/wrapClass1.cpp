// wrapClass1.cpp
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
#include <cstddef>
#include <cstring>
#include "wrapClass1.h"

// splicer begin class.Class1.CXX_definitions
// splicer end class.Class1.CXX_definitions

extern "C" {


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


// start helper string_to_cdesc
// helper string_to_cdesc
// Save std::string metadata into array to allow Fortran to access values.
// CHARACTER(len=elem_size) src
static void ShroudStringToCdesc(CLA_SHROUD_array *cdesc,
    const std::string * src)
{
    if (src->empty()) {
        cdesc->addr.ccharp = NULL;
        cdesc->elem_len = 0;
    } else {
        cdesc->addr.ccharp = src->data();
        cdesc->elem_len = src->length();
    }
    cdesc->size = 1;
    cdesc->rank = 0;  // scalar
}
// end helper string_to_cdesc
// splicer begin class.Class1.C_definitions
// splicer end class.Class1.C_definitions

// ----------------------------------------
// Function:  Class1
// Attrs:     +api(capptr)+intent(ctor)
// Statement: f_ctor_shadow_scalar_capptr
// start CLA_Class1_ctor_default
CLA_Class1 * CLA_Class1_ctor_default(CLA_Class1 * SHC_rv)
{
    // splicer begin class.Class1.method.ctor_default
    classes::Class1 *SHCXX_rv = new classes::Class1();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
    // splicer end class.Class1.method.ctor_default
}
// end CLA_Class1_ctor_default

// ----------------------------------------
// Function:  Class1
// Attrs:     +api(capptr)+intent(ctor)
// Statement: f_ctor_shadow_scalar_capptr
// ----------------------------------------
// Argument:  int flag +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// start CLA_Class1_ctor_flag
CLA_Class1 * CLA_Class1_ctor_flag(int flag, CLA_Class1 * SHC_rv)
{
    // splicer begin class.Class1.method.ctor_flag
    classes::Class1 *SHCXX_rv = new classes::Class1(flag);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
    // splicer end class.Class1.method.ctor_flag
}
// end CLA_Class1_ctor_flag

// ----------------------------------------
// Function:  ~Class1 +name(delete)
// Attrs:     +intent(dtor)
// Statement: f_dtor
// start CLA_Class1_delete
void CLA_Class1_delete(CLA_Class1 * self)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.delete
    delete SH_this;
    self->addr = nullptr;
    // splicer end class.Class1.method.delete
}
// end CLA_Class1_delete

/**
 * \brief returns the value of flag member
 *
 */
// ----------------------------------------
// Function:  int Method1
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
// start CLA_Class1_Method1
int CLA_Class1_Method1(CLA_Class1 * self)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.Method1
    int SHC_rv = SH_this->Method1();
    return SHC_rv;
    // splicer end class.Class1.method.Method1
}
// end CLA_Class1_Method1

/**
 * \brief Pass in reference to instance
 *
 */
// ----------------------------------------
// Function:  bool equivalent
// Attrs:     +intent(function)
// Statement: f_function_bool_scalar
// ----------------------------------------
// Argument:  const Class1 & obj2
// Attrs:     +intent(in)
// Statement: f_in_shadow_&
// start CLA_Class1_equivalent
bool CLA_Class1_equivalent(const CLA_Class1 * self, CLA_Class1 * obj2)
{
    const classes::Class1 *SH_this =
        static_cast<const classes::Class1 *>(self->addr);
    // splicer begin class.Class1.method.equivalent
    const classes::Class1 * SHCXX_obj2 =
        static_cast<const classes::Class1 *>(obj2->addr);
    bool SHC_rv = SH_this->equivalent(*SHCXX_obj2);
    return SHC_rv;
    // splicer end class.Class1.method.equivalent
}
// end CLA_Class1_equivalent

/**
 * \brief Return pointer to 'this' to allow chaining calls
 *
 */
// Generated by return_this
// ----------------------------------------
// Function:  void returnThis
// Attrs:     +api(capptr)+intent(subroutine)
// Statement: f_subroutine
// start CLA_Class1_returnThis
void CLA_Class1_returnThis(CLA_Class1 * self)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.returnThis
    SH_this->returnThis();
    // splicer end class.Class1.method.returnThis
}
// end CLA_Class1_returnThis

/**
 * \brief Return pointer to 'this' to allow chaining calls
 *
 */
// ----------------------------------------
// Function:  Class1 * returnThisBuffer
// Attrs:     +api(capptr)+intent(function)
// Statement: f_function_shadow_*_capptr
// ----------------------------------------
// Argument:  std::string & name +intent(in)
// Attrs:     +intent(in)
// Statement: f_in_string_&
// ----------------------------------------
// Argument:  bool flag +value
// Attrs:     +intent(in)
// Statement: f_in_bool_scalar
// start CLA_Class1_returnThisBuffer
CLA_Class1 * CLA_Class1_returnThisBuffer(CLA_Class1 * self, char * name,
    bool flag, CLA_Class1 * SHC_rv)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.returnThisBuffer
    std::string SHCXX_name(name);
    classes::Class1 * SHCXX_rv = SH_this->returnThisBuffer(SHCXX_name,
        flag);
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end class.Class1.method.returnThisBuffer
}
// end CLA_Class1_returnThisBuffer

/**
 * \brief Return pointer to 'this' to allow chaining calls
 *
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  Class1 * returnThisBuffer
// Attrs:     +api(capptr)+intent(function)
// Statement: f_function_shadow_*_capptr
// ----------------------------------------
// Argument:  std::string & name +intent(in)
// Attrs:     +api(buf)+intent(in)
// Statement: f_in_string_&_buf
// ----------------------------------------
// Argument:  bool flag +value
// Attrs:     +intent(in)
// Statement: f_in_bool_scalar
// start CLA_Class1_returnThisBuffer_bufferify
CLA_Class1 * CLA_Class1_returnThisBuffer_bufferify(CLA_Class1 * self,
    char *name, int SHT_name_len, bool flag, CLA_Class1 * SHC_rv)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.returnThisBuffer_bufferify
    std::string SHCXX_name(name, ShroudCharLenTrim(name, SHT_name_len));
    classes::Class1 * SHCXX_rv = SH_this->returnThisBuffer(SHCXX_name,
        flag);
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end class.Class1.method.returnThisBuffer_bufferify
}
// end CLA_Class1_returnThisBuffer_bufferify

/**
 * \brief Test const method
 *
 */
// ----------------------------------------
// Function:  Class1 * getclass3
// Attrs:     +api(capptr)+intent(function)
// Statement: f_function_shadow_*_capptr
// start CLA_Class1_getclass3
CLA_Class1 * CLA_Class1_getclass3(const CLA_Class1 * self,
    CLA_Class1 * SHC_rv)
{
    const classes::Class1 *SH_this =
        static_cast<const classes::Class1 *>(self->addr);
    // splicer begin class.Class1.method.getclass3
    classes::Class1 * SHCXX_rv = SH_this->getclass3();
    SHC_rv->addr = SHCXX_rv;
    SHC_rv->idtor = 0;
    return SHC_rv;
    // splicer end class.Class1.method.getclass3
}
// end CLA_Class1_getclass3

/**
 * \brief test helper
 *
 */
// ----------------------------------------
// Function:  const std::string & getName
// Attrs:     +deref(allocatable)+intent(function)
// Statement: f_function_string_&_allocatable
// start CLA_Class1_getName
const char * CLA_Class1_getName(CLA_Class1 * self)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.getName
    const std::string & SHCXX_rv = SH_this->getName();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end class.Class1.method.getName
}
// end CLA_Class1_getName

/**
 * \brief test helper
 *
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  const std::string & getName
// Attrs:     +api(cdesc)+deref(allocatable)+intent(function)
// Statement: f_function_string_&_cdesc_allocatable
// start CLA_Class1_getName_bufferify
void CLA_Class1_getName_bufferify(CLA_Class1 * self,
    CLA_SHROUD_array *SHT_rv_cdesc,
    CLA_SHROUD_capsule_data *SHT_rv_capsule)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.getName_bufferify
    const std::string & SHCXX_rv = SH_this->getName();
    ShroudStringToCdesc(SHT_rv_cdesc, &SHCXX_rv);
    SHT_rv_capsule->addr  = const_cast<std::string *>(&SHCXX_rv);
    SHT_rv_capsule->idtor = 0;
    // splicer end class.Class1.method.getName_bufferify
}
// end CLA_Class1_getName_bufferify

// ----------------------------------------
// Function:  DIRECTION directionFunc
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
// ----------------------------------------
// Argument:  DIRECTION arg +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// start CLA_Class1_directionFunc
int CLA_Class1_directionFunc(CLA_Class1 * self, int arg)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.directionFunc
    classes::Class1::DIRECTION SHCXX_arg =
        static_cast<classes::Class1::DIRECTION>(arg);
    classes::Class1::DIRECTION SHCXX_rv = SH_this->directionFunc(
        SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
    // splicer end class.Class1.method.directionFunc
}
// end CLA_Class1_directionFunc

// Generated by getter/setter
// ----------------------------------------
// Function:  int get_m_flag
// Attrs:     +intent(getter)
// Statement: f_getter_native_scalar
// start CLA_Class1_get_m_flag
int CLA_Class1_get_m_flag(CLA_Class1 * self)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.get_m_flag
    // skip call c_getter
    return SH_this->m_flag;
    // splicer end class.Class1.method.get_m_flag
}
// end CLA_Class1_get_m_flag

// Generated by getter/setter
// ----------------------------------------
// Function:  int get_test
// Attrs:     +intent(getter)
// Statement: f_getter_native_scalar
// start CLA_Class1_get_test
int CLA_Class1_get_test(CLA_Class1 * self)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.get_test
    // skip call c_getter
    return SH_this->m_test;
    // splicer end class.Class1.method.get_test
}
// end CLA_Class1_get_test

// Generated by getter/setter
// ----------------------------------------
// Function:  void set_test
// Attrs:     +intent(setter)
// Statement: f_setter
// ----------------------------------------
// Argument:  int val +intent(in)+value
// Attrs:     +intent(setter)
// Statement: f_setter_native_scalar
// start CLA_Class1_set_test
void CLA_Class1_set_test(CLA_Class1 * self, int val)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.set_test
    // skip call c_setter
    SH_this->m_test = val;
    // splicer end class.Class1.method.set_test
}
// end CLA_Class1_set_test

// Generated by arg_to_buffer - getter/setter
// ----------------------------------------
// Function:  std::string get_m_name
// Attrs:     +api(cdesc)+deref(allocatable)+intent(getter)
// Statement: f_getter_string_scalar_cdesc_allocatable
// start CLA_Class1_get_m_name_bufferify
void CLA_Class1_get_m_name_bufferify(CLA_Class1 * self,
    CLA_SHROUD_array *SHT_rv_cdesc)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.get_m_name_bufferify
    SHT_rv_cdesc->addr.base = SH_this->m_name.data();
    SHT_rv_cdesc->type = 0; // SH_CHAR;
    SHT_rv_cdesc->elem_len = SH_this->m_name.size();
    SHT_rv_cdesc->rank = 0;
    // splicer end class.Class1.method.get_m_name_bufferify
}
// end CLA_Class1_get_m_name_bufferify

// Generated by arg_to_buffer - getter/setter
// ----------------------------------------
// Function:  void set_m_name
// Attrs:     +intent(setter)
// Statement: f_setter
// ----------------------------------------
// Argument:  std::string val +intent(in)
// Attrs:     +api(buf)+intent(setter)
// Statement: f_setter_string_scalar_buf
// start CLA_Class1_set_m_name_bufferify
void CLA_Class1_set_m_name_bufferify(CLA_Class1 * self, char *val,
    int SHT_val_len)
{
    classes::Class1 *SH_this = static_cast<classes::Class1 *>
        (self->addr);
    // splicer begin class.Class1.method.set_m_name_bufferify
    // skip call c_setter
    SH_this->m_name = std::string(val, SHT_val_len);
    // splicer end class.Class1.method.set_m_name_bufferify
}
// end CLA_Class1_set_m_name_bufferify

}  // extern "C"
