// wrapTutorial.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "tutorial.hpp"
// typemap
#include <string>
// shroud
#include <cstddef>
#include <cstring>
#include "wrapTutorial.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper ShroudLenTrim
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
static int ShroudLenTrim(const char *src, int nsrc) {
    int i;

    for (i = nsrc - 1; i >= 0; i--) {
        if (src[i] != ' ') {
            break;
        }
    }

    return i + 1;
}


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

// start helper ShroudStrToArray
// helper ShroudStrToArray
// Save str metadata into array to allow Fortran to access values.
// CHARACTER(len=elem_size) src
static void ShroudStrToArray(TUT_SHROUD_array *array, const std::string * src, int idtor)
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
// end helper ShroudStrToArray
// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  void NoReturnNoArguments
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// start TUT_NoReturnNoArguments
void TUT_NoReturnNoArguments(void)
{
    // splicer begin function.NoReturnNoArguments
    tutorial::NoReturnNoArguments();
    // splicer end function.NoReturnNoArguments
}
// end TUT_NoReturnNoArguments

// ----------------------------------------
// Function:  double PassByValue
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  double arg1 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int arg2 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
double TUT_PassByValue(double arg1, int arg2)
{
    // splicer begin function.PassByValue
    double SHC_rv = tutorial::PassByValue(arg1, arg2);
    return SHC_rv;
    // splicer end function.PassByValue
}

/**
 * Note that since a reference is returned, no intermediate string
 * is allocated.  It is assumed +owner(library).
 */
// ----------------------------------------
// Function:  const std::string ConcatenateStrings
// Attrs:     +api(cdesc)+deref(allocatable)+intent(function)
// Exact:     c_function_string_scalar_cdesc_allocatable
// ----------------------------------------
// Argument:  const std::string & arg1
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_string_&_buf
// ----------------------------------------
// Argument:  const std::string & arg2
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_string_&_buf
void TUT_ConcatenateStrings_bufferify(char *arg1, int SHT_arg1_len,
    char *arg2, int SHT_arg2_len, TUT_SHROUD_array *SHT_rv_cdesc)
{
    // splicer begin function.ConcatenateStrings_bufferify
    const std::string SHCXX_arg1(arg1,
        ShroudLenTrim(arg1, SHT_arg1_len));
    const std::string SHCXX_arg2(arg2,
        ShroudLenTrim(arg2, SHT_arg2_len));
    std::string * SHCXX_rv = new std::string;
    *SHCXX_rv = tutorial::ConcatenateStrings(SHCXX_arg1, SHCXX_arg2);
    ShroudStrToArray(SHT_rv_cdesc, SHCXX_rv, 1);
    // splicer end function.ConcatenateStrings_bufferify
}

// ----------------------------------------
// Function:  double UseDefaultArguments
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// start TUT_UseDefaultArguments
double TUT_UseDefaultArguments(void)
{
    // splicer begin function.UseDefaultArguments
    double SHC_rv = tutorial::UseDefaultArguments();
    return SHC_rv;
    // splicer end function.UseDefaultArguments
}
// end TUT_UseDefaultArguments

// ----------------------------------------
// Function:  double UseDefaultArguments
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  double arg1=3.1415 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// start TUT_UseDefaultArguments_arg1
double TUT_UseDefaultArguments_arg1(double arg1)
{
    // splicer begin function.UseDefaultArguments_arg1
    double SHC_rv = tutorial::UseDefaultArguments(arg1);
    return SHC_rv;
    // splicer end function.UseDefaultArguments_arg1
}
// end TUT_UseDefaultArguments_arg1

// ----------------------------------------
// Function:  double UseDefaultArguments
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  double arg1=3.1415 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  bool arg2=true +value
// Attrs:     +intent(in)
// Requested: c_in_bool_scalar
// Match:     c_default
// start TUT_UseDefaultArguments_arg1_arg2
double TUT_UseDefaultArguments_arg1_arg2(double arg1, bool arg2)
{
    // splicer begin function.UseDefaultArguments_arg1_arg2
    double SHC_rv = tutorial::UseDefaultArguments(arg1, arg2);
    return SHC_rv;
    // splicer end function.UseDefaultArguments_arg1_arg2
}
// end TUT_UseDefaultArguments_arg1_arg2

// ----------------------------------------
// Function:  void OverloadedFunction
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const std::string & name
// Attrs:     +intent(in)
// Exact:     c_in_string_&
void TUT_OverloadedFunction_from_name(const char * name)
{
    // splicer begin function.OverloadedFunction_from_name
    const std::string SHCXX_name(name);
    tutorial::OverloadedFunction(SHCXX_name);
    // splicer end function.OverloadedFunction_from_name
}

// ----------------------------------------
// Function:  void OverloadedFunction
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const std::string & name
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_string_&_buf
void TUT_OverloadedFunction_from_name_bufferify(char *name,
    int SHT_name_len)
{
    // splicer begin function.OverloadedFunction_from_name_bufferify
    const std::string SHCXX_name(name,
        ShroudLenTrim(name, SHT_name_len));
    tutorial::OverloadedFunction(SHCXX_name);
    // splicer end function.OverloadedFunction_from_name_bufferify
}

// ----------------------------------------
// Function:  void OverloadedFunction
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int indx +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void TUT_OverloadedFunction_from_index(int indx)
{
    // splicer begin function.OverloadedFunction_from_index
    tutorial::OverloadedFunction(indx);
    // splicer end function.OverloadedFunction_from_index
}

// ----------------------------------------
// Function:  void TemplateArgument
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int arg +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void TUT_TemplateArgument_int(int arg)
{
    // splicer begin function.TemplateArgument_int
    tutorial::TemplateArgument<int>(arg);
    // splicer end function.TemplateArgument_int
}

// ----------------------------------------
// Function:  void TemplateArgument
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  double arg +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void TUT_TemplateArgument_double(double arg)
{
    // splicer begin function.TemplateArgument_double
    tutorial::TemplateArgument<double>(arg);
    // splicer end function.TemplateArgument_double
}

// ----------------------------------------
// Function:  int TemplateReturn
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
int TUT_TemplateReturn_int(void)
{
    // splicer begin function.TemplateReturn_int
    int SHC_rv = tutorial::TemplateReturn<int>();
    return SHC_rv;
    // splicer end function.TemplateReturn_int
}

// ----------------------------------------
// Function:  double TemplateReturn
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
double TUT_TemplateReturn_double(void)
{
    // splicer begin function.TemplateReturn_double
    double SHC_rv = tutorial::TemplateReturn<double>();
    return SHC_rv;
    // splicer end function.TemplateReturn_double
}

// ----------------------------------------
// Function:  void FortranGenericOverloaded
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
void TUT_FortranGenericOverloaded_0(void)
{
    // splicer begin function.FortranGenericOverloaded_0
    tutorial::FortranGenericOverloaded();
    // splicer end function.FortranGenericOverloaded_0
}

// ----------------------------------------
// Function:  void FortranGenericOverloaded
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const std::string & name
// Attrs:     +intent(in)
// Exact:     c_in_string_&
// ----------------------------------------
// Argument:  double arg2 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void TUT_FortranGenericOverloaded_1(const char * name, double arg2)
{
    // splicer begin function.FortranGenericOverloaded_1
    const std::string SHCXX_name(name);
    tutorial::FortranGenericOverloaded(SHCXX_name, arg2);
    // splicer end function.FortranGenericOverloaded_1
}

// ----------------------------------------
// Function:  void FortranGenericOverloaded
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const std::string & name
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_string_&_buf
// ----------------------------------------
// Argument:  float arg2 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void TUT_FortranGenericOverloaded_1_float_bufferify(char *name,
    int SHT_name_len, float arg2)
{
    // splicer begin function.FortranGenericOverloaded_1_float_bufferify
    const std::string SHCXX_name(name,
        ShroudLenTrim(name, SHT_name_len));
    tutorial::FortranGenericOverloaded(SHCXX_name, arg2);
    // splicer end function.FortranGenericOverloaded_1_float_bufferify
}

// ----------------------------------------
// Function:  void FortranGenericOverloaded
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  const std::string & name
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_string_&_buf
// ----------------------------------------
// Argument:  double arg2 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void TUT_FortranGenericOverloaded_1_double_bufferify(char *name,
    int SHT_name_len, double arg2)
{
    // splicer begin function.FortranGenericOverloaded_1_double_bufferify
    const std::string SHCXX_name(name,
        ShroudLenTrim(name, SHT_name_len));
    tutorial::FortranGenericOverloaded(SHCXX_name, arg2);
    // splicer end function.FortranGenericOverloaded_1_double_bufferify
}

// ----------------------------------------
// Function:  int UseDefaultOverload
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  int num +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int TUT_UseDefaultOverload_num(int num)
{
    // splicer begin function.UseDefaultOverload_num
    int SHC_rv = tutorial::UseDefaultOverload(num);
    return SHC_rv;
    // splicer end function.UseDefaultOverload_num
}

// ----------------------------------------
// Function:  int UseDefaultOverload
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  int num +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int offset=0 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int TUT_UseDefaultOverload_num_offset(int num, int offset)
{
    // splicer begin function.UseDefaultOverload_num_offset
    int SHC_rv = tutorial::UseDefaultOverload(num, offset);
    return SHC_rv;
    // splicer end function.UseDefaultOverload_num_offset
}

// ----------------------------------------
// Function:  int UseDefaultOverload
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  int num +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int offset=0 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int stride=1 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int TUT_UseDefaultOverload_num_offset_stride(int num, int offset,
    int stride)
{
    // splicer begin function.UseDefaultOverload_num_offset_stride
    int SHC_rv = tutorial::UseDefaultOverload(num, offset, stride);
    return SHC_rv;
    // splicer end function.UseDefaultOverload_num_offset_stride
}

// ----------------------------------------
// Function:  int UseDefaultOverload
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  double type +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int num +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int TUT_UseDefaultOverload_3(double type, int num)
{
    // splicer begin function.UseDefaultOverload_3
    int SHC_rv = tutorial::UseDefaultOverload(type, num);
    return SHC_rv;
    // splicer end function.UseDefaultOverload_3
}

// ----------------------------------------
// Function:  int UseDefaultOverload
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  double type +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int num +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int offset=0 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int TUT_UseDefaultOverload_4(double type, int num, int offset)
{
    // splicer begin function.UseDefaultOverload_4
    int SHC_rv = tutorial::UseDefaultOverload(type, num, offset);
    return SHC_rv;
    // splicer end function.UseDefaultOverload_4
}

// ----------------------------------------
// Function:  int UseDefaultOverload
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  double type +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int num +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int offset=0 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int stride=1 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int TUT_UseDefaultOverload_5(double type, int num, int offset,
    int stride)
{
    // splicer begin function.UseDefaultOverload_5
    int SHC_rv = tutorial::UseDefaultOverload(type, num, offset,
        stride);
    return SHC_rv;
    // splicer end function.UseDefaultOverload_5
}

// ----------------------------------------
// Function:  TypeID typefunc
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  TypeID arg +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
TUT_TypeID TUT_typefunc(TUT_TypeID arg)
{
    // splicer begin function.typefunc
    tutorial::TypeID SHC_rv = tutorial::typefunc(arg);
    return SHC_rv;
    // splicer end function.typefunc
}

// ----------------------------------------
// Function:  EnumTypeID enumfunc
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  EnumTypeID arg +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
TUT_EnumTypeID TUT_enumfunc(TUT_EnumTypeID arg)
{
    // splicer begin function.enumfunc
    tutorial::EnumTypeID SHCXX_arg =
        static_cast<tutorial::EnumTypeID>(arg);
    tutorial::EnumTypeID SHCXX_rv = tutorial::enumfunc(SHCXX_arg);
    TUT_EnumTypeID SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
    // splicer end function.enumfunc
}

// ----------------------------------------
// Function:  Color colorfunc
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  Color arg +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
int TUT_colorfunc(int arg)
{
    // splicer begin function.colorfunc
    tutorial::Color SHCXX_arg = static_cast<tutorial::Color>(arg);
    tutorial::Color SHCXX_rv = tutorial::colorfunc(SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
    // splicer end function.colorfunc
}

/**
 * \brief Pass in reference to scalar
 *
 */
// ----------------------------------------
// Function:  void getMinMax
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int & min +intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_&
// Match:     c_default
// ----------------------------------------
// Argument:  int & max +intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_&
// Match:     c_default
// start TUT_getMinMax
void TUT_getMinMax(int * min, int * max)
{
    // splicer begin function.getMinMax
    tutorial::getMinMax(*min, *max);
    // splicer end function.getMinMax
}
// end TUT_getMinMax

/**
 * \brief Test function pointer
 *
 */
// ----------------------------------------
// Function:  int callback1
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  int in +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  int ( * incr)(int +value) +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// start TUT_callback1
int TUT_callback1(int in, int ( * incr)(int))
{
    // splicer begin function.callback1
    int SHC_rv = tutorial::callback1(in, incr);
    return SHC_rv;
    // splicer end function.callback1
}
// end TUT_callback1

// ----------------------------------------
// Function:  const std::string & LastFunctionCalled +len(30)
// Attrs:     +deref(copy)+intent(function)
// Requested: c_function_string_&_copy
// Match:     c_function_string_&
const char * TUT_LastFunctionCalled(void)
{
    // splicer begin function.LastFunctionCalled
    const std::string & SHCXX_rv = tutorial::LastFunctionCalled();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
    // splicer end function.LastFunctionCalled
}

// ----------------------------------------
// Function:  const std::string & LastFunctionCalled +len(30)
// Attrs:     +api(buf)+deref(copy)+intent(function)
// Requested: c_function_string_&_buf_copy
// Match:     c_function_string_&_buf
void TUT_LastFunctionCalled_bufferify(char *SHC_rv, int SHT_rv_len)
{
    // splicer begin function.LastFunctionCalled_bufferify
    const std::string & SHCXX_rv = tutorial::LastFunctionCalled();
    if (SHCXX_rv.empty()) {
        ShroudStrCopy(SHC_rv, SHT_rv_len, nullptr, 0);
    } else {
        ShroudStrCopy(SHC_rv, SHT_rv_len, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    // splicer end function.LastFunctionCalled_bufferify
}

}  // extern "C"
