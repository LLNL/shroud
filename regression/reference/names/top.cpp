// top.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "top.h"

// typemap
#include <string>
// shroud
#include <cstring>
#include <cstdlib>

// splicer begin CXX_definitions
// Add some text from splicer
// And another line
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


// helper ShroudStrAlloc
// Copy src into new memory and null terminate.
static char *ShroudStrAlloc(const char *src, int nsrc, int ntrim)
{
   char *rv = (char *) std::malloc(nsrc + 1);
   if (ntrim == -1) {
      ntrim = ShroudLenTrim(src, nsrc);
   }
   if (ntrim > 0) {
     std::memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\0';
   return rv;
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

// helper ShroudStrFree
// Release memory allocated by ShroudStrAlloc
static void ShroudStrFree(char *src)
{
   free(src);
}
// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  void getName
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  char * name +len(worklen)+len_trim(worktrim)
// Attrs:     +intent(inout)
// Requested: c_inout_char_*
// Match:     c_default
void TES_get_name(char * name)
{
    // splicer begin function.get_name
    getName(name);
    // splicer end function.get_name
}

// ----------------------------------------
// Function:  void getName
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  char * name +len(worklen)+len_trim(worktrim)
// Attrs:     +api(buf)+intent(inout)
// Exact:     c_inout_char_*_buf
void TES_get_name_bufferify(char *name, int SHT_name_len)
{
    // splicer begin function.get_name_bufferify
    char * ARG_name = ShroudStrAlloc(name, SHT_name_len, -1);
    getName(ARG_name);
    ShroudStrCopy(name, SHT_name_len, ARG_name, -1);
    ShroudStrFree(ARG_name);
    // splicer end function.get_name_bufferify
}

// ----------------------------------------
// Function:  void function1
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
void YYY_TES_function1(void)
{
    // splicer begin function.function1
    function1();
    // splicer end function.function1
}

// ----------------------------------------
// Function:  void function2
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
void c_name_special(void)
{
    // splicer begin function.function2
    function2();
    // splicer end function.function2
}

// ----------------------------------------
// Function:  void function3a
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int i +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void YYY_TES_function3a_0(int i)
{
    // splicer begin function.function3a_0
    function3a(i);
    // splicer end function.function3a_0
}

// ----------------------------------------
// Function:  void function3a
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  long i +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void YYY_TES_function3a_1(long i)
{
    // splicer begin function.function3a_1
    function3a(i);
    // splicer end function.function3a_1
}

// ----------------------------------------
// Function:  int function4
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const std::string & rv
// Attrs:     +intent(in)
// Exact:     c_in_string_&
int YYY_TES_function4(const char * rv)
{
    // splicer begin function.function4
    const std::string ARG_rv(rv);
    int SHC_rv = function4(ARG_rv);
    return SHC_rv;
    // splicer end function.function4
}

// ----------------------------------------
// Function:  int function4
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const std::string & rv
// Attrs:     +api(buf)+intent(in)
// Exact:     c_in_string_&_buf
int YYY_TES_function4_bufferify(char *rv, int SHT_rv_len)
{
    // splicer begin function.function4_bufferify
    const std::string ARG_rv(rv, ShroudLenTrim(rv, SHT_rv_len));
    int SHC_rv = function4(ARG_rv);
    return SHC_rv;
    // splicer end function.function4_bufferify
}

// ----------------------------------------
// Function:  void function5 +name(fiveplus)
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
void YYY_TES_fiveplus(void)
{
    // splicer begin function.fiveplus
    fiveplus();
    // splicer end function.fiveplus
}

/**
 * Use std::string argument to get bufferified function.
 */
// ----------------------------------------
// Function:  void TestMultilineSplicer
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::string & name
// Attrs:     +intent(inout)
// Exact:     c_inout_string_&
// ----------------------------------------
// Argument:  int * value +intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_*
// Match:     c_default
void TES_test_multiline_splicer(char * name, int * value)
{
    // splicer begin function.test_multiline_splicer
    // line 1
    // line 2
    // splicer end function.test_multiline_splicer
}

/**
 * Use std::string argument to get bufferified function.
 */
// ----------------------------------------
// Function:  void TestMultilineSplicer
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  std::string & name
// Attrs:     +api(buf)+intent(inout)
// Exact:     c_inout_string_&_buf
// ----------------------------------------
// Argument:  int * value +intent(out)
// Attrs:     +intent(out)
// Requested: c_out_native_*
// Match:     c_default
void TES_test_multiline_splicer_bufferify(char *name, int SHT_name_len,
    int * value)
{
    // splicer begin function.test_multiline_splicer_bufferify
    // buf line 1
    // buf line 2
    // splicer end function.test_multiline_splicer_bufferify
}

/**
 * \brief Function template with two template parameters.
 *
 */
// ----------------------------------------
// Function:  void FunctionTU
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  int arg1 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  long arg2 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void c_name_instantiation1(int arg1, long arg2)
{
    // splicer begin function.function_tu_0
    FunctionTU<int, long>(arg1, arg2);
    // splicer end function.function_tu_0
}

/**
 * \brief Function template with two template parameters.
 *
 */
// ----------------------------------------
// Function:  void FunctionTU
// Attrs:     +intent(subroutine)
// Exact:     c_subroutine
// ----------------------------------------
// Argument:  float arg1 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
// ----------------------------------------
// Argument:  double arg2 +value
// Attrs:     +intent(in)
// Requested: c_in_native_scalar
// Match:     c_default
void TES_function_tu_instantiation2(float arg1, double arg2)
{
    // splicer begin function.function_tu_instantiation2
    FunctionTU<float, double>(arg1, arg2);
    // splicer end function.function_tu_instantiation2
}

/**
 * \brief Function which uses a templated T in the implemetation.
 *
 */
// ----------------------------------------
// Function:  int UseImplWorker
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
int TES_use_impl_worker_instantiation3(void)
{
    // splicer begin function.use_impl_worker_instantiation3
    int SHC_rv = UseImplWorker<internal::ImplWorker1>();
    return SHC_rv;
    // splicer end function.use_impl_worker_instantiation3
}

// ----------------------------------------
// Function:  int Cstruct_as_class_sum
// Attrs:     +intent(function)
// Requested: c_function_native_scalar
// Match:     c_function
// ----------------------------------------
// Argument:  const Cstruct_as_class * point +pass
// Attrs:     +intent(in)
// Requested: c_in_shadow_*
// Match:     c_in_shadow
int TES_cstruct_as_class_sum(TES_Cstruct_as_class * point)
{
    // splicer begin function.cstruct_as_class_sum
    const Cstruct_as_class * ARG_point =
        static_cast<const Cstruct_as_class *>(point->addr);
    int SHC_rv = Cstruct_as_class_sum(ARG_point);
    return SHC_rv;
    // splicer end function.cstruct_as_class_sum
}

}  // extern "C"
