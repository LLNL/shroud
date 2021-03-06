// top.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
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
#include "typestestnames.hh"
#include <cstdlib>

// splicer begin CXX_definitions
// Add some text from splicer
// And another line
// splicer end CXX_definitions

extern "C" {


// helper ShroudStrAlloc
// Copy src into new memory and null terminate.
static char *ShroudStrAlloc(const char *src, int nsrc, int ntrim)
{
   char *rv = (char *) std::malloc(nsrc + 1);
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  char * name +intent(inout)+len(worklen)+len_trim(worktrim)
// Requested: c_char_*_inout
// Match:     c_default
void TES_get_name(char * name)
{
    // splicer begin function.get_name
    getName(name);
    // splicer end function.get_name
}

// ----------------------------------------
// Function:  void getName
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  char * name +intent(inout)+len(worklen)+len_trim(worktrim)
// Exact:     c_char_*_inout_buf
void TES_get_name_bufferify(char * name, int worktrim, int worklen)
{
    // splicer begin function.get_name_bufferify
    char * ARG_name = ShroudStrAlloc(name, worklen, worktrim);
    getName(ARG_name);
    ShroudStrCopy(name, worklen, ARG_name, -1);
    ShroudStrFree(ARG_name);
    // splicer end function.get_name_bufferify
}

// ----------------------------------------
// Function:  void function1
// Requested: c
// Match:     c_default
void YYY_TES_function1(void)
{
    // splicer begin function.function1
    function1();
    // splicer end function.function1
}

// ----------------------------------------
// Function:  void function2
// Requested: c
// Match:     c_default
void c_name_special(void)
{
    // splicer begin function.function2
    function2();
    // splicer end function.function2
}

// ----------------------------------------
// Function:  void function3a
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int i +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
void YYY_TES_function3a_0(int i)
{
    // splicer begin function.function3a_0
    function3a(i);
    // splicer end function.function3a_0
}

// ----------------------------------------
// Function:  void function3a
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  long i +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
void YYY_TES_function3a_1(long i)
{
    // splicer begin function.function3a_1
    function3a(i);
    // splicer end function.function3a_1
}

// ----------------------------------------
// Function:  int function4
// Requested: c_native_scalar_result
// Match:     c_default
// ----------------------------------------
// Argument:  const std::string & rv +intent(in)
// Requested: c_string_&_in
// Match:     c_string_in
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
// Requested: c_native_scalar_result_buf
// Match:     c_default
// ----------------------------------------
// Argument:  const std::string & rv +intent(in)+len_trim(Lrv)
// Requested: c_string_&_in_buf
// Match:     c_string_in_buf
int YYY_TES_function4_bufferify(const char * rv, int Lrv)
{
    // splicer begin function.function4_bufferify
    const std::string ARG_rv(rv, Lrv);
    int SHC_rv = function4(ARG_rv);
    return SHC_rv;
    // splicer end function.function4_bufferify
}

// ----------------------------------------
// Function:  void function5 +name(fiveplus)
// Requested: c
// Match:     c_default
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  std::string & name +intent(inout)
// Requested: c_string_&_inout
// Match:     c_string_inout
// ----------------------------------------
// Argument:  int * value +intent(out)
// Requested: c_native_*_out
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  std::string & name +intent(inout)+len(Nname)+len_trim(Lname)
// Requested: c_string_&_inout_buf
// Match:     c_string_inout_buf
// ----------------------------------------
// Argument:  int * value +intent(out)
// Requested: c_native_*_out_buf
// Match:     c_default
void TES_test_multiline_splicer_bufferify(char * name, int Lname,
    int Nname, int * value)
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  int arg1 +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// ----------------------------------------
// Argument:  long arg2 +intent(in)+value
// Requested: c_native_scalar_in
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
// Requested: c
// Match:     c_default
// ----------------------------------------
// Argument:  float arg1 +intent(in)+value
// Requested: c_native_scalar_in
// Match:     c_default
// ----------------------------------------
// Argument:  double arg2 +intent(in)+value
// Requested: c_native_scalar_in
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
// Requested: c_native_scalar_result
// Match:     c_default
int TES_use_impl_worker_instantiation3(void)
{
    // splicer begin function.use_impl_worker_instantiation3
    int SHC_rv = UseImplWorker<internal::ImplWorker1>();
    return SHC_rv;
    // splicer end function.use_impl_worker_instantiation3
}

// ----------------------------------------
// Function:  int Cstruct_as_class_sum
// Requested: c_native_scalar_result
// Match:     c_default
// ----------------------------------------
// Argument:  const Cstruct_as_class * point +intent(in)+pass
// Requested: c_shadow_*_in
// Match:     c_shadow_in
int TES_cstruct_as_class_sum(TES_Cstruct_as_class * point)
{
    // splicer begin function.cstruct_as_class_sum
    const Cstruct_as_class * ARG_point =
        static_cast<const Cstruct_as_class *>(point->addr);
    int SHC_rv = Cstruct_as_class_sum(ARG_point);
    return SHC_rv;
    // splicer end function.cstruct_as_class_sum
}

// Release library allocated memory.
void TES_SHROUD_memory_destructor(TES_SHROUD_capsule_data *cap)
{
    void *ptr = cap->addr;
    switch (cap->idtor) {
    case 0:   // --none--
    {
        // Nothing to delete
        break;
    }
    case 1:   // ns0::Names
    {
        ns0::Names *cxx_ptr = reinterpret_cast<ns0::Names *>(ptr);
        delete cxx_ptr;
        break;
    }
    default:
    {
        // Unexpected case in destructor
        break;
    }
    }
    cap->addr = nullptr;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
