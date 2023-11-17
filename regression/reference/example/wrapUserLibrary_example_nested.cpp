// wrapUserLibrary_example_nested.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// typemap
#include <string>
// shroud
#include "wrapUserLibrary_example_nested.h"

// splicer begin namespace.example::nested.CXX_definitions
// splicer end namespace.example::nested.CXX_definitions

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

// splicer begin namespace.example::nested.C_definitions
// splicer end namespace.example::nested.C_definitions

// ----------------------------------------
// Function:  void local_function1
// Statement: c_subroutine
void AA_example_nested_local_function1(void)
{
    // splicer begin namespace.example::nested.function.local_function1
    example::nested::local_function1();
    // splicer end namespace.example::nested.function.local_function1
}

// ----------------------------------------
// Function:  bool isNameValid
// Statement: c_function_bool_scalar
// ----------------------------------------
// Argument:  const std::string & name
// Statement: c_in_string_&
bool AA_example_nested_isNameValid(const char * name)
{
    // splicer begin namespace.example::nested.function.isNameValid
    return name != NULL;
    // splicer end namespace.example::nested.function.isNameValid
}

// ----------------------------------------
// Function:  bool isNameValid
// Statement: f_function_bool_scalar
// ----------------------------------------
// Argument:  const std::string & name
// Statement: f_in_string_&_buf
bool AA_example_nested_isNameValid_bufferify(char *name,
    int SHT_name_len)
{
    // splicer begin namespace.example::nested.function.isNameValid_bufferify
    return name != NULL;
    // splicer end namespace.example::nested.function.isNameValid_bufferify
}

// ----------------------------------------
// Function:  bool isInitialized
// Statement: c_function_bool_scalar
bool AA_example_nested_isInitialized(void)
{
    // splicer begin namespace.example::nested.function.isInitialized
    bool SHC_rv = example::nested::isInitialized();
    return SHC_rv;
    // splicer end namespace.example::nested.function.isInitialized
}

// ----------------------------------------
// Function:  void test_names
// Statement: c_subroutine
// ----------------------------------------
// Argument:  const std::string & name
// Statement: c_in_string_&
void AA_example_nested_test_names(const char * name)
{
    // splicer begin namespace.example::nested.function.test_names
    const std::string SHCXX_name(name);
    example::nested::test_names(SHCXX_name);
    // splicer end namespace.example::nested.function.test_names
}

// ----------------------------------------
// Function:  void test_names
// Statement: f_subroutine
// ----------------------------------------
// Argument:  const std::string & name
// Statement: f_in_string_&_buf
void AA_example_nested_test_names_bufferify(char *name,
    int SHT_name_len)
{
    // splicer begin namespace.example::nested.function.test_names_bufferify
    const std::string SHCXX_name(name,
        ShroudCharLenTrim(name, SHT_name_len));
    example::nested::test_names(SHCXX_name);
    // splicer end namespace.example::nested.function.test_names_bufferify
}

// ----------------------------------------
// Function:  void test_names
// Statement: c_subroutine
// ----------------------------------------
// Argument:  const std::string & name
// Statement: c_in_string_&
// ----------------------------------------
// Argument:  int flag +value
// Statement: c_in_native_scalar
void AA_example_nested_test_names_flag(const char * name, int flag)
{
    // splicer begin namespace.example::nested.function.test_names_flag
    const std::string SHCXX_name(name);
    example::nested::test_names(SHCXX_name, flag);
    // splicer end namespace.example::nested.function.test_names_flag
}

// ----------------------------------------
// Function:  void test_names
// Statement: f_subroutine
// ----------------------------------------
// Argument:  const std::string & name
// Statement: f_in_string_&_buf
// ----------------------------------------
// Argument:  int flag +value
// Statement: f_in_native_scalar
void AA_example_nested_test_names_flag_bufferify(char *name,
    int SHT_name_len, int flag)
{
    // splicer begin namespace.example::nested.function.test_names_flag_bufferify
    const std::string SHCXX_name(name,
        ShroudCharLenTrim(name, SHT_name_len));
    example::nested::test_names(SHCXX_name, flag);
    // splicer end namespace.example::nested.function.test_names_flag_bufferify
}

// Generated by has_default_arg
// ----------------------------------------
// Function:  void testoptional
// Statement: c_subroutine
void AA_example_nested_testoptional_0(void)
{
    // splicer begin namespace.example::nested.function.testoptional_0
    example::nested::testoptional();
    // splicer end namespace.example::nested.function.testoptional_0
}

// Generated by has_default_arg
// ----------------------------------------
// Function:  void testoptional
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int i=1 +value
// Statement: c_in_native_scalar
void AA_example_nested_testoptional_1(int i)
{
    // splicer begin namespace.example::nested.function.testoptional_1
    example::nested::testoptional(i);
    // splicer end namespace.example::nested.function.testoptional_1
}

// ----------------------------------------
// Function:  void testoptional
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int i=1 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  long j=2 +value
// Statement: c_in_native_scalar
void AA_example_nested_testoptional_2(int i, long j)
{
    // splicer begin namespace.example::nested.function.testoptional_2
    example::nested::testoptional(i, j);
    // splicer end namespace.example::nested.function.testoptional_2
}

// ----------------------------------------
// Function:  size_t test_size_t
// Statement: c_function_native_scalar
size_t AA_example_nested_test_size_t(void)
{
    // splicer begin namespace.example::nested.function.test_size_t
    size_t SHC_rv = example::nested::test_size_t();
    return SHC_rv;
    // splicer end namespace.example::nested.function.test_size_t
}

#ifdef HAVE_MPI
// ----------------------------------------
// Function:  void testmpi
// Statement: c_subroutine
// ----------------------------------------
// Argument:  MPI_Comm comm +value
// Statement: c_in_unknown_scalar
void AA_example_nested_testmpi_mpi(MPI_Fint comm)
{
    // splicer begin namespace.example::nested.function.testmpi_mpi
    MPI_Comm SHCXX_comm = MPI_Comm_f2c(comm);
    example::nested::testmpi(SHCXX_comm);
    // splicer end namespace.example::nested.function.testmpi_mpi
}
#endif  // ifdef HAVE_MPI

#ifndef HAVE_MPI
// ----------------------------------------
// Function:  void testmpi
// Statement: c_subroutine
void AA_example_nested_testmpi_serial(void)
{
    // splicer begin namespace.example::nested.function.testmpi_serial
    example::nested::testmpi();
    // splicer end namespace.example::nested.function.testmpi_serial
}
#endif  // ifndef HAVE_MPI

/**
 * \brief subroutine
 *
 */
// ----------------------------------------
// Function:  void FuncPtr1
// Statement: c_subroutine
// ----------------------------------------
// Argument:  void ( * get)(void) +value
// Statement: c_in_void_scalar
void AA_example_nested_FuncPtr1(void ( * get)(void))
{
    // splicer begin namespace.example::nested.function.FuncPtr1
    example::nested::FuncPtr1(get);
    // splicer end namespace.example::nested.function.FuncPtr1
}

/**
 * \brief return a pointer
 *
 */
// ----------------------------------------
// Function:  void FuncPtr2
// Statement: c_subroutine
// ----------------------------------------
// Argument:  double * ( * get)(void)
// Statement: c_in_native_*
void AA_example_nested_FuncPtr2(double * ( * get)(void))
{
    // splicer begin namespace.example::nested.function.FuncPtr2
    example::nested::FuncPtr2(get);
    // splicer end namespace.example::nested.function.FuncPtr2
}

/**
 * \brief abstract argument
 *
 */
// ----------------------------------------
// Function:  void FuncPtr3
// Statement: c_subroutine
// ----------------------------------------
// Argument:  double ( * get)(int i +value, int +value) +value
// Statement: c_in_native_scalar
void AA_example_nested_FuncPtr3(double ( * get)(int i, int))
{
    // splicer begin namespace.example::nested.function.FuncPtr3
    example::nested::FuncPtr3(get);
    // splicer end namespace.example::nested.function.FuncPtr3
}

/**
 * \brief abstract argument
 *
 */
// ----------------------------------------
// Function:  void FuncPtr4
// Statement: c_subroutine
// ----------------------------------------
// Argument:  double ( * get)(double +value, int +value) +value
// Statement: c_in_native_scalar
void AA_example_nested_FuncPtr4(double ( * get)(double, int))
{
    // splicer begin namespace.example::nested.function.FuncPtr4
    example::nested::FuncPtr4(get);
    // splicer end namespace.example::nested.function.FuncPtr4
}

// ----------------------------------------
// Function:  void FuncPtr5
// Statement: c_subroutine
// ----------------------------------------
// Argument:  void ( * get)(int verylongname1 +value, int verylongname2 +value, int verylongname3 +value, int verylongname4 +value, int verylongname5 +value, int verylongname6 +value, int verylongname7 +value, int verylongname8 +value, int verylongname9 +value, int verylongname10 +value) +value
// Statement: c_in_void_scalar
void AA_example_nested_FuncPtr5(void ( * get)(int verylongname1,
    int verylongname2, int verylongname3, int verylongname4,
    int verylongname5, int verylongname6, int verylongname7,
    int verylongname8, int verylongname9, int verylongname10))
{
    // splicer begin namespace.example::nested.function.FuncPtr5
    example::nested::FuncPtr5(get);
    // splicer end namespace.example::nested.function.FuncPtr5
}

// ----------------------------------------
// Function:  void verylongfunctionname1
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * verylongname1 +intent(inout)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname2 +intent(inout)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname3 +intent(inout)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname4 +intent(inout)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname5 +intent(inout)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname6 +intent(inout)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname7 +intent(inout)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname8 +intent(inout)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname9 +intent(inout)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int * verylongname10 +intent(inout)
// Statement: c_inout_native_*
void AA_example_nested_verylongfunctionname1(int * verylongname1,
    int * verylongname2, int * verylongname3, int * verylongname4,
    int * verylongname5, int * verylongname6, int * verylongname7,
    int * verylongname8, int * verylongname9, int * verylongname10)
{
    // splicer begin namespace.example::nested.function.verylongfunctionname1
    example::nested::verylongfunctionname1(verylongname1, verylongname2,
        verylongname3, verylongname4, verylongname5, verylongname6,
        verylongname7, verylongname8, verylongname9, verylongname10);
    // splicer end namespace.example::nested.function.verylongfunctionname1
}

// ----------------------------------------
// Function:  int verylongfunctionname2
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  int verylongname1 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname2 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname3 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname4 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname5 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname6 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname7 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname8 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname9 +value
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int verylongname10 +value
// Statement: c_in_native_scalar
int AA_example_nested_verylongfunctionname2(int verylongname1,
    int verylongname2, int verylongname3, int verylongname4,
    int verylongname5, int verylongname6, int verylongname7,
    int verylongname8, int verylongname9, int verylongname10)
{
    // splicer begin namespace.example::nested.function.verylongfunctionname2
    int SHC_rv = example::nested::verylongfunctionname2(verylongname1,
        verylongname2, verylongname3, verylongname4, verylongname5,
        verylongname6, verylongname7, verylongname8, verylongname9,
        verylongname10);
    return SHC_rv;
    // splicer end namespace.example::nested.function.verylongfunctionname2
}

/**
 * \brief Test multidimensional arrays with allocatable
 *
 */
// ----------------------------------------
// Function:  void cos_doubles
// Statement: c_subroutine
// ----------------------------------------
// Argument:  double * in +intent(in)+rank(2)
// Statement: c_in_native_*
// ----------------------------------------
// Argument:  double * out +dimension(shape(in))+intent(out)
// Statement: c_out_native_*
// ----------------------------------------
// Argument:  int sizein +implied(size(in))+value
// Statement: c_in_native_scalar
void AA_example_nested_cos_doubles(double * in, double * out,
    int sizein)
{
    // splicer begin namespace.example::nested.function.cos_doubles
    example::nested::cos_doubles(in, out, sizein);
    // splicer end namespace.example::nested.function.cos_doubles
}

}  // extern "C"
