// wrappointers.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "pointers.h"
// shroud
#include "wrappointers.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// ----------------------------------------
// Function:  void intargs_in
// Statement: c_subroutine
// ----------------------------------------
// Argument:  const int * arg
// Statement: c_in_native_*
// start POI_intargs_in
void POI_intargs_in(const int * arg)
{
    // splicer begin function.intargs_in
    intargs_in(arg);
    // splicer end function.intargs_in
}
// end POI_intargs_in

/**
 * Argument is modified by library, defaults to intent(inout).
 */
// ----------------------------------------
// Function:  void intargs_inout
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * arg
// Statement: c_inout_native_*
// start POI_intargs_inout
void POI_intargs_inout(int * arg)
{
    // splicer begin function.intargs_inout
    intargs_inout(arg);
    // splicer end function.intargs_inout
}
// end POI_intargs_inout

// ----------------------------------------
// Function:  void intargs_out
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * arg +intent(out)
// Statement: c_out_native_*
// start POI_intargs_out
void POI_intargs_out(int * arg)
{
    // splicer begin function.intargs_out
    intargs_out(arg);
    // splicer end function.intargs_out
}
// end POI_intargs_out

// ----------------------------------------
// Function:  void intargs
// Statement: c_subroutine
// ----------------------------------------
// Argument:  const int argin +intent(in)
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int * arginout +intent(inout)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int * argout +intent(out)
// Statement: c_out_native_*
// start POI_intargs
void POI_intargs(const int argin, int * arginout, int * argout)
{
    // splicer begin function.intargs
    intargs(argin, arginout, argout);
    // splicer end function.intargs
}
// end POI_intargs

/**
 * \brief compute cos of IN and save in OUT
 *
 * allocate OUT same type as IN implied size of array
 */
// ----------------------------------------
// Function:  void cos_doubles
// Statement: c_subroutine
// ----------------------------------------
// Argument:  double * in +intent(in)+rank(1)
// Statement: c_in_native_*
// ----------------------------------------
// Argument:  double * out +dimension(size(in))+intent(out)
// Statement: c_out_native_*
// ----------------------------------------
// Argument:  int sizein +implied(size(in))
// Statement: c_in_native_scalar
// start POI_cos_doubles
void POI_cos_doubles(double * in, double * out, int sizein)
{
    // splicer begin function.cos_doubles
    cos_doubles(in, out, sizein);
    // splicer end function.cos_doubles
}
// end POI_cos_doubles

/**
 * \brief truncate IN argument and save in OUT
 *
 * allocate OUT different type as IN
 * implied size of array
 */
// ----------------------------------------
// Function:  void truncate_to_int
// Statement: c_subroutine
// ----------------------------------------
// Argument:  double * in +intent(in)+rank(1)
// Statement: c_in_native_*
// ----------------------------------------
// Argument:  int * out +dimension(size(in))+intent(out)
// Statement: c_out_native_*
// ----------------------------------------
// Argument:  int sizein +implied(size(in))
// Statement: c_in_native_scalar
// start POI_truncate_to_int
void POI_truncate_to_int(double * in, int * out, int sizein)
{
    // splicer begin function.truncate_to_int
    truncate_to_int(in, out, sizein);
    // splicer end function.truncate_to_int
}
// end POI_truncate_to_int

/**
 * \brief fill values into array
 *
 * The function knows how long the array must be.
 * Fortran will treat the dimension as assumed-length.
 * The Python wrapper will create a NumPy array or list so it must
 * have an explicit dimension (not assumed-length).
 */
// ----------------------------------------
// Function:  void get_values
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * nvalues +intent(OUT)
// Statement: c_out_native_*
// ----------------------------------------
// Argument:  int * values +dimension(3)+intent(out)
// Statement: c_out_native_*
// start POI_get_values
void POI_get_values(int * nvalues, int * values)
{
    // splicer begin function.get_values
    get_values(nvalues, values);
    // splicer end function.get_values
}
// end POI_get_values

/**
 * \brief fill values into two arrays
 *
 * Test two intent(out) arguments.
 * Make sure error handling works with C++.
 */
// ----------------------------------------
// Function:  void get_values2
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * arg1 +dimension(3)+intent(out)
// Statement: c_out_native_*
// ----------------------------------------
// Argument:  int * arg2 +dimension(3)+intent(out)
// Statement: c_out_native_*
// start POI_get_values2
void POI_get_values2(int * arg1, int * arg2)
{
    // splicer begin function.get_values2
    get_values2(arg1, arg2);
    // splicer end function.get_values2
}
// end POI_get_values2

// ----------------------------------------
// Function:  void iota_dimension
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int nvar
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  int * values +dimension(nvar)+intent(out)
// Statement: c_out_native_*
// start POI_iota_dimension
void POI_iota_dimension(int nvar, int * values)
{
    // splicer begin function.iota_dimension
    iota_dimension(nvar, values);
    // splicer end function.iota_dimension
}
// end POI_iota_dimension

// ----------------------------------------
// Function:  void Sum
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int len +implied(size(values))
// Statement: c_in_native_scalar
// ----------------------------------------
// Argument:  const int * values +rank(1)
// Statement: c_in_native_*
// ----------------------------------------
// Argument:  int * result +intent(out)
// Statement: c_out_native_*
// start POI_Sum
void POI_Sum(int len, const int * values, int * result)
{
    // splicer begin function.Sum
    Sum(len, values, result);
    // splicer end function.Sum
}
// end POI_Sum

/**
 * Return three values into memory the user provides.
 */
// ----------------------------------------
// Function:  void fillIntArray
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * out +dimension(3)+intent(out)
// Statement: c_out_native_*
// start POI_fillIntArray
void POI_fillIntArray(int * out)
{
    // splicer begin function.fillIntArray
    fillIntArray(out);
    // splicer end function.fillIntArray
}
// end POI_fillIntArray

/**
 * Increment array in place using intent(INOUT).
 */
// ----------------------------------------
// Function:  void incrementIntArray
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * array +intent(inout)+rank(1)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int sizein +implied(size(array))
// Statement: c_in_native_scalar
// start POI_incrementIntArray
void POI_incrementIntArray(int * array, int sizein)
{
    // splicer begin function.incrementIntArray
    incrementIntArray(array, sizein);
    // splicer end function.incrementIntArray
}
// end POI_incrementIntArray

// ----------------------------------------
// Function:  void fill_with_zeros
// Statement: c_subroutine
// ----------------------------------------
// Argument:  double * x +rank(1)
// Statement: c_inout_native_*
// ----------------------------------------
// Argument:  int x_length +implied(size(x))
// Statement: c_in_native_scalar
// start POI_fill_with_zeros
void POI_fill_with_zeros(double * x, int x_length)
{
    // splicer begin function.fill_with_zeros
    fill_with_zeros(x, x_length);
    // splicer end function.fill_with_zeros
}
// end POI_fill_with_zeros

// ----------------------------------------
// Function:  int accumulate
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  const int * arr +rank(1)
// Statement: c_in_native_*
// ----------------------------------------
// Argument:  size_t len +implied(size(arr))
// Statement: c_in_native_scalar
// start POI_accumulate
int POI_accumulate(const int * arr, size_t len)
{
    // splicer begin function.accumulate
    int SHC_rv = accumulate(arr, len);
    return SHC_rv;
    // splicer end function.accumulate
}
// end POI_accumulate

/**
 * Return strlen of the first index as a check.
 */
// ----------------------------------------
// Function:  int acceptCharArrayIn
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  char * * names +intent(in)
// Statement: c_in_char_**
// start POI_acceptCharArrayIn
int POI_acceptCharArrayIn(char **names)
{
    // splicer begin function.acceptCharArrayIn
    int SHC_rv = acceptCharArrayIn(names);
    return SHC_rv;
    // splicer end function.acceptCharArrayIn
}
// end POI_acceptCharArrayIn

// ----------------------------------------
// Function:  void setGlobalInt
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int value
// Statement: c_in_native_scalar
// start POI_setGlobalInt
void POI_setGlobalInt(int value)
{
    // splicer begin function.setGlobalInt
    setGlobalInt(value);
    // splicer end function.setGlobalInt
}
// end POI_setGlobalInt

/**
 * Used to test values global_array.
 */
// ----------------------------------------
// Function:  int sumFixedArray
// Statement: c_function_native_scalar
// start POI_sumFixedArray
int POI_sumFixedArray(void)
{
    // splicer begin function.sumFixedArray
    int SHC_rv = sumFixedArray();
    return SHC_rv;
    // splicer end function.sumFixedArray
}
// end POI_sumFixedArray

// ----------------------------------------
// Function:  void getPtrToScalar
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * * nitems +intent(out)
// Statement: c_out_native_**
// start POI_getPtrToScalar
void POI_getPtrToScalar(int * * nitems)
{
    // splicer begin function.getPtrToScalar
    getPtrToScalar(nitems);
    // splicer end function.getPtrToScalar
}
// end POI_getPtrToScalar

/**
 * Return a Fortran pointer to an array which is always the same length.
 */
// ----------------------------------------
// Function:  void getPtrToFixedArray
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * * count +dimension(10)+intent(out)
// Statement: c_out_native_**
// start POI_getPtrToFixedArray
void POI_getPtrToFixedArray(int * * count)
{
    // splicer begin function.getPtrToFixedArray
    getPtrToFixedArray(count);
    // splicer end function.getPtrToFixedArray
}
// end POI_getPtrToFixedArray

/**
 * Return a Fortran pointer to an array which is the length of
 * the argument ncount.
 */
// ----------------------------------------
// Function:  void getPtrToDynamicArray
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * * count +dimension(ncount)+intent(out)
// Statement: c_out_native_**
// ----------------------------------------
// Argument:  int * ncount +hidden+intent(out)
// Statement: c_out_native_*
// start POI_getPtrToDynamicArray
void POI_getPtrToDynamicArray(int * * count, int * ncount)
{
    // splicer begin function.getPtrToDynamicArray
    getPtrToDynamicArray(count, ncount);
    // splicer end function.getPtrToDynamicArray
}
// end POI_getPtrToDynamicArray

/**
 * Return a Fortran pointer to an array which is the length
 * is computed by C++ function getLen.
 * getLen will be called from C/C++ to compute the shape.
 */
// ----------------------------------------
// Function:  void getPtrToFuncArray
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * * count +dimension(getLen())+intent(out)
// Statement: c_out_native_**
// start POI_getPtrToFuncArray
void POI_getPtrToFuncArray(int * * count)
{
    // splicer begin function.getPtrToFuncArray
    getPtrToFuncArray(count);
    // splicer end function.getPtrToFuncArray
}
// end POI_getPtrToFuncArray

// ----------------------------------------
// Function:  void getPtrToConstScalar
// Statement: c_subroutine
// ----------------------------------------
// Argument:  const int * * nitems +intent(out)
// Statement: c_out_native_**
// start POI_getPtrToConstScalar
void POI_getPtrToConstScalar(const int * * nitems)
{
    // splicer begin function.getPtrToConstScalar
    getPtrToConstScalar(nitems);
    // splicer end function.getPtrToConstScalar
}
// end POI_getPtrToConstScalar

// ----------------------------------------
// Function:  void getPtrToFixedConstArray
// Statement: c_subroutine
// ----------------------------------------
// Argument:  const int * * count +dimension(10)+intent(out)
// Statement: c_out_native_**
// start POI_getPtrToFixedConstArray
void POI_getPtrToFixedConstArray(const int * * count)
{
    // splicer begin function.getPtrToFixedConstArray
    getPtrToFixedConstArray(count);
    // splicer end function.getPtrToFixedConstArray
}
// end POI_getPtrToFixedConstArray

// ----------------------------------------
// Function:  void getPtrToDynamicConstArray
// Statement: c_subroutine
// ----------------------------------------
// Argument:  const int * * count +dimension(ncount)+intent(out)
// Statement: c_out_native_**
// ----------------------------------------
// Argument:  int * ncount +hidden+intent(out)
// Statement: c_out_native_*
// start POI_getPtrToDynamicConstArray
void POI_getPtrToDynamicConstArray(const int * * count, int * ncount)
{
    // splicer begin function.getPtrToDynamicConstArray
    getPtrToDynamicConstArray(count, ncount);
    // splicer end function.getPtrToDynamicConstArray
}
// end POI_getPtrToDynamicConstArray

/**
 * Called directly via an interface in Fortran.
 */
// ----------------------------------------
// Function:  void getRawPtrToScalar
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * * nitems +deref(raw)+intent(out)
// Statement: c_out_native_**
// start POI_getRawPtrToScalar
void POI_getRawPtrToScalar(int * * nitems)
{
    // splicer begin function.getRawPtrToScalar
    getRawPtrToScalar(nitems);
    // splicer end function.getRawPtrToScalar
}
// end POI_getRawPtrToScalar

/**
 * Create a Fortran wrapper.
 */
// ----------------------------------------
// Function:  void getRawPtrToScalarForce
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * * nitems +deref(raw)+intent(out)
// Statement: c_out_native_**
// start POI_getRawPtrToScalarForce
void POI_getRawPtrToScalarForce(int * * nitems)
{
    // splicer begin function.getRawPtrToScalarForce
    getRawPtrToScalarForce(nitems);
    // splicer end function.getRawPtrToScalarForce
}
// end POI_getRawPtrToScalarForce

/**
 * Return a type(C_PTR) to an array which is always the same length.
 * Called directly via an interface in Fortran.
 * # Uses +deref(raw) instead of +dimension(10) like getPtrToFixedArray.
 */
// ----------------------------------------
// Function:  void getRawPtrToFixedArray
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * * count +deref(raw)+intent(out)
// Statement: c_out_native_**
// start POI_getRawPtrToFixedArray
void POI_getRawPtrToFixedArray(int * * count)
{
    // splicer begin function.getRawPtrToFixedArray
    getRawPtrToFixedArray(count);
    // splicer end function.getRawPtrToFixedArray
}
// end POI_getRawPtrToFixedArray

/**
 * Return a type(C_PTR) to an array which is always the same length.
 * Create a Fortran wrapper.
 */
// ----------------------------------------
// Function:  void getRawPtrToFixedArrayForce
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * * count +deref(raw)+intent(out)
// Statement: c_out_native_**
// start POI_getRawPtrToFixedArrayForce
void POI_getRawPtrToFixedArrayForce(int * * count)
{
    // splicer begin function.getRawPtrToFixedArrayForce
    getRawPtrToFixedArrayForce(count);
    // splicer end function.getRawPtrToFixedArrayForce
}
// end POI_getRawPtrToFixedArrayForce

/**
 * Test multiple layers of indirection.
 */
// ----------------------------------------
// Function:  void getRawPtrToInt2d
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * * * arg +intent(out)
// Statement: c_out_native_***
// start POI_getRawPtrToInt2d
void POI_getRawPtrToInt2d(int * * * arg)
{
    // splicer begin function.getRawPtrToInt2d
    getRawPtrToInt2d(arg);
    // splicer end function.getRawPtrToInt2d
}
// end POI_getRawPtrToInt2d

/**
 * Check results of getRawPtrToInt2d.
 */
// ----------------------------------------
// Function:  int checkInt2d
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  int * * arg +intent(in)
// Statement: c_in_native_**
// start POI_checkInt2d
int POI_checkInt2d(int **arg)
{
    // splicer begin function.checkInt2d
    int SHC_rv = checkInt2d(arg);
    return SHC_rv;
    // splicer end function.checkInt2d
}
// end POI_checkInt2d

/**
 * Test +dimension(10,20) +intent(in) together.
 * This will not use assumed-shape in the Fortran wrapper.
 */
// ----------------------------------------
// Function:  void DimensionIn
// Statement: c_subroutine
// ----------------------------------------
// Argument:  const int * arg +dimension(10,20)
// Statement: c_in_native_*
// start POI_DimensionIn
void POI_DimensionIn(const int * arg)
{
    // splicer begin function.DimensionIn
    DimensionIn(arg);
    // splicer end function.DimensionIn
}
// end POI_DimensionIn

/**
 * Return a Fortran pointer to an array which is always the same length.
 */
// ----------------------------------------
// Function:  void getAllocToFixedArray
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int * * count +deref(allocatable)+dimension(10)+intent(out)
// Statement: c_out_native_**
// start POI_getAllocToFixedArray
void POI_getAllocToFixedArray(int * * count)
{
    // splicer begin function.getAllocToFixedArray
    getAllocToFixedArray(count);
    // splicer end function.getAllocToFixedArray
}
// end POI_getAllocToFixedArray

// ----------------------------------------
// Function:  void * returnAddress1
// Statement: c_function_void_*
// ----------------------------------------
// Argument:  int flag
// Statement: c_in_native_scalar
// start POI_returnAddress1
void * POI_returnAddress1(int flag)
{
    // splicer begin function.returnAddress1
    void * SHC_rv = returnAddress1(flag);
    return SHC_rv;
    // splicer end function.returnAddress1
}
// end POI_returnAddress1

// ----------------------------------------
// Function:  void * returnAddress2
// Statement: c_function_void_*
// ----------------------------------------
// Argument:  int flag
// Statement: c_in_native_scalar
// start POI_returnAddress2
void * POI_returnAddress2(int flag)
{
    // splicer begin function.returnAddress2
    void * SHC_rv = returnAddress2(flag);
    return SHC_rv;
    // splicer end function.returnAddress2
}
// end POI_returnAddress2

// ----------------------------------------
// Function:  void fetchVoidPtr
// Statement: c_subroutine
// ----------------------------------------
// Argument:  void * * addr +intent(out)
// Statement: c_out_void_**
// start POI_fetchVoidPtr
void POI_fetchVoidPtr(void **addr)
{
    // splicer begin function.fetchVoidPtr
    fetchVoidPtr(addr);
    // splicer end function.fetchVoidPtr
}
// end POI_fetchVoidPtr

// ----------------------------------------
// Function:  void updateVoidPtr
// Statement: c_subroutine
// ----------------------------------------
// Argument:  void * * addr +intent(inout)
// Statement: c_inout_void_**
// start POI_updateVoidPtr
void POI_updateVoidPtr(void **addr)
{
    // splicer begin function.updateVoidPtr
    updateVoidPtr(addr);
    // splicer end function.updateVoidPtr
}
// end POI_updateVoidPtr

// ----------------------------------------
// Function:  int VoidPtrArray
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  void * * addr +rank(1)
// Statement: c_in_void_**
// start POI_VoidPtrArray
int POI_VoidPtrArray(void **addr)
{
    // splicer begin function.VoidPtrArray
    int SHC_rv = VoidPtrArray(addr);
    return SHC_rv;
    // splicer end function.VoidPtrArray
}
// end POI_VoidPtrArray

// ----------------------------------------
// Function:  int * returnIntPtrToScalar
// Statement: c_function_native_*
// start POI_returnIntPtrToScalar
int * POI_returnIntPtrToScalar(void)
{
    // splicer begin function.returnIntPtrToScalar
    int * SHC_rv = returnIntPtrToScalar();
    return SHC_rv;
    // splicer end function.returnIntPtrToScalar
}
// end POI_returnIntPtrToScalar

// ----------------------------------------
// Function:  int * returnIntPtrToFixedArray +dimension(10)
// Statement: c_function_native_*
// start POI_returnIntPtrToFixedArray
int * POI_returnIntPtrToFixedArray(void)
{
    // splicer begin function.returnIntPtrToFixedArray
    int * SHC_rv = returnIntPtrToFixedArray();
    return SHC_rv;
    // splicer end function.returnIntPtrToFixedArray
}
// end POI_returnIntPtrToFixedArray

// ----------------------------------------
// Function:  const int * returnIntPtrToConstScalar
// Statement: c_function_native_*
// start POI_returnIntPtrToConstScalar
const int * POI_returnIntPtrToConstScalar(void)
{
    // splicer begin function.returnIntPtrToConstScalar
    const int * SHC_rv = returnIntPtrToConstScalar();
    return SHC_rv;
    // splicer end function.returnIntPtrToConstScalar
}
// end POI_returnIntPtrToConstScalar

// ----------------------------------------
// Function:  const int * returnIntPtrToFixedConstArray +dimension(10)
// Statement: c_function_native_*
// start POI_returnIntPtrToFixedConstArray
const int * POI_returnIntPtrToFixedConstArray(void)
{
    // splicer begin function.returnIntPtrToFixedConstArray
    const int * SHC_rv = returnIntPtrToFixedConstArray();
    return SHC_rv;
    // splicer end function.returnIntPtrToFixedConstArray
}
// end POI_returnIntPtrToFixedConstArray

// ----------------------------------------
// Function:  int * returnIntScalar +deref(scalar)
// Statement: c_function_native_*
// start POI_returnIntScalar
int * POI_returnIntScalar(void)
{
    // splicer begin function.returnIntScalar
    int * SHC_rv = returnIntScalar();
    return SHC_rv;
    // splicer end function.returnIntScalar
}
// end POI_returnIntScalar

/**
 * Call directly via interface.
 */
// ----------------------------------------
// Function:  int * returnIntRaw +deref(raw)
// Statement: c_function_native_*
// start POI_returnIntRaw
int * POI_returnIntRaw(void)
{
    // splicer begin function.returnIntRaw
    int * SHC_rv = returnIntRaw();
    return SHC_rv;
    // splicer end function.returnIntRaw
}
// end POI_returnIntRaw

/**
 * Like returnIntRaw but with another argument to force a wrapper.
 * Uses fc_statements f_function_native_*_raw.
 */
// ----------------------------------------
// Function:  int * returnIntRawWithArgs +deref(raw)
// Statement: c_function_native_*
// ----------------------------------------
// Argument:  const char * name
// Statement: c_in_char_*
// start POI_returnIntRawWithArgs
int * POI_returnIntRawWithArgs(const char * name)
{
    // splicer begin function.returnIntRawWithArgs
    int * SHC_rv = returnIntRawWithArgs(name);
    return SHC_rv;
    // splicer end function.returnIntRawWithArgs
}
// end POI_returnIntRawWithArgs

/**
 * Test multiple layers of indirection.
 * # getRawPtrToInt2d
 */
// ----------------------------------------
// Function:  int * * returnRawPtrToInt2d
// Statement: c_function_native_**
// start POI_returnRawPtrToInt2d
int * * POI_returnRawPtrToInt2d(void)
{
    // splicer begin function.returnRawPtrToInt2d
    int * * SHC_rv = returnRawPtrToInt2d();
    return SHC_rv;
    // splicer end function.returnRawPtrToInt2d
}
// end POI_returnRawPtrToInt2d

// ----------------------------------------
// Function:  int * returnIntAllocToFixedArray +deref(allocatable)+dimension(10)
// Statement: c_function_native_*
// start POI_returnIntAllocToFixedArray
int * POI_returnIntAllocToFixedArray(void)
{
    // splicer begin function.returnIntAllocToFixedArray
    int * SHC_rv = returnIntAllocToFixedArray();
    return SHC_rv;
    // splicer end function.returnIntAllocToFixedArray
}
// end POI_returnIntAllocToFixedArray

}  // extern "C"
