// wrappointers.cpp
// This is generated code, do not edit
// #######################################################################
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
#include "wrappointers.h"
#include <stdlib.h>
#include "pointers.hpp"
#include "typespointers.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// void intargs(const int argin +intent(in)+value, int * arginout +intent(inout), int * argout +intent(out))
void POI_intargs(const int argin, int * arginout, int * argout)
{
// splicer begin function.intargs
    intargs(argin, arginout, argout);
    return;
// splicer end function.intargs
}

// void cos_doubles(double * in +dimension(:)+intent(in), double * out +allocatable(mold=in)+intent(out), int sizein +implied(size(in))+intent(in)+value)
/**
 * \brief compute cos of IN and save in OUT
 *
 * allocate OUT same type as IN implied size of array
 */
void POI_cos_doubles(double * in, double * out, int sizein)
{
// splicer begin function.cos_doubles
    cos_doubles(in, out, sizein);
    return;
// splicer end function.cos_doubles
}

// void truncate_to_int(double * in +dimension(:)+intent(in), int * out +allocatable(mold=in)+intent(out), int sizein +implied(size(in))+intent(in)+value)
/**
 * \brief truncate IN argument and save in OUT
 *
 * allocate OUT different type as IN
 * implied size of array
 */
void POI_truncate_to_int(double * in, int * out, int sizein)
{
// splicer begin function.truncate_to_int
    truncate_to_int(in, out, sizein);
    return;
// splicer end function.truncate_to_int
}

// void get_values(int * nvalues +intent(out), int * values +dimension(3)+intent(out))
/**
 * \brief fill values into array
 *
 * The function knows how long the array must be.
 * Fortran will treat the dimension as assumed-length.
 * The Python wrapper will create a NumPy array or list so it must
 * have an explicit dimension (not assumed-length).
 */
void POI_get_values(int * nvalues, int * values)
{
// splicer begin function.get_values
    get_values(nvalues, values);
    return;
// splicer end function.get_values
}

// void get_values2(int * arg1 +dimension(3)+intent(out), int * arg2 +dimension(3)+intent(out))
/**
 * \brief fill values into two arrays
 *
 * Test two intent(out) arguments.
 * Make sure error handling works with C++.
 */
void POI_get_values2(int * arg1, int * arg2)
{
// splicer begin function.get_values2
    get_values2(arg1, arg2);
    return;
// splicer end function.get_values2
}

// void Sum(int len +implied(size(values))+intent(in)+value, int * values +dimension(:)+intent(in), int * result +intent(out))
// start POI_sum
void POI_sum(int len, int * values, int * result)
{
// splicer begin function.sum
    Sum(len, values, result);
    return;
// splicer end function.sum
}
// end POI_sum

// void fillIntArray(int * out +dimension(3)+intent(out))
/**
 * Return three values into memory the user provides.
 */
void POI_fill_int_array(int * out)
{
// splicer begin function.fill_int_array
    fillIntArray(out);
    return;
// splicer end function.fill_int_array
}

// void incrementIntArray(int * array +dimension(:)+intent(inout), int sizein +implied(size(array))+intent(in)+value)
/**
 * Increment array in place using intent(INOUT).
 */
void POI_increment_int_array(int * array, int sizein)
{
// splicer begin function.increment_int_array
    incrementIntArray(array, sizein);
    return;
// splicer end function.increment_int_array
}

// Release C++ allocated memory.
void POI_SHROUD_memory_destructor(POI_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
