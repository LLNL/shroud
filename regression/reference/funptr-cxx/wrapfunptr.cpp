// wrapfunptr.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#include "funptr.h"
#include "wrapfunptr.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

/**
 * \brief Create abstract interface for function
 *
 */
void FUN_callback1(void ( * incr)(void))
{
    // splicer begin function.callback1
    callback1(incr);
    // splicer end function.callback1
}

/**
 * \brief Create abstract interface for function
 *
 * Create a Fortran wrapper to call the bind(C) interface.
 */
void FUN_callback1_wrap(void ( * incr)(void))
{
    // splicer begin function.callback1_wrap
    callback1_wrap(incr);
    // splicer end function.callback1_wrap
}

/**
 * \brief Declare callback as external
 *
 */
void FUN_callback1_external(void ( * incr)(void))
{
    // splicer begin function.callback1_external
    callback1_external(incr);
    // splicer end function.callback1_external
}

/**
 * \brief Declare callback as c_funptr
 *
 * The caller is responsible for using c_funloc to pass the function address.
 */
// start FUN_callback1_funptr
void FUN_callback1_funptr(void ( * incr)(void))
{
    // splicer begin function.callback1_funptr
    callback1_funptr(incr);
    // splicer end function.callback1_funptr
}
// end FUN_callback1_funptr

/**
 * \brief Create abstract interface for function
 *
 */
void FUN_callback2(const char * name, int ival, FUN_incrtype incr)
{
    // splicer begin function.callback2
    callback2(name, ival, incr);
    // splicer end function.callback2
}

/**
 * \brief Declare callback as external
 *
 */
void FUN_callback2_external(const char * name, int ival,
    FUN_incrtype incr)
{
    // splicer begin function.callback2_external
    callback2_external(name, ival, incr);
    // splicer end function.callback2_external
}

/**
 * \brief Declare callback as c_funptr
 *
 * The caller is responsible for using c_funloc to pass the function address.
 * Allows any function to be passed as an argument.
 */
void FUN_callback2_funptr(const char * name, int ival,
    FUN_incrtype incr)
{
    // splicer begin function.callback2_funptr
    callback2_funptr(name, ival, incr);
    // splicer end function.callback2_funptr
}

/**
 * \brief Test function pointer with assumedtype
 *
 */
void FUN_callback3(int type, void * in, void ( * incr)(void))
{
    // splicer begin function.callback3
    callback3(type, in, incr);
    // splicer end function.callback3
}

}  // extern "C"
