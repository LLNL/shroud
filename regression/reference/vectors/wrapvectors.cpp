// wrapvectors.cpp
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
// All rights reserved.
//
// This file is part of Shroud.  For details, see
// https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the disclaimer (as noted below)
//   in the documentation and/or other materials provided with the
//   distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
// LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// #######################################################################
#include "wrapvectors.h"
#include <cstring>
#include <stdlib.h>
#include <string>
#include "typesvectors.h"
#include "vectors.hpp"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper function
// Returns the length of character string a with length ls,
// ignoring any trailing blanks.
int ShroudLenTrim(const char *s, int ls) {
    int i;

    for (i = ls - 1; i >= 0; i--) {
        if (s[i] != ' ') {
            break;
        }
    }

    return i + 1;
}


// helper function
// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
void VEC_ShroudCopyArray(VEC_SHROUD_array *data, void *c_var, 
    size_t c_var_size)
{
    const void *cxx_var = data->addr.cvoidp;
    int n = c_var_size < data->size ? c_var_size : data->size;
    n *= data->len;
    std::memcpy(c_var, cxx_var, n);
    VEC_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}
// splicer begin C_definitions
// splicer end C_definitions

// int vector_sum(const std::vector<int> & arg +dimension(:)+intent(in)+size(Sarg))
int VEC_vector_sum_bufferify(const int * arg, long Sarg)
{
// splicer begin function.vector_sum_bufferify
    const std::vector<int> SH_arg(arg, arg + Sarg);
    int SHC_rv = vector_sum(SH_arg);
    return SHC_rv;
// splicer end function.vector_sum_bufferify
}

// void vector_iota_out(std::vector<int> & arg +context(Darg)+dimension(:)+intent(out))
/**
 * \brief Copy vector into Fortran input array
 *
 */
void VEC_vector_iota_out_bufferify(VEC_SHROUD_array *Darg)
{
// splicer begin function.vector_iota_out_bufferify
    std::vector<int> *SH_arg = new std::vector<int>;
    vector_iota_out(*SH_arg);
    Darg->cxx.addr  = static_cast<void *>(SH_arg);
    Darg->cxx.idtor = 1;
    Darg->addr.cvoidp = SH_arg->empty() ? NULL : &SH_arg->front();
    Darg->len = sizeof(int);
    Darg->size = SH_arg->size();
    return;
// splicer end function.vector_iota_out_bufferify
}

// void vector_iota_out_alloc(std::vector<int> & arg +context(Darg)+deref(allocatable)+dimension(:)+intent(out))
/**
 * \brief Copy vector into Fortran allocatable array
 *
 */
void VEC_vector_iota_out_alloc_bufferify(VEC_SHROUD_array *Darg)
{
// splicer begin function.vector_iota_out_alloc_bufferify
    std::vector<int> *SH_arg = new std::vector<int>;
    vector_iota_out_alloc(*SH_arg);
    Darg->cxx.addr  = static_cast<void *>(SH_arg);
    Darg->cxx.idtor = 1;
    Darg->addr.cvoidp = SH_arg->empty() ? NULL : &SH_arg->front();
    Darg->len = sizeof(int);
    Darg->size = SH_arg->size();
    return;
// splicer end function.vector_iota_out_alloc_bufferify
}

// void vector_iota_inout_alloc(std::vector<int> & arg +context(Darg)+deref(allocatable)+dimension(:)+intent(inout)+size(Sarg))
/**
 * \brief Copy vector into Fortran allocatable array
 *
 */
void VEC_vector_iota_inout_alloc_bufferify(int * arg, long Sarg,
    VEC_SHROUD_array *Darg)
{
// splicer begin function.vector_iota_inout_alloc_bufferify
    std::vector<int> *SH_arg = new std::vector<int>(arg, arg + Sarg);
    vector_iota_inout_alloc(*SH_arg);
    Darg->cxx.addr  = static_cast<void *>(SH_arg);
    Darg->cxx.idtor = 1;
    Darg->addr.cvoidp = SH_arg->empty() ? NULL : &SH_arg->front();
    Darg->len = sizeof(int);
    Darg->size = SH_arg->size();
    return;
// splicer end function.vector_iota_inout_alloc_bufferify
}

// void vector_increment(std::vector<int> & arg +context(Darg)+dimension(:)+intent(inout)+size(Sarg))
void VEC_vector_increment_bufferify(int * arg, long Sarg,
    VEC_SHROUD_array *Darg)
{
// splicer begin function.vector_increment_bufferify
    std::vector<int> *SH_arg = new std::vector<int>(arg, arg + Sarg);
    vector_increment(*SH_arg);
    Darg->cxx.addr  = static_cast<void *>(SH_arg);
    Darg->cxx.idtor = 1;
    Darg->addr.cvoidp = SH_arg->empty() ? NULL : &SH_arg->front();
    Darg->len = sizeof(int);
    Darg->size = SH_arg->size();
    return;
// splicer end function.vector_increment_bufferify
}

// int vector_string_count(const std::vector<std::string> & arg +dimension(:)+intent(in)+len(Narg)+size(Sarg))
/**
 * \brief count number of underscore in vector of strings
 *
 */
int VEC_vector_string_count_bufferify(const char * arg, long Sarg,
    int Narg)
{
// splicer begin function.vector_string_count_bufferify
    std::vector<std::string> SH_arg;
    {
          const char * BBB = arg;
          std::vector<std::string>::size_type
            SHT_i = 0,
            SHT_n = Sarg;
        for(; SHT_i < SHT_n; SHT_i++) {
            SH_arg.push_back(std::string(BBB,ShroudLenTrim(BBB, Narg)));
            BBB += Narg;
        }
    }
    int SHC_rv = vector_string_count(SH_arg);
    return SHC_rv;
// splicer end function.vector_string_count_bufferify
}

// Release C++ allocated memory.
void VEC_SHROUD_memory_destructor(VEC_SHROUD_capsule_data *cap)
{
    void *ptr = cap->addr;
    switch (cap->idtor) {
    case 0:   // --none--
    {
        // Nothing to delete
        break;
    }
    case 1:   // std_vector_int
    {
        std::vector<int> *cxx_ptr = 
            reinterpret_cast<std::vector<int> *>(ptr);
        delete cxx_ptr;
        break;
    }
    default:
    {
        // Unexpected case in destructor
        break;
    }
    }
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
