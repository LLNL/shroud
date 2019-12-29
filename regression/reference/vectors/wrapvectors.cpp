// wrapvectors.cpp
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
#include "wrapvectors.h"
#include <cstdlib>
#include <cstring>
#include "typesvectors.h"
#include "vectors.hpp"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper function
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
int ShroudLenTrim(const char *src, int nsrc) {
    int i;

    for (i = nsrc - 1; i >= 0; i--) {
        if (src[i] != ' ') {
            break;
        }
    }

    return i + 1;
}


// helper function
// start helper copy_array
// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
// Called from Fortran.
void VEC_ShroudCopyArray(VEC_SHROUD_array *data, void *c_var, 
    size_t c_var_size)
{
    const void *cxx_var = data->addr.base;
    int n = c_var_size < data->size ? c_var_size : data->size;
    n *= data->elem_len;
    std::memcpy(c_var, cxx_var, n);
    VEC_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}
// end helper copy_array
// splicer begin C_definitions
// splicer end C_definitions

// int vector_sum(const std::vector<int> & arg +dimension(:)+intent(in)+size(Sarg))
// start VEC_vector_sum_bufferify
int VEC_vector_sum_bufferify(const int * arg, long Sarg)
{
    // splicer begin function.vector_sum_bufferify
    const std::vector<int> SHCXX_arg(arg, arg + Sarg);
    int SHC_rv = vector_sum(SHCXX_arg);
    return SHC_rv;
    // splicer end function.vector_sum_bufferify
}
// end VEC_vector_sum_bufferify

// void vector_iota_out(std::vector<int> & arg +context(Darg)+dimension(:)+intent(out))
/**
 * \brief Copy vector into Fortran input array
 *
 */
// start VEC_vector_iota_out_bufferify
void VEC_vector_iota_out_bufferify(VEC_SHROUD_array *Darg)
{
    // splicer begin function.vector_iota_out_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>;
    vector_iota_out(*SHCXX_arg);
    Darg->cxx.addr  = static_cast<void *>(SHCXX_arg);
    Darg->cxx.idtor = 1;
    Darg->addr.base = SHCXX_arg->empty() ? NULL : &SHCXX_arg->front();
    Darg->elem_len = sizeof(int);
    Darg->size = SHCXX_arg->size();
    return;
    // splicer end function.vector_iota_out_bufferify
}
// end VEC_vector_iota_out_bufferify

// void vector_iota_out_alloc(std::vector<int> & arg +context(Darg)+deref(allocatable)+dimension(:)+intent(out))
/**
 * \brief Copy vector into Fortran allocatable array
 *
 */
// start VEC_vector_iota_out_alloc_bufferify
void VEC_vector_iota_out_alloc_bufferify(VEC_SHROUD_array *Darg)
{
    // splicer begin function.vector_iota_out_alloc_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>;
    vector_iota_out_alloc(*SHCXX_arg);
    Darg->cxx.addr  = static_cast<void *>(SHCXX_arg);
    Darg->cxx.idtor = 1;
    Darg->addr.base = SHCXX_arg->empty() ? NULL : &SHCXX_arg->front();
    Darg->elem_len = sizeof(int);
    Darg->size = SHCXX_arg->size();
    return;
    // splicer end function.vector_iota_out_alloc_bufferify
}
// end VEC_vector_iota_out_alloc_bufferify

// void vector_iota_inout_alloc(std::vector<int> & arg +context(Darg)+deref(allocatable)+dimension(:)+intent(inout)+size(Sarg))
/**
 * \brief Copy vector into Fortran allocatable array
 *
 */
// start VEC_vector_iota_inout_alloc_bufferify
void VEC_vector_iota_inout_alloc_bufferify(int * arg, long Sarg,
    VEC_SHROUD_array *Darg)
{
    // splicer begin function.vector_iota_inout_alloc_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>(arg, arg + Sarg);
    vector_iota_inout_alloc(*SHCXX_arg);
    Darg->cxx.addr  = static_cast<void *>(SHCXX_arg);
    Darg->cxx.idtor = 1;
    Darg->addr.base = SHCXX_arg->empty() ? NULL : &SHCXX_arg->front();
    Darg->elem_len = sizeof(int);
    Darg->size = SHCXX_arg->size();
    return;
    // splicer end function.vector_iota_inout_alloc_bufferify
}
// end VEC_vector_iota_inout_alloc_bufferify

// void vector_increment(std::vector<int> & arg +context(Darg)+dimension(:)+intent(inout)+size(Sarg))
void VEC_vector_increment_bufferify(int * arg, long Sarg,
    VEC_SHROUD_array *Darg)
{
    // splicer begin function.vector_increment_bufferify
    std::vector<int> *SHCXX_arg = new std::vector<int>(arg, arg + Sarg);
    vector_increment(*SHCXX_arg);
    Darg->cxx.addr  = static_cast<void *>(SHCXX_arg);
    Darg->cxx.idtor = 1;
    Darg->addr.base = SHCXX_arg->empty() ? NULL : &SHCXX_arg->front();
    Darg->elem_len = sizeof(int);
    Darg->size = SHCXX_arg->size();
    return;
    // splicer end function.vector_increment_bufferify
}

// void vector_iota_out_d(std::vector<double> & arg +context(Darg)+dimension(:)+intent(out))
/**
 * \brief Copy vector into Fortran input array
 *
 */
void VEC_vector_iota_out_d_bufferify(VEC_SHROUD_array *Darg)
{
    // splicer begin function.vector_iota_out_d_bufferify
    std::vector<double> *SHCXX_arg = new std::vector<double>;
    vector_iota_out_d(*SHCXX_arg);
    Darg->cxx.addr  = static_cast<void *>(SHCXX_arg);
    Darg->cxx.idtor = 2;
    Darg->addr.base = SHCXX_arg->empty() ? NULL : &SHCXX_arg->front();
    Darg->elem_len = sizeof(double);
    Darg->size = SHCXX_arg->size();
    return;
    // splicer end function.vector_iota_out_d_bufferify
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
    std::vector<std::string> SHCXX_arg;
    {
        const char * BBB = arg;
        std::vector<std::string>::size_type
            SHT_i = 0,
            SHT_n = Sarg;
        for(; SHT_i < SHT_n; SHT_i++) {
            SHCXX_arg.push_back(std::string(BBB,ShroudLenTrim(BBB, Narg)));
            BBB += Narg;
        }
    }
    int SHC_rv = vector_string_count(SHCXX_arg);
    return SHC_rv;
    // splicer end function.vector_string_count_bufferify
}

// void ReturnVectorAlloc(int n +intent(in)+value, std::vector<int> * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out))
/**
 * Implement iota function.
 * Return a vector as an ALLOCATABLE array.
 * Copy results into the new array.
 */
void VEC_return_vector_alloc_bufferify(int n, VEC_SHROUD_array *DSHF_rv)
{
    // splicer begin function.return_vector_alloc_bufferify
    std::vector<int> *SHC_rv = new std::vector<int>;
    *SHC_rv = ReturnVectorAlloc(n);
    DSHF_rv->cxx.addr  = static_cast<void *>(SHC_rv);
    DSHF_rv->cxx.idtor = 1;
    DSHF_rv->addr.base = SHC_rv->empty() ? NULL : &SHC_rv->front();
    DSHF_rv->elem_len = sizeof(int);
    DSHF_rv->size = SHC_rv->size();
    return;
    // splicer end function.return_vector_alloc_bufferify
}

// start release allocated memory
// Release library allocated memory.
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
    case 2:   // std_vector_double
    {
        std::vector<double> *cxx_ptr = 
            reinterpret_cast<std::vector<double> *>(ptr);
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
// end release allocated memory

}  // extern "C"
