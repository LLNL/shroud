// utilvectors.cpp
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// typemap
#include <vector>
// shroud
#include "typesvectors.h"
#include <cstddef>
#include <cstring>


#ifdef __cplusplus
extern "C" {
#endif

// start helper copy_array
// helper copy_array
// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
// Called from Fortran.
void VEC_ShroudCopyArray(VEC_SHROUD_array *data, void *c_var, 
    size_t c_var_size)
{
    const void *cxx_var = data->base_addr;
    int n = c_var_size < data->size ? c_var_size : data->size;
    n *= data->elem_len;
    std::memcpy(c_var, cxx_var, n);
}
// end helper copy_array

// start helper vector_string_allocatable
// helper vector_string_allocatable
// Copy the std::vector<std::string> into Fortran array.
// Called by Fortran to deal with allocatable character.
// out is already blank filled.
void VEC_ShroudVectorStringAllocatable(VEC_SHROUD_array *dest, VEC_SHROUD_capsule_data *src)
{
    std::vector<std::string> *cxxvec =
        static_cast< std::vector<std::string> * >(src->addr);
    VEC_ShroudVectorStringOut(dest, *cxxvec);
}
// end helper vector_string_allocatable


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
    cap->addr = nullptr;
    cap->idtor = 0;  // avoid deleting again
}
// end release allocated memory

#ifdef __cplusplus
}
#endif

// start helper vector_string_out
// helper vector_string_out
// Copy the std::vector<std::string> into Fortran array argument.
// Called by C++.
void VEC_ShroudVectorStringOut(VEC_SHROUD_array *outdesc, std::vector<std::string> &in)
{
    size_t nvect = outdesc->size;
    size_t len = outdesc->elem_len;
    char *dest = static_cast<char *>(outdesc->base_addr);
    // Clear user memory
    std::memset(dest, ' ', nvect*len);

    // Copy into user memory
    nvect = std::min(nvect, in.size());
    //char *dest = static_cast<char *>(outdesc->cxx.addr);
    for (size_t i = 0; i < nvect; ++i) {
        std::memcpy(dest, in[i].data(), std::min(len, in[i].length()));
        dest += outdesc->elem_len;
    }
}
// end helper vector_string_out


// start helper vector_string_out_len
// helper vector_string_out_len
// Return the maximum string length in a std::vector<std::string>.
size_t VEC_ShroudVectorStringOutSize(std::vector<std::string> &in)
{
    size_t nvect = in.size();
    size_t len = 0;
    for (size_t i = 0; i < nvect; ++i) {
        len = std::max(len, in[i].length());
    }
    return len;
}
// end helper vector_string_out_len

