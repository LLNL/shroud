// wrapownership.cpp
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
#include "wrapownership.h"
#include <cstring>
#include <stdlib.h>
#include "ownership.hpp"
#include "typesownership.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
void OWN_ShroudCopyArray(OWN_SHROUD_array *data, void *c_var, 
    size_t c_var_size)
{
    const void *cxx_var = data->addr.cvoidp;
    int n = c_var_size < data->size ? c_var_size : data->size;
    n *= data->len;
    std::memcpy(c_var, cxx_var, n);
    OWN_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}
// splicer begin C_definitions
// splicer end C_definitions

// int * ReturnIntPtrRaw() +deref(raw)
// function_index=2
int * OWN_return_int_ptr_raw()
{
// splicer begin function.return_int_ptr_raw
    int * SHC_rv = ReturnIntPtrRaw();
    return SHC_rv;
// splicer end function.return_int_ptr_raw
}

// int * ReturnIntPtrScalar() +deref(scalar)
// function_index=3
int OWN_return_int_ptr_scalar()
{
// splicer begin function.return_int_ptr_scalar
    int * SHC_rv = ReturnIntPtrScalar();
    return *SHC_rv;
// splicer end function.return_int_ptr_scalar
}

// int * ReturnIntPtrPointer() +deref(pointer)
// function_index=4
int * OWN_return_int_ptr_pointer()
{
// splicer begin function.return_int_ptr_pointer
    int * SHC_rv = ReturnIntPtrPointer();
    return SHC_rv;
// splicer end function.return_int_ptr_pointer
}

// int * ReturnIntPtrDimRaw(int * len +intent(out)) +deref(raw)
// function_index=5
int * OWN_return_int_ptr_dim_raw(int * len)
{
// splicer begin function.return_int_ptr_dim_raw
    int * SHC_rv = ReturnIntPtrDimRaw(len);
    return SHC_rv;
// splicer end function.return_int_ptr_dim_raw
}

// int * ReturnIntPtrDimPointer(int * len +hidden+intent(out)) +deref(pointer)+dimension(len)
// function_index=6
int * OWN_return_int_ptr_dim_pointer(int * len)
{
// splicer begin function.return_int_ptr_dim_pointer
    int * SHC_rv = ReturnIntPtrDimPointer(len);
    return SHC_rv;
// splicer end function.return_int_ptr_dim_pointer
}

// int * ReturnIntPtrDimAlloc(int * len +hidden+intent(out)) +deref(allocatable)+dimension(len)
// function_index=7
int * OWN_return_int_ptr_dim_alloc(int * len)
{
// splicer begin function.return_int_ptr_dim_alloc
    int * SHC_rv = ReturnIntPtrDimAlloc(len);
    return SHC_rv;
// splicer end function.return_int_ptr_dim_alloc
}

// int * ReturnIntPtrDimAlloc(int * len +hidden+intent(out)) +context(DSHC_rv)+deref(allocatable)+dimension(len)
// function_index=24
int * OWN_return_int_ptr_dim_alloc_bufferify(OWN_SHROUD_array *DSHC_rv,
    int * len)
{
// splicer begin function.return_int_ptr_dim_alloc_bufferify
    int * SHC_rv = ReturnIntPtrDimAlloc(len);
    DSHC_rv->cxx.addr  = SHC_rv;
    DSHC_rv->cxx.idtor = 0;
    DSHC_rv->addr.cvoidp = SHC_rv;
    DSHC_rv->len = sizeof(int);
    DSHC_rv->size = *len;
    return SHC_rv;
// splicer end function.return_int_ptr_dim_alloc_bufferify
}

// int * ReturnIntPtrDimDefault(int * len +hidden+intent(out)) +dimension(len)
// function_index=8
int * OWN_return_int_ptr_dim_default(int * len)
{
// splicer begin function.return_int_ptr_dim_default
    int * SHC_rv = ReturnIntPtrDimDefault(len);
    return SHC_rv;
// splicer end function.return_int_ptr_dim_default
}

// int * ReturnIntPtrDimRawNew(int * len +hidden+intent(out)) +dimension(len)+owner(caller)
// function_index=9
int * OWN_return_int_ptr_dim_raw_new(int * len)
{
// splicer begin function.return_int_ptr_dim_raw_new
    int * SHC_rv = ReturnIntPtrDimRawNew(len);
    return SHC_rv;
// splicer end function.return_int_ptr_dim_raw_new
}

// int * ReturnIntPtrDimPointerNew(int * len +hidden+intent(out)) +deref(pointer)+dimension(len)+owner(caller)
// function_index=10
int * OWN_return_int_ptr_dim_pointer_new(int * len)
{
// splicer begin function.return_int_ptr_dim_pointer_new
    int * SHC_rv = ReturnIntPtrDimPointerNew(len);
    return SHC_rv;
// splicer end function.return_int_ptr_dim_pointer_new
}

// int * ReturnIntPtrDimAllocNew(int * len +hidden+intent(out)) +deref(allocatable)+dimension(len)+owner(caller)
// function_index=11
int * OWN_return_int_ptr_dim_alloc_new(int * len)
{
// splicer begin function.return_int_ptr_dim_alloc_new
    int * SHC_rv = ReturnIntPtrDimAllocNew(len);
    return SHC_rv;
// splicer end function.return_int_ptr_dim_alloc_new
}

// int * ReturnIntPtrDimDefaultNew(int * len +hidden+intent(out)) +dimension(len)+owner(caller)
// function_index=12
int * OWN_return_int_ptr_dim_default_new(int * len)
{
// splicer begin function.return_int_ptr_dim_default_new
    int * SHC_rv = ReturnIntPtrDimDefaultNew(len);
    return SHC_rv;
// splicer end function.return_int_ptr_dim_default_new
}

// void createClassStatic(int flag +intent(in)+value)
// function_index=21
void OWN_create_class_static(int flag)
{
// splicer begin function.create_class_static
    createClassStatic(flag);
    return;
// splicer end function.create_class_static
}

// Class1 * getClassStatic() +owner(library)
// function_index=22
OWN_class1 OWN_get_class_static()
{
// splicer begin function.get_class_static
    Class1 * SHCXX_rv = getClassStatic();
    OWN_class1 SHC_rv;
    SHC_rv.addr = static_cast<void *>(SHCXX_rv);
    SHC_rv.idtor = 0;
    return SHC_rv;
// splicer end function.get_class_static
}

// Class1 * getClassNew(int flag +intent(in)+value) +owner(caller)
// function_index=23
/**
 * \brief Return pointer to new Class1 instance.
 *
 */
OWN_class1 OWN_get_class_new(int flag)
{
// splicer begin function.get_class_new
    Class1 * SHCXX_rv = getClassNew(flag);
    OWN_class1 SHC_rv;
    SHC_rv.addr = static_cast<void *>(SHCXX_rv);
    SHC_rv.idtor = 0;
    return SHC_rv;
// splicer end function.get_class_new
}

// Release C++ allocated memory.
void OWN_SHROUD_memory_destructor(OWN_SHROUD_capsule_data *cap)
{
    void *ptr = cap->addr;
    switch (cap->idtor) {
    case 0:   // --none--
    {
        // Nothing to delete
        break;
    }
    case 1:   // Class1
    {
        Class1 *cxx_ptr = reinterpret_cast<Class1 *>(ptr);
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
