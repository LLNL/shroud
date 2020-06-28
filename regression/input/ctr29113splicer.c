

// splicer begin C_definitions
#include "ISO_Fortran_binding.h"


void TR2_get_const_string_ptr_alloc_tr_bufferify(CFI_cdesc_t * rv)
{
    const std::string * SHCXX_rv = getConstStringPtrAlloc();

    CFI_allocate(rv, nullptr, nullptr, SHCXX_rv->length());
    std::memcpy(rv->base_addr, SHCXX_rv->data(), rv->elem_len);
}

// Test unitialized string and zero-length allocate.
void TR2_get_const_string_ptr_alloc_tr_bufferify_zerolength(CFI_cdesc_t * rv)
{
    const std::string SHCXX_rv;

    CFI_allocate(rv, nullptr, nullptr, SHCXX_rv.length());
    std::memcpy(rv->base_addr, SHCXX_rv.data(), rv->elem_len);
}
// splicer end C_definitions
