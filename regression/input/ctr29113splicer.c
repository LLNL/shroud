

// splicer begin C_definitions
#include "ISO_Fortran_binding.h"


void TR2_get_const_string_ptr_alloc_tr_bufferify(CFI_cdesc_t * rv)
{
    const std::string * SHCXX_rv = getConstStringPtrAlloc();

    //    CFI_index_t lower[1], upper[1];
    //    CFI_allocate(rv, lower, upper, SHCXX_rv->length());
    if (! SHCXX_rv->empty()) {
        CFI_allocate(rv, nullptr, nullptr, SHCXX_rv->length());
        std::memcpy(rv->base_addr, SHCXX_rv->data(), rv->elem_len);
    }
}
// splicer end C_definitions
