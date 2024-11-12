
---------- array_context ----------
{
    "c_fmtname": "LIB_SHROUD_array",
    "dependent_helpers": [
        "type_defines"
    ],
    "f_fmtname": "LIB_SHROUD_array",
    "include": [
        "<stddef.h>"
    ],
    "modules": {
        "iso_c_binding": [
            "C_NULL_PTR",
            "C_PTR",
            "C_SIZE_T",
            "C_INT",
            "C_LONG"
        ]
    },
    "name": "array_context",
    "scope": "cwrap_include"
}

##### start array_context source

// helper array_context
struct s_LIB_SHROUD_array {
    void * base_addr;
    int type;        /* type of element */
    size_t elem_len; /* bytes-per-item or character len in c++ */
    size_t size;     /* size of data in c++ */
    int rank;        /* number of dimensions, 0=scalar */
    long shape[7];
};
typedef struct s_LIB_SHROUD_array LIB_SHROUD_array;
##### end array_context source

##### start array_context derived_type

! helper array_context
type, bind(C) :: LIB_SHROUD_array
    ! address of data
    type(C_PTR) :: base_addr = C_NULL_PTR
    ! type of element
    integer(C_INT) :: type
    ! bytes-per-item or character len of data in cxx
    integer(C_SIZE_T) :: elem_len = 0_C_SIZE_T
    ! size of data in cxx
    integer(C_SIZE_T) :: size = 0_C_SIZE_T
    ! number of dimensions
    integer(C_INT) :: rank = -1
    integer(C_LONG) :: shape(7) = 0
end type LIB_SHROUD_array
##### end array_context derived_type

---------- array_string_allocatable ----------
{
    "api": "c",
    "c_fmtname": "LIB_ShroudArrayStringAllocatable",
    "dependent_helpers": [
        "capsule_data_helper",
        "array_context",
        "array_string_out"
    ],
    "f_fmtname": "LIB_SHROUD_array_string_allocatable",
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudArrayStringAllocatable",
        "cnameproto": "void {cnamefunc}({C_array_type} *dest, {C_capsule_data_type} *src)",
        "fnamefunc": "{C_prefix}SHROUD_array_string_allocatable"
    },
    "name": "array_string_allocatable",
    "proto": "void LIB_ShroudArrayStringAllocatable(LIB_SHROUD_array *dest, LIB_SHROUD_capsule_data *src);",
    "scope": "cwrap_impl"
}

##### start array_string_allocatable source

// helper array_string_allocatable
// Copy the std::string array into Fortran array.
// Called by Fortran to deal with allocatable character.
// out is already blank filled.
void LIB_ShroudArrayStringAllocatable(LIB_SHROUD_array *dest, LIB_SHROUD_capsule_data *src)
{
    std::string *cxxvec = static_cast< std::string *>(src->addr);
    LIB_ShroudArrayStringOut(dest, cxxvec, dest->size);
}

##### end array_string_allocatable source

##### start array_string_allocatable interface

interface
    ! helper array_string_allocatable
    subroutine LIB_SHROUD_array_string_allocatable(dest, src) &
         bind(c,name="LIB_ShroudArrayStringAllocatable")
        import LIB_SHROUD_array, LIB_SHROUD_capsule_data
        type(LIB_SHROUD_array), intent(IN) :: dest
        type(LIB_SHROUD_capsule_data), intent(IN) :: src
    end subroutine LIB_SHROUD_array_string_allocatable
end interface
##### end array_string_allocatable interface

---------- array_string_out ----------
{
    "api": "cxx",
    "c_fmtname": "LIB_ShroudArrayStringOut",
    "cxx_include": [
        "<cstring>",
        "<cstddef>"
    ],
    "dependent_helpers": [
        "array_context"
    ],
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudArrayStringOut",
        "cnamefunc_array_string_out": "{cnamefunc}",
        "cnameproto": "void {cnamefunc}({C_array_type} *outdesc, std::string *in, size_t nsize)"
    },
    "name": "array_string_out",
    "proto": "void LIB_ShroudArrayStringOut(LIB_SHROUD_array *outdesc, std::string *in, size_t nsize);",
    "proto_include": [
        "<string>",
        "<vector>"
    ],
    "scope": "cwrap_impl"
}

##### start array_string_out source

// helper array_string_out
// Copy the std::vector<std::string> into Fortran array argument.
// Called by C++.
void LIB_ShroudArrayStringOut(LIB_SHROUD_array *outdesc, std::string *in, size_t nsize)
{
    size_t nvect = outdesc->size;
    size_t len = outdesc->elem_len;
    char *dest = static_cast<char *>(outdesc->base_addr);
    // Clear user memory
    std::memset(dest, ' ', nvect*len);

    // Copy into user memory
    nvect = std::min(nvect, nsize);
    for (size_t i = 0; i < nvect; ++i) {
        std::memcpy(dest, in[i].data(), std::min(len, in[i].length()));
        dest += outdesc->elem_len;
    }
}
##### end array_string_out source

---------- array_string_out_len ----------
{
    "api": "cxx",
    "c_fmtname": "LIB_ShroudArrayStringOutSize",
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudArrayStringOutSize",
        "cnameproto": "size_t {cnamefunc}(std::string *in, size_t nsize)"
    },
    "name": "array_string_out_len",
    "proto": "size_t LIB_ShroudArrayStringOutSize(std::string *in, size_t nsize);",
    "proto_include": [
        "<string>",
        "<vector>"
    ],
    "scope": "cwrap_impl"
}

##### start array_string_out_len source

// helper array_string_out_len
// Return the maximum string length in a std::vector<std::string>.
size_t LIB_ShroudArrayStringOutSize(std::string *in, size_t nsize)
{
    size_t len = 0;
    for (size_t i = 0; i < nsize; ++i) {
        len = std::max(len, in[i].length());
    }
    return len;
}
##### end array_string_out_len source

---------- capsule_data_helper ----------
{
    "f_fmtname": "LIB_SHROUD_capsule_data",
    "modules": {
        "iso_c_binding": [
            "C_PTR",
            "C_INT",
            "C_NULL_PTR"
        ]
    },
    "name": "capsule_data_helper",
    "scope": "cwrap_include"
}

##### start capsule_data_helper derived_type

! helper capsule_data_helper
type, bind(C) :: LIB_SHROUD_capsule_data
    type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
    integer(C_INT) :: idtor = 0       ! index of destructor
end type LIB_SHROUD_capsule_data
##### end capsule_data_helper derived_type

##### start capsule_data_helper source

// helper capsule_data_helper
struct s_LIB_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_LIB_SHROUD_capsule_data LIB_SHROUD_capsule_data;
##### end capsule_data_helper source

---------- capsule_helper ----------
{
    "dependent_helpers": [
        "capsule_data_helper",
        "capsule_dtor"
    ],
    "name": "capsule_helper"
}

##### start capsule_helper derived_type

! helper capsule_helper
type :: LIB_SHROUD_capsule
    private
    type(LIB_SHROUD_capsule_data) :: mem
contains
    final :: SHROUD_capsule_final
    procedure :: delete => SHROUD_capsule_delete
end type LIB_SHROUD_capsule
##### end capsule_helper derived_type

##### start capsule_helper f_source

! helper capsule_helper
! finalize a static LIB_SHROUD_capsule_data
subroutine SHROUD_capsule_final(cap)
    type(LIB_SHROUD_capsule), intent(INOUT) :: cap
    call LIB_SHROUD_capsule_dtor(cap%mem)
end subroutine SHROUD_capsule_final

subroutine SHROUD_capsule_delete(cap)
    class(LIB_SHROUD_capsule) :: cap
    call LIB_SHROUD_capsule_dtor(cap%mem)
end subroutine SHROUD_capsule_delete
##### end capsule_helper f_source

---------- pointer_string ----------
{
    "dependent_helpers": [
        "array_context"
    ],
    "f_fmtname": "LIB_SHROUD_pointer_string",
    "fmtdict": {
        "fnamefunc": "{C_prefix}SHROUD_pointer_string"
    },
    "name": "pointer_string"
}

##### start pointer_string f_source

! helper pointer_string
! Assign context to an assumed-length character pointer
subroutine LIB_SHROUD_pointer_string(context, var)
    use iso_c_binding, only : c_f_pointer, C_PTR
    implicit none
    type(LIB_SHROUD_array), intent(IN) :: context
    character(len=:), pointer, intent(OUT) :: var
    character(len=context%elem_len), pointer :: fptr
    call c_f_pointer(context%base_addr, fptr)
    var => fptr
end subroutine LIB_SHROUD_pointer_string
##### end pointer_string f_source

---------- string_to_cdesc ----------
{
    "c_fmtname": "ShroudStringToCdesc",
    "cxx_include": [
        "<cstring>",
        "<cstddef>"
    ],
    "dependent_helpers": [
        "array_context"
    ],
    "fmtdict": {
        "cnamefunc": "ShroudStringToCdesc"
    },
    "name": "string_to_cdesc"
}

##### start string_to_cdesc source

// helper string_to_cdesc
// Save std::string metadata into array to allow Fortran to access values.
// CHARACTER(len=elem_size) src
static void ShroudStringToCdesc(LIB_SHROUD_array *cdesc,&
    const std::string * src)
{
    if (src->empty()) {
        cdesc->base_addr = NULL;
        cdesc->elem_len = 0;
    } else {
        cdesc->base_addr = const_cast<char *>(src->data());
        cdesc->elem_len = src->length();
    }
    cdesc->size = 1;
    cdesc->rank = 0;  // scalar
}
##### end string_to_cdesc source

---------- vector_string_allocatable ----------
{
    "api": "c",
    "c_fmtname": "LIB_ShroudVectorStringAllocatable",
    "dependent_helpers": [
        "capsule_data_helper",
        "array_context",
        "vector_string_out"
    ],
    "f_fmtname": "LIB_SHROUD_vector_string_allocatable",
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudVectorStringAllocatable",
        "cnameproto": "void {cnamefunc}({C_array_type} *dest, {C_capsule_data_type} *src)",
        "fnamefunc": "{C_prefix}SHROUD_vector_string_allocatable"
    },
    "name": "vector_string_allocatable",
    "proto": "void LIB_ShroudVectorStringAllocatable(LIB_SHROUD_array *dest, LIB_SHROUD_capsule_data *src);",
    "scope": "cwrap_impl"
}

##### start vector_string_allocatable source

// helper vector_string_allocatable
// Copy the std::vector<std::string> into Fortran array.
// Called by Fortran to deal with allocatable character.
// out is already blank filled.
void LIB_ShroudVectorStringAllocatable(LIB_SHROUD_array *dest, LIB_SHROUD_capsule_data *src)
{
    std::vector<std::string> *cxxvec =&
        static_cast< std::vector<std::string> * >(src->addr);
    LIB_ShroudVectorStringOut(dest, *cxxvec);
}
##### end vector_string_allocatable source

##### start vector_string_allocatable interface

interface
    ! helper vector_string_allocatable
    ! Copy the char* or std::string in context into c_var.
    subroutine LIB_SHROUD_vector_string_allocatable(dest, src) &
         bind(c,name="LIB_ShroudVectorStringAllocatable")
        import LIB_SHROUD_capsule_data, LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: dest
        type(LIB_SHROUD_capsule_data), intent(IN) :: src
    end subroutine LIB_SHROUD_vector_string_allocatable
end interface
##### end vector_string_allocatable interface

---------- vector_string_out ----------
{
    "api": "cxx",
    "c_fmtname": "LIB_ShroudVectorStringOut",
    "cxx_include": [
        "<cstring>",
        "<cstddef>"
    ],
    "dependent_helpers": [
        "array_context"
    ],
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudVectorStringOut",
        "cnamefunc_vector_string_out": "{cnamefunc}",
        "cnameproto": "void {cnamefunc}({C_array_type} *outdesc, std::vector<std::string> &in)",
        "fnamefunc": "{C_prefix}shroud_vector_string_out"
    },
    "name": "vector_string_out",
    "proto": "void LIB_ShroudVectorStringOut(LIB_SHROUD_array *outdesc, std::vector<std::string> &in);",
    "proto_include": [
        "<string>",
        "<vector>"
    ],
    "scope": "cwrap_impl"
}

##### start vector_string_out source

// helper vector_string_out
// Copy the std::vector<std::string> into Fortran array argument.
// Called by C++.
void LIB_ShroudVectorStringOut(LIB_SHROUD_array *outdesc, std::vector<std::string> &in)
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
##### end vector_string_out source

---------- vector_string_out_len ----------
{
    "api": "cxx",
    "c_fmtname": "LIB_ShroudVectorStringOutSize",
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudVectorStringOutSize",
        "cnameproto": "size_t {cnamefunc}(std::vector<std::string> &in)"
    },
    "name": "vector_string_out_len",
    "proto": "size_t LIB_ShroudVectorStringOutSize(std::vector<std::string> &in);",
    "proto_include": [
        "<string>",
        "<vector>"
    ],
    "scope": "cwrap_impl"
}

##### start vector_string_out_len source

// helper vector_string_out_len
// Return the maximum string length in a std::vector<std::string>.
size_t LIB_ShroudVectorStringOutSize(std::vector<std::string> &in)
{
    size_t nvect = in.size();
    size_t len = 0;
    for (size_t i = 0; i < nvect; ++i) {
        len = std::max(len, in[i].length());
    }
    return len;
}
##### end vector_string_out_len source
