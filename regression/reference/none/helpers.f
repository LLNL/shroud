
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
