
---------- ShroudTypeDefines ----------
{}

##### start ShroudTypeDefines derived_type

! helper ShroudTypeDefines
! Shroud type defines from helper ShroudTypeDefines
integer, parameter, private :: &
    SH_TYPE_SIGNED_CHAR= 1, &
    SH_TYPE_SHORT      = 2, &
    SH_TYPE_INT        = 3, &
    SH_TYPE_LONG       = 4, &
    SH_TYPE_LONG_LONG  = 5, &
    SH_TYPE_SIZE_T     = 6, &
    SH_TYPE_UNSIGNED_SHORT      = SH_TYPE_SHORT + 100, &
    SH_TYPE_UNSIGNED_INT        = SH_TYPE_INT + 100, &
    SH_TYPE_UNSIGNED_LONG       = SH_TYPE_LONG + 100, &
    SH_TYPE_UNSIGNED_LONG_LONG  = SH_TYPE_LONG_LONG + 100, &
    SH_TYPE_INT8_T    =  7, &
    SH_TYPE_INT16_T   =  8, &
    SH_TYPE_INT32_T   =  9, &
    SH_TYPE_INT64_T   = 10, &
    SH_TYPE_UINT8_T  =  SH_TYPE_INT8_T + 100, &
    SH_TYPE_UINT16_T =  SH_TYPE_INT16_T + 100, &
    SH_TYPE_UINT32_T =  SH_TYPE_INT32_T + 100, &
    SH_TYPE_UINT64_T =  SH_TYPE_INT64_T + 100, &
    SH_TYPE_FLOAT       = 22, &
    SH_TYPE_DOUBLE      = 23, &
    SH_TYPE_LONG_DOUBLE = 24, &
    SH_TYPE_FLOAT_COMPLEX      = 25, &
    SH_TYPE_DOUBLE_COMPLEX     = 26, &
    SH_TYPE_LONG_DOUBLE_COMPLEX= 27, &
    SH_TYPE_BOOL      = 28, &
    SH_TYPE_CHAR      = 29, &
    SH_TYPE_CPTR      = 30, &
    SH_TYPE_STRUCT    = 31, &
    SH_TYPE_OTHER     = 32
##### end ShroudTypeDefines derived_type

---------- array_context ----------
{
    "dependent_helpers": [
        "capsule_data_helper"
    ],
    "modules": {
        "iso_c_binding": [
            "C_NULL_PTR",
            "C_PTR",
            "C_SIZE_T",
            "C_INT",
            "C_LONG"
        ]
    }
}

##### start array_context derived_type

! helper array_context
type, bind(C) :: LIB_SHROUD_array
    ! address of C++ memory
    type(LIB_SHROUD_capsule_data) :: cxx
    ! address of data in cxx
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
    "dependent_helpers": [
        "array_context"
    ],
    "name": "LIB_SHROUD_array_string_allocatable"
}

##### start array_string_allocatable interface

interface
    ! helper array_string_allocatable
    ! Copy the char* or std::string in context into c_var.
    subroutine LIB_SHROUD_array_string_allocatable(out, in) &
         bind(c,name="LIB_ShroudArrayStringAllocatable")
        import LIB_SHROUD_array, LIB_SHROUD_capsule_data
        type(LIB_SHROUD_array), intent(IN) :: out
        type(LIB_SHROUD_array), intent(IN) :: in
    end subroutine LIB_SHROUD_array_string_allocatable
end interface
##### end array_string_allocatable interface

---------- capsule_data_helper ----------
{
    "modules": {
        "iso_c_binding": [
            "C_PTR",
            "C_INT",
            "C_NULL_PTR"
        ]
    }
}

##### start capsule_data_helper derived_type

! helper capsule_data_helper
type, bind(C) :: LIB_SHROUD_capsule_data
    type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
    integer(C_INT) :: idtor = 0       ! index of destructor
end type LIB_SHROUD_capsule_data
##### end capsule_data_helper derived_type

---------- capsule_dtor ----------
{
    "dependent_helpers": [
        "capsule_data_helper"
    ],
    "name": "LIB_SHROUD_capsule_dtor"
}

##### start capsule_dtor interface

interface
    ! helper capsule_dtor
    ! Delete memory in a capsule.
    subroutine LIB_SHROUD_capsule_dtor(ptr)&
        bind(C, name="LIB_SHROUD_memory_destructor")
        import LIB_SHROUD_capsule_data
        implicit none
        type(LIB_SHROUD_capsule_data), intent(INOUT) :: ptr
    end subroutine LIB_SHROUD_capsule_dtor
end interface
##### end capsule_dtor interface

---------- capsule_helper ----------
{
    "dependent_helpers": [
        "capsule_data_helper",
        "capsule_dtor"
    ]
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

##### start capsule_helper source

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
##### end capsule_helper source

---------- copy_array ----------
{
    "dependent_helpers": [
        "array_context"
    ],
    "name": "LIB_SHROUD_copy_array"
}

##### start copy_array interface

interface
    ! helper copy_array
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_PTR, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        type(C_PTR), intent(IN), value :: c_var
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array
end interface
##### end copy_array interface

---------- copy_string ----------
{
    "dependent_helpers": [
        "array_context"
    ],
    "name": "LIB_SHROUD_copy_string_and_free"
}

##### start copy_string interface

interface
    ! helper copy_string
    ! Copy the char* or std::string in context into c_var.
    subroutine LIB_SHROUD_copy_string_and_free(context, c_var, c_var_size) &
         bind(c,name="LIB_ShroudCopyStringAndFree")
        use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        character(kind=C_CHAR), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_string_and_free
end interface
##### end copy_string interface

---------- pointer_string ----------
{
    "dependent_helpers": [
        "array_context"
    ],
    "name": "LIB_SHROUD_pointer_string"
}

##### start pointer_string source

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
##### end pointer_string source

---------- vector_string_allocatable ----------
{
    "dependent_helpers": [
        "array_context"
    ],
    "name": "LIB_SHROUD_vector_string_allocatable"
}

##### start vector_string_allocatable interface

interface
    ! helper vector_string_allocatable
    ! Copy the char* or std::string in context into c_var.
    subroutine LIB_SHROUD_vector_string_allocatable(out, in) &
         bind(c,name="LIB_ShroudVectorStringAllocatable")
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: out
        type(LIB_SHROUD_array), intent(IN) :: in
    end subroutine LIB_SHROUD_vector_string_allocatable
end interface
##### end vector_string_allocatable interface
