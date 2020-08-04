
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

##### start array_context derived_type

! helper array_context
type, bind(C) :: LIB_SHROUD_array
    ! address of C++ memory
    type(SHROUD_capsule_data) :: cxx
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

##### start capsule_data_helper derived_type

! helper capsule_data_helper
type, bind(C) :: SHROUD_capsule_data
    type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
    integer(C_INT) :: idtor = 0       ! index of destructor
end type SHROUD_capsule_data
##### end capsule_data_helper derived_type

##### start capsule_dtor interface

interface
    ! helper capsule_dtor
    ! Delete memory in a capsule.
    subroutine LIB_SHROUD_capsule_dtor(ptr)&
        bind(C, name="LIB_SHROUD_memory_destructor")
        import SHROUD_capsule_data
        implicit none
        type(SHROUD_capsule_data), intent(INOUT) :: ptr
    end subroutine LIB_SHROUD_capsule_dtor
end interface
##### end capsule_dtor interface

##### start capsule_helper derived_type

! helper capsule_helper
type :: LIB_SHROUD_capsule
    private
    type(SHROUD_capsule_data) :: mem
contains
    final :: SHROUD_capsule_final
    procedure :: delete => SHROUD_capsule_delete
end type LIB_SHROUD_capsule
##### end capsule_helper derived_type

##### start capsule_helper source

! helper capsule_helper
! finalize a static SHROUD_capsule_data
subroutine SHROUD_capsule_final(cap)
    type(LIB_SHROUD_capsule), intent(INOUT) :: cap
    call LIB_SHROUD_capsule_dtor(cap%mem)
end subroutine SHROUD_capsule_final

subroutine SHROUD_capsule_delete(cap)
    class(LIB_SHROUD_capsule) :: cap
    call LIB_SHROUD_capsule_dtor(cap%mem)
end subroutine SHROUD_capsule_delete
##### end capsule_helper source

##### start copy_array_double interface

interface
    ! helper copy_array_double
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_double(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_DOUBLE, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        real(C_DOUBLE), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_double
end interface
##### end copy_array_double interface

##### start copy_array_double_complex interface

interface
    ! helper copy_array_double_complex
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_double_complex(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_DOUBLE_COMPLEX, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        complex(C_DOUBLE_COMPLEX), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_double_complex
end interface
##### end copy_array_double_complex interface

##### start copy_array_float interface

interface
    ! helper copy_array_float
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_float(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_FLOAT, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        real(C_FLOAT), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_float
end interface
##### end copy_array_float interface

##### start copy_array_float_complex interface

interface
    ! helper copy_array_float_complex
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_float_complex(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_FLOAT_COMPLEX, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        complex(C_FLOAT_COMPLEX), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_float_complex
end interface
##### end copy_array_float_complex interface

##### start copy_array_int interface

interface
    ! helper copy_array_int
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_int(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_INT), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_int
end interface
##### end copy_array_int interface

##### start copy_array_int16_t interface

interface
    ! helper copy_array_int16_t
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_int16_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT16_T, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_INT16_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_int16_t
end interface
##### end copy_array_int16_t interface

##### start copy_array_int32_t interface

interface
    ! helper copy_array_int32_t
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_int32_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT32_T, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_INT32_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_int32_t
end interface
##### end copy_array_int32_t interface

##### start copy_array_int64_t interface

interface
    ! helper copy_array_int64_t
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_int64_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT64_T, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_INT64_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_int64_t
end interface
##### end copy_array_int64_t interface

##### start copy_array_int8_t interface

interface
    ! helper copy_array_int8_t
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_int8_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT8_T, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_INT8_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_int8_t
end interface
##### end copy_array_int8_t interface

##### start copy_array_long interface

interface
    ! helper copy_array_long
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_long(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_LONG, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_LONG), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_long
end interface
##### end copy_array_long interface

##### start copy_array_long_long interface

interface
    ! helper copy_array_long_long
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_long_long(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_LONG_LONG, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_LONG_LONG), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_long_long
end interface
##### end copy_array_long_long interface

##### start copy_array_short interface

interface
    ! helper copy_array_short
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_short(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_SHORT, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_SHORT), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_short
end interface
##### end copy_array_short interface

##### start copy_array_size_t interface

interface
    ! helper copy_array_size_t
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_size_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_SIZE_T, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_SIZE_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_size_t
end interface
##### end copy_array_size_t interface

##### start copy_array_uint16_t interface

interface
    ! helper copy_array_uint16_t
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_uint16_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT16_T, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_INT16_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_uint16_t
end interface
##### end copy_array_uint16_t interface

##### start copy_array_uint32_t interface

interface
    ! helper copy_array_uint32_t
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_uint32_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT32_T, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_INT32_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_uint32_t
end interface
##### end copy_array_uint32_t interface

##### start copy_array_uint64_t interface

interface
    ! helper copy_array_uint64_t
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_uint64_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT64_T, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_INT64_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_uint64_t
end interface
##### end copy_array_uint64_t interface

##### start copy_array_uint8_t interface

interface
    ! helper copy_array_uint8_t
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_uint8_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT8_T, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_INT8_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_uint8_t
end interface
##### end copy_array_uint8_t interface

##### start copy_array_unsigned_int interface

interface
    ! helper copy_array_unsigned_int
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_unsigned_int(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_INT), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_unsigned_int
end interface
##### end copy_array_unsigned_int interface

##### start copy_array_unsigned_long interface

interface
    ! helper copy_array_unsigned_long
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_unsigned_long(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_LONG, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_LONG), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_unsigned_long
end interface
##### end copy_array_unsigned_long interface

##### start copy_array_unsigned_long_long interface

interface
    ! helper copy_array_unsigned_long_long
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_unsigned_long_long(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_LONG_LONG, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_LONG_LONG), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_unsigned_long_long
end interface
##### end copy_array_unsigned_long_long interface

##### start copy_array_unsigned_short interface

interface
    ! helper copy_array_unsigned_short
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array_unsigned_short(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_SHORT, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        integer(C_SHORT), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array_unsigned_short
end interface
##### end copy_array_unsigned_short interface

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
