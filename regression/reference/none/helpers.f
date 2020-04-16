
##### start capsule_helper source

! helper capsule_helper
! finalize a static SHROUD_capsule_data
subroutine SHROUD_capsule_final(cap)
    use iso_c_binding, only : C_BOOL
    type(SHROUD_capsule), intent(INOUT) :: cap
    interface
        subroutine array_destructor(ptr, gc)&
            bind(C, name="LIB_SHROUD_memory_destructor")
            use iso_c_binding, only : C_BOOL
            import SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(INOUT) :: ptr
            logical(C_BOOL), value, intent(IN) :: gc
        end subroutine array_destructor
    end interface
    call array_destructor(cap%mem, .false._C_BOOL)
end subroutine SHROUD_capsule_final
            
##### end capsule_helper source

##### start copy_array_double interface

interface
    ! helper copy_array_double
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_double(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_DOUBLE, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        real(C_DOUBLE), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_double
end interface
##### end copy_array_double interface

##### start copy_array_float interface

interface
    ! helper copy_array_float
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_float(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_FLOAT, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        real(C_FLOAT), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_float
end interface
##### end copy_array_float interface

##### start copy_array_int interface

interface
    ! helper copy_array_int
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_int(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_INT), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_int
end interface
##### end copy_array_int interface

##### start copy_array_int16_t interface

interface
    ! helper copy_array_int16_t
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_int16_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT16_T, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_INT16_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_int16_t
end interface
##### end copy_array_int16_t interface

##### start copy_array_int32_t interface

interface
    ! helper copy_array_int32_t
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_int32_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT32_T, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_INT32_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_int32_t
end interface
##### end copy_array_int32_t interface

##### start copy_array_int64_t interface

interface
    ! helper copy_array_int64_t
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_int64_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT64_T, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_INT64_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_int64_t
end interface
##### end copy_array_int64_t interface

##### start copy_array_int8_t interface

interface
    ! helper copy_array_int8_t
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_int8_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT8_T, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_INT8_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_int8_t
end interface
##### end copy_array_int8_t interface

##### start copy_array_long interface

interface
    ! helper copy_array_long
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_long(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_LONG, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_LONG), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_long
end interface
##### end copy_array_long interface

##### start copy_array_long_long interface

interface
    ! helper copy_array_long_long
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_long_long(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_LONG_LONG, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_LONG_LONG), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_long_long
end interface
##### end copy_array_long_long interface

##### start copy_array_short interface

interface
    ! helper copy_array_short
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_short(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_SHORT, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_SHORT), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_short
end interface
##### end copy_array_short interface

##### start copy_array_size_t interface

interface
    ! helper copy_array_size_t
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_size_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_SIZE_T, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_SIZE_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_size_t
end interface
##### end copy_array_size_t interface

##### start copy_array_uint16_t interface

interface
    ! helper copy_array_uint16_t
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_uint16_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT16_T, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_INT16_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_uint16_t
end interface
##### end copy_array_uint16_t interface

##### start copy_array_uint32_t interface

interface
    ! helper copy_array_uint32_t
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_uint32_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT32_T, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_INT32_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_uint32_t
end interface
##### end copy_array_uint32_t interface

##### start copy_array_uint64_t interface

interface
    ! helper copy_array_uint64_t
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_uint64_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT64_T, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_INT64_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_uint64_t
end interface
##### end copy_array_uint64_t interface

##### start copy_array_uint8_t interface

interface
    ! helper copy_array_uint8_t
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_uint8_t(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT8_T, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_INT8_T), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_uint8_t
end interface
##### end copy_array_uint8_t interface

##### start copy_array_unsigned_int interface

interface
    ! helper copy_array_unsigned_int
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_unsigned_int(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_INT, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_INT), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_unsigned_int
end interface
##### end copy_array_unsigned_int interface

##### start copy_array_unsigned_long interface

interface
    ! helper copy_array_unsigned_long
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_unsigned_long(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_LONG, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_LONG), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_unsigned_long
end interface
##### end copy_array_unsigned_long interface

##### start copy_array_unsigned_long_long interface

interface
    ! helper copy_array_unsigned_long_long
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_unsigned_long_long(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_LONG_LONG, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_LONG_LONG), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_unsigned_long_long
end interface
##### end copy_array_unsigned_long_long interface

##### start copy_array_unsigned_short interface

interface
    ! helper copy_array_unsigned_short
    ! Copy contents of context into c_var.
    subroutine SHROUD_copy_array_unsigned_short(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_SHORT, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        integer(C_SHORT), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_array_unsigned_short
end interface
##### end copy_array_unsigned_short interface

##### start copy_string interface

interface
    ! helper copy_string
    ! Copy the char* or std::string in context into c_var.
    subroutine SHROUD_copy_string_and_free(context, c_var, c_var_size) &
         bind(c,name="LIB_ShroudCopyStringAndFree")
        use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
        import SHROUD_array
        type(SHROUD_array), intent(IN) :: context
        character(kind=C_CHAR), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine SHROUD_copy_string_and_free
end interface
##### end copy_string interface
