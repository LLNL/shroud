
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
