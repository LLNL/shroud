! wrapfns.f
! This is generated code, do not edit
!>
!! \file wrapfns.f
!! \brief Shroud generated wrapper for ns library
!<
! splicer begin file_top
! splicer end file_top
module ns_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

    type, bind(C) :: SHROUD_array
        type(SHROUD_capsule_data) :: cxx       ! address of C++ memory
        type(C_PTR) :: addr = C_NULL_PTR       ! address of data in cxx
        integer(C_SIZE_T) :: len = 0_C_SIZE_T  ! bytes-per-item or character len of data in cxx
        integer(C_SIZE_T) :: size = 0_C_SIZE_T ! size of data in cxx
    end type SHROUD_array

    interface

        function c_last_function_called() &
                result(SHT_rv) &
                bind(C, name="NS_last_function_called")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR) SHT_rv
        end function c_last_function_called

        subroutine c_last_function_called_bufferify(DSHF_rv) &
                bind(C, name="NS_last_function_called_bufferify")
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_last_function_called_bufferify

        subroutine one() &
                bind(C, name="NS_one")
            implicit none
        end subroutine one

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface
        ! helper function
        ! Copy the char* or std::string in context into c_var.
        subroutine SHROUD_copy_string_and_free(context, c_var, c_var_size) &
             bind(c,name="NS_ShroudCopyStringAndFree")
            use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            character(kind=C_CHAR), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_string_and_free
    end interface

contains

    ! const std::string & LastFunctionCalled() +deref(allocatable)
    ! arg_to_buffer
    function last_function_called() &
            result(SHT_rv)
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin function.last_function_called
        call c_last_function_called_bufferify(DSHF_rv)
        ! splicer end function.last_function_called
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function last_function_called

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module ns_mod
