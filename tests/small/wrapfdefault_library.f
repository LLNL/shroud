! wrapfdefault_library.f
! This is generated code, do not edit
!>
!! \file wrapfdefault_library.f
!! \brief Shroud generated wrapper for default_library library
!<
! splicer begin file_top
! splicer end file_top
module default_library_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none


    interface

        subroutine c_vector1_bufferify(arg, Narg) &
                bind(C, name="DEF_vector1_bufferify")
            use iso_c_binding, only : C_INT, C_LONG
            implicit none
            integer(C_INT), intent(IN) :: arg(*)
            integer(C_LONG), value, intent(IN) :: Narg
        end subroutine c_vector1_bufferify

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    subroutine vector1(arg)
        use iso_c_binding, only : C_INT, C_LONG
        integer(C_INT), intent(IN) :: arg(:)
        ! splicer begin function.vector1
        call c_vector1_bufferify(  &
            arg,  &
            size(arg, kind=C_LONG))
        ! splicer end function.vector1
    end subroutine vector1

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module default_library_mod
