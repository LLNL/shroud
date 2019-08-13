! wrapfns_outer.f
! This is generated code, do not edit
!>
!! \file wrapfns_outer.f
!! \brief Shroud generated wrapper for outer namespace
!<
! splicer begin file_top
! splicer end file_top
module ns_outer_mod
    use iso_c_binding, only : C_DOUBLE, C_INT
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top


    type, bind(C) :: cstruct1
        integer(C_INT) :: ifield
        real(C_DOUBLE) :: dfield
    end type cstruct1

    interface

        subroutine one() &
                bind(C, name="NS_outer_one")
            implicit none
        end subroutine one

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module ns_outer_mod
