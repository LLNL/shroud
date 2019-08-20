! wrapfns_outer.f
! This is generated code, do not edit
!>
!! \file wrapfns_outer.f
!! \brief Shroud generated wrapper for outer namespace
!<
! splicer begin namespace.outer.file_top
! splicer end namespace.outer.file_top
module ns_outer_mod
    use iso_c_binding, only : C_DOUBLE, C_INT
    ! splicer begin namespace.outer.module_use
    ! splicer end namespace.outer.module_use
    implicit none

    ! splicer begin namespace.outer.module_top
    ! splicer end namespace.outer.module_top


    type, bind(C) :: cstruct1
        integer(C_INT) :: ifield
        real(C_DOUBLE) :: dfield
    end type cstruct1

    interface

        subroutine one() &
                bind(C, name="NS_outer_one")
            implicit none
        end subroutine one

        ! splicer begin namespace.outer.additional_interfaces
        ! splicer end namespace.outer.additional_interfaces
    end interface

contains

    ! splicer begin namespace.outer.additional_functions
    ! splicer end namespace.outer.additional_functions

end module ns_outer_mod
