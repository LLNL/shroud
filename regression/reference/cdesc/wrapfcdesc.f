! wrapfcdesc.f
! This is generated code, do not edit
! Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfcdesc.f
!! \brief Shroud generated wrapper for cdesc library
!<
! splicer begin file_top
! splicer end file_top
module cdesc_mod
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
    end type SHROUD_array

    interface

        subroutine c_rank2_in(Darg) &
                bind(C, name="CDE_rank2_in")
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_rank2_in

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! void Rank2In(int * arg +cdesc+context(Darg)+intent(in)+rank(2))
    subroutine rank2_in(arg)
        use iso_c_binding, only : C_INT, C_LOC
        integer(C_INT), intent(IN), target :: arg(:,:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.rank2_in
        Darg%base_addr = C_LOC(arg)
        ! Darg%type = 
        ! Darg%elem_len = 
        Darg%size = size(arg)
        ! Darg%rank = -1
        call c_rank2_in(Darg)
        ! splicer end function.rank2_in
    end subroutine rank2_in

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module cdesc_mod
