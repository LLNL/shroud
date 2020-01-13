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
        ! number of dimensions
        integer(C_INT) :: rank = -1
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
        Darg%type = SH_TYPE_INT;
        ! Darg%elem_len = C_SIZEOF()
        Darg%size = size(arg)
        Darg%rank = 2
        call c_rank2_in(Darg)
        ! splicer end function.rank2_in
    end subroutine rank2_in

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module cdesc_mod
