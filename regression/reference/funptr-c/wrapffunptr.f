! wrapffunptr.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapffunptr.f
!! \brief Shroud generated wrapper for funptr library
!<
! splicer begin file_top
! splicer end file_top
module funptr_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    abstract interface

        subroutine callback1_external_incr() bind(C)
            implicit none
        end subroutine callback1_external_incr

        subroutine callback1_incr() bind(C)
            implicit none
        end subroutine callback1_incr

        subroutine callback1_wrap_incr() bind(C)
            implicit none
        end subroutine callback1_wrap_incr

    end interface

    interface

        subroutine callback1(incr) &
                bind(C, name="callback1")
            import :: callback1_incr
            implicit none
            procedure(callback1_incr) :: incr
        end subroutine callback1

        subroutine c_callback1_wrap(incr) &
                bind(C, name="callback1_wrap")
            import :: callback1_wrap_incr
            implicit none
            procedure(callback1_wrap_incr) :: incr
        end subroutine c_callback1_wrap

        subroutine c_callback1_external(incr) &
                bind(C, name="callback1_external")
            import :: callback1_external_incr
            implicit none
            procedure(callback1_external_incr) :: incr
        end subroutine c_callback1_external
    end interface

    ! splicer begin additional_declarations
    ! splicer end additional_declarations

contains

    !>
    !! \brief Create abstract interface for function
    !!
    !! Create a Fortran wrapper to call the bind(C) interface.
    !<
    subroutine callback1_wrap(incr)
        procedure(callback1_wrap_incr) :: incr
        ! splicer begin function.callback1_wrap
        call c_callback1_wrap(incr)
        ! splicer end function.callback1_wrap
    end subroutine callback1_wrap

    !>
    !! \brief Declare callback as external
    !!
    !<
    subroutine callback1_external(incr)
        external :: incr
        ! splicer begin function.callback1_external
        call c_callback1_external(incr)
        ! splicer end function.callback1_external
    end subroutine callback1_external

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module funptr_mod
