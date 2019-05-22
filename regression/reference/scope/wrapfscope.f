! wrapfscope.f
! This is generated code, do not edit
! #######################################################################
! Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
!
! Produced at the Lawrence Livermore National Laboratory
!
! LLNL-CODE-738041.
!
! All rights reserved.
!
! This file is part of Shroud.
!
! For details about use and distribution, please read LICENSE.
!
! #######################################################################
!>
!! \file wrapfscope.f
!! \brief Shroud generated wrapper for scope library
!<
! splicer begin file_top
! splicer end file_top
module scope_mod
    use iso_c_binding, only : C_INT
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    !  Color
    integer(C_INT), parameter :: color_red = 0
    integer(C_INT), parameter :: color_blue = 1
    integer(C_INT), parameter :: color_white = 2

    !  Color
    integer(C_INT), parameter :: color_red = 0
    integer(C_INT), parameter :: color_blue = 1
    integer(C_INT), parameter :: color_white = 2

    !  Color
    integer(C_INT), parameter :: color_red = 0
    integer(C_INT), parameter :: color_blue = 1
    integer(C_INT), parameter :: color_white = 2

    interface

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module scope_mod
