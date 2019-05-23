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

    !  enum Color
    integer(C_INT), parameter :: red = 10
    integer(C_INT), parameter :: blue = 11
    integer(C_INT), parameter :: white = 12

    !  enum ns1Enum::Color
    integer(C_INT), parameter :: ns1enum_color_red = 20
    integer(C_INT), parameter :: ns1enum_color_blue = 21
    integer(C_INT), parameter :: ns1enum_color_white = 22

    !  enum ns2Enum::Color
    integer(C_INT), parameter :: ns2enum_color_red = 30
    integer(C_INT), parameter :: ns2enum_color_blue = 31
    integer(C_INT), parameter :: ns2enum_color_white = 32

    !  enum ColorEnum
    integer(C_INT), parameter :: colorenum_red = 60
    integer(C_INT), parameter :: colorenum_blue = 61
    integer(C_INT), parameter :: colorenum_white = 62

    interface

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module scope_mod
