! wrapfenum.f
! This is generated code, do not edit
! Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfenum.f
!! \brief Shroud generated wrapper for enum library
!<
! splicer begin file_top
! splicer end file_top
module enum_mod
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

    !  enum val
    integer(C_INT), parameter :: a1 = 0
    integer(C_INT), parameter :: b1 = 3
    integer(C_INT), parameter :: c1 = 4
    integer(C_INT), parameter :: d1 = b1-a1
    integer(C_INT), parameter :: e1 = d1
    integer(C_INT), parameter :: f1 = d1+1
    integer(C_INT), parameter :: g1 = d1+2
    integer(C_INT), parameter :: h1 = 100

    interface

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module enum_mod
