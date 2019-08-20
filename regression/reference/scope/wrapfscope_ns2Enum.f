! wrapfscope_ns2Enum.f
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
!! \file wrapfscope_ns2Enum.f
!! \brief Shroud generated wrapper for ns2Enum namespace
!<
! splicer begin namespace.ns2Enum.file_top
! splicer end namespace.ns2Enum.file_top
module scope_ns2enum_mod
    use iso_c_binding, only : C_INT
    ! splicer begin namespace.ns2Enum.module_use
    ! splicer end namespace.ns2Enum.module_use
    implicit none

    ! splicer begin namespace.ns2Enum.module_top
    ! splicer end namespace.ns2Enum.module_top

    !  enum ns2Enum::Color
    integer(C_INT), parameter :: red = 30
    integer(C_INT), parameter :: blue = 31
    integer(C_INT), parameter :: white = 32

    interface

        ! splicer begin namespace.ns2Enum.additional_interfaces
        ! splicer end namespace.ns2Enum.additional_interfaces
    end interface

contains

    ! splicer begin namespace.ns2Enum.additional_functions
    ! splicer end namespace.ns2Enum.additional_functions

end module scope_ns2enum_mod
