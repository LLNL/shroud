! wrapfscope_ns1.f
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
!! \file wrapfscope_ns1.f
!! \brief Shroud generated wrapper for ns1 namespace
!<
! splicer begin namespace.ns1.file_top
! splicer end namespace.ns1.file_top
module scope_ns1_mod
    use iso_c_binding, only : C_INT
    ! splicer begin namespace.ns1.module_use
    ! splicer end namespace.ns1.module_use
    implicit none

    ! splicer begin namespace.ns1.module_top
    ! splicer end namespace.ns1.module_top

    !  enum ns1::Color
    integer(C_INT), parameter :: red = 20
    integer(C_INT), parameter :: blue = 21
    integer(C_INT), parameter :: white = 22

    interface

        ! splicer begin namespace.ns1.additional_interfaces
        ! splicer end namespace.ns1.additional_interfaces
    end interface

contains

    ! splicer begin namespace.ns1.additional_functions
    ! splicer end namespace.ns1.additional_functions

end module scope_ns1_mod
