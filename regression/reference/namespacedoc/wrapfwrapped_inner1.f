! wrapfwrapped_inner1.f
! This file is generated by Shroud 0.12.1. Do not edit.
! Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfwrapped_inner1.f
!! \brief Shroud generated wrapper for inner1 namespace
!<
! splicer begin namespace.inner1.file_top
! splicer end namespace.inner1.file_top
module wrapped_inner1_mod
    ! splicer begin namespace.inner1.module_use
    ! splicer end namespace.inner1.module_use
    implicit none

    ! splicer begin namespace.inner1.module_top
    ! splicer end namespace.inner1.module_top

    interface

        ! ----------------------------------------
        ! Function:  void worker
        ! Requested: c_void_scalar_result
        ! Match:     c_default
        subroutine worker() &
                bind(C, name="WWW_inner1_worker")
            implicit none
        end subroutine worker

        ! splicer begin namespace.inner1.additional_interfaces
        ! splicer end namespace.inner1.additional_interfaces
    end interface

contains

    ! splicer begin namespace.inner1.additional_functions
    ! splicer end namespace.inner1.additional_functions

end module wrapped_inner1_mod
