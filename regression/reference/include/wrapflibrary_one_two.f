! wrapflibrary_one_two.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapflibrary_one_two.f
!! \brief Shroud generated wrapper for two namespace
!<
module library_one_two_mod
    implicit none


    interface

        ! ----------------------------------------
        ! Function:  void function1
        ! Attrs:     +intent(subroutine)
        ! Exact:     f_subroutine_void_scalar
        subroutine function1() &
                bind(C, name="LIB_one_two_function1")
            implicit none
        end subroutine function1
    end interface


contains

#if 0
    ! Only the interface is needed
    ! ----------------------------------------
    ! Function:  void function1
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    subroutine function1()
        call c_function1()
    end subroutine function1
#endif


end module library_one_two_mod
