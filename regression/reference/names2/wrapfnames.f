! wrapfnames.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfnames.f
!! \brief Shroud generated wrapper for ignore2 namespace
!<
! splicer begin file_top
! splicer end file_top
module worker_names
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    interface

        ! ----------------------------------------
        ! Function:  void AFunction
        ! Attrs:     +intent(subroutine)
        ! Exact:     f_subroutine_void_scalar
        subroutine a_function() &
                bind(C, name="NAM_AFunction")
            implicit none
        end subroutine a_function
    end interface

    ! splicer begin additional_declarations
    ! splicer end additional_declarations

contains

#if 0
    ! Only the interface is needed
    ! ----------------------------------------
    ! Function:  void AFunction
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    subroutine a_function()
        ! splicer begin function.a_function
        call c_a_function()
        ! splicer end function.a_function
    end subroutine a_function
#endif

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module worker_names
