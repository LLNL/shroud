! wrapflibrary.f
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
!! \file wrapflibrary.f
!! \brief Shroud generated wrapper for library library
!<
module library_mod
    implicit none


    interface

        subroutine function1() &
                bind(C, name="LIB_function1")
            implicit none
        end subroutine function1

    end interface

contains


end module library_mod
