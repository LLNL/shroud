! wrapftestnames_ns1.F
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
!! \file wrapftestnames_ns1.F
!! \brief Shroud generated wrapper for ns1 namespace
!<
! splicer begin namespace.ns1.file_top
! splicer end namespace.ns1.file_top
module testnames_ns1_mod
    ! splicer begin namespace.ns1.module_use
    ! splicer end namespace.ns1.module_use
    implicit none

    ! splicer begin namespace.ns1.module_top
    ! splicer end namespace.ns1.module_top

    interface

        subroutine c_init_ns1() &
                bind(C, name="TES_ns1_init_ns1")
            implicit none
        end subroutine c_init_ns1

        ! splicer begin namespace.ns1.additional_interfaces
        ! splicer end namespace.ns1.additional_interfaces
    end interface

contains

    ! void init_ns1()
    subroutine testnames_init_ns1()
        ! splicer begin namespace.ns1.function.init_ns1
        call c_init_ns1()
        ! splicer end namespace.ns1.function.init_ns1
    end subroutine testnames_init_ns1

    ! splicer begin namespace.ns1.additional_functions
    ! splicer end namespace.ns1.additional_functions

end module testnames_ns1_mod
