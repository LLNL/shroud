! wrapfExClass3.f
! This is generated code, do not edit
! Copyright (c) 2017, Lawrence Livermore National Security, LLC.
! Produced at the Lawrence Livermore National Laboratory
!
! LLNL-CODE-738041.
! All rights reserved.
!
! This file is part of Shroud.  For details, see
! https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are
! met:
!
! * Redistributions of source code must retain the above copyright
!   notice, this list of conditions and the disclaimer below.
!
! * Redistributions in binary form must reproduce the above copyright
!   notice, this list of conditions and the disclaimer (as noted below)
!   in the documentation and/or other materials provided with the
!   distribution.
!
! * Neither the name of the LLNS/LLNL nor the names of its contributors
!   may be used to endorse or promote products derived from this
!   software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
! A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
! LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
! EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
! PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
! LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
! NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
! SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!
! #######################################################################
!>
!! \file wrapfExClass3.f
!! \brief Shroud generated wrapper for ExClass3 class
!<
! splicer begin file_top
! splicer end file_top
module exclass3_mod
    use iso_c_binding, only : C_PTR
    ! splicer begin class.ExClass3.module_use
    ! splicer end class.ExClass3.module_use
    implicit none


    ! splicer begin class.ExClass3.module_top
    ! splicer end class.ExClass3.module_top

    type exclass3
        type(C_PTR), private :: voidptr
        ! splicer begin class.ExClass3.component_part
        ! splicer end class.ExClass3.component_part
    contains
        procedure :: exfunc => exclass3_exfunc
        procedure :: dtor => exclass3_dtor
        procedure :: yadda => exclass3_yadda
        procedure :: associated => exclass3_associated
        ! splicer begin class.ExClass3.type_bound_procedure_part
        ! splicer end class.ExClass3.type_bound_procedure_part
    end type exclass3

    interface operator (.eq.)
        module procedure exclass3_eq
    end interface

    interface operator (.ne.)
        module procedure exclass3_ne
    end interface

    interface

        subroutine c_exclass3_exfunc(self) &
                bind(C, name="AA_exclass3_exfunc")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
        end subroutine c_exclass3_exfunc

        function c_exclass3_ctor() &
                result(SHT_rv) &
                bind(C, name="AA_exclass3_ctor")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR) :: SHT_rv
        end function c_exclass3_ctor

        subroutine c_exclass3_dtor(self) &
                bind(C, name="AA_exclass3_dtor")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
        end subroutine c_exclass3_dtor

        ! splicer begin class.ExClass3.additional_interfaces
        ! splicer end class.ExClass3.additional_interfaces
    end interface

contains

    ! void exfunc()
    ! function_index=48
    subroutine exclass3_exfunc(obj)
        class(exclass3) :: obj
        ! splicer begin class.ExClass3.method.exfunc
        call c_exclass3_exfunc(obj%voidptr)
        ! splicer end class.ExClass3.method.exfunc
    end subroutine exclass3_exfunc

    ! ExClass3()
    ! function_index=49
    function exclass3_ctor() &
            result(SHT_rv)
        type(exclass3) :: SHT_rv
        ! splicer begin class.ExClass3.method.ctor
        SHT_rv%voidptr = c_exclass3_ctor()
        ! splicer end class.ExClass3.method.ctor
    end function exclass3_ctor

    ! ~ExClass3()
    ! function_index=50
    subroutine exclass3_dtor(obj)
        use iso_c_binding, only : C_NULL_PTR
        class(exclass3) :: obj
        ! splicer begin class.ExClass3.method.dtor
        call c_exclass3_dtor(obj%voidptr)
        obj%voidptr = C_NULL_PTR
        ! splicer end class.ExClass3.method.dtor
    end subroutine exclass3_dtor

    function exclass3_yadda(obj) result (voidptr)
        use iso_c_binding, only: C_PTR
        class(exclass3), intent(IN) :: obj
        type(C_PTR) :: voidptr
        voidptr = obj%voidptr
    end function exclass3_yadda

    function exclass3_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(exclass3), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%voidptr)
    end function exclass3_associated

    ! splicer begin class.ExClass3.additional_functions
    ! splicer end class.ExClass3.additional_functions

    function exclass3_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(exclass3), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%voidptr, b%voidptr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function exclass3_eq

    function exclass3_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(exclass3), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%voidptr, b%voidptr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function exclass3_ne

end module exclass3_mod
