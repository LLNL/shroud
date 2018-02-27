! foo.f
! This is generated code, do not edit
! #######################################################################
! Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
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
!! \file foo.f
!! \brief Shroud generated wrapper for Names class
!<
! splicer begin file_top
! splicer end file_top
module name_module
    use iso_c_binding, only : C_PTR
    ! splicer begin class.Names.module_use
    ! splicer end class.Names.module_use
    implicit none


    ! splicer begin class.Names.module_top
    ! splicer end class.Names.module_top

    type FNames
        type(C_PTR), private :: voidptr
        ! splicer begin class.Names.component_part
        ! splicer end class.Names.component_part
    contains
        procedure :: type_method1 => names_method1
        procedure :: method2 => names_method2
        procedure :: get_instance => names_get_instance
        procedure :: set_instance => names_set_instance
        procedure :: associated => names_associated
        ! splicer begin class.Names.type_bound_procedure_part
        ! splicer end class.Names.type_bound_procedure_part
    end type FNames

    interface operator (.eq.)
        module procedure names_eq
    end interface

    interface operator (.ne.)
        module procedure names_ne
    end interface

    interface

        subroutine xxx_tes_names_method1(self) &
                bind(C, name="XXX_TES_names_method1")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
        end subroutine xxx_tes_names_method1

        subroutine xxx_tes_names_method2(self2) &
                bind(C, name="XXX_TES_names_method2")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self2
        end subroutine xxx_tes_names_method2

        ! splicer begin class.Names.additional_interfaces
        ! splicer end class.Names.additional_interfaces
    end interface

contains

    ! void method1()
    ! function_index=0
    subroutine names_method1(obj)
        class(FNames) :: obj
        ! splicer begin class.Names.method.type_method1
        call xxx_tes_names_method1(obj%voidptr)
        ! splicer end class.Names.method.type_method1
    end subroutine names_method1

    ! void method2()
    ! function_index=1
    subroutine names_method2(obj2)
        class(FNames) :: obj2
        ! splicer begin class.Names.method.method2
        call xxx_tes_names_method2(obj2%voidptr)
        ! splicer end class.Names.method.method2
    end subroutine names_method2

    function names_get_instance(obj) result (voidptr)
        use iso_c_binding, only: C_PTR
        class(FNames), intent(IN) :: obj
        type(C_PTR) :: voidptr
        voidptr = obj%voidptr
    end function names_get_instance

    subroutine names_set_instance(obj, voidptr)
        use iso_c_binding, only: C_PTR
        class(FNames), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: voidptr
        obj%voidptr = voidptr
    end subroutine names_set_instance

    function names_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(FNames), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%voidptr)
    end function names_associated

    ! splicer begin class.Names.additional_functions
    ! splicer end class.Names.additional_functions

    function names_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(FNames), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%voidptr, b%voidptr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function names_eq

    function names_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(FNames), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%voidptr, b%voidptr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function names_ne

end module name_module
