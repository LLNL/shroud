! wrapfNames2.F
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
!! \file wrapfNames2.F
!! \brief Shroud generated wrapper for Names2 class
!<
! splicer begin file_top
! splicer end file_top
module names2_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.Names2.module_use
    ! splicer end class.Names2.module_use
    implicit none


    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
        integer(C_INT) :: refcount = 0    ! reference count
    end type SHROUD_capsule_data

    ! splicer begin class.Names2.module_top
    ! splicer end class.Names2.module_top

    type names2
        type(C_PTR), private :: cxxptr = C_NULL_PTR
        type(SHROUD_capsule_data), pointer :: cxxmem => null()
        ! splicer begin class.Names2.component_part
        ! splicer end class.Names2.component_part
    contains
        procedure :: get_instance => names2_get_instance
        procedure :: set_instance => names2_set_instance
        procedure :: associated => names2_associated
        procedure :: names2_assign
        generic :: assignment(=) => names2_assign
        final :: names2_final
        ! splicer begin class.Names2.type_bound_procedure_part
        ! splicer end class.Names2.type_bound_procedure_part
    end type names2

    interface operator (.eq.)
        module procedure names2_eq
    end interface

    interface operator (.ne.)
        module procedure names2_ne
    end interface

    interface

        ! splicer begin class.Names2.additional_interfaces
        ! splicer end class.Names2.additional_interfaces
    end interface

contains

    ! Return pointer to C++ memory if allocated, else C_NULL_PTR.
    function names2_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: c_associated, C_NULL_PTR, C_PTR
        class(names2), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        if (c_associated(obj%cxxptr)) then
            cxxptr = obj%cxxmem%addr
        else
            cxxptr = C_NULL_PTR
        endif
    end function names2_get_instance

    subroutine names2_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(names2), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine names2_set_instance

    function names2_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(names2), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function names2_associated

    subroutine names2_assign(lhs, rhs)
        use iso_c_binding, only : c_associated, c_f_pointer
        class(names2), intent(INOUT) :: lhs
        class(names2), intent(IN) :: rhs

        lhs%cxxptr = rhs%cxxptr
        if (c_associated(lhs%cxxptr)) then
            call c_f_pointer(lhs%cxxptr, lhs%cxxmem)
            lhs%cxxmem%refcount = lhs%cxxmem%refcount + 1
        else
            nullify(lhs%cxxmem)
        endif
    end subroutine names2_assign

    subroutine names2_final(obj)
        use iso_c_binding, only : c_associated, C_BOOL, C_NULL_PTR
        type(names2), intent(INOUT) :: obj
        interface
            subroutine array_destructor(ptr, gc) &
                bind(C, name="TES_SHROUD_array_destructor_function")
                use iso_c_binding, only : C_BOOL, C_INT, C_PTR
                implicit none
                type(C_PTR), value, intent(IN) :: ptr
                logical(C_BOOL), value, intent(IN) :: gc
            end subroutine array_destructor
        end interface
        if (c_associated(obj%cxxptr)) then
            call array_destructor(obj%cxxptr, .true._C_BOOL)
            obj%cxxptr = C_NULL_PTR
            nullify(obj%cxxmem)
        endif
    end subroutine names2_final

    ! splicer begin class.Names2.additional_functions
    ! splicer end class.Names2.additional_functions

    function names2_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(names2), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function names2_eq

    function names2_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(names2), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function names2_ne

end module names2_mod
