! wrapfExClass3.f
! This is generated code, do not edit
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
!! \file wrapfExClass3.f
!! \brief Shroud generated wrapper for ExClass3 class
!<
! splicer begin file_top
! splicer end file_top
module exclass3_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.ExClass3.module_use
    ! splicer end class.ExClass3.module_use
    implicit none


    ! splicer begin class.ExClass3.module_top
    ! splicer end class.ExClass3.module_top

    type, bind(C) :: SHROUD_exclass3_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_exclass3_capsule

    type exclass3
        type(SHROUD_exclass3_capsule) :: cxxmem
        ! splicer begin class.ExClass3.component_part
        ! splicer end class.ExClass3.component_part
    contains
        procedure :: exfunc_0 => exclass3_exfunc_0
        procedure :: exfunc_1 => exclass3_exfunc_1
        procedure :: yadda => exclass3_yadda
        procedure :: associated => exclass3_associated
#ifdef USE_CLASS3_A
        generic :: exfunc => exfunc_0
#endif
#ifndef USE_CLASS3_A
        generic :: exfunc => exfunc_1
#endif
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

#ifdef USE_CLASS3_A
        subroutine c_exclass3_exfunc_0(self) &
                bind(C, name="AA_exclass3_exfunc_0")
            import :: SHROUD_exclass3_capsule
            implicit none
            type(SHROUD_exclass3_capsule), intent(IN) :: self
        end subroutine c_exclass3_exfunc_0
#endif

#ifndef USE_CLASS3_A
        subroutine c_exclass3_exfunc_1(self, flag) &
                bind(C, name="AA_exclass3_exfunc_1")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass3_capsule
            implicit none
            type(SHROUD_exclass3_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: flag
        end subroutine c_exclass3_exfunc_1
#endif

        ! splicer begin class.ExClass3.additional_interfaces
        ! splicer end class.ExClass3.additional_interfaces
    end interface

contains

#ifdef USE_CLASS3_A
    ! void exfunc()
    subroutine exclass3_exfunc_0(obj)
        class(exclass3) :: obj
        ! splicer begin class.ExClass3.method.exfunc_0
        call c_exclass3_exfunc_0(obj%cxxmem)
        ! splicer end class.ExClass3.method.exfunc_0
    end subroutine exclass3_exfunc_0
#endif

#ifndef USE_CLASS3_A
    ! void exfunc(int flag +intent(in)+value)
    subroutine exclass3_exfunc_1(obj, flag)
        use iso_c_binding, only : C_INT
        class(exclass3) :: obj
        integer(C_INT), value, intent(IN) :: flag
        ! splicer begin class.ExClass3.method.exfunc_1
        call c_exclass3_exfunc_1(obj%cxxmem, flag)
        ! splicer end class.ExClass3.method.exfunc_1
    end subroutine exclass3_exfunc_1
#endif

    ! Return pointer to C++ memory.
    function exclass3_yadda(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(exclass3), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function exclass3_yadda

    function exclass3_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(exclass3), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function exclass3_associated

    ! splicer begin class.ExClass3.additional_functions
    ! splicer end class.ExClass3.additional_functions

    function exclass3_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(exclass3), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function exclass3_eq

    function exclass3_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(exclass3), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function exclass3_ne

end module exclass3_mod
