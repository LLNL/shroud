! wrapfUser2.f
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
!! \file wrapfUser2.f
!! \brief Shroud generated wrapper for User2 class
!<
! splicer begin file_top
! splicer end file_top
module user2_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.User2.module_use
    ! splicer end class.User2.module_use
    implicit none


    ! splicer begin class.User2.module_top
    ! splicer end class.User2.module_top

#ifdef USE_USER2
    type, bind(C) :: SHROUD_user2_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_user2_capsule

    type user2
        type(SHROUD_user2_capsule) :: cxxmem
        ! splicer begin class.User2.component_part
        ! splicer end class.User2.component_part
    contains
#ifdef USE_CLASS3_A
        procedure :: exfunc_0 => user2_exfunc_0
#endif
#ifndef USE_CLASS3_A
        procedure :: exfunc_1 => user2_exfunc_1
#endif
        procedure :: get_instance => user2_get_instance
        procedure :: set_instance => user2_set_instance
        procedure :: associated => user2_associated
#ifdef USE_CLASS3_A
        generic :: exfunc => exfunc_0
#endif
#ifndef USE_CLASS3_A
        generic :: exfunc => exfunc_1
#endif
        ! splicer begin class.User2.type_bound_procedure_part
        ! splicer end class.User2.type_bound_procedure_part
    end type user2
#endif

    interface operator (.eq.)
#ifdef USE_USER2
        module procedure user2_eq
#endif
    end interface

    interface operator (.ne.)
#ifdef USE_USER2
        module procedure user2_ne
#endif
    end interface

    interface
#ifdef USE_USER2

#ifdef USE_CLASS3_A
        subroutine c_user2_exfunc_0(self) &
                bind(C, name="PRE_user2_exfunc_0")
            import :: SHROUD_user2_capsule
            implicit none
            type(SHROUD_user2_capsule), intent(IN) :: self
        end subroutine c_user2_exfunc_0
#endif

#ifndef USE_CLASS3_A
        subroutine c_user2_exfunc_1(self, flag) &
                bind(C, name="PRE_user2_exfunc_1")
            use iso_c_binding, only : C_INT
            import :: SHROUD_user2_capsule
            implicit none
            type(SHROUD_user2_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: flag
        end subroutine c_user2_exfunc_1
#endif
#endif

        ! splicer begin class.User2.additional_interfaces
        ! splicer end class.User2.additional_interfaces
    end interface

contains
#ifdef USE_USER2

#ifdef USE_CLASS3_A
    ! void exfunc()
    subroutine user2_exfunc_0(obj)
        class(user2) :: obj
        ! splicer begin class.User2.method.exfunc_0
        call c_user2_exfunc_0(obj%cxxmem)
        ! splicer end class.User2.method.exfunc_0
    end subroutine user2_exfunc_0
#endif

#ifndef USE_CLASS3_A
    ! void exfunc(int flag +intent(in)+value)
    subroutine user2_exfunc_1(obj, flag)
        use iso_c_binding, only : C_INT
        class(user2) :: obj
        integer(C_INT), value, intent(IN) :: flag
        ! splicer begin class.User2.method.exfunc_1
        call c_user2_exfunc_1(obj%cxxmem, flag)
        ! splicer end class.User2.method.exfunc_1
    end subroutine user2_exfunc_1
#endif

    ! Return pointer to C++ memory.
    function user2_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(user2), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function user2_get_instance

    subroutine user2_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(user2), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine user2_set_instance

    function user2_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(user2), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function user2_associated

    ! splicer begin class.User2.additional_functions
    ! splicer end class.User2.additional_functions
#endif
#ifdef USE_USER2

    function user2_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(user2), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function user2_eq

    function user2_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(user2), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function user2_ne
#endif

end module user2_mod
