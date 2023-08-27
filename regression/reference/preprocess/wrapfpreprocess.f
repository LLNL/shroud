! wrapfpreprocess.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfpreprocess.f
!! \brief Shroud generated wrapper for preprocess library
!<
! splicer begin file_top
! splicer end file_top
module preprocess_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! helper capsule_data_helper
    type, bind(C) :: PRE_SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type PRE_SHROUD_capsule_data

    type user1
        type(PRE_SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.User1.component_part
        ! splicer end class.User1.component_part
    contains
        procedure :: method1 => user1_method1
#if defined(USE_TWO)
        procedure :: method2 => user1_method2
#endif
#if defined(USE_THREE)
        procedure :: method3def_0 => user1_method3def_0
#endif
#if defined(USE_THREE)
        procedure :: method3def_1 => user1_method3def_1
#endif
        procedure :: get_instance => user1_get_instance
        procedure :: set_instance => user1_set_instance
        procedure :: associated => user1_associated
#if defined(USE_THREE)
        generic :: method3def => method3def_0
#endif
#if defined(USE_THREE)
        generic :: method3def => method3def_1
#endif
        ! splicer begin class.User1.type_bound_procedure_part
        ! splicer end class.User1.type_bound_procedure_part
    end type user1

#ifdef USE_USER2
    type user2
        type(PRE_SHROUD_capsule_data) :: cxxmem
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
        module procedure user1_eq
#ifdef USE_USER2
        module procedure user2_eq
#endif
    end interface

    interface operator (.ne.)
        module procedure user1_ne
#ifdef USE_USER2
        module procedure user2_ne
#endif
    end interface

    interface

        ! ----------------------------------------
        ! Function:  void method1
        ! Attrs:     +intent(subroutine)
        ! Requested: c_subroutine_void_scalar
        ! Match:     c_subroutine
        subroutine c_user1_method1(self) &
                bind(C, name="PRE_User1_method1")
            import :: PRE_SHROUD_capsule_data
            implicit none
            type(PRE_SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_user1_method1

#if defined(USE_TWO)
        ! ----------------------------------------
        ! Function:  void method2
        ! Attrs:     +intent(subroutine)
        ! Requested: c_subroutine_void_scalar
        ! Match:     c_subroutine
        subroutine c_user1_method2(self) &
                bind(C, name="PRE_User1_method2")
            import :: PRE_SHROUD_capsule_data
            implicit none
            type(PRE_SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_user1_method2
#endif

#if defined(USE_THREE)
        ! ----------------------------------------
        ! Function:  void method3def
        ! Attrs:     +intent(subroutine)
        ! Requested: c_subroutine_void_scalar
        ! Match:     c_subroutine
        subroutine c_user1_method3def_0(self) &
                bind(C, name="PRE_User1_method3def_0")
            import :: PRE_SHROUD_capsule_data
            implicit none
            type(PRE_SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_user1_method3def_0
#endif

#if defined(USE_THREE)
        ! ----------------------------------------
        ! Function:  void method3def
        ! Attrs:     +intent(subroutine)
        ! Requested: c_subroutine_void_scalar
        ! Match:     c_subroutine
        ! ----------------------------------------
        ! Argument:  int i=0 +value
        ! Attrs:     +intent(in)
        ! Exact:     c_in_native_scalar
        subroutine c_user1_method3def_1(self, i) &
                bind(C, name="PRE_User1_method3def_1")
            use iso_c_binding, only : C_INT
            import :: PRE_SHROUD_capsule_data
            implicit none
            type(PRE_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: i
        end subroutine c_user1_method3def_1
#endif
#ifdef USE_USER2

#ifdef USE_CLASS3_A
        ! ----------------------------------------
        ! Function:  void exfunc
        ! Attrs:     +intent(subroutine)
        ! Requested: c_subroutine_void_scalar
        ! Match:     c_subroutine
        subroutine c_user2_exfunc_0(self) &
                bind(C, name="PRE_User2_exfunc_0")
            import :: PRE_SHROUD_capsule_data
            implicit none
            type(PRE_SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_user2_exfunc_0
#endif

#ifndef USE_CLASS3_A
        ! ----------------------------------------
        ! Function:  void exfunc
        ! Attrs:     +intent(subroutine)
        ! Requested: c_subroutine_void_scalar
        ! Match:     c_subroutine
        ! ----------------------------------------
        ! Argument:  int flag +value
        ! Attrs:     +intent(in)
        ! Exact:     c_in_native_scalar
        subroutine c_user2_exfunc_1(self, flag) &
                bind(C, name="PRE_User2_exfunc_1")
            use iso_c_binding, only : C_INT
            import :: PRE_SHROUD_capsule_data
            implicit none
            type(PRE_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: flag
        end subroutine c_user2_exfunc_1
#endif
#endif
    end interface

#if defined(USE_THREE)
    interface user1_method3def
        module procedure user1_method3def_0
        module procedure user1_method3def_1
    end interface user1_method3def
#endif

#ifdef USE_USER2
    interface user2_exfunc
#ifdef USE_CLASS3_A
        module procedure user2_exfunc_0
#endif
#ifndef USE_CLASS3_A
        module procedure user2_exfunc_1
#endif
    end interface user2_exfunc
#endif

    ! splicer begin additional_declarations
    ! splicer end additional_declarations

contains

    ! ----------------------------------------
    ! Function:  void method1
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    subroutine user1_method1(obj)
        class(user1) :: obj
        ! splicer begin class.User1.method.method1
        call c_user1_method1(obj%cxxmem)
        ! splicer end class.User1.method.method1
    end subroutine user1_method1

#if defined(USE_TWO)
    ! ----------------------------------------
    ! Function:  void method2
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    subroutine user1_method2(obj)
        class(user1) :: obj
        ! splicer begin class.User1.method.method2
        call c_user1_method2(obj%cxxmem)
        ! splicer end class.User1.method.method2
    end subroutine user1_method2
#endif

#if defined(USE_THREE)
    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  void method3def
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    subroutine user1_method3def_0(obj)
        class(user1) :: obj
        ! splicer begin class.User1.method.method3def_0
        call c_user1_method3def_0(obj%cxxmem)
        ! splicer end class.User1.method.method3def_0
    end subroutine user1_method3def_0
#endif

#if defined(USE_THREE)
    ! ----------------------------------------
    ! Function:  void method3def
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  int i=0 +value
    ! Attrs:     +intent(in)
    ! Exact:     f_in_native_scalar
    ! Attrs:     +intent(in)
    ! Exact:     c_in_native_scalar
    subroutine user1_method3def_1(obj, i)
        use iso_c_binding, only : C_INT
        class(user1) :: obj
        integer(C_INT), value, intent(IN) :: i
        ! splicer begin class.User1.method.method3def_1
        call c_user1_method3def_1(obj%cxxmem, i)
        ! splicer end class.User1.method.method3def_1
    end subroutine user1_method3def_1
#endif

    ! Return pointer to C++ memory.
    function user1_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(user1), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function user1_get_instance

    subroutine user1_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(user1), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine user1_set_instance

    function user1_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(user1), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function user1_associated

    ! splicer begin class.User1.additional_functions
    ! splicer end class.User1.additional_functions
#ifdef USE_USER2

#ifdef USE_CLASS3_A
    ! ----------------------------------------
    ! Function:  void exfunc
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    subroutine user2_exfunc_0(obj)
        class(user2) :: obj
        ! splicer begin class.User2.method.exfunc_0
        call c_user2_exfunc_0(obj%cxxmem)
        ! splicer end class.User2.method.exfunc_0
    end subroutine user2_exfunc_0
#endif

#ifndef USE_CLASS3_A
    ! ----------------------------------------
    ! Function:  void exfunc
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  int flag +value
    ! Attrs:     +intent(in)
    ! Exact:     f_in_native_scalar
    ! Attrs:     +intent(in)
    ! Exact:     c_in_native_scalar
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

    ! splicer begin additional_functions
    ! splicer end additional_functions

    function user1_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(user1), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function user1_eq

    function user1_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(user1), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function user1_ne
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

end module preprocess_mod
