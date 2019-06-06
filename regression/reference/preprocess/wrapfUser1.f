! wrapfUser1.f
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
!! \file wrapfUser1.f
!! \brief Shroud generated wrapper for User1 class
!<
! splicer begin file_top
! splicer end file_top
module user1_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.User1.module_use
    ! splicer end class.User1.module_use
    implicit none


    ! splicer begin class.User1.module_top
    ! splicer end class.User1.module_top

    type, bind(C) :: SHROUD_user1_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_user1_capsule

    type user1
        type(SHROUD_user1_capsule) :: cxxmem
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

    interface operator (.eq.)
        module procedure user1_eq
    end interface

    interface operator (.ne.)
        module procedure user1_ne
    end interface

    interface

        subroutine c_user1_method1(self) &
                bind(C, name="PRE_user1_method1")
            import :: SHROUD_user1_capsule
            implicit none
            type(SHROUD_user1_capsule), intent(IN) :: self
        end subroutine c_user1_method1

#if defined(USE_TWO)
        subroutine c_user1_method2(self) &
                bind(C, name="PRE_user1_method2")
            import :: SHROUD_user1_capsule
            implicit none
            type(SHROUD_user1_capsule), intent(IN) :: self
        end subroutine c_user1_method2
#endif

#if defined(USE_THREE)
        subroutine c_user1_method3def_0(self) &
                bind(C, name="PRE_user1_method3def_0")
            import :: SHROUD_user1_capsule
            implicit none
            type(SHROUD_user1_capsule), intent(IN) :: self
        end subroutine c_user1_method3def_0
#endif

#if defined(USE_THREE)
        subroutine c_user1_method3def_1(self, i) &
                bind(C, name="PRE_user1_method3def_1")
            use iso_c_binding, only : C_INT
            import :: SHROUD_user1_capsule
            implicit none
            type(SHROUD_user1_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: i
        end subroutine c_user1_method3def_1
#endif

        ! splicer begin class.User1.additional_interfaces
        ! splicer end class.User1.additional_interfaces
    end interface

contains

    ! void method1()
    subroutine user1_method1(obj)
        class(user1) :: obj
        ! splicer begin class.User1.method.method1
        call c_user1_method1(obj%cxxmem)
        ! splicer end class.User1.method.method1
    end subroutine user1_method1

#if defined(USE_TWO)
    ! void method2()
    subroutine user1_method2(obj)
        class(user1) :: obj
        ! splicer begin class.User1.method.method2
        call c_user1_method2(obj%cxxmem)
        ! splicer end class.User1.method.method2
    end subroutine user1_method2
#endif

#if defined(USE_THREE)
    ! void method3def()
    ! has_default_arg
    subroutine user1_method3def_0(obj)
        class(user1) :: obj
        ! splicer begin class.User1.method.method3def_0
        call c_user1_method3def_0(obj%cxxmem)
        ! splicer end class.User1.method.method3def_0
    end subroutine user1_method3def_0
#endif

#if defined(USE_THREE)
    ! void method3def(int i=0 +intent(in)+value)
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

end module user1_mod
