! wrapflibrary_three.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapflibrary_three.f
!! \brief Shroud generated wrapper for three namespace
!<
module library_three_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    implicit none


    ! helper capsule_data_helper
    type, bind(C) :: LIB_SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type LIB_SHROUD_capsule_data

    type class1
        type(LIB_SHROUD_capsule_data) :: cxxmem
    contains
        procedure :: method1 => class1_method1
        procedure :: get_instance => class1_get_instance
        procedure :: set_instance => class1_set_instance
        procedure :: associated => class1_associated
    end type class1

    interface operator (.eq.)
        module procedure class1_eq
    end interface

    interface operator (.ne.)
        module procedure class1_ne
    end interface

    interface

        ! ----------------------------------------
        ! Function:  void method1
        ! Attrs:     +intent(subroutine)
        ! Exact:     f_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  CustomType arg1 +value
        ! Attrs:     +intent(in)
        ! Exact:     f_in_native_scalar
        subroutine c_class1_method1(self, arg1) &
                bind(C, name="LIB_three_Class1_method1")
            import :: LIB_SHROUD_capsule_data, custom_type
            implicit none
            type(LIB_SHROUD_capsule_data), intent(IN) :: self
            integer(custom_type), value, intent(IN) :: arg1
        end subroutine c_class1_method1
    end interface


contains

    ! ----------------------------------------
    ! Function:  void method1
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! ----------------------------------------
    ! Argument:  CustomType arg1 +value
    ! Attrs:     +intent(in)
    ! Exact:     f_in_native_scalar
    subroutine class1_method1(obj, arg1)
        use library_mod, only : custom_type
        class(class1) :: obj
        integer(custom_type), value, intent(IN) :: arg1
        call c_class1_method1(obj%cxxmem, arg1)
    end subroutine class1_method1

    ! Return pointer to C++ memory.
    function class1_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(class1), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function class1_get_instance

    subroutine class1_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(class1), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine class1_set_instance

    function class1_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(class1), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function class1_associated



    function class1_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(class1), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function class1_eq

    function class1_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(class1), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function class1_ne

end module library_three_mod
