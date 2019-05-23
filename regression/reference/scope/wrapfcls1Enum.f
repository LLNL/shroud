! wrapfcls1Enum.f
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
!! \file wrapfcls1Enum.f
!! \brief Shroud generated wrapper for cls1Enum class
!<
! splicer begin file_top
! splicer end file_top
module cls1enum_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.cls1Enum.module_use
    ! splicer end class.cls1Enum.module_use
    implicit none


    !  enum cls1Enum::Color
    integer(C_INT), parameter :: cls1enum_color_red = 40
    integer(C_INT), parameter :: cls1enum_color_blue = 41
    integer(C_INT), parameter :: cls1enum_color_white = 42

    ! splicer begin class.cls1Enum.module_top
    ! splicer end class.cls1Enum.module_top

    type, bind(C) :: SHROUD_cls1enum_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_cls1enum_capsule

    type cls1enum
        type(SHROUD_cls1enum_capsule) :: cxxmem
        ! splicer begin class.cls1Enum.component_part
        ! splicer end class.cls1Enum.component_part
    contains
        procedure :: get_instance => cls1enum_get_instance
        procedure :: set_instance => cls1enum_set_instance
        procedure :: associated => cls1enum_associated
        ! splicer begin class.cls1Enum.type_bound_procedure_part
        ! splicer end class.cls1Enum.type_bound_procedure_part
    end type cls1enum

    interface operator (.eq.)
        module procedure cls1enum_eq
    end interface

    interface operator (.ne.)
        module procedure cls1enum_ne
    end interface

    interface

        ! splicer begin class.cls1Enum.additional_interfaces
        ! splicer end class.cls1Enum.additional_interfaces
    end interface

contains

    ! Return pointer to C++ memory.
    function cls1enum_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(cls1enum), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function cls1enum_get_instance

    subroutine cls1enum_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(cls1enum), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine cls1enum_set_instance

    function cls1enum_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(cls1enum), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function cls1enum_associated

    ! splicer begin class.cls1Enum.additional_functions
    ! splicer end class.cls1Enum.additional_functions

    function cls1enum_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(cls1enum), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function cls1enum_eq

    function cls1enum_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(cls1enum), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function cls1enum_ne

end module cls1enum_mod
