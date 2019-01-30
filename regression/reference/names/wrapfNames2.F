! wrapfNames2.F
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


    ! splicer begin class.Names2.module_top
    ! splicer end class.Names2.module_top

    type, bind(C) :: SHROUD_names2_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_names2_capsule

    type names2
        type(SHROUD_names2_capsule) :: cxxmem
        ! splicer begin class.Names2.component_part
        ! splicer end class.Names2.component_part
    contains
        procedure :: get_instance => names2_get_instance
        procedure :: set_instance => names2_set_instance
        procedure :: associated => names2_associated
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

    ! Return pointer to C++ memory.
    function names2_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(names2), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
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
