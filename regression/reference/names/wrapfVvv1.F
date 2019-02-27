! wrapfVvv1.F
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
!! \file wrapfVvv1.F
!! \brief Shroud generated wrapper for vector class
!<
! splicer begin file_top
! splicer end file_top
module vvv1_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.Vvv1.module_use
    ! splicer end class.Vvv1.module_use
    implicit none


    ! splicer begin class.Vvv1.module_top
    ! splicer end class.Vvv1.module_top

    type, bind(C) :: SHROUD_vector_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_vector_capsule

    type FFvvv1
        type(SHROUD_vector_capsule) :: cxxmem
        ! splicer begin class.Vvv1.component_part
        ! splicer end class.Vvv1.component_part
    contains
        procedure :: get_instance => vvv1_get_instance
        procedure :: set_instance => vvv1_set_instance
        procedure :: associated => vvv1_associated
        ! splicer begin class.Vvv1.type_bound_procedure_part
        ! splicer end class.Vvv1.type_bound_procedure_part
    end type FFvvv1

    interface operator (.eq.)
        module procedure vvv1_eq
    end interface

    interface operator (.ne.)
        module procedure vvv1_ne
    end interface

    interface

        ! splicer begin class.Vvv1.additional_interfaces
        ! splicer end class.Vvv1.additional_interfaces
    end interface

contains

    ! Return pointer to C++ memory.
    function vvv1_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(FFvvv1), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function vvv1_get_instance

    subroutine vvv1_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(FFvvv1), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine vvv1_set_instance

    function vvv1_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(FFvvv1), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function vvv1_associated

    ! splicer begin class.Vvv1.additional_functions
    ! splicer end class.Vvv1.additional_functions

    function vvv1_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(FFvvv1), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vvv1_eq

    function vvv1_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(FFvvv1), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vvv1_ne

end module vvv1_mod
