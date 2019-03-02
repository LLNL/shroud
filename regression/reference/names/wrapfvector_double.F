! wrapfvector_double.F
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
!! \file wrapfvector_double.F
!! \brief Shroud generated wrapper for vector class
!<
! splicer begin file_top
! splicer end file_top
module vector_double_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.vector_double.module_use
    ! splicer end class.vector_double.module_use
    implicit none


    ! splicer begin class.vector_double.module_top
    ! splicer end class.vector_double.module_top

    type, bind(C) :: SHROUD_vector_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_vector_capsule

    type vector_double
        type(SHROUD_vector_capsule) :: cxxmem
        ! splicer begin class.vector_double.component_part
        ! splicer end class.vector_double.component_part
    contains
        procedure :: get_instance => vector_double_get_instance
        procedure :: set_instance => vector_double_set_instance
        procedure :: associated => vector_double_associated
        ! splicer begin class.vector_double.type_bound_procedure_part
        ! splicer end class.vector_double.type_bound_procedure_part
    end type vector_double

    interface operator (.eq.)
        module procedure vector_double_eq
    end interface

    interface operator (.ne.)
        module procedure vector_double_ne
    end interface

    interface

        ! splicer begin class.vector_double.additional_interfaces
        ! splicer end class.vector_double.additional_interfaces
    end interface

contains

    ! Return pointer to C++ memory.
    function vector_double_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(vector_double), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function vector_double_get_instance

    subroutine vector_double_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(vector_double), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine vector_double_set_instance

    function vector_double_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(vector_double), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function vector_double_associated

    ! splicer begin class.vector_double.additional_functions
    ! splicer end class.vector_double.additional_functions

    function vector_double_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_double), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_double_eq

    function vector_double_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_double), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_double_ne

end module vector_double_mod
