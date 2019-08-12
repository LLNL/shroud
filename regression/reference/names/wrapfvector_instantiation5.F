! wrapfvector_instantiation5.F
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
!! \file wrapfvector_instantiation5.F
!! \brief Shroud generated wrapper for vector class
!<
! splicer begin file_top
! splicer end file_top
module vector_instantiation5_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.vector_instantiation5.module_use
    ! splicer end class.vector_instantiation5.module_use
    implicit none


    ! splicer begin class.vector_instantiation5.module_top
    ! splicer end class.vector_instantiation5.module_top

    type, bind(C) :: SHROUD_vector_instantiation5_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_vector_instantiation5_capsule

    type vector_instantiation5
        type(SHROUD_vector_instantiation5_capsule) :: cxxmem
        ! splicer begin class.vector_instantiation5.component_part
        ! splicer end class.vector_instantiation5.component_part
    contains
        procedure :: get_instance_instantiation5 => vector_instantiation5_get_instance_instantiation5
        procedure :: set_instance_instantiation5 => vector_instantiation5_set_instance_instantiation5
        procedure :: associated_instantiation5 => vector_instantiation5_associated_instantiation5
        ! splicer begin class.vector_instantiation5.type_bound_procedure_part
        ! splicer end class.vector_instantiation5.type_bound_procedure_part
    end type vector_instantiation5

    interface operator (.eq.)
        module procedure vector_instantiation5_eq
    end interface

    interface operator (.ne.)
        module procedure vector_instantiation5_ne
    end interface

    interface

        ! splicer begin class.vector_instantiation5.additional_interfaces
        ! splicer end class.vector_instantiation5.additional_interfaces
    end interface

contains

    ! Return pointer to C++ memory.
    function vector_instantiation5_get_instance_instantiation5(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(vector_instantiation5), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function vector_instantiation5_get_instance_instantiation5

    subroutine vector_instantiation5_set_instance_instantiation5(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(vector_instantiation5), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine vector_instantiation5_set_instance_instantiation5

    function vector_instantiation5_associated_instantiation5(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(vector_instantiation5), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function vector_instantiation5_associated_instantiation5

    ! splicer begin class.vector_instantiation5.additional_functions
    ! splicer end class.vector_instantiation5.additional_functions

    function vector_instantiation5_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_instantiation5), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_instantiation5_eq

    function vector_instantiation5_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_instantiation5), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_instantiation5_ne

end module vector_instantiation5_mod
