! wrapftwoTs_instantiation4.F
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
!! \file wrapftwoTs_instantiation4.F
!! \brief Shroud generated wrapper for twoTs class
!<
! splicer begin file_top
! splicer end file_top
module twots_instantiation4_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.twoTs_instantiation4.module_use
    ! splicer end class.twoTs_instantiation4.module_use
    implicit none


    ! splicer begin class.twoTs_instantiation4.module_top
    ! splicer end class.twoTs_instantiation4.module_top

    type, bind(C) :: SHROUD_twots_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_twots_capsule

    type twots_instantiation4
        type(SHROUD_twots_capsule) :: cxxmem
        ! splicer begin class.twoTs_instantiation4.component_part
        ! splicer end class.twoTs_instantiation4.component_part
    contains
        procedure :: get_instance_instantiation4 => twots_instantiation4_get_instance_instantiation4
        procedure :: set_instance_instantiation4 => twots_instantiation4_set_instance_instantiation4
        procedure :: associated_instantiation4 => twots_instantiation4_associated_instantiation4
        ! splicer begin class.twoTs_instantiation4.type_bound_procedure_part
        ! splicer end class.twoTs_instantiation4.type_bound_procedure_part
    end type twots_instantiation4

    interface operator (.eq.)
        module procedure twots_instantiation4_eq
    end interface

    interface operator (.ne.)
        module procedure twots_instantiation4_ne
    end interface

    interface

        ! splicer begin class.twoTs_instantiation4.additional_interfaces
        ! splicer end class.twoTs_instantiation4.additional_interfaces
    end interface

contains

    ! Return pointer to C++ memory.
    function twots_instantiation4_get_instance_instantiation4(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(twots_instantiation4), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function twots_instantiation4_get_instance_instantiation4

    subroutine twots_instantiation4_set_instance_instantiation4(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(twots_instantiation4), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine twots_instantiation4_set_instance_instantiation4

    function twots_instantiation4_associated_instantiation4(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(twots_instantiation4), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function twots_instantiation4_associated_instantiation4

    ! splicer begin class.twoTs_instantiation4.additional_functions
    ! splicer end class.twoTs_instantiation4.additional_functions

    function twots_instantiation4_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(twots_instantiation4), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function twots_instantiation4_eq

    function twots_instantiation4_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(twots_instantiation4), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function twots_instantiation4_ne

end module twots_instantiation4_mod
