! wrapftwoTs_0.F
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
!! \file wrapftwoTs_0.F
!! \brief Shroud generated wrapper for twoTs class
!<
! splicer begin file_top
! splicer end file_top
module twots_0_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.twoTs_0.module_use
    ! splicer end class.twoTs_0.module_use
    implicit none


    ! splicer begin class.twoTs_0.module_top
    ! splicer end class.twoTs_0.module_top

    type, bind(C) :: SHROUD_twots_0_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_twots_0_capsule

    type twots_0
        type(SHROUD_twots_0_capsule) :: cxxmem
        ! splicer begin class.twoTs_0.component_part
        ! splicer end class.twoTs_0.component_part
    contains
        procedure :: get_instance => twots_0_get_instance
        procedure :: set_instance => twots_0_set_instance
        procedure :: associated => twots_0_associated
        ! splicer begin class.twoTs_0.type_bound_procedure_part
        ! splicer end class.twoTs_0.type_bound_procedure_part
    end type twots_0

    interface operator (.eq.)
        module procedure twots_0_eq
    end interface

    interface operator (.ne.)
        module procedure twots_0_ne
    end interface

    interface

        ! splicer begin class.twoTs_0.additional_interfaces
        ! splicer end class.twoTs_0.additional_interfaces
    end interface

contains

    ! Return pointer to C++ memory.
    function twots_0_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(twots_0), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function twots_0_get_instance

    subroutine twots_0_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(twots_0), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine twots_0_set_instance

    function twots_0_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(twots_0), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function twots_0_associated

    ! splicer begin class.twoTs_0.additional_functions
    ! splicer end class.twoTs_0.additional_functions

    function twots_0_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(twots_0), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function twots_0_eq

    function twots_0_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(twots_0), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function twots_0_ne

end module twots_0_mod
