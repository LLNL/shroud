! wrapftestnames_std.F
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
!! \file wrapftestnames_std.F
!! \brief Shroud generated wrapper for std namespace
!<
! splicer begin file_top
! splicer end file_top
module testnames_std_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! splicer begin class.Vvv1.module_top
    ! splicer end class.Vvv1.module_top

    type, bind(C) :: SHROUD_vvv1_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_vvv1_capsule

    type FFvvv1
        type(SHROUD_vvv1_capsule) :: cxxmem
        ! splicer begin class.Vvv1.component_part
        ! splicer end class.Vvv1.component_part
    contains
        procedure :: get_instance => vvv1_get_instance
        procedure :: set_instance => vvv1_set_instance
        procedure :: associated => vvv1_associated
        ! splicer begin class.Vvv1.type_bound_procedure_part
        ! splicer end class.Vvv1.type_bound_procedure_part
    end type FFvvv1

    ! splicer begin class.vector_double.module_top
    ! splicer end class.vector_double.module_top

    type, bind(C) :: SHROUD_vector_double_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_vector_double_capsule

    type vector_double
        type(SHROUD_vector_double_capsule) :: cxxmem
        ! splicer begin class.vector_double.component_part
        ! splicer end class.vector_double.component_part
    contains
        procedure :: get_instance => vector_double_get_instance
        procedure :: set_instance => vector_double_set_instance
        procedure :: associated => vector_double_associated
        ! splicer begin class.vector_double.type_bound_procedure_part
        ! splicer end class.vector_double.type_bound_procedure_part
    end type vector_double

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

    ! splicer begin class.vector_instantiation3.module_top
    ! splicer end class.vector_instantiation3.module_top

    type, bind(C) :: SHROUD_vector_instantiation3_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_vector_instantiation3_capsule

    type vector_instantiation3
        type(SHROUD_vector_instantiation3_capsule) :: cxxmem
        ! splicer begin class.vector_instantiation3.component_part
        ! splicer end class.vector_instantiation3.component_part
    contains
        procedure :: get_instance => vector_instantiation3_get_instance
        procedure :: set_instance => vector_instantiation3_set_instance
        procedure :: associated => vector_instantiation3_associated
        ! splicer begin class.vector_instantiation3.type_bound_procedure_part
        ! splicer end class.vector_instantiation3.type_bound_procedure_part
    end type vector_instantiation3

    interface operator (.eq.)
        module procedure vvv1_eq
        module procedure vector_double_eq
        module procedure vector_instantiation5_eq
        module procedure vector_instantiation3_eq
    end interface

    interface operator (.ne.)
        module procedure vvv1_ne
        module procedure vector_double_ne
        module procedure vector_instantiation5_ne
        module procedure vector_instantiation3_ne
    end interface

    interface

        ! splicer begin class.Vvv1.additional_interfaces
        ! splicer end class.Vvv1.additional_interfaces

        ! splicer begin class.vector_double.additional_interfaces
        ! splicer end class.vector_double.additional_interfaces

        ! splicer begin class.vector_instantiation5.additional_interfaces
        ! splicer end class.vector_instantiation5.additional_interfaces

        ! splicer begin class.vector_instantiation3.additional_interfaces
        ! splicer end class.vector_instantiation3.additional_interfaces
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

    ! Return pointer to C++ memory.
    function vector_instantiation3_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(vector_instantiation3), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function vector_instantiation3_get_instance

    subroutine vector_instantiation3_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(vector_instantiation3), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine vector_instantiation3_set_instance

    function vector_instantiation3_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(vector_instantiation3), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function vector_instantiation3_associated

    ! splicer begin class.vector_instantiation3.additional_functions
    ! splicer end class.vector_instantiation3.additional_functions

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

    function vector_instantiation3_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_instantiation3), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_instantiation3_eq

    function vector_instantiation3_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_instantiation3), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_instantiation3_ne

end module testnames_std_mod