! foo.f
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
!! \file foo.f
!! \brief Shroud generated wrapper for ns0 namespace
!<
! splicer begin file_top
! splicer end file_top
module name_module
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! splicer begin class.Names.module_top
    ! splicer end class.Names.module_top

    type, bind(C) :: SHROUD_names_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_names_capsule

    type FNames
        type(SHROUD_names_capsule) :: cxxmem
        ! splicer begin class.Names.component_part
        ! splicer end class.Names.component_part
    contains
        procedure :: type_method1 => names_method1
        procedure :: method2 => names_method2
        procedure :: get_instance => names_get_instance
        procedure :: set_instance => names_set_instance
        procedure :: associated => names_associated
        ! splicer begin class.Names.type_bound_procedure_part
        ! splicer end class.Names.type_bound_procedure_part
    end type FNames

    interface operator (.eq.)
        module procedure names_eq
    end interface

    interface operator (.ne.)
        module procedure names_ne
    end interface

    interface

        subroutine xxx_tes_names_method1(self) &
                bind(C, name="XXX_TES_ns0_Names_method1")
            import :: SHROUD_names_capsule
            implicit none
            type(SHROUD_names_capsule), intent(IN) :: self
        end subroutine xxx_tes_names_method1

        subroutine xxx_tes_names_method2(self2) &
                bind(C, name="XXX_TES_ns0_Names_method2")
            import :: SHROUD_names_capsule
            implicit none
            type(SHROUD_names_capsule), intent(IN) :: self2
        end subroutine xxx_tes_names_method2

        ! splicer begin class.Names.additional_interfaces
        ! splicer end class.Names.additional_interfaces
    end interface

contains

    ! void method1()
    subroutine names_method1(obj)
        class(FNames) :: obj
        ! splicer begin class.Names.method.type_method1
        call xxx_tes_names_method1(obj%cxxmem)
        ! splicer end class.Names.method.type_method1
    end subroutine names_method1

    ! void method2()
    subroutine names_method2(obj2)
        class(FNames) :: obj2
        ! splicer begin class.Names.method.method2
        call xxx_tes_names_method2(obj2%cxxmem)
        ! splicer end class.Names.method.method2
    end subroutine names_method2

    ! Return pointer to C++ memory.
    function names_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(FNames), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function names_get_instance

    subroutine names_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(FNames), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine names_set_instance

    function names_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(FNames), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function names_associated

    ! splicer begin class.Names.additional_functions
    ! splicer end class.Names.additional_functions

    function names_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(FNames), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function names_eq

    function names_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(FNames), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function names_ne

end module name_module
