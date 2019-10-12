! wrapflibrary_outer2.f
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
!! \file wrapflibrary_outer2.f
!! \brief Shroud generated wrapper for outer2 namespace
!<
module library_outer2_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    implicit none


    type, bind(C) :: SHROUD_classa_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_classa_capsule

    type classa
        type(SHROUD_classa_capsule) :: cxxmem
    contains
        procedure :: method => classa_method
        procedure :: get_instance => classa_get_instance
        procedure :: set_instance => classa_set_instance
        procedure :: associated => classa_associated
    end type classa

    interface operator (.eq.)
        module procedure classa_eq
    end interface

    interface operator (.ne.)
        module procedure classa_ne
    end interface

    interface

        subroutine c_classa_method(self) &
                bind(C, name="LIB_outer2_classA_method")
            import :: SHROUD_classa_capsule
            implicit none
            type(SHROUD_classa_capsule), intent(IN) :: self
        end subroutine c_classa_method


        subroutine outer_func() &
                bind(C, name="LIB_outer2_outer_func")
            implicit none
        end subroutine outer_func

    end interface

contains

    subroutine classa_method(obj)
        class(classa) :: obj
        call c_classa_method(obj%cxxmem)
    end subroutine classa_method

    ! Return pointer to C++ memory.
    function classa_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(classa), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function classa_get_instance

    subroutine classa_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(classa), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine classa_set_instance

    function classa_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(classa), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function classa_associated



    function classa_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(classa), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function classa_eq

    function classa_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(classa), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function classa_ne

end module library_outer2_mod
