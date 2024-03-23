! wrapffunptr.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapffunptr.f
!! \brief Shroud generated wrapper for funptr library
!<
! splicer begin file_top
! splicer end file_top
module funptr_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    abstract interface

        subroutine callback1_external_incr() bind(C)
            implicit none
        end subroutine callback1_external_incr

        subroutine callback1_incr() bind(C)
            implicit none
        end subroutine callback1_incr

        subroutine callback1_wrap_incr() bind(C)
            implicit none
        end subroutine callback1_wrap_incr

        subroutine incrtype(i) bind(C)
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value :: i
        end subroutine incrtype

    end interface

    interface

        subroutine callback1(incr) &
                bind(C, name="callback1")
            import :: callback1_incr
            implicit none
            procedure(callback1_incr) :: incr
        end subroutine callback1

        subroutine c_callback1_wrap(incr) &
                bind(C, name="callback1_wrap")
            import :: callback1_wrap_incr
            implicit none
            procedure(callback1_wrap_incr) :: incr
        end subroutine c_callback1_wrap

        subroutine c_callback1_external(incr) &
                bind(C, name="callback1_external")
            import :: callback1_external_incr
            implicit none
            procedure(callback1_external_incr) :: incr
        end subroutine c_callback1_external

        ! start callback1_funptr
        subroutine callback1_funptr(incr) &
                bind(C, name="callback1_funptr")
            use iso_c_binding, only : C_FUNPTR
            implicit none
            type(C_FUNPTR), value :: incr
        end subroutine callback1_funptr
        ! end callback1_funptr

        subroutine c_callback2(name, ival, incr) &
                bind(C, name="callback2")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: incrtype
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: ival
            procedure(incrtype) :: incr
        end subroutine c_callback2

        subroutine c_callback2_external(name, ival, incr) &
                bind(C, name="callback2_external")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: incrtype
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: ival
            procedure(incrtype) :: incr
        end subroutine c_callback2_external

        subroutine c_callback2_funptr(name, ival, incr) &
                bind(C, name="callback2_funptr")
            use iso_c_binding, only : C_CHAR, C_FUNPTR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: ival
            type(C_FUNPTR), value :: incr
        end subroutine c_callback2_funptr

        subroutine callback3(type, in, incr) &
                bind(C, name="callback3")
            use iso_c_binding, only : C_FUNPTR, C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: type
            type(*) :: in
            type(C_FUNPTR), value :: incr
        end subroutine callback3
    end interface

    ! splicer begin additional_declarations
    ! splicer end additional_declarations

contains

    !>
    !! \brief Create abstract interface for function
    !!
    !! Create a Fortran wrapper to call the bind(C) interface.
    !<
    subroutine callback1_wrap(incr)
        procedure(callback1_wrap_incr) :: incr
        ! splicer begin function.callback1_wrap
        call c_callback1_wrap(incr)
        ! splicer end function.callback1_wrap
    end subroutine callback1_wrap

    !>
    !! \brief Declare callback as external
    !!
    !<
    subroutine callback1_external(incr)
        external :: incr
        ! splicer begin function.callback1_external
        call c_callback1_external(incr)
        ! splicer end function.callback1_external
    end subroutine callback1_external

    !>
    !! \brief Create abstract interface for function
    !!
    !<
    subroutine callback2(name, ival, incr)
        use iso_c_binding, only : C_INT, C_NULL_CHAR
        character(len=*), intent(IN) :: name
        integer(C_INT), value, intent(IN) :: ival
        procedure(incrtype) :: incr
        ! splicer begin function.callback2
        call c_callback2(trim(name)//C_NULL_CHAR, ival, incr)
        ! splicer end function.callback2
    end subroutine callback2

    !>
    !! \brief Declare callback as external
    !!
    !<
    subroutine callback2_external(name, ival, incr)
        use iso_c_binding, only : C_INT, C_NULL_CHAR
        character(len=*), intent(IN) :: name
        integer(C_INT), value, intent(IN) :: ival
        external :: incr
        ! splicer begin function.callback2_external
        call c_callback2_external(trim(name)//C_NULL_CHAR, ival, incr)
        ! splicer end function.callback2_external
    end subroutine callback2_external

    !>
    !! \brief Declare callback as c_funptr
    !!
    !! The caller is responsible for using c_funloc to pass the function address.
    !! Allows any function to be passed as an argument.
    !<
    subroutine callback2_funptr(name, ival, incr)
        use iso_c_binding, only : C_FUNPTR, C_INT, C_NULL_CHAR
        character(len=*), intent(IN) :: name
        integer(C_INT), value, intent(IN) :: ival
        type(C_FUNPTR) :: incr
        ! splicer begin function.callback2_funptr
        call c_callback2_funptr(trim(name)//C_NULL_CHAR, ival, incr)
        ! splicer end function.callback2_funptr
    end subroutine callback2_funptr

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module funptr_mod
