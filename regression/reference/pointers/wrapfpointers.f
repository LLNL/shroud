! wrapfpointers.f
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
!! \file wrapfpointers.f
!! \brief Shroud generated wrapper for pointers library
!<
! splicer begin file_top
! splicer end file_top
module pointers_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    interface

        subroutine intargs(argin, arginout, argout) &
                bind(C, name="POI_intargs")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: argin
            integer(C_INT), intent(INOUT) :: arginout
            integer(C_INT), intent(OUT) :: argout
        end subroutine intargs

        subroutine c_cos_doubles(in, out, sizein) &
                bind(C, name="POI_cos_doubles")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), intent(IN) :: in(*)
            real(C_DOUBLE), intent(OUT) :: out(*)
            integer(C_INT), value, intent(IN) :: sizein
        end subroutine c_cos_doubles

        subroutine c_truncate_to_int(in, out, sizein) &
                bind(C, name="POI_truncate_to_int")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), intent(IN) :: in(*)
            integer(C_INT), intent(OUT) :: out(*)
            integer(C_INT), value, intent(IN) :: sizein
        end subroutine c_truncate_to_int

        subroutine c_increment(array, sizein) &
                bind(C, name="POI_increment")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(INOUT) :: array(*)
            integer(C_INT), value, intent(IN) :: sizein
        end subroutine c_increment

        subroutine get_values(nvalues, values) &
                bind(C, name="POI_get_values")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(OUT) :: nvalues
            integer(C_INT), intent(OUT) :: values(*)
        end subroutine get_values

        subroutine get_values2(arg1, arg2) &
                bind(C, name="POI_get_values2")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(OUT) :: arg1(*)
            integer(C_INT), intent(OUT) :: arg2(*)
        end subroutine get_values2

        subroutine c_sum(len, values, result) &
                bind(C, name="POI_sum")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: len
            integer(C_INT), intent(IN) :: values(*)
            integer(C_INT), intent(OUT) :: result
        end subroutine c_sum

        subroutine fill_int_array(out) &
                bind(C, name="POI_fill_int_array")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(OUT) :: out(*)
        end subroutine fill_int_array

        subroutine c_increment_int_array(values, len) &
                bind(C, name="POI_increment_int_array")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(INOUT) :: values(*)
            integer(C_INT), value, intent(IN) :: len
        end subroutine c_increment_int_array

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! void cos_doubles(double * in +dimension(:)+intent(in), double * out +allocatable(mold=in)+intent(out), int sizein +implied(size(in))+intent(in)+value)
    !>
    !! \brief compute cos of IN and save in OUT
    !!
    !! allocate OUT same type as IN implied size of array
    !<
    subroutine cos_doubles(in, out)
        use iso_c_binding, only : C_DOUBLE, C_INT
        real(C_DOUBLE), intent(IN) :: in(:)
        real(C_DOUBLE), intent(OUT), allocatable :: out(:)
        integer(C_INT) :: sizein
        allocate(out(lbound(in,1):ubound(in,1)))
        sizein = size(in,kind=C_INT)
        ! splicer begin function.cos_doubles
        call c_cos_doubles(in, out, sizein)
        ! splicer end function.cos_doubles
    end subroutine cos_doubles

    ! void truncate_to_int(double * in +dimension(:)+intent(in), int * out +allocatable(mold=in)+intent(out), int sizein +implied(size(in))+intent(in)+value)
    !>
    !! \brief truncate IN argument and save in OUT
    !!
    !! allocate OUT different type as IN
    !! implied size of array
    !<
    subroutine truncate_to_int(in, out)
        use iso_c_binding, only : C_DOUBLE, C_INT
        real(C_DOUBLE), intent(IN) :: in(:)
        integer(C_INT), intent(OUT), allocatable :: out(:)
        integer(C_INT) :: sizein
        allocate(out(lbound(in,1):ubound(in,1)))
        sizein = size(in,kind=C_INT)
        ! splicer begin function.truncate_to_int
        call c_truncate_to_int(in, out, sizein)
        ! splicer end function.truncate_to_int
    end subroutine truncate_to_int

    ! void increment(int * array +dimension(:)+intent(inout), int sizein +implied(size(array))+intent(in)+value)
    !>
    !! \brief None
    !!
    !! array with intent(INOUT)
    !<
    subroutine increment(array)
        use iso_c_binding, only : C_INT
        integer(C_INT), intent(INOUT) :: array(:)
        integer(C_INT) :: sizein
        sizein = size(array,kind=C_INT)
        ! splicer begin function.increment
        call c_increment(array, sizein)
        ! splicer end function.increment
    end subroutine increment

    ! void Sum(int len +implied(size(values))+intent(in)+value, int * values +dimension(:)+intent(in), int * result +intent(out))
    subroutine sum(values, result)
        use iso_c_binding, only : C_INT
        integer(C_INT) :: len
        integer(C_INT), intent(IN) :: values(:)
        integer(C_INT), intent(OUT) :: result
        len = size(values,kind=C_INT)
        ! splicer begin function.sum
        call c_sum(len, values, result)
        ! splicer end function.sum
    end subroutine sum

    ! void incrementIntArray(int * values +dimension(:)+intent(inout), int len +implied(size(values))+intent(in)+value)
    !>
    !! Increment array in place.
    !<
    subroutine increment_int_array(values)
        use iso_c_binding, only : C_INT
        integer(C_INT), intent(INOUT) :: values(:)
        integer(C_INT) :: len
        len = size(values,kind=C_INT)
        ! splicer begin function.increment_int_array
        call c_increment_int_array(values, len)
        ! splicer end function.increment_int_array
    end subroutine increment_int_array

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module pointers_mod
