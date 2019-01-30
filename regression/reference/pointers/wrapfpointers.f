! wrapfpointers.f
! This is generated code, do not edit
! #######################################################################
! Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
! Produced at the Lawrence Livermore National Laboratory
!
! LLNL-CODE-738041.
! All rights reserved.
!
! This file is part of Shroud.  For details, see
! https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are
! met:
!
! * Redistributions of source code must retain the above copyright
!   notice, this list of conditions and the disclaimer below.
!
! * Redistributions in binary form must reproduce the above copyright
!   notice, this list of conditions and the disclaimer (as noted below)
!   in the documentation and/or other materials provided with the
!   distribution.
!
! * Neither the name of the LLNS/LLNL nor the names of its contributors
!   may be used to endorse or promote products derived from this
!   software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
! A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
! LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
! EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
! PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
! LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
! NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
! SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module pointers_mod
