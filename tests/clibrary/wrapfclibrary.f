! wrapfclibrary.f
! This is generated code, do not edit
! #######################################################################
! Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
!! \file wrapfclibrary.f
!! \brief Shroud generated wrapper for Clibrary library
!<
! splicer begin file_top
! splicer end file_top
module clibrary_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    interface

        subroutine function1() &
                bind(C, name="Function1")
            implicit none
        end subroutine function1

        function function2(arg1, arg2) &
                result(SHT_rv) &
                bind(C, name="Function2")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg1
            integer(C_INT), value, intent(IN) :: arg2
            real(C_DOUBLE) :: SHT_rv
        end function function2

        subroutine c_sum(len, values, result) &
                bind(C, name="Sum")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: len
            integer(C_INT), intent(IN) :: values(*)
            integer(C_INT), intent(OUT) :: result
        end subroutine c_sum

        function c_function3(arg) &
                result(SHT_rv) &
                bind(C, name="Function3")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL), value, intent(IN) :: arg
            logical(C_BOOL) :: SHT_rv
        end function c_function3

        subroutine c_function3b(arg1, arg2, arg3) &
                bind(C, name="Function3b")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL), value, intent(IN) :: arg1
            logical(C_BOOL), intent(OUT) :: arg2
            logical(C_BOOL), intent(INOUT) :: arg3
        end subroutine c_function3b

        function c_function4a(arg1, arg2) &
                result(SHT_rv) &
                bind(C, name="Function4a")
            use iso_c_binding, only : C_CHAR, C_PTR
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg1(*)
            character(kind=C_CHAR), intent(IN) :: arg2(*)
            type(C_PTR) SHT_rv
        end function c_function4a

        subroutine c_function4a_bufferify(arg1, Larg1, arg2, Larg2, &
                SHF_rv, NSHF_rv) &
                bind(C, name="CLI_function4a_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg1(*)
            integer(C_INT), value, intent(IN) :: Larg1
            character(kind=C_CHAR), intent(IN) :: arg2(*)
            integer(C_INT), value, intent(IN) :: Larg2
            character(kind=C_CHAR), intent(OUT) :: SHF_rv(*)
            integer(C_INT), value, intent(IN) :: NSHF_rv
        end subroutine c_function4a_bufferify

        subroutine intargs(argin, arginout, argout) &
                bind(C, name="intargs")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: argin
            integer(C_INT), intent(INOUT) :: arginout
            integer(C_INT), intent(OUT) :: argout
        end subroutine intargs

        subroutine c_cos_doubles(in, out, sizein) &
                bind(C, name="cos_doubles")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), intent(IN) :: in(*)
            real(C_DOUBLE), intent(OUT) :: out(*)
            integer(C_INT), value, intent(IN) :: sizein
        end subroutine c_cos_doubles

        subroutine c_truncate_to_int(in, out, sizein) &
                bind(C, name="truncate_to_int")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), intent(IN) :: in(*)
            integer(C_INT), intent(OUT) :: out(*)
            integer(C_INT), value, intent(IN) :: sizein
        end subroutine c_truncate_to_int

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! void Sum(int len +implied(size(values))+intent(in)+value, int * values +dimension(:)+intent(in), int * result +intent(out))
    ! function_index=2
    subroutine sum(values, result)
        use iso_c_binding, only : C_INT
        integer(C_INT) :: len
        integer(C_INT), intent(IN) :: values(:)
        integer(C_INT), intent(OUT) :: result
        len = size(values)
        ! splicer begin function.sum
        call c_sum(len, values, result)
        ! splicer end function.sum
    end subroutine sum

    ! bool Function3(bool arg +intent(in)+value)
    ! function_index=3
    function function3(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        logical, value, intent(IN) :: arg
        logical(C_BOOL) SH_arg
        logical :: SHT_rv
        SH_arg = arg  ! coerce to C_BOOL
        ! splicer begin function.function3
        SHT_rv = c_function3(SH_arg)
        ! splicer end function.function3
    end function function3

    ! void Function3b(const bool arg1 +intent(in)+value, bool * arg2 +intent(out), bool * arg3 +intent(inout))
    ! function_index=4
    subroutine function3b(arg1, arg2, arg3)
        use iso_c_binding, only : C_BOOL
        logical, value, intent(IN) :: arg1
        logical(C_BOOL) SH_arg1
        logical, intent(OUT) :: arg2
        logical(C_BOOL) SH_arg2
        logical, intent(INOUT) :: arg3
        logical(C_BOOL) SH_arg3
        SH_arg1 = arg1  ! coerce to C_BOOL
        SH_arg3 = arg3  ! coerce to C_BOOL
        ! splicer begin function.function3b
        call c_function3b(SH_arg1, SH_arg2, SH_arg3)
        ! splicer end function.function3b
        arg2 = SH_arg2  ! coerce to logical
        arg3 = SH_arg3  ! coerce to logical
    end subroutine function3b

    ! char * Function4a +len(30)(const char * arg1 +intent(in), const char * arg2 +intent(in))
    ! arg_to_buffer
    ! function_index=5
    function function4a(arg1, arg2) &
            result(SHT_rv)
        use iso_c_binding, only : C_CHAR, C_INT
        character(*), intent(IN) :: arg1
        character(*), intent(IN) :: arg2
        character(kind=C_CHAR, len=30) :: SHT_rv
        ! splicer begin function.function4a
        call c_function4a_bufferify(arg1, len_trim(arg1, kind=C_INT), &
            arg2, len_trim(arg2, kind=C_INT), SHT_rv, &
            len(SHT_rv, kind=C_INT))
        ! splicer end function.function4a
    end function function4a

    ! void cos_doubles(double * in +dimension(:)+intent(in), double * out +allocatable(mold=in)+intent(out), int sizein +implied(size(in))+intent(in)+value)
    ! function_index=7
    subroutine cos_doubles(in, out)
        use iso_c_binding, only : C_DOUBLE, C_INT
        real(C_DOUBLE), intent(IN) :: in(:)
        real(C_DOUBLE), intent(OUT), allocatable :: out(:)
        integer(C_INT) :: sizein
        allocate(out(lbound(in,1):ubound(in,1)))
        sizein = size(in)
        ! splicer begin function.cos_doubles
        call c_cos_doubles(in, out, sizein)
        ! splicer end function.cos_doubles
    end subroutine cos_doubles

    ! void truncate_to_int(double * in +dimension(:)+intent(in), int * out +allocatable(mold=in)+intent(out), int sizein +implied(size(in))+intent(in)+value)
    ! function_index=8
    !>
    !! \brief truncate IN argument
    !!
    !! Different types for in and out.
    !<
    subroutine truncate_to_int(in, out)
        use iso_c_binding, only : C_DOUBLE, C_INT
        real(C_DOUBLE), intent(IN) :: in(:)
        integer(C_INT), intent(OUT), allocatable :: out(:)
        integer(C_INT) :: sizein
        allocate(out(lbound(in,1):ubound(in,1)))
        sizein = size(in)
        ! splicer begin function.truncate_to_int
        call c_truncate_to_int(in, out, sizein)
        ! splicer end function.truncate_to_int
    end subroutine truncate_to_int

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module clibrary_mod
