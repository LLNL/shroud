! top.f
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
!! \file top.f
!! \brief Shroud generated wrapper for testnames library
!<
! splicer begin file_top
! splicer end file_top
module top_module
    use iso_c_binding, only : C_INT
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    !  Color
    integer(C_INT), parameter :: color_red = 0
    integer(C_INT), parameter :: color_blue = 1
    integer(C_INT), parameter :: color_white = 2

    interface

        subroutine yyy_tes_function1() &
                bind(C, name="YYY_TES_function1")
            implicit none
        end subroutine yyy_tes_function1

        subroutine f_c_name_special() &
                bind(C, name="c_name_special")
            implicit none
        end subroutine f_c_name_special

        subroutine yyy_tes_function3a_0(i) &
                bind(C, name="YYY_TES_function3a_0")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: i
        end subroutine yyy_tes_function3a_0

        subroutine yyy_tes_function3a_1(i) &
                bind(C, name="YYY_TES_function3a_1")
            use iso_c_binding, only : C_LONG
            implicit none
            integer(C_LONG), value, intent(IN) :: i
        end subroutine yyy_tes_function3a_1

        function yyy_tes_function4(rv) &
                result(SHT_rv) &
                bind(C, name="YYY_TES_function4")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: rv(*)
            integer(C_INT) :: SHT_rv
        end function yyy_tes_function4

        function yyy_tes_function4_bufferify(rv, Lrv) &
                result(SHT_rv) &
                bind(C, name="YYY_TES_function4_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: rv(*)
            integer(C_INT), value, intent(IN) :: Lrv
            integer(C_INT) :: SHT_rv
        end function yyy_tes_function4_bufferify

        subroutine yyy_tes_fiveplus() &
                bind(C, name="YYY_TES_fiveplus")
            implicit none
        end subroutine yyy_tes_fiveplus

        subroutine c_init_ns1() &
                bind(C, name="TES_init_ns1")
            implicit none
        end subroutine c_init_ns1

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface generic3
        module procedure F_name_function3a_int
        module procedure F_name_function3a_long
    end interface generic3

contains

    ! void function1()
    ! function_index=2
    subroutine testnames_function1()
        ! splicer begin function.function1
        call yyy_tes_function1()
        ! splicer end function.function1
    end subroutine testnames_function1

    ! void function2()
    ! function_index=3
    subroutine f_name_special()
        ! splicer begin function.function2
        call f_c_name_special()
        ! splicer end function.function2
    end subroutine f_name_special

    ! void function3a(int i +intent(in)+value)
    ! function_index=4
    subroutine F_name_function3a_int(i)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: i
        ! splicer begin function.function3a_0
        call yyy_tes_function3a_0(i)
        ! splicer end function.function3a_0
    end subroutine F_name_function3a_int

    ! void function3a(long i +intent(in)+value)
    ! function_index=5
    subroutine F_name_function3a_long(i)
        use iso_c_binding, only : C_LONG
        integer(C_LONG), value, intent(IN) :: i
        ! splicer begin function.function3a_1
        call yyy_tes_function3a_1(i)
        ! splicer end function.function3a_1
    end subroutine F_name_function3a_long

    ! int function4(const std::string & rv +intent(in))
    ! arg_to_buffer
    ! function_index=6
    function testnames_function4(rv) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        character(len=*), intent(IN) :: rv
        integer(C_INT) :: SHT_rv
        ! splicer begin function.function4
        SHT_rv = yyy_tes_function4_bufferify(rv, &
            len_trim(rv, kind=C_INT))
        ! splicer end function.function4
    end function testnames_function4

    ! void function5() +name(fiveplus)
    ! function_index=7
    subroutine testnames_fiveplus()
        ! splicer begin function.fiveplus
        call yyy_tes_fiveplus()
        ! splicer end function.fiveplus
    end subroutine testnames_fiveplus

    ! void init_ns1()
    ! function_index=8
    subroutine testnames_init_ns1()
        ! splicer begin function.init_ns1
        call c_init_ns1()
        ! splicer end function.init_ns1
    end subroutine testnames_init_ns1

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module top_module
