! top.f
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

        subroutine f_c_name_instantiation1(arg1, arg2) &
                bind(C, name="c_name_instantiation1")
            use iso_c_binding, only : C_INT, C_LONG
            implicit none
            integer(C_INT), value, intent(IN) :: arg1
            integer(C_LONG), value, intent(IN) :: arg2
        end subroutine f_c_name_instantiation1

        subroutine f_c_name_instantiation2(arg1, arg2) &
                bind(C, name="c_name_instantiation2")
            use iso_c_binding, only : C_DOUBLE, C_FLOAT
            implicit none
            real(C_FLOAT), value, intent(IN) :: arg1
            real(C_DOUBLE), value, intent(IN) :: arg2
        end subroutine f_c_name_instantiation2

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface function_tu
        module procedure f_name_instantiation1
        module procedure f_name_instantiation2
    end interface function_tu

    interface generic3
        module procedure F_name_function3a_int
        module procedure F_name_function3a_long
    end interface generic3

contains

    ! void function1()
    subroutine testnames_function1()
        ! splicer begin function.function1
        call yyy_tes_function1()
        ! splicer end function.function1
    end subroutine testnames_function1

    ! void function2()
    subroutine f_name_special()
        ! splicer begin function.function2
        call f_c_name_special()
        ! splicer end function.function2
    end subroutine f_name_special

    ! void function3a(int i +intent(in)+value)
    subroutine F_name_function3a_int(i)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: i
        ! splicer begin function.function3a_0
        call yyy_tes_function3a_0(i)
        ! splicer end function.function3a_0
    end subroutine F_name_function3a_int

    ! void function3a(long i +intent(in)+value)
    subroutine F_name_function3a_long(i)
        use iso_c_binding, only : C_LONG
        integer(C_LONG), value, intent(IN) :: i
        ! splicer begin function.function3a_1
        call yyy_tes_function3a_1(i)
        ! splicer end function.function3a_1
    end subroutine F_name_function3a_long

    ! int function4(const std::string & rv +intent(in))
    ! arg_to_buffer
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
    subroutine testnames_fiveplus()
        ! splicer begin function.fiveplus
        call yyy_tes_fiveplus()
        ! splicer end function.fiveplus
    end subroutine testnames_fiveplus

    ! void init_ns1()
    subroutine testnames_init_ns1()
        ! splicer begin function.init_ns1
        call c_init_ns1()
        ! splicer end function.init_ns1
    end subroutine testnames_init_ns1

    ! void FunctionTU(int arg1 +intent(in)+value, long arg2 +intent(in)+value)
    ! cxx_template
    !>
    !! \brief Function template with two template parameters.
    !!
    !<
    subroutine f_name_instantiation1(arg1, arg2)
        use iso_c_binding, only : C_INT, C_LONG
        integer(C_INT), value, intent(IN) :: arg1
        integer(C_LONG), value, intent(IN) :: arg2
        ! splicer begin function.function_tu_0
        call f_c_name_instantiation1(arg1, arg2)
        ! splicer end function.function_tu_0
    end subroutine f_name_instantiation1

    ! void FunctionTU(float arg1 +intent(in)+value, double arg2 +intent(in)+value)
    ! cxx_template
    !>
    !! \brief Function template with two template parameters.
    !!
    !<
    subroutine f_name_instantiation2(arg1, arg2)
        use iso_c_binding, only : C_DOUBLE, C_FLOAT
        real(C_FLOAT), value, intent(IN) :: arg1
        real(C_DOUBLE), value, intent(IN) :: arg2
        ! splicer begin function.function_tu_1
        call f_c_name_instantiation2(arg1, arg2)
        ! splicer end function.function_tu_1
    end subroutine f_name_instantiation2

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module top_module
