! wrapftutorial.f
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
!! \file wrapftutorial.f
!! \brief Shroud generated wrapper for Tutorial library
!<
! splicer begin file_top
! splicer end file_top
module tutorial_mod
    use iso_c_binding, only : C_DOUBLE, C_INT, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

    type, bind(C) :: SHROUD_array
        type(SHROUD_capsule_data) :: cxx       ! address of C++ memory
        type(C_PTR) :: addr = C_NULL_PTR       ! address of data in cxx
        integer(C_SIZE_T) :: len = 0_C_SIZE_T  ! bytes-per-item or character len of data in cxx
        integer(C_SIZE_T) :: size = 0_C_SIZE_T ! size of data in cxx
    end type SHROUD_array

    !  DIRECTION
    integer(C_INT), parameter :: class1_direction_up = 2
    integer(C_INT), parameter :: class1_direction_down = 3
    integer(C_INT), parameter :: class1_direction_left = 100
    integer(C_INT), parameter :: class1_direction_right = 101

    !  Color
    integer(C_INT), parameter :: color_red = 0
    integer(C_INT), parameter :: color_blue = 1
    integer(C_INT), parameter :: color_white = 2


    type, bind(C) :: struct1
        integer(C_INT) :: ifield
        real(C_DOUBLE) :: dfield
    end type struct1

    ! splicer begin class.Class1.module_top
    ! splicer end class.Class1.module_top

    type class1
        type(SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.Class1.component_part
        ! splicer end class.Class1.component_part
    contains
        procedure :: delete => class1_delete
        procedure :: method1 => class1_method1
        procedure :: equivalent => class1_equivalent
        procedure :: return_this => class1_return_this
        procedure :: return_this_buffer => class1_return_this_buffer
        procedure :: direction_func => class1_direction_func
        procedure :: get_m_flag => class1_get_m_flag
        procedure :: get_test => class1_get_test
        procedure :: set_test => class1_set_test
        procedure :: get_instance => class1_get_instance
        procedure :: set_instance => class1_set_instance
        procedure :: associated => class1_associated
        ! splicer begin class.Class1.type_bound_procedure_part
        ! splicer end class.Class1.type_bound_procedure_part
    end type class1

    ! splicer begin class.Singleton.module_top
    ! splicer end class.Singleton.module_top

    type singleton
        type(SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.Singleton.component_part
        ! splicer end class.Singleton.component_part
    contains
        procedure, nopass :: get_reference => singleton_get_reference
        procedure :: get_instance => singleton_get_instance
        procedure :: set_instance => singleton_set_instance
        procedure :: associated => singleton_associated
        ! splicer begin class.Singleton.type_bound_procedure_part
        ! splicer end class.Singleton.type_bound_procedure_part
    end type singleton

    interface operator (.eq.)
        module procedure class1_eq
        module procedure singleton_eq
    end interface

    interface operator (.ne.)
        module procedure class1_ne
        module procedure singleton_ne
    end interface

    abstract interface

        function callback1_incr(arg0) bind(C)
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value :: arg0
            integer(C_INT) :: callback1_incr
        end function callback1_incr

    end interface

    interface

        function c_class1_new_default() &
                result(SHT_rv) &
                bind(C, name="TUT_class1_new_default")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_class1_new_default

        function c_class1_new_flag(flag) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_new_flag")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            integer(C_INT), value, intent(IN) :: flag
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_class1_new_flag

        subroutine c_class1_delete(self) &
                bind(C, name="TUT_class1_delete")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_class1_delete

        function c_class1_method1(self) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_method1")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_class1_method1

        pure function c_class1_equivalent(self, obj2) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_equivalent")
            use iso_c_binding, only : C_BOOL
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            type(SHROUD_capsule_data), intent(IN) :: obj2
            logical(C_BOOL) :: SHT_rv
        end function c_class1_equivalent

        subroutine c_class1_return_this(self) &
                bind(C, name="TUT_class1_return_this")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_class1_return_this

        function c_class1_return_this_buffer(self, name, flag) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_return_this_buffer")
            use iso_c_binding, only : C_BOOL, C_CHAR
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            logical(C_BOOL), value, intent(IN) :: flag
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_class1_return_this_buffer

        function c_class1_return_this_buffer_bufferify(self, name, &
                Lname, flag) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_return_this_buffer_bufferify")
            use iso_c_binding, only : C_BOOL, C_CHAR, C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            logical(C_BOOL), value, intent(IN) :: flag
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_class1_return_this_buffer_bufferify

        function c_class1_direction_func(self, arg) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_direction_func")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: arg
            integer(C_INT) :: SHT_rv
        end function c_class1_direction_func

        function c_class1_get_m_flag(self) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_get_m_flag")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_class1_get_m_flag

        function c_class1_get_test(self) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_get_test")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_class1_get_test

        subroutine c_class1_set_test(self, val) &
                bind(C, name="TUT_class1_set_test")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: val
        end subroutine c_class1_set_test

        ! splicer begin class.Class1.additional_interfaces
        ! splicer end class.Class1.additional_interfaces

        function c_singleton_get_reference() &
                result(SHT_rv) &
                bind(C, name="TUT_singleton_get_reference")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_singleton_get_reference

        ! splicer begin class.Singleton.additional_interfaces
        ! splicer end class.Singleton.additional_interfaces

        subroutine function1() &
                bind(C, name="TUT_function1")
            implicit none
        end subroutine function1

        function function2(arg1, arg2) &
                result(SHT_rv) &
                bind(C, name="TUT_function2")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg1
            integer(C_INT), value, intent(IN) :: arg2
            real(C_DOUBLE) :: SHT_rv
        end function function2

        subroutine c_sum(len, values, result) &
                bind(C, name="TUT_sum")
            use iso_c_binding, only : C_INT, C_SIZE_T
            implicit none
            integer(C_SIZE_T), value, intent(IN) :: len
            integer(C_INT), intent(IN) :: values(*)
            integer(C_INT), intent(OUT) :: result
        end subroutine c_sum

        function type_long_long(arg1) &
                result(SHT_rv) &
                bind(C, name="TUT_type_long_long")
            use iso_c_binding, only : C_LONG_LONG
            implicit none
            integer(C_LONG_LONG), value, intent(IN) :: arg1
            integer(C_LONG_LONG) :: SHT_rv
        end function type_long_long

        function c_function3(arg) &
                result(SHT_rv) &
                bind(C, name="TUT_function3")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL), value, intent(IN) :: arg
            logical(C_BOOL) :: SHT_rv
        end function c_function3

        subroutine c_function3b(arg1, arg2, arg3) &
                bind(C, name="TUT_function3b")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL), value, intent(IN) :: arg1
            logical(C_BOOL), intent(OUT) :: arg2
            logical(C_BOOL), intent(INOUT) :: arg3
        end subroutine c_function3b

        subroutine c_function4a_bufferify(arg1, Larg1, arg2, Larg2, &
                DSHF_rv) &
                bind(C, name="TUT_function4a_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_array
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg1(*)
            integer(C_INT), value, intent(IN) :: Larg1
            character(kind=C_CHAR), intent(IN) :: arg2(*)
            integer(C_INT), value, intent(IN) :: Larg2
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_function4a_bufferify

        function c_function4b(arg1, arg2) &
                result(SHT_rv) &
                bind(C, name="TUT_function4b")
            use iso_c_binding, only : C_CHAR, C_PTR
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg1(*)
            character(kind=C_CHAR), intent(IN) :: arg2(*)
            type(C_PTR) SHT_rv
        end function c_function4b

        subroutine c_function4b_bufferify(arg1, Larg1, arg2, Larg2, &
                output, Noutput) &
                bind(C, name="TUT_function4b_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg1(*)
            integer(C_INT), value, intent(IN) :: Larg1
            character(kind=C_CHAR), intent(IN) :: arg2(*)
            integer(C_INT), value, intent(IN) :: Larg2
            character(kind=C_CHAR), intent(OUT) :: output(*)
            integer(C_INT), value, intent(IN) :: Noutput
        end subroutine c_function4b_bufferify

        function c_function4c(arg1, arg2) &
                result(SHT_rv) &
                bind(C, name="TUT_function4c")
            use iso_c_binding, only : C_CHAR, C_PTR
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg1(*)
            character(kind=C_CHAR), intent(IN) :: arg2(*)
            type(C_PTR) SHT_rv
        end function c_function4c

        subroutine c_function4c_bufferify(arg1, Larg1, arg2, Larg2, &
                DSHF_rv) &
                bind(C, name="TUT_function4c_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_array
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg1(*)
            integer(C_INT), value, intent(IN) :: Larg1
            character(kind=C_CHAR), intent(IN) :: arg2(*)
            integer(C_INT), value, intent(IN) :: Larg2
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_function4c_bufferify

        function c_function5() &
                result(SHT_rv) &
                bind(C, name="TUT_function5")
            use iso_c_binding, only : C_DOUBLE
            implicit none
            real(C_DOUBLE) :: SHT_rv
        end function c_function5

        function c_function5_arg1(arg1) &
                result(SHT_rv) &
                bind(C, name="TUT_function5_arg1")
            use iso_c_binding, only : C_DOUBLE
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg1
            real(C_DOUBLE) :: SHT_rv
        end function c_function5_arg1

        function c_function5_arg1_arg2(arg1, arg2) &
                result(SHT_rv) &
                bind(C, name="TUT_function5_arg1_arg2")
            use iso_c_binding, only : C_BOOL, C_DOUBLE
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg1
            logical(C_BOOL), value, intent(IN) :: arg2
            real(C_DOUBLE) :: SHT_rv
        end function c_function5_arg1_arg2

        subroutine c_function6_from_name(name) &
                bind(C, name="TUT_function6_from_name")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_function6_from_name

        subroutine c_function6_from_name_bufferify(name, Lname) &
                bind(C, name="TUT_function6_from_name_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
        end subroutine c_function6_from_name_bufferify

        subroutine c_function6_from_index(indx) &
                bind(C, name="TUT_function6_from_index")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: indx
        end subroutine c_function6_from_index

        subroutine c_function7_int(arg) &
                bind(C, name="TUT_function7_int")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg
        end subroutine c_function7_int

        subroutine c_function7_double(arg) &
                bind(C, name="TUT_function7_double")
            use iso_c_binding, only : C_DOUBLE
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg
        end subroutine c_function7_double

        function c_function8_int() &
                result(SHT_rv) &
                bind(C, name="TUT_function8_int")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT) :: SHT_rv
        end function c_function8_int

        function c_function8_double() &
                result(SHT_rv) &
                bind(C, name="TUT_function8_double")
            use iso_c_binding, only : C_DOUBLE
            implicit none
            real(C_DOUBLE) :: SHT_rv
        end function c_function8_double

        subroutine c_function9(arg) &
                bind(C, name="TUT_function9")
            use iso_c_binding, only : C_DOUBLE
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg
        end subroutine c_function9

        subroutine c_function10_0() &
                bind(C, name="TUT_function10_0")
            implicit none
        end subroutine c_function10_0

        subroutine c_function10_1(name, arg2) &
                bind(C, name="TUT_function10_1")
            use iso_c_binding, only : C_CHAR, C_DOUBLE
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            real(C_DOUBLE), value, intent(IN) :: arg2
        end subroutine c_function10_1

        subroutine c_function10_1_bufferify(name, Lname, arg2) &
                bind(C, name="TUT_function10_1_bufferify")
            use iso_c_binding, only : C_CHAR, C_DOUBLE, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            real(C_DOUBLE), value, intent(IN) :: arg2
        end subroutine c_function10_1_bufferify

        function c_overload1_num(num) &
                result(SHT_rv) &
                bind(C, name="TUT_overload1_num")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: num
            integer(C_INT) :: SHT_rv
        end function c_overload1_num

        function c_overload1_num_offset(num, offset) &
                result(SHT_rv) &
                bind(C, name="TUT_overload1_num_offset")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: num
            integer(C_INT), value, intent(IN) :: offset
            integer(C_INT) :: SHT_rv
        end function c_overload1_num_offset

        function c_overload1_num_offset_stride(num, offset, stride) &
                result(SHT_rv) &
                bind(C, name="TUT_overload1_num_offset_stride")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: num
            integer(C_INT), value, intent(IN) :: offset
            integer(C_INT), value, intent(IN) :: stride
            integer(C_INT) :: SHT_rv
        end function c_overload1_num_offset_stride

        function c_overload1_3(type, num) &
                result(SHT_rv) &
                bind(C, name="TUT_overload1_3")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), value, intent(IN) :: type
            integer(C_INT), value, intent(IN) :: num
            integer(C_INT) :: SHT_rv
        end function c_overload1_3

        function c_overload1_4(type, num, offset) &
                result(SHT_rv) &
                bind(C, name="TUT_overload1_4")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), value, intent(IN) :: type
            integer(C_INT), value, intent(IN) :: num
            integer(C_INT), value, intent(IN) :: offset
            integer(C_INT) :: SHT_rv
        end function c_overload1_4

        function c_overload1_5(type, num, offset, stride) &
                result(SHT_rv) &
                bind(C, name="TUT_overload1_5")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), value, intent(IN) :: type
            integer(C_INT), value, intent(IN) :: num
            integer(C_INT), value, intent(IN) :: offset
            integer(C_INT), value, intent(IN) :: stride
            integer(C_INT) :: SHT_rv
        end function c_overload1_5

        function typefunc(arg) &
                result(SHT_rv) &
                bind(C, name="TUT_typefunc")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg
            integer(C_INT) :: SHT_rv
        end function typefunc

        function enumfunc(arg) &
                result(SHT_rv) &
                bind(C, name="TUT_enumfunc")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg
            integer(C_INT) :: SHT_rv
        end function enumfunc

        function colorfunc(arg) &
                result(SHT_rv) &
                bind(C, name="TUT_colorfunc")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg
            integer(C_INT) :: SHT_rv
        end function colorfunc

        subroutine get_min_max(min, max) &
                bind(C, name="TUT_get_min_max")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(OUT) :: min
            integer(C_INT), intent(OUT) :: max
        end subroutine get_min_max

        function direction_func(arg) &
                result(SHT_rv) &
                bind(C, name="TUT_direction_func")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg
            integer(C_INT) :: SHT_rv
        end function direction_func

        function c_useclass(arg1) &
                result(SHT_rv) &
                bind(C, name="TUT_useclass")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: arg1
            integer(C_INT) :: SHT_rv
        end function c_useclass

        function c_getclass2() &
                result(SHT_rv) &
                bind(C, name="TUT_getclass2")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_getclass2

        function c_getclass3() &
                result(SHT_rv) &
                bind(C, name="TUT_getclass3")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_getclass3

        function c_get_class_copy(flag) &
                result(SHT_rv) &
                bind(C, name="TUT_get_class_copy")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            integer(C_INT), value, intent(IN) :: flag
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_get_class_copy

        function callback1(in, incr) &
                result(SHT_rv) &
                bind(C, name="TUT_callback1")
            use iso_c_binding, only : C_INT
            import :: callback1_incr
            implicit none
            integer(C_INT), value, intent(IN) :: in
            procedure(callback1_incr) :: incr
            integer(C_INT) :: SHT_rv
        end function callback1

        function return_struct(i, d) &
                result(SHT_rv) &
                bind(C, name="TUT_return_struct")
            use iso_c_binding, only : C_DOUBLE, C_INT
            import :: struct1
            implicit none
            integer(C_INT), value, intent(IN) :: i
            real(C_DOUBLE), value, intent(IN) :: d
            type(struct1) :: SHT_rv
        end function return_struct

        function c_return_struct_ptr(i, d) &
                result(SHT_rv) &
                bind(C, name="TUT_return_struct_ptr")
            use iso_c_binding, only : C_DOUBLE, C_INT, C_PTR
            import :: struct1
            implicit none
            integer(C_INT), value, intent(IN) :: i
            real(C_DOUBLE), value, intent(IN) :: d
            type(C_PTR) SHT_rv
        end function c_return_struct_ptr

        function accept_struct_in(arg) &
                result(SHT_rv) &
                bind(C, name="TUT_accept_struct_in")
            use iso_c_binding, only : C_DOUBLE
            import :: struct1
            implicit none
            type(struct1), value, intent(IN) :: arg
            real(C_DOUBLE) :: SHT_rv
        end function accept_struct_in

        function accept_struct_in_ptr(arg) &
                result(SHT_rv) &
                bind(C, name="TUT_accept_struct_in_ptr")
            use iso_c_binding, only : C_DOUBLE
            import :: struct1
            implicit none
            type(struct1), intent(IN) :: arg
            real(C_DOUBLE) :: SHT_rv
        end function accept_struct_in_ptr

        subroutine accept_struct_out_ptr(arg, i, d) &
                bind(C, name="TUT_accept_struct_out_ptr")
            use iso_c_binding, only : C_DOUBLE, C_INT
            import :: struct1
            implicit none
            type(struct1), intent(OUT) :: arg
            integer(C_INT), value, intent(IN) :: i
            real(C_DOUBLE), value, intent(IN) :: d
        end subroutine accept_struct_out_ptr

        subroutine accept_struct_in_out_ptr(arg) &
                bind(C, name="TUT_accept_struct_in_out_ptr")
            import :: struct1
            implicit none
            type(struct1), intent(INOUT) :: arg
        end subroutine accept_struct_in_out_ptr

        function c_last_function_called() &
                result(SHT_rv) &
                bind(C, name="TUT_last_function_called")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR) SHT_rv
        end function c_last_function_called

        subroutine c_last_function_called_bufferify(SHF_rv, NSHF_rv) &
                bind(C, name="TUT_last_function_called_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(OUT) :: SHF_rv(*)
            integer(C_INT), value, intent(IN) :: NSHF_rv
        end subroutine c_last_function_called_bufferify

        ! splicer begin additional_interfaces
        subroutine all_test1(array)
          implicit none
          integer, dimension(:), allocatable :: array
        end subroutine all_test1
        ! splicer end additional_interfaces
    end interface

    interface class1_new
        module procedure class1_new_default
        module procedure class1_new_flag
    end interface class1_new

    interface function10
        module procedure function10_0
        module procedure function10_1_float
        module procedure function10_1_double
    end interface function10

    interface function5
        module procedure function5
        module procedure function5_arg1
        module procedure function5_arg1_arg2
    end interface function5

    interface function6
        module procedure function6_from_name
        module procedure function6_from_index
    end interface function6

    interface function7
        module procedure function7_int
        module procedure function7_double
    end interface function7

    interface function9
        module procedure function9_float
        module procedure function9_double
    end interface function9

    interface overload1
        module procedure overload1_num
        module procedure overload1_num_offset
        module procedure overload1_num_offset_stride
        module procedure overload1_3
        module procedure overload1_4
        module procedure overload1_5
    end interface overload1

    interface
        ! Copy the std::string in context into c_var.
        subroutine SHROUD_copy_string_and_free(context, c_var, c_var_size) &
             bind(c,name="TUT_ShroudCopyStringAndFree")
            use, intrinsic :: iso_c_binding, only : C_CHAR, C_LONG
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            character(kind=C_CHAR), intent(OUT) :: c_var(*)
            integer(C_LONG), value :: c_var_size
        end subroutine SHROUD_copy_string_and_free
    end interface

contains

    ! Class1() +name(new)
    ! function_index=0
    function class1_new_default() &
            result(SHT_rv)
        type(class1) :: SHT_rv
        ! splicer begin class.Class1.method.new_default
        SHT_rv%cxxmem = c_class1_new_default()
        ! splicer end class.Class1.method.new_default
    end function class1_new_default

    ! Class1(int flag +intent(in)+value) +name(new)
    ! function_index=1
    function class1_new_flag(flag) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: flag
        type(class1) :: SHT_rv
        ! splicer begin class.Class1.method.new_flag
        SHT_rv%cxxmem = c_class1_new_flag(flag)
        ! splicer end class.Class1.method.new_flag
    end function class1_new_flag

    ! ~Class1() +name(delete)
    ! function_index=2
    subroutine class1_delete(obj)
        use iso_c_binding, only : C_NULL_PTR
        class(class1) :: obj
        ! splicer begin class.Class1.method.delete
        call c_class1_delete(obj%cxxmem)
        obj%cxxmem%addr = C_NULL_PTR
        ! splicer end class.Class1.method.delete
    end subroutine class1_delete

    ! int Method1()
    ! function_index=3
    !>
    !! \brief returns the value of flag member
    !!
    !<
    function class1_method1(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.method1
        SHT_rv = c_class1_method1(obj%cxxmem)
        ! splicer end class.Class1.method.method1
    end function class1_method1

    ! bool equivalent(const Class1 & obj2 +intent(in)) const
    ! function_index=4
    !>
    !! \brief Pass in reference to instance
    !!
    !<
    function class1_equivalent(obj, obj2) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        class(class1) :: obj
        type(class1), intent(IN) :: obj2
        logical :: SHT_rv
        ! splicer begin class.Class1.method.equivalent
        SHT_rv = c_class1_equivalent(obj%cxxmem, obj2%cxxmem)
        ! splicer end class.Class1.method.equivalent
    end function class1_equivalent

    ! Class1 * returnThis()
    ! function_index=5
    !>
    !! \brief Return pointer to 'this' to allow chaining calls
    !!
    !<
    subroutine class1_return_this(obj)
        class(class1) :: obj
        ! splicer begin class.Class1.method.return_this
        call c_class1_return_this(obj%cxxmem)
        ! splicer end class.Class1.method.return_this
    end subroutine class1_return_this

    ! Class1 * returnThisBuffer(std::string & name +intent(in), bool flag +intent(in)+value)
    ! arg_to_buffer
    ! function_index=6
    !>
    !! \brief Return pointer to 'this' to allow chaining calls
    !!
    !<
    function class1_return_this_buffer(obj, name, flag) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_INT
        class(class1) :: obj
        character(len=*), intent(IN) :: name
        logical, value, intent(IN) :: flag
        logical(C_BOOL) SH_flag
        type(class1) :: SHT_rv
        SH_flag = flag  ! coerce to C_BOOL
        ! splicer begin class.Class1.method.return_this_buffer
        SHT_rv%cxxmem = c_class1_return_this_buffer_bufferify(obj%cxxmem, &
            name, len_trim(name, kind=C_INT), SH_flag)
        ! splicer end class.Class1.method.return_this_buffer
    end function class1_return_this_buffer

    ! DIRECTION directionFunc(DIRECTION arg +intent(in)+value)
    ! function_index=7
    function class1_direction_func(obj, arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT), value, intent(IN) :: arg
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.direction_func
        SHT_rv = c_class1_direction_func(obj%cxxmem, arg)
        ! splicer end class.Class1.method.direction_func
    end function class1_direction_func

    ! int getM_flag()
    ! function_index=8
    function class1_get_m_flag(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.get_m_flag
        SHT_rv = c_class1_get_m_flag(obj%cxxmem)
        ! splicer end class.Class1.method.get_m_flag
    end function class1_get_m_flag

    ! int getTest()
    ! function_index=9
    function class1_get_test(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.get_test
        SHT_rv = c_class1_get_test(obj%cxxmem)
        ! splicer end class.Class1.method.get_test
    end function class1_get_test

    ! void setTest(int val +intent(in)+value)
    ! function_index=10
    subroutine class1_set_test(obj, val)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT), value, intent(IN) :: val
        ! splicer begin class.Class1.method.set_test
        call c_class1_set_test(obj%cxxmem, val)
        ! splicer end class.Class1.method.set_test
    end subroutine class1_set_test

    ! Return pointer to C++ memory.
    function class1_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: c_associated, C_NULL_PTR, C_PTR
        class(class1), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function class1_get_instance

    subroutine class1_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(class1), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine class1_set_instance

    function class1_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(class1), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function class1_associated

    ! splicer begin class.Class1.additional_functions
    ! splicer end class.Class1.additional_functions

    ! static Singleton & getReference()
    ! function_index=12
    function singleton_get_reference() &
            result(SHT_rv)
        type(singleton) :: SHT_rv
        ! splicer begin class.Singleton.method.get_reference
        SHT_rv%cxxmem = c_singleton_get_reference()
        ! splicer end class.Singleton.method.get_reference
    end function singleton_get_reference

    ! Return pointer to C++ memory.
    function singleton_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: c_associated, C_NULL_PTR, C_PTR
        class(singleton), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function singleton_get_instance

    subroutine singleton_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(singleton), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine singleton_set_instance

    function singleton_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(singleton), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function singleton_associated

    ! splicer begin class.Singleton.additional_functions
    ! splicer end class.Singleton.additional_functions

    ! void Sum(size_t len +implied(size(values))+intent(in)+value, int * values +dimension(:)+intent(in), int * result +intent(out))
    ! function_index=15
    subroutine sum(values, result)
        use iso_c_binding, only : C_INT, C_SIZE_T
        integer(C_SIZE_T) :: len
        integer(C_INT), intent(IN) :: values(:)
        integer(C_INT), intent(OUT) :: result
        len = size(values,kind=C_SIZE_T)
        ! splicer begin function.sum
        call c_sum(len, values, result)
        ! splicer end function.sum
    end subroutine sum

    ! bool Function3(bool arg +intent(in)+value)
    ! function_index=17
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
    ! function_index=18
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

    ! const std::string Function4a(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in)) +deref(allocatable)+len(30)
    ! arg_to_buffer
    ! function_index=19
    function function4a(arg1, arg2) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        character(len=*), intent(IN) :: arg1
        character(len=*), intent(IN) :: arg2
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin function.function4a
        call c_function4a_bufferify(arg1, len_trim(arg1, kind=C_INT), &
            arg2, len_trim(arg2, kind=C_INT), DSHF_rv)
        ! splicer end function.function4a
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function function4a

    ! void Function4b(const std::string & arg1 +intent(in)+len_trim(Larg1), const std::string & arg2 +intent(in)+len_trim(Larg2), std::string & output +intent(out)+len(Noutput))
    ! arg_to_buffer - arg_to_buffer
    ! function_index=61
    subroutine function4b(arg1, arg2, output)
        use iso_c_binding, only : C_INT
        character(len=*), intent(IN) :: arg1
        character(len=*), intent(IN) :: arg2
        character(len=*), intent(OUT) :: output
        ! splicer begin function.function4b
        call c_function4b_bufferify(arg1, len_trim(arg1, kind=C_INT), &
            arg2, len_trim(arg2, kind=C_INT), output, &
            len(output, kind=C_INT))
        ! splicer end function.function4b
    end subroutine function4b

    ! const std::string & Function4c(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in)) +deref(allocatable)
    ! arg_to_buffer
    ! function_index=21
    function function4c(arg1, arg2) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        character(len=*), intent(IN) :: arg1
        character(len=*), intent(IN) :: arg2
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin function.function4c
        call c_function4c_bufferify(arg1, len_trim(arg1, kind=C_INT), &
            arg2, len_trim(arg2, kind=C_INT), DSHF_rv)
        ! splicer end function.function4c
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function function4c

    ! double Function5()
    ! has_default_arg
    ! function_index=49
    function function5() &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE) :: SHT_rv
        ! splicer begin function.function5
        SHT_rv = c_function5()
        ! splicer end function.function5
    end function function5

    ! double Function5(double arg1=3.1415 +intent(in)+value)
    ! has_default_arg
    ! function_index=50
    function function5_arg1(arg1) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg1
        real(C_DOUBLE) :: SHT_rv
        ! splicer begin function.function5_arg1
        SHT_rv = c_function5_arg1(arg1)
        ! splicer end function.function5_arg1
    end function function5_arg1

    ! double Function5(double arg1=3.1415 +intent(in)+value, bool arg2=true +intent(in)+value)
    ! function_index=22
    function function5_arg1_arg2(arg1, arg2) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg1
        logical, value, intent(IN) :: arg2
        logical(C_BOOL) SH_arg2
        real(C_DOUBLE) :: SHT_rv
        SH_arg2 = arg2  ! coerce to C_BOOL
        ! splicer begin function.function5_arg1_arg2
        SHT_rv = c_function5_arg1_arg2(arg1, SH_arg2)
        ! splicer end function.function5_arg1_arg2
    end function function5_arg1_arg2

    ! void Function6(const std::string & name +intent(in))
    ! arg_to_buffer
    ! function_index=23
    subroutine function6_from_name(name)
        use iso_c_binding, only : C_INT
        character(len=*), intent(IN) :: name
        ! splicer begin function.function6_from_name
        call c_function6_from_name_bufferify(name, &
            len_trim(name, kind=C_INT))
        ! splicer end function.function6_from_name
    end subroutine function6_from_name

    ! void Function6(int indx +intent(in)+value)
    ! function_index=24
    subroutine function6_from_index(indx)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: indx
        ! splicer begin function.function6_from_index
        call c_function6_from_index(indx)
        ! splicer end function.function6_from_index
    end subroutine function6_from_index

    ! void Function7(int arg +intent(in)+value)
    ! cxx_template
    ! function_index=51
    subroutine function7_int(arg)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: arg
        ! splicer begin function.function7_int
        call c_function7_int(arg)
        ! splicer end function.function7_int
    end subroutine function7_int

    ! void Function7(double arg +intent(in)+value)
    ! cxx_template
    ! function_index=52
    subroutine function7_double(arg)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg
        ! splicer begin function.function7_double
        call c_function7_double(arg)
        ! splicer end function.function7_double
    end subroutine function7_double

    ! int Function8()
    ! cxx_template
    ! function_index=53
    function function8_int() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT) :: SHT_rv
        ! splicer begin function.function8_int
        SHT_rv = c_function8_int()
        ! splicer end function.function8_int
    end function function8_int

    ! double Function8()
    ! cxx_template
    ! function_index=54
    function function8_double() &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE) :: SHT_rv
        ! splicer begin function.function8_double
        SHT_rv = c_function8_double()
        ! splicer end function.function8_double
    end function function8_double

    ! void Function9(float arg +intent(in)+value)
    ! fortran_generic
    ! function_index=66
    subroutine function9_float(arg)
        use iso_c_binding, only : C_DOUBLE, C_FLOAT
        real(C_FLOAT), value, intent(IN) :: arg
        ! splicer begin function.function9_float
        call c_function9(real(arg, C_DOUBLE))
        ! splicer end function.function9_float
    end subroutine function9_float

    ! void Function9(double arg +intent(in)+value)
    ! fortran_generic
    ! function_index=67
    subroutine function9_double(arg)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg
        ! splicer begin function.function9_double
        call c_function9(arg)
        ! splicer end function.function9_double
    end subroutine function9_double

    ! void Function10()
    ! function_index=28
    subroutine function10_0()
        ! splicer begin function.function10_0
        call c_function10_0()
        ! splicer end function.function10_0
    end subroutine function10_0

    ! void Function10(const std::string & name +intent(in), float arg2 +intent(in)+value)
    ! fortran_generic - arg_to_buffer
    ! function_index=68
    subroutine function10_1_float(name, arg2)
        use iso_c_binding, only : C_DOUBLE, C_FLOAT, C_INT
        character(len=*), intent(IN) :: name
        real(C_FLOAT), value, intent(IN) :: arg2
        ! splicer begin function.function10_1_float
        call c_function10_1_bufferify(name, len_trim(name, kind=C_INT), &
            real(arg2, C_DOUBLE))
        ! splicer end function.function10_1_float
    end subroutine function10_1_float

    ! void Function10(const std::string & name +intent(in), double arg2 +intent(in)+value)
    ! fortran_generic - arg_to_buffer
    ! function_index=69
    subroutine function10_1_double(name, arg2)
        use iso_c_binding, only : C_DOUBLE, C_INT
        character(len=*), intent(IN) :: name
        real(C_DOUBLE), value, intent(IN) :: arg2
        ! splicer begin function.function10_1_double
        call c_function10_1_bufferify(name, len_trim(name, kind=C_INT), &
            arg2)
        ! splicer end function.function10_1_double
    end subroutine function10_1_double

    ! int overload1(int num +intent(in)+value)
    ! has_default_arg
    ! function_index=55
    function overload1_num(num) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: num
        integer(C_INT) :: SHT_rv
        ! splicer begin function.overload1_num
        SHT_rv = c_overload1_num(num)
        ! splicer end function.overload1_num
    end function overload1_num

    ! int overload1(int num +intent(in)+value, int offset=0 +intent(in)+value)
    ! has_default_arg
    ! function_index=56
    function overload1_num_offset(num, offset) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: num
        integer(C_INT), value, intent(IN) :: offset
        integer(C_INT) :: SHT_rv
        ! splicer begin function.overload1_num_offset
        SHT_rv = c_overload1_num_offset(num, offset)
        ! splicer end function.overload1_num_offset
    end function overload1_num_offset

    ! int overload1(int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
    ! function_index=30
    function overload1_num_offset_stride(num, offset, stride) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: num
        integer(C_INT), value, intent(IN) :: offset
        integer(C_INT), value, intent(IN) :: stride
        integer(C_INT) :: SHT_rv
        ! splicer begin function.overload1_num_offset_stride
        SHT_rv = c_overload1_num_offset_stride(num, offset, stride)
        ! splicer end function.overload1_num_offset_stride
    end function overload1_num_offset_stride

    ! int overload1(double type +intent(in)+value, int num +intent(in)+value)
    ! has_default_arg
    ! function_index=57
    function overload1_3(type, num) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, C_INT
        real(C_DOUBLE), value, intent(IN) :: type
        integer(C_INT), value, intent(IN) :: num
        integer(C_INT) :: SHT_rv
        ! splicer begin function.overload1_3
        SHT_rv = c_overload1_3(type, num)
        ! splicer end function.overload1_3
    end function overload1_3

    ! int overload1(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value)
    ! has_default_arg
    ! function_index=58
    function overload1_4(type, num, offset) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, C_INT
        real(C_DOUBLE), value, intent(IN) :: type
        integer(C_INT), value, intent(IN) :: num
        integer(C_INT), value, intent(IN) :: offset
        integer(C_INT) :: SHT_rv
        ! splicer begin function.overload1_4
        SHT_rv = c_overload1_4(type, num, offset)
        ! splicer end function.overload1_4
    end function overload1_4

    ! int overload1(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
    ! function_index=31
    function overload1_5(type, num, offset, stride) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, C_INT
        real(C_DOUBLE), value, intent(IN) :: type
        integer(C_INT), value, intent(IN) :: num
        integer(C_INT), value, intent(IN) :: offset
        integer(C_INT), value, intent(IN) :: stride
        integer(C_INT) :: SHT_rv
        ! splicer begin function.overload1_5
        SHT_rv = c_overload1_5(type, num, offset, stride)
        ! splicer end function.overload1_5
    end function overload1_5

    ! int useclass(const Class1 * arg1 +intent(in))
    ! function_index=37
    function useclass(arg1) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        type(class1), intent(IN) :: arg1
        integer(C_INT) :: SHT_rv
        ! splicer begin function.useclass
        SHT_rv = c_useclass(arg1%cxxmem)
        ! splicer end function.useclass
    end function useclass

    ! const Class1 * getclass2()
    ! function_index=38
    function getclass2() &
            result(SHT_rv)
        type(class1) :: SHT_rv
        ! splicer begin function.getclass2
        SHT_rv%cxxmem = c_getclass2()
        ! splicer end function.getclass2
    end function getclass2

    ! Class1 * getclass3()
    ! function_index=39
    function getclass3() &
            result(SHT_rv)
        type(class1) :: SHT_rv
        ! splicer begin function.getclass3
        SHT_rv%cxxmem = c_getclass3()
        ! splicer end function.getclass3
    end function getclass3

    ! Class1 getClassCopy(int flag +intent(in)+value)
    ! function_index=40
    !>
    !! \brief Return Class1 instance by value, uses copy constructor
    !!
    !<
    function get_class_copy(flag) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: flag
        type(class1) :: SHT_rv
        ! splicer begin function.get_class_copy
        SHT_rv%cxxmem = c_get_class_copy(flag)
        ! splicer end function.get_class_copy
    end function get_class_copy

    ! struct1 * returnStructPtr(int i +intent(in)+value, double d +intent(in)+value)
    ! function_index=43
    function return_struct_ptr(i, d) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, C_INT, C_PTR, c_f_pointer
        integer(C_INT), value, intent(IN) :: i
        real(C_DOUBLE), value, intent(IN) :: d
        type(struct1), pointer :: SHT_rv
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_struct_ptr
        SHT_ptr = c_return_struct_ptr(i, d)
        call c_f_pointer(SHT_ptr, SHT_rv)
        ! splicer end function.return_struct_ptr
    end function return_struct_ptr

    ! const std::string & LastFunctionCalled() +deref(result_as_arg)+len(30)
    ! arg_to_buffer
    ! function_index=48
    function last_function_called() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        character(len=30) :: SHT_rv
        ! splicer begin function.last_function_called
        call c_last_function_called_bufferify(SHT_rv, &
            len(SHT_rv, kind=C_INT))
        ! splicer end function.last_function_called
    end function last_function_called

    ! splicer begin additional_functions
    ! splicer end additional_functions

    function class1_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(class1), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function class1_eq

    function class1_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(class1), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function class1_ne

    function singleton_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(singleton), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function singleton_eq

    function singleton_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(singleton), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function singleton_ne

end module tutorial_mod
