! wrapftutorial.f
! This is generated code, do not edit
! Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapftutorial.f
!! \brief Shroud generated wrapper for Tutorial library
!<
! splicer begin file_top
! splicer end file_top
module tutorial_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR, C_SIZE_T
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

    !  enum tutorial::Class1::DIRECTION
    integer(C_INT), parameter :: tutorial_class1_direction_up = 2
    integer(C_INT), parameter :: tutorial_class1_direction_down = 3
    integer(C_INT), parameter :: tutorial_class1_direction_left = 100
    integer(C_INT), parameter :: tutorial_class1_direction_right = 101

    !  enum tutorial::Color
    integer(C_INT), parameter :: tutorial_color_red = 0
    integer(C_INT), parameter :: tutorial_color_blue = 1
    integer(C_INT), parameter :: tutorial_color_white = 2

    ! splicer begin class.Class1.module_top
    ! splicer end class.Class1.module_top

    ! start derived-type SHROUD_class1_capsule
    type, bind(C) :: SHROUD_class1_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_class1_capsule
    ! end derived-type SHROUD_class1_capsule

    type class1
        type(SHROUD_class1_capsule) :: cxxmem
        ! splicer begin class.Class1.component_part
        ! splicer end class.Class1.component_part
    contains
        procedure :: delete => class1_delete
        procedure :: method1 => class1_method1
        procedure :: equivalent => class1_equivalent
        procedure :: return_this => class1_return_this
        procedure :: return_this_buffer => class1_return_this_buffer
        procedure :: getclass3 => class1_getclass3
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

    type, bind(C) :: SHROUD_singleton_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_singleton_capsule

    type singleton
        type(SHROUD_singleton_capsule) :: cxxmem
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

        ! start abstract callback1_incr
        function callback1_incr(arg0) bind(C)
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value :: arg0
            integer(C_INT) :: callback1_incr
        end function callback1_incr
        ! end abstract callback1_incr

    end interface

    interface

        ! start c_class1_new_default
        function c_class1_new_default(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_new_default")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_class1_new_default
        ! end c_class1_new_default

        ! start c_class1_new_flag
        function c_class1_new_flag(flag, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_new_flag")
            use iso_c_binding, only : C_INT, C_PTR
            import :: SHROUD_class1_capsule
            implicit none
            integer(C_INT), value, intent(IN) :: flag
            type(SHROUD_class1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_class1_new_flag
        ! end c_class1_new_flag

        ! start c_class1_delete
        subroutine c_class1_delete(self) &
                bind(C, name="TUT_class1_delete")
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
        end subroutine c_class1_delete
        ! end c_class1_delete

        ! start c_class1_method1
        function c_class1_method1(self) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_method1")
            use iso_c_binding, only : C_INT
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_class1_method1
        ! end c_class1_method1

        ! start c_class1_equivalent
        pure function c_class1_equivalent(self, obj2) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_equivalent")
            use iso_c_binding, only : C_BOOL
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
            type(SHROUD_class1_capsule), intent(IN) :: obj2
            logical(C_BOOL) :: SHT_rv
        end function c_class1_equivalent
        ! end c_class1_equivalent

        ! start c_class1_return_this
        subroutine c_class1_return_this(self) &
                bind(C, name="TUT_class1_return_this")
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
        end subroutine c_class1_return_this
        ! end c_class1_return_this

        ! start c_class1_return_this_buffer
        function c_class1_return_this_buffer(self, name, flag, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_return_this_buffer")
            use iso_c_binding, only : C_BOOL, C_CHAR, C_PTR
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            logical(C_BOOL), value, intent(IN) :: flag
            type(SHROUD_class1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_class1_return_this_buffer
        ! end c_class1_return_this_buffer

        ! start c_class1_return_this_buffer_bufferify
        function c_class1_return_this_buffer_bufferify(self, name, &
                Lname, flag, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_return_this_buffer_bufferify")
            use iso_c_binding, only : C_BOOL, C_CHAR, C_INT, C_PTR
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            logical(C_BOOL), value, intent(IN) :: flag
            type(SHROUD_class1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_class1_return_this_buffer_bufferify
        ! end c_class1_return_this_buffer_bufferify

        ! start c_class1_getclass3
        function c_class1_getclass3(self, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_getclass3")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
            type(SHROUD_class1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_class1_getclass3
        ! end c_class1_getclass3

        ! start c_class1_direction_func
        function c_class1_direction_func(self, arg) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_direction_func")
            use iso_c_binding, only : C_INT
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: arg
            integer(C_INT) :: SHT_rv
        end function c_class1_direction_func
        ! end c_class1_direction_func

        ! start c_class1_get_m_flag
        function c_class1_get_m_flag(self) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_get_m_flag")
            use iso_c_binding, only : C_INT
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_class1_get_m_flag
        ! end c_class1_get_m_flag

        ! start c_class1_get_test
        function c_class1_get_test(self) &
                result(SHT_rv) &
                bind(C, name="TUT_class1_get_test")
            use iso_c_binding, only : C_INT
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_class1_get_test
        ! end c_class1_get_test

        ! start c_class1_set_test
        subroutine c_class1_set_test(self, val) &
                bind(C, name="TUT_class1_set_test")
            use iso_c_binding, only : C_INT
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: val
        end subroutine c_class1_set_test
        ! end c_class1_set_test

        ! splicer begin class.Class1.additional_interfaces
        ! splicer end class.Class1.additional_interfaces

        function c_singleton_get_reference(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TUT_singleton_get_reference")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_singleton_capsule
            implicit none
            type(SHROUD_singleton_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_singleton_get_reference

        ! splicer begin class.Singleton.additional_interfaces
        ! splicer end class.Singleton.additional_interfaces

        ! start no_return_no_arguments
        subroutine no_return_no_arguments() &
                bind(C, name="TUT_no_return_no_arguments")
            implicit none
        end subroutine no_return_no_arguments
        ! end no_return_no_arguments

        function pass_by_value(arg1, arg2) &
                result(SHT_rv) &
                bind(C, name="TUT_pass_by_value")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg1
            integer(C_INT), value, intent(IN) :: arg2
            real(C_DOUBLE) :: SHT_rv
        end function pass_by_value

        subroutine c_concatenate_strings_bufferify(arg1, Larg1, arg2, &
                Larg2, DSHF_rv) &
                bind(C, name="TUT_concatenate_strings_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_array
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg1(*)
            integer(C_INT), value, intent(IN) :: Larg1
            character(kind=C_CHAR), intent(IN) :: arg2(*)
            integer(C_INT), value, intent(IN) :: Larg2
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_concatenate_strings_bufferify

        function c_use_default_arguments() &
                result(SHT_rv) &
                bind(C, name="TUT_use_default_arguments")
            use iso_c_binding, only : C_DOUBLE
            implicit none
            real(C_DOUBLE) :: SHT_rv
        end function c_use_default_arguments

        function c_use_default_arguments_arg1(arg1) &
                result(SHT_rv) &
                bind(C, name="TUT_use_default_arguments_arg1")
            use iso_c_binding, only : C_DOUBLE
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg1
            real(C_DOUBLE) :: SHT_rv
        end function c_use_default_arguments_arg1

        function c_use_default_arguments_arg1_arg2(arg1, arg2) &
                result(SHT_rv) &
                bind(C, name="TUT_use_default_arguments_arg1_arg2")
            use iso_c_binding, only : C_BOOL, C_DOUBLE
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg1
            logical(C_BOOL), value, intent(IN) :: arg2
            real(C_DOUBLE) :: SHT_rv
        end function c_use_default_arguments_arg1_arg2

        subroutine c_overloaded_function_from_name(name) &
                bind(C, name="TUT_overloaded_function_from_name")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_overloaded_function_from_name

        subroutine c_overloaded_function_from_name_bufferify(name, &
                Lname) &
                bind(C, name="TUT_overloaded_function_from_name_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
        end subroutine c_overloaded_function_from_name_bufferify

        subroutine c_overloaded_function_from_index(indx) &
                bind(C, name="TUT_overloaded_function_from_index")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: indx
        end subroutine c_overloaded_function_from_index

        subroutine c_template_argument_int(arg) &
                bind(C, name="TUT_template_argument_int")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg
        end subroutine c_template_argument_int

        subroutine c_template_argument_double(arg) &
                bind(C, name="TUT_template_argument_double")
            use iso_c_binding, only : C_DOUBLE
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg
        end subroutine c_template_argument_double

        function c_template_return_int() &
                result(SHT_rv) &
                bind(C, name="TUT_template_return_int")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT) :: SHT_rv
        end function c_template_return_int

        function c_template_return_double() &
                result(SHT_rv) &
                bind(C, name="TUT_template_return_double")
            use iso_c_binding, only : C_DOUBLE
            implicit none
            real(C_DOUBLE) :: SHT_rv
        end function c_template_return_double

        subroutine c_fortran_generic(arg) &
                bind(C, name="TUT_fortran_generic")
            use iso_c_binding, only : C_DOUBLE
            implicit none
            real(C_DOUBLE), value, intent(IN) :: arg
        end subroutine c_fortran_generic

        subroutine c_fortran_generic_overloaded_0() &
                bind(C, name="TUT_fortran_generic_overloaded_0")
            implicit none
        end subroutine c_fortran_generic_overloaded_0

        subroutine c_fortran_generic_overloaded_1(name, arg2) &
                bind(C, name="TUT_fortran_generic_overloaded_1")
            use iso_c_binding, only : C_CHAR, C_DOUBLE
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            real(C_DOUBLE), value, intent(IN) :: arg2
        end subroutine c_fortran_generic_overloaded_1

        subroutine c_fortran_generic_overloaded_1_bufferify(name, Lname, &
                arg2) &
                bind(C, name="TUT_fortran_generic_overloaded_1_bufferify")
            use iso_c_binding, only : C_CHAR, C_DOUBLE, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            real(C_DOUBLE), value, intent(IN) :: arg2
        end subroutine c_fortran_generic_overloaded_1_bufferify

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

        ! start get_min_max
        subroutine get_min_max(min, max) &
                bind(C, name="TUT_get_min_max")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(OUT) :: min
            integer(C_INT), intent(OUT) :: max
        end subroutine get_min_max
        ! end get_min_max

        function direction_func(arg) &
                result(SHT_rv) &
                bind(C, name="TUT_direction_func")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg
            integer(C_INT) :: SHT_rv
        end function direction_func

        subroutine c_pass_class_by_value(arg) &
                bind(C, name="TUT_pass_class_by_value")
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), value, intent(IN) :: arg
        end subroutine c_pass_class_by_value

        function c_useclass(arg) &
                result(SHT_rv) &
                bind(C, name="TUT_useclass")
            use iso_c_binding, only : C_INT
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(IN) :: arg
            integer(C_INT) :: SHT_rv
        end function c_useclass

        function c_getclass2(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TUT_getclass2")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_getclass2

        function c_getclass3(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TUT_getclass3")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_class1_capsule
            implicit none
            type(SHROUD_class1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_getclass3

        function c_get_class_copy(flag, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TUT_get_class_copy")
            use iso_c_binding, only : C_INT, C_PTR
            import :: SHROUD_class1_capsule
            implicit none
            integer(C_INT), value, intent(IN) :: flag
            type(SHROUD_class1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_get_class_copy

        ! start callback1
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
        ! end callback1

        subroutine set_global_flag(arg) &
                bind(C, name="TUT_set_global_flag")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg
        end subroutine set_global_flag

        function get_global_flag() &
                result(SHT_rv) &
                bind(C, name="TUT_get_global_flag")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT) :: SHT_rv
        end function get_global_flag

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

    interface fortran_generic
        module procedure fortran_generic_float
        module procedure fortran_generic_double
    end interface fortran_generic

    interface fortran_generic_overloaded
        module procedure fortran_generic_overloaded_0
        module procedure fortran_generic_overloaded_1_float
        module procedure fortran_generic_overloaded_1_double
    end interface fortran_generic_overloaded

    interface overload1
        module procedure overload1_num
        module procedure overload1_num_offset
        module procedure overload1_num_offset_stride
        module procedure overload1_3
        module procedure overload1_4
        module procedure overload1_5
    end interface overload1

    interface overloaded_function
        module procedure overloaded_function_from_name
        module procedure overloaded_function_from_index
    end interface overloaded_function

    interface template_argument
        module procedure template_argument_int
        module procedure template_argument_double
    end interface template_argument

    interface use_default_arguments
        module procedure use_default_arguments
        module procedure use_default_arguments_arg1
        module procedure use_default_arguments_arg1_arg2
    end interface use_default_arguments

    interface
        ! helper function
        ! Copy the char* or std::string in context into c_var.
        subroutine SHROUD_copy_string_and_free(context, c_var, c_var_size) &
             bind(c,name="TUT_ShroudCopyStringAndFree")
            use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            character(kind=C_CHAR), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_string_and_free
    end interface

contains

    ! Class1() +name(new)
    ! start class1_new_default
    function class1_new_default() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(C_PTR) :: SHT_prv
        type(class1) :: SHT_rv
        ! splicer begin class.Class1.method.new_default
        SHT_prv = c_class1_new_default(SHT_rv%cxxmem)
        ! splicer end class.Class1.method.new_default
    end function class1_new_default
    ! end class1_new_default

    ! Class1(int flag +intent(in)+value) +name(new)
    ! start class1_new_flag
    function class1_new_flag(flag) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR
        integer(C_INT), value, intent(IN) :: flag
        type(C_PTR) :: SHT_prv
        type(class1) :: SHT_rv
        ! splicer begin class.Class1.method.new_flag
        SHT_prv = c_class1_new_flag(flag, SHT_rv%cxxmem)
        ! splicer end class.Class1.method.new_flag
    end function class1_new_flag
    ! end class1_new_flag

    ! ~Class1() +name(delete)
    ! start class1_delete
    subroutine class1_delete(obj)
        class(class1) :: obj
        ! splicer begin class.Class1.method.delete
        call c_class1_delete(obj%cxxmem)
        ! splicer end class.Class1.method.delete
    end subroutine class1_delete
    ! end class1_delete

    ! int Method1()
    !>
    !! \brief returns the value of flag member
    !!
    !<
    ! start class1_method1
    function class1_method1(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.method1
        SHT_rv = c_class1_method1(obj%cxxmem)
        ! splicer end class.Class1.method.method1
    end function class1_method1
    ! end class1_method1

    ! bool equivalent(const Class1 & obj2 +intent(in)) const
    !>
    !! \brief Pass in reference to instance
    !!
    !<
    ! start class1_equivalent
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
    ! end class1_equivalent

    ! Class1 * returnThis()
    !>
    !! \brief Return pointer to 'this' to allow chaining calls
    !!
    !<
    ! start class1_return_this
    subroutine class1_return_this(obj)
        class(class1) :: obj
        ! splicer begin class.Class1.method.return_this
        call c_class1_return_this(obj%cxxmem)
        ! splicer end class.Class1.method.return_this
    end subroutine class1_return_this
    ! end class1_return_this

    ! Class1 * returnThisBuffer(std::string & name +intent(in), bool flag +intent(in)+value)
    ! arg_to_buffer
    !>
    !! \brief Return pointer to 'this' to allow chaining calls
    !!
    !<
    ! start class1_return_this_buffer
    function class1_return_this_buffer(obj, name, flag) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_INT, C_PTR
        class(class1) :: obj
        character(len=*), intent(IN) :: name
        logical, value, intent(IN) :: flag
        logical(C_BOOL) SH_flag
        type(C_PTR) :: SHT_prv
        type(class1) :: SHT_rv
        SH_flag = flag  ! coerce to C_BOOL
        ! splicer begin class.Class1.method.return_this_buffer
        SHT_prv = c_class1_return_this_buffer_bufferify(obj%cxxmem, &
            name, len_trim(name, kind=C_INT), SH_flag, SHT_rv%cxxmem)
        ! splicer end class.Class1.method.return_this_buffer
    end function class1_return_this_buffer
    ! end class1_return_this_buffer

    ! Class1 * getclass3() const
    !>
    !! \brief Test const method
    !!
    !<
    ! start class1_getclass3
    function class1_getclass3(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(class1) :: obj
        type(C_PTR) :: SHT_prv
        type(class1) :: SHT_rv
        ! splicer begin class.Class1.method.getclass3
        SHT_prv = c_class1_getclass3(obj%cxxmem, SHT_rv%cxxmem)
        ! splicer end class.Class1.method.getclass3
    end function class1_getclass3
    ! end class1_getclass3

    ! DIRECTION directionFunc(DIRECTION arg +intent(in)+value)
    ! start class1_direction_func
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
    ! end class1_direction_func

    ! int getM_flag()
    ! start class1_get_m_flag
    function class1_get_m_flag(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.get_m_flag
        SHT_rv = c_class1_get_m_flag(obj%cxxmem)
        ! splicer end class.Class1.method.get_m_flag
    end function class1_get_m_flag
    ! end class1_get_m_flag

    ! int getTest()
    ! start class1_get_test
    function class1_get_test(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.get_test
        SHT_rv = c_class1_get_test(obj%cxxmem)
        ! splicer end class.Class1.method.get_test
    end function class1_get_test
    ! end class1_get_test

    ! void setTest(int val +intent(in)+value)
    ! start class1_set_test
    subroutine class1_set_test(obj, val)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT), value, intent(IN) :: val
        ! splicer begin class.Class1.method.set_test
        call c_class1_set_test(obj%cxxmem, val)
        ! splicer end class.Class1.method.set_test
    end subroutine class1_set_test
    ! end class1_set_test

    ! Return pointer to C++ memory.
    function class1_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
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
    function singleton_get_reference() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(C_PTR) :: SHT_prv
        type(singleton) :: SHT_rv
        ! splicer begin class.Singleton.method.get_reference
        SHT_prv = c_singleton_get_reference(SHT_rv%cxxmem)
        ! splicer end class.Singleton.method.get_reference
    end function singleton_get_reference

    ! Return pointer to C++ memory.
    function singleton_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
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

    ! const std::string ConcatenateStrings(const std::string & arg1 +intent(in), const std::string & arg2 +intent(in)) +deref(allocatable)
    ! arg_to_buffer
    !>
    !! Note that since a reference is returned, no intermediate string
    !! is allocated.  It is assumed +owner(library).
    !<
    function concatenate_strings(arg1, arg2) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        character(len=*), intent(IN) :: arg1
        character(len=*), intent(IN) :: arg2
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin function.concatenate_strings
        call c_concatenate_strings_bufferify(arg1, &
            len_trim(arg1, kind=C_INT), arg2, &
            len_trim(arg2, kind=C_INT), DSHF_rv)
        ! splicer end function.concatenate_strings
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function concatenate_strings

    ! double UseDefaultArguments()
    ! has_default_arg
    function use_default_arguments() &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE) :: SHT_rv
        ! splicer begin function.use_default_arguments
        SHT_rv = c_use_default_arguments()
        ! splicer end function.use_default_arguments
    end function use_default_arguments

    ! double UseDefaultArguments(double arg1=3.1415 +intent(in)+value)
    ! has_default_arg
    function use_default_arguments_arg1(arg1) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg1
        real(C_DOUBLE) :: SHT_rv
        ! splicer begin function.use_default_arguments_arg1
        SHT_rv = c_use_default_arguments_arg1(arg1)
        ! splicer end function.use_default_arguments_arg1
    end function use_default_arguments_arg1

    ! double UseDefaultArguments(double arg1=3.1415 +intent(in)+value, bool arg2=true +intent(in)+value)
    function use_default_arguments_arg1_arg2(arg1, arg2) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg1
        logical, value, intent(IN) :: arg2
        logical(C_BOOL) SH_arg2
        real(C_DOUBLE) :: SHT_rv
        SH_arg2 = arg2  ! coerce to C_BOOL
        ! splicer begin function.use_default_arguments_arg1_arg2
        SHT_rv = c_use_default_arguments_arg1_arg2(arg1, SH_arg2)
        ! splicer end function.use_default_arguments_arg1_arg2
    end function use_default_arguments_arg1_arg2

    ! void OverloadedFunction(const std::string & name +intent(in))
    ! arg_to_buffer
    subroutine overloaded_function_from_name(name)
        use iso_c_binding, only : C_INT
        character(len=*), intent(IN) :: name
        ! splicer begin function.overloaded_function_from_name
        call c_overloaded_function_from_name_bufferify(name, &
            len_trim(name, kind=C_INT))
        ! splicer end function.overloaded_function_from_name
    end subroutine overloaded_function_from_name

    ! void OverloadedFunction(int indx +intent(in)+value)
    subroutine overloaded_function_from_index(indx)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: indx
        ! splicer begin function.overloaded_function_from_index
        call c_overloaded_function_from_index(indx)
        ! splicer end function.overloaded_function_from_index
    end subroutine overloaded_function_from_index

    ! void TemplateArgument(int arg +intent(in)+value)
    ! cxx_template
    subroutine template_argument_int(arg)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: arg
        ! splicer begin function.template_argument_int
        call c_template_argument_int(arg)
        ! splicer end function.template_argument_int
    end subroutine template_argument_int

    ! void TemplateArgument(double arg +intent(in)+value)
    ! cxx_template
    subroutine template_argument_double(arg)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg
        ! splicer begin function.template_argument_double
        call c_template_argument_double(arg)
        ! splicer end function.template_argument_double
    end subroutine template_argument_double

    ! int TemplateReturn()
    ! cxx_template
    function template_return_int() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT) :: SHT_rv
        ! splicer begin function.template_return_int
        SHT_rv = c_template_return_int()
        ! splicer end function.template_return_int
    end function template_return_int

    ! double TemplateReturn()
    ! cxx_template
    function template_return_double() &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE) :: SHT_rv
        ! splicer begin function.template_return_double
        SHT_rv = c_template_return_double()
        ! splicer end function.template_return_double
    end function template_return_double

    ! void FortranGeneric(float arg +intent(in)+value)
    ! fortran_generic
    subroutine fortran_generic_float(arg)
        use iso_c_binding, only : C_DOUBLE, C_FLOAT
        real(C_FLOAT), value, intent(IN) :: arg
        ! splicer begin function.fortran_generic_float
        call c_fortran_generic(real(arg, C_DOUBLE))
        ! splicer end function.fortran_generic_float
    end subroutine fortran_generic_float

    ! void FortranGeneric(double arg +intent(in)+value)
    ! fortran_generic
    subroutine fortran_generic_double(arg)
        use iso_c_binding, only : C_DOUBLE
        real(C_DOUBLE), value, intent(IN) :: arg
        ! splicer begin function.fortran_generic_double
        call c_fortran_generic(arg)
        ! splicer end function.fortran_generic_double
    end subroutine fortran_generic_double

    ! void FortranGenericOverloaded()
    subroutine fortran_generic_overloaded_0()
        ! splicer begin function.fortran_generic_overloaded_0
        call c_fortran_generic_overloaded_0()
        ! splicer end function.fortran_generic_overloaded_0
    end subroutine fortran_generic_overloaded_0

    ! void FortranGenericOverloaded(const std::string & name +intent(in), float arg2 +intent(in)+value)
    ! fortran_generic - arg_to_buffer
    subroutine fortran_generic_overloaded_1_float(name, arg2)
        use iso_c_binding, only : C_DOUBLE, C_FLOAT, C_INT
        character(len=*), intent(IN) :: name
        real(C_FLOAT), value, intent(IN) :: arg2
        ! splicer begin function.fortran_generic_overloaded_1_float
        call c_fortran_generic_overloaded_1_bufferify(name, &
            len_trim(name, kind=C_INT), real(arg2, C_DOUBLE))
        ! splicer end function.fortran_generic_overloaded_1_float
    end subroutine fortran_generic_overloaded_1_float

    ! void FortranGenericOverloaded(const std::string & name +intent(in), double arg2 +intent(in)+value)
    ! fortran_generic - arg_to_buffer
    subroutine fortran_generic_overloaded_1_double(name, arg2)
        use iso_c_binding, only : C_DOUBLE, C_INT
        character(len=*), intent(IN) :: name
        real(C_DOUBLE), value, intent(IN) :: arg2
        ! splicer begin function.fortran_generic_overloaded_1_double
        call c_fortran_generic_overloaded_1_bufferify(name, &
            len_trim(name, kind=C_INT), arg2)
        ! splicer end function.fortran_generic_overloaded_1_double
    end subroutine fortran_generic_overloaded_1_double

    ! int overload1(int num +intent(in)+value)
    ! has_default_arg
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

    ! void passClassByValue(Class1 arg +intent(in)+value)
    !>
    !! \brief Pass arguments to a function.
    !!
    !<
    subroutine pass_class_by_value(arg)
        type(class1), value, intent(IN) :: arg
        ! splicer begin function.pass_class_by_value
        call c_pass_class_by_value(arg%cxxmem)
        ! splicer end function.pass_class_by_value
    end subroutine pass_class_by_value

    ! int useclass(const Class1 * arg +intent(in))
    function useclass(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        type(class1), intent(IN) :: arg
        integer(C_INT) :: SHT_rv
        ! splicer begin function.useclass
        SHT_rv = c_useclass(arg%cxxmem)
        ! splicer end function.useclass
    end function useclass

    ! const Class1 * getclass2()
    function getclass2() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(C_PTR) :: SHT_prv
        type(class1) :: SHT_rv
        ! splicer begin function.getclass2
        SHT_prv = c_getclass2(SHT_rv%cxxmem)
        ! splicer end function.getclass2
    end function getclass2

    ! Class1 * getclass3()
    function getclass3() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(C_PTR) :: SHT_prv
        type(class1) :: SHT_rv
        ! splicer begin function.getclass3
        SHT_prv = c_getclass3(SHT_rv%cxxmem)
        ! splicer end function.getclass3
    end function getclass3

    ! Class1 getClassCopy(int flag +intent(in)+value)
    !>
    !! \brief Return Class1 instance by value, uses copy constructor
    !!
    !<
    function get_class_copy(flag) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR
        integer(C_INT), value, intent(IN) :: flag
        type(C_PTR) :: SHT_prv
        type(class1) :: SHT_rv
        ! splicer begin function.get_class_copy
        SHT_prv = c_get_class_copy(flag, SHT_rv%cxxmem)
        ! splicer end function.get_class_copy
    end function get_class_copy

    ! const std::string & LastFunctionCalled() +deref(result_as_arg)+len(30)
    ! arg_to_buffer
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
