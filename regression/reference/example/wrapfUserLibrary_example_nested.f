! wrapfUserLibrary_example_nested.f
! This is generated code, do not edit
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
!! \file wrapfUserLibrary_example_nested.f
!! \brief Shroud generated wrapper for nested namespace
!<
! splicer begin namespace.example::nested.file_top
! splicer end namespace.example::nested.file_top
module userlibrary_example_nested_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin namespace.example::nested.module_use
    ! splicer end namespace.example::nested.module_use
    implicit none

    ! splicer begin namespace.example::nested.module_top
    top of module namespace example splicer  3
    ! splicer end namespace.example::nested.module_top

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

    type, bind(C) :: SHROUD_exclass1_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_exclass1_capsule

    type exclass1
        type(SHROUD_exclass1_capsule) :: cxxmem
        ! splicer begin namespace.example::nested.class.ExClass1.component_part
          component part 1a
          component part 1b
        ! splicer end namespace.example::nested.class.ExClass1.component_part
    contains
        procedure :: delete => exclass1_dtor
        procedure :: increment_count => exclass1_increment_count
        procedure :: get_name_error_pattern => exclass1_get_name_error_pattern
        procedure :: get_name_length => exclass1_get_name_length
        procedure :: get_name_error_check => exclass1_get_name_error_check
        procedure :: get_name_arg => exclass1_get_name_arg
        procedure :: get_root => exclass1_get_root
        procedure :: get_value_from_int => exclass1_get_value_from_int
        procedure :: get_value_1 => exclass1_get_value_1
        procedure :: get_addr => exclass1_get_addr
        procedure :: has_addr => exclass1_has_addr
        procedure :: splicer_special => exclass1_splicer_special
        procedure :: yadda => exclass1_yadda
        procedure :: associated => exclass1_associated
        generic :: get_value => get_value_from_int, get_value_1
        ! splicer begin namespace.example::nested.class.ExClass1.type_bound_procedure_part
          type bound procedure part 1
        ! splicer end namespace.example::nested.class.ExClass1.type_bound_procedure_part
    end type exclass1

    type, bind(C) :: SHROUD_exclass2_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_exclass2_capsule

    type exclass2
        type(SHROUD_exclass2_capsule) :: cxxmem
        ! splicer begin namespace.example::nested.class.ExClass2.component_part
        ! splicer end namespace.example::nested.class.ExClass2.component_part
    contains
        procedure :: delete => exclass2_dtor
        procedure :: get_name => exclass2_get_name
        procedure :: get_name2 => exclass2_get_name2
        procedure :: get_name3 => exclass2_get_name3
        procedure :: get_name4 => exclass2_get_name4
        procedure :: get_name_length => exclass2_get_name_length
        procedure :: get_class1 => exclass2_get_class1
        procedure :: declare_0_int => exclass2_declare_0_int
        procedure :: declare_0_long => exclass2_declare_0_long
        procedure :: declare_1_int => exclass2_declare_1_int
        procedure :: declare_1_long => exclass2_declare_1_long
        procedure :: destroyall => exclass2_destroyall
        procedure :: get_type_id => exclass2_get_type_id
        procedure :: set_value_int => exclass2_set_value_int
        procedure :: set_value_long => exclass2_set_value_long
        procedure :: set_value_float => exclass2_set_value_float
        procedure :: set_value_double => exclass2_set_value_double
        procedure :: get_value_int => exclass2_get_value_int
        procedure :: get_value_double => exclass2_get_value_double
        procedure :: yadda => exclass2_yadda
        procedure :: associated => exclass2_associated
        generic :: declare => declare_0_int, declare_0_long,  &
            declare_1_int, declare_1_long
        generic :: set_value => set_value_int, set_value_long,  &
            set_value_float, set_value_double
        ! splicer begin namespace.example::nested.class.ExClass2.type_bound_procedure_part
        ! splicer end namespace.example::nested.class.ExClass2.type_bound_procedure_part
    end type exclass2

    interface operator (.eq.)
        module procedure exclass1_eq
        module procedure exclass2_eq
    end interface

    interface operator (.ne.)
        module procedure exclass1_ne
        module procedure exclass2_ne
    end interface

    abstract interface

        function custom_funptr(XX0arg, XX1arg) bind(C)
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), value :: XX0arg
            integer(C_INT), value :: XX1arg
            type(C_PTR) :: custom_funptr
        end function custom_funptr

        subroutine func_ptr1_get() bind(C)
            implicit none
        end subroutine func_ptr1_get

        function func_ptr2_get() bind(C)
            implicit none
            type(C_PTR) :: func_ptr2_get
        end function func_ptr2_get

        function func_ptr3_get(i, arg1) bind(C)
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value :: i
            integer(C_INT), value :: arg1
            type(C_PTR) :: func_ptr3_get
        end function func_ptr3_get

        subroutine func_ptr5_get(verylongname1, verylongname2, &
            verylongname3, verylongname4, verylongname5, verylongname6, &
            verylongname7, verylongname8, verylongname9, &
            verylongname10) bind(C)
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value :: verylongname1
            integer(C_INT), value :: verylongname2
            integer(C_INT), value :: verylongname3
            integer(C_INT), value :: verylongname4
            integer(C_INT), value :: verylongname5
            integer(C_INT), value :: verylongname6
            integer(C_INT), value :: verylongname7
            integer(C_INT), value :: verylongname8
            integer(C_INT), value :: verylongname9
            integer(C_INT), value :: verylongname10
        end subroutine func_ptr5_get

    end interface

    interface

        function c_exclass1_ctor_0(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_ctor_0")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_exclass1_ctor_0

        function c_exclass1_ctor_1(name, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_ctor_1")
            use iso_c_binding, only : C_CHAR, C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_exclass1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_exclass1_ctor_1

        function c_exclass1_ctor_1_bufferify(name, Lname, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_ctor_1_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(SHROUD_exclass1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_exclass1_ctor_1_bufferify

        subroutine c_exclass1_dtor(self) &
                bind(C, name="AA_example_nested_ExClass1_dtor")
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
        end subroutine c_exclass1_dtor

        function c_exclass1_increment_count(self, incr) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_increment_count")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: incr
            integer(C_INT) :: SHT_rv
        end function c_exclass1_increment_count

        pure function c_exclass1_get_name_error_pattern(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_get_name_error_pattern")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass1_get_name_error_pattern

        subroutine c_exclass1_get_name_error_pattern_bufferify(self, &
                SHF_rv, NSHF_rv) &
                bind(C, name="AA_example_nested_ExClass1_get_name_error_pattern_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(OUT) :: SHF_rv(*)
            integer(C_INT), value, intent(IN) :: NSHF_rv
        end subroutine c_exclass1_get_name_error_pattern_bufferify

        pure function c_exclass1_get_name_length(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_get_name_length")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_exclass1_get_name_length

        pure function c_exclass1_get_name_error_check(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_get_name_error_check")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass1_get_name_error_check

        subroutine c_exclass1_get_name_error_check_bufferify(self, &
                DSHF_rv) &
                bind(C, name="AA_example_nested_ExClass1_get_name_error_check_bufferify")
            import :: SHROUD_array, SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_exclass1_get_name_error_check_bufferify

        pure function c_exclass1_get_name_arg(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_get_name_arg")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass1_get_name_arg

        subroutine c_exclass1_get_name_arg_bufferify(self, name, Nname) &
                bind(C, name="AA_example_nested_ExClass1_get_name_arg_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(OUT) :: name(*)
            integer(C_INT), value, intent(IN) :: Nname
        end subroutine c_exclass1_get_name_arg_bufferify

        function c_exclass1_get_root(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_get_root")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(C_PTR) :: SHT_rv
        end function c_exclass1_get_root

        function c_exclass1_get_value_from_int(self, value) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_get_value_from_int")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: value
            integer(C_INT) :: SHT_rv
        end function c_exclass1_get_value_from_int

        function c_exclass1_get_value_1(self, value) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_get_value_1")
            use iso_c_binding, only : C_LONG
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            integer(C_LONG), value, intent(IN) :: value
            integer(C_LONG) :: SHT_rv
        end function c_exclass1_get_value_1

        function c_exclass1_get_addr(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_get_addr")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(C_PTR) :: SHT_rv
        end function c_exclass1_get_addr

        function c_exclass1_has_addr(self, in) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass1_has_addr")
            use iso_c_binding, only : C_BOOL
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            logical(C_BOOL), value, intent(IN) :: in
            logical(C_BOOL) :: SHT_rv
        end function c_exclass1_has_addr

        subroutine c_exclass1_splicer_special(self) &
                bind(C, name="AA_example_nested_ExClass1_splicer_special")
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
        end subroutine c_exclass1_splicer_special

        ! splicer begin namespace.example::nested.class.ExClass1.additional_interfaces
        ! splicer end namespace.example::nested.class.ExClass1.additional_interfaces

        function c_exclass2_ctor(name, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_ctor")
            use iso_c_binding, only : C_CHAR, C_PTR
            import :: SHROUD_exclass2_capsule
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_exclass2_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_exclass2_ctor

        function c_exclass2_ctor_bufferify(name, trim_name, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_ctor_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            import :: SHROUD_exclass2_capsule
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: trim_name
            type(SHROUD_exclass2_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_exclass2_ctor_bufferify

        subroutine c_exclass2_dtor(self) &
                bind(C, name="AA_example_nested_ExClass2_dtor")
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
        end subroutine c_exclass2_dtor

        pure function c_exclass2_get_name(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_name")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass2_get_name

        subroutine c_exclass2_get_name_bufferify(self, SHF_rv, NSHF_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_name_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(OUT) :: SHF_rv(*)
            integer(C_INT), value, intent(IN) :: NSHF_rv
        end subroutine c_exclass2_get_name_bufferify

        function c_exclass2_get_name2(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_name2")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass2_get_name2

        subroutine c_exclass2_get_name2_bufferify(self, DSHF_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_name2_bufferify")
            import :: SHROUD_array, SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_exclass2_get_name2_bufferify

        pure function c_exclass2_get_name3(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_name3")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass2_get_name3

        subroutine c_exclass2_get_name3_bufferify(self, DSHF_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_name3_bufferify")
            import :: SHROUD_array, SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_exclass2_get_name3_bufferify

        function c_exclass2_get_name4(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_name4")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass2_get_name4

        subroutine c_exclass2_get_name4_bufferify(self, DSHF_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_name4_bufferify")
            import :: SHROUD_array, SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_exclass2_get_name4_bufferify

        pure function c_exclass2_get_name_length(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_name_length")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_exclass2_get_name_length

        function c_exclass2_get_class1(self, in, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_class1")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule, SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            type(SHROUD_exclass1_capsule), intent(IN) :: in
            type(SHROUD_exclass1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_exclass2_get_class1

        subroutine c_exclass2_declare_0(self, type) &
                bind(C, name="AA_example_nested_ExClass2_declare_0")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: type
        end subroutine c_exclass2_declare_0

        subroutine c_exclass2_declare_1(self, type, len) &
                bind(C, name="AA_example_nested_ExClass2_declare_1")
            use iso_c_binding, only : C_INT, C_LONG
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: type
            integer(C_LONG), value, intent(IN) :: len
        end subroutine c_exclass2_declare_1

        subroutine c_exclass2_destroyall(self) &
                bind(C, name="AA_example_nested_ExClass2_destroyall")
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
        end subroutine c_exclass2_destroyall

        pure function c_exclass2_get_type_id(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_type_id")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_exclass2_get_type_id

        subroutine c_exclass2_set_value_int(self, value) &
                bind(C, name="AA_example_nested_ExClass2_set_value_int")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: value
        end subroutine c_exclass2_set_value_int

        subroutine c_exclass2_set_value_long(self, value) &
                bind(C, name="AA_example_nested_ExClass2_set_value_long")
            use iso_c_binding, only : C_LONG
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            integer(C_LONG), value, intent(IN) :: value
        end subroutine c_exclass2_set_value_long

        subroutine c_exclass2_set_value_float(self, value) &
                bind(C, name="AA_example_nested_ExClass2_set_value_float")
            use iso_c_binding, only : C_FLOAT
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            real(C_FLOAT), value, intent(IN) :: value
        end subroutine c_exclass2_set_value_float

        subroutine c_exclass2_set_value_double(self, value) &
                bind(C, name="AA_example_nested_ExClass2_set_value_double")
            use iso_c_binding, only : C_DOUBLE
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            real(C_DOUBLE), value, intent(IN) :: value
        end subroutine c_exclass2_set_value_double

        function c_exclass2_get_value_int(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_value_int")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_exclass2_get_value_int

        function c_exclass2_get_value_double(self) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_ExClass2_get_value_double")
            use iso_c_binding, only : C_DOUBLE
            import :: SHROUD_exclass2_capsule
            implicit none
            type(SHROUD_exclass2_capsule), intent(IN) :: self
            real(C_DOUBLE) :: SHT_rv
        end function c_exclass2_get_value_double

        ! splicer begin namespace.example::nested.class.ExClass2.additional_interfaces
        ! splicer end namespace.example::nested.class.ExClass2.additional_interfaces

        subroutine local_function1() &
                bind(C, name="AA_example_nested_local_function1")
            implicit none
        end subroutine local_function1

        function c_is_name_valid(name) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_is_name_valid")
            use iso_c_binding, only : C_BOOL, C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            logical(C_BOOL) :: SHT_rv
        end function c_is_name_valid

        function c_is_name_valid_bufferify(name, Lname) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_is_name_valid_bufferify")
            use iso_c_binding, only : C_BOOL, C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            logical(C_BOOL) :: SHT_rv
        end function c_is_name_valid_bufferify

        function c_is_initialized() &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_is_initialized")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL) :: SHT_rv
        end function c_is_initialized

        subroutine c_test_names(name) &
                bind(C, name="AA_example_nested_test_names")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_test_names

        subroutine c_test_names_bufferify(name, Lname) &
                bind(C, name="AA_example_nested_test_names_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
        end subroutine c_test_names_bufferify

        subroutine c_test_names_flag(name, flag) &
                bind(C, name="AA_example_nested_test_names_flag")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: flag
        end subroutine c_test_names_flag

        subroutine c_test_names_flag_bufferify(name, Lname, flag) &
                bind(C, name="AA_example_nested_test_names_flag_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            integer(C_INT), value, intent(IN) :: flag
        end subroutine c_test_names_flag_bufferify

        subroutine c_testoptional_0() &
                bind(C, name="AA_example_nested_testoptional_0")
            implicit none
        end subroutine c_testoptional_0

        subroutine c_testoptional_1(i) &
                bind(C, name="AA_example_nested_testoptional_1")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: i
        end subroutine c_testoptional_1

        subroutine c_testoptional_2(i, j) &
                bind(C, name="AA_example_nested_testoptional_2")
            use iso_c_binding, only : C_INT, C_LONG
            implicit none
            integer(C_INT), value, intent(IN) :: i
            integer(C_LONG), value, intent(IN) :: j
        end subroutine c_testoptional_2

        function test_size_t() &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_test_size_t")
            use iso_c_binding, only : C_SIZE_T
            implicit none
            integer(C_SIZE_T) :: SHT_rv
        end function test_size_t

#ifdef HAVE_MPI
        subroutine c_testmpi_mpi(comm) &
                bind(C, name="AA_example_nested_testmpi_mpi")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: comm
        end subroutine c_testmpi_mpi
#endif

#ifndef HAVE_MPI
        subroutine c_testmpi_serial() &
                bind(C, name="AA_example_nested_testmpi_serial")
            implicit none
        end subroutine c_testmpi_serial
#endif

        subroutine c_testgroup1(grp) &
                bind(C, name="AA_example_nested_testgroup1")
            use sidre_mod, only : SHROUD_group_capsule
            implicit none
            type(SHROUD_group_capsule), intent(IN) :: grp
        end subroutine c_testgroup1

        subroutine c_testgroup2(grp) &
                bind(C, name="AA_example_nested_testgroup2")
            use sidre_mod, only : SHROUD_group_capsule
            implicit none
            type(SHROUD_group_capsule), intent(IN) :: grp
        end subroutine c_testgroup2

        subroutine func_ptr1(get) &
                bind(C, name="AA_example_nested_func_ptr1")
            use iso_c_binding, only : C_PTR
            import :: func_ptr1_get
            implicit none
            procedure(func_ptr1_get) :: get
        end subroutine func_ptr1

        subroutine func_ptr2(get) &
                bind(C, name="AA_example_nested_func_ptr2")
            use iso_c_binding, only : C_DOUBLE
            import :: func_ptr2_get
            implicit none
            procedure(func_ptr2_get) :: get
        end subroutine func_ptr2

        subroutine c_func_ptr3(get) &
                bind(C, name="AA_example_nested_func_ptr3")
            use iso_c_binding, only : C_DOUBLE
            import :: func_ptr3_get
            implicit none
            procedure(func_ptr3_get) :: get
        end subroutine c_func_ptr3

        subroutine c_func_ptr4(get) &
                bind(C, name="AA_example_nested_func_ptr4")
            use iso_c_binding, only : C_DOUBLE
            import :: custom_funptr
            implicit none
            procedure(custom_funptr) :: get
        end subroutine c_func_ptr4

        subroutine func_ptr5(get) &
                bind(C, name="AA_example_nested_func_ptr5")
            use iso_c_binding, only : C_PTR
            import :: func_ptr5_get
            implicit none
            procedure(func_ptr5_get) :: get
        end subroutine func_ptr5

        subroutine c_verylongfunctionname1(verylongname1, verylongname2, &
                verylongname3, verylongname4, verylongname5, &
                verylongname6, verylongname7, verylongname8, &
                verylongname9, verylongname10) &
                bind(C, name="AA_example_nested_verylongfunctionname1")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(INOUT) :: verylongname1
            integer(C_INT), intent(INOUT) :: verylongname2
            integer(C_INT), intent(INOUT) :: verylongname3
            integer(C_INT), intent(INOUT) :: verylongname4
            integer(C_INT), intent(INOUT) :: verylongname5
            integer(C_INT), intent(INOUT) :: verylongname6
            integer(C_INT), intent(INOUT) :: verylongname7
            integer(C_INT), intent(INOUT) :: verylongname8
            integer(C_INT), intent(INOUT) :: verylongname9
            integer(C_INT), intent(INOUT) :: verylongname10
        end subroutine c_verylongfunctionname1

        function c_verylongfunctionname2(verylongname1, verylongname2, &
                verylongname3, verylongname4, verylongname5, &
                verylongname6, verylongname7, verylongname8, &
                verylongname9, verylongname10) &
                result(SHT_rv) &
                bind(C, name="AA_example_nested_verylongfunctionname2")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: verylongname1
            integer(C_INT), value, intent(IN) :: verylongname2
            integer(C_INT), value, intent(IN) :: verylongname3
            integer(C_INT), value, intent(IN) :: verylongname4
            integer(C_INT), value, intent(IN) :: verylongname5
            integer(C_INT), value, intent(IN) :: verylongname6
            integer(C_INT), value, intent(IN) :: verylongname7
            integer(C_INT), value, intent(IN) :: verylongname8
            integer(C_INT), value, intent(IN) :: verylongname9
            integer(C_INT), value, intent(IN) :: verylongname10
            integer(C_INT) :: SHT_rv
        end function c_verylongfunctionname2

        subroutine c_cos_doubles(in, out, sizein) &
                bind(C, name="AA_example_nested_cos_doubles")
            use iso_c_binding, only : C_DOUBLE, C_INT
            implicit none
            real(C_DOUBLE), intent(IN) :: in(*)
            real(C_DOUBLE), intent(OUT) :: out(*)
            integer(C_INT), value, intent(IN) :: sizein
        end subroutine c_cos_doubles

        ! splicer begin namespace.example::nested.additional_interfaces
        ! splicer end namespace.example::nested.additional_interfaces
    end interface

    interface exclass1_ctor
        module procedure exclass1_ctor_0
        module procedure exclass1_ctor_1
    end interface exclass1_ctor

    interface test_names
        module procedure test_names
        module procedure test_names_flag
    end interface test_names

    interface testmpi
#ifdef HAVE_MPI
        module procedure testmpi_mpi
#endif
#ifndef HAVE_MPI
        module procedure testmpi_serial
#endif
    end interface testmpi

    interface testoptional
        module procedure testoptional_0
        module procedure testoptional_1
        module procedure testoptional_2
    end interface testoptional

    interface
        ! helper function
        ! Copy the char* or std::string in context into c_var.
        subroutine SHROUD_copy_string_and_free(context, c_var, c_var_size) &
             bind(c,name="AA_ShroudCopyStringAndFree")
            use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            character(kind=C_CHAR), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_string_and_free
    end interface

contains

    ! ExClass1()
    function exclass1_ctor_0() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(C_PTR) :: SHT_prv
        type(exclass1) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass1.method.ctor_0
        SHT_prv = c_exclass1_ctor_0(SHT_rv%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass1.method.ctor_0
    end function exclass1_ctor_0

    ! ExClass1(const string * name +intent(in))
    ! arg_to_buffer
    !>
    !! \brief constructor
    !!
    !! longer description
    !! usually multiple lines
    !!
    !! \return return new instance
    !<
    function exclass1_ctor_1(name) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR
        character(len=*), intent(IN) :: name
        type(C_PTR) :: SHT_prv
        type(exclass1) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass1.method.ctor_1
        SHT_prv = c_exclass1_ctor_1_bufferify(name, &
            len_trim(name, kind=C_INT), SHT_rv%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass1.method.ctor_1
    end function exclass1_ctor_1

    ! ~ExClass1()
    !>
    !! \brief destructor
    !!
    !! longer description joined with previous line
    !<
    subroutine exclass1_dtor(obj)
        class(exclass1) :: obj
        ! splicer begin namespace.example::nested.class.ExClass1.method.delete
        call c_exclass1_dtor(obj%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass1.method.delete
    end subroutine exclass1_dtor

    ! int incrementCount(int incr +intent(in)+value)
    function exclass1_increment_count(obj, incr) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        integer(C_INT), value, intent(IN) :: incr
        integer(C_INT) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass1.method.increment_count
        SHT_rv = c_exclass1_increment_count(obj%cxxmem, incr)
        ! splicer end namespace.example::nested.class.ExClass1.method.increment_count
    end function exclass1_increment_count

    ! const string & getNameErrorPattern() const +deref(result_as_arg)+len(aa_exclass1_get_name_length({F_this}%{F_derived_member}))
    ! arg_to_buffer
    function exclass1_get_name_error_pattern(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        character(len=aa_exclass1_get_name_length({F_this}%{F_derived_member})) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass1.method.get_name_error_pattern
        call c_exclass1_get_name_error_pattern_bufferify(obj%cxxmem, &
            SHT_rv, len(SHT_rv, kind=C_INT))
        ! splicer end namespace.example::nested.class.ExClass1.method.get_name_error_pattern
    end function exclass1_get_name_error_pattern

    ! int GetNameLength() const
    !>
    !! \brief helper function for Fortran to get length of name.
    !!
    !<
    function exclass1_get_name_length(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass1.method.get_name_length
        SHT_rv = c_exclass1_get_name_length(obj%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass1.method.get_name_length
    end function exclass1_get_name_length

    ! const string & getNameErrorCheck() const +deref(allocatable)
    ! arg_to_buffer
    function exclass1_get_name_error_check(obj) &
            result(SHT_rv)
        class(exclass1) :: obj
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass1.method.get_name_error_check
        call c_exclass1_get_name_error_check_bufferify(obj%cxxmem, &
            DSHF_rv)
        ! splicer end namespace.example::nested.class.ExClass1.method.get_name_error_check
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function exclass1_get_name_error_check

    ! void getNameArg(string & name +intent(out)+len(Nname)) const
    ! arg_to_buffer - arg_to_buffer
    subroutine exclass1_get_name_arg(obj, name)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        character(len=*), intent(OUT) :: name
        ! splicer begin namespace.example::nested.class.ExClass1.method.get_name_arg
        call c_exclass1_get_name_arg_bufferify(obj%cxxmem, name, &
            len(name, kind=C_INT))
        ! splicer end namespace.example::nested.class.ExClass1.method.get_name_arg
    end subroutine exclass1_get_name_arg

    ! void * getRoot()
    function exclass1_get_root(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(exclass1) :: obj
        type(C_PTR) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass1.method.get_root
        SHT_rv = c_exclass1_get_root(obj%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass1.method.get_root
    end function exclass1_get_root

    ! int getValue(int value +intent(in)+value)
    function exclass1_get_value_from_int(obj, value) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        integer(C_INT), value, intent(IN) :: value
        integer(C_INT) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass1.method.get_value_from_int
        SHT_rv = c_exclass1_get_value_from_int(obj%cxxmem, value)
        ! splicer end namespace.example::nested.class.ExClass1.method.get_value_from_int
    end function exclass1_get_value_from_int

    ! long getValue(long value +intent(in)+value)
    function exclass1_get_value_1(obj, value) &
            result(SHT_rv)
        use iso_c_binding, only : C_LONG
        class(exclass1) :: obj
        integer(C_LONG), value, intent(IN) :: value
        integer(C_LONG) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass1.method.get_value_1
        SHT_rv = c_exclass1_get_value_1(obj%cxxmem, value)
        ! splicer end namespace.example::nested.class.ExClass1.method.get_value_1
    end function exclass1_get_value_1

    ! void * getAddr()
    function exclass1_get_addr(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(exclass1) :: obj
        type(C_PTR) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass1.method.get_addr
        SHT_rv = c_exclass1_get_addr(obj%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass1.method.get_addr
    end function exclass1_get_addr

    ! bool hasAddr(bool in +intent(in)+value)
    function exclass1_has_addr(obj, in) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        class(exclass1) :: obj
        logical, value, intent(IN) :: in
        logical(C_BOOL) SH_in
        logical :: SHT_rv
        SH_in = in  ! coerce to C_BOOL
        ! splicer begin namespace.example::nested.class.ExClass1.method.has_addr
        SHT_rv = c_exclass1_has_addr(obj%cxxmem, SH_in)
        ! splicer end namespace.example::nested.class.ExClass1.method.has_addr
    end function exclass1_has_addr

    ! void SplicerSpecial()
    subroutine exclass1_splicer_special(obj)
        class(exclass1) :: obj
        ! splicer begin namespace.example::nested.class.ExClass1.method.splicer_special
        blah blah blah
        ! splicer end namespace.example::nested.class.ExClass1.method.splicer_special
    end subroutine exclass1_splicer_special

    ! Return pointer to C++ memory.
    function exclass1_yadda(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(exclass1), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function exclass1_yadda

    function exclass1_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(exclass1), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function exclass1_associated

    ! splicer begin namespace.example::nested.class.ExClass1.additional_functions
      insert extra functions here
    ! splicer end namespace.example::nested.class.ExClass1.additional_functions

    ! ExClass2(const string * name +intent(in)+len_trim(trim_name))
    ! arg_to_buffer
    !>
    !! \brief constructor
    !!
    !<
    function exclass2_ctor(name) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR
        character(len=*), intent(IN) :: name
        type(C_PTR) :: SHT_prv
        type(exclass2) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass2.method.ctor
        SHT_prv = c_exclass2_ctor_bufferify(name, &
            len_trim(name, kind=C_INT), SHT_rv%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass2.method.ctor
    end function exclass2_ctor

    ! ~ExClass2()
    !>
    !! \brief destructor
    !!
    !<
    subroutine exclass2_dtor(obj)
        class(exclass2) :: obj
        ! splicer begin namespace.example::nested.class.ExClass2.method.delete
        call c_exclass2_dtor(obj%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass2.method.delete
    end subroutine exclass2_dtor

    ! const string & getName() const +deref(result_as_arg)+len(aa_exclass2_get_name_length({F_this}%{F_derived_member}))
    ! arg_to_buffer
    function exclass2_get_name(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        character(len=aa_exclass2_get_name_length({F_this}%{F_derived_member})) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass2.method.get_name
        call c_exclass2_get_name_bufferify(obj%cxxmem, SHT_rv, &
            len(SHT_rv, kind=C_INT))
        ! splicer end namespace.example::nested.class.ExClass2.method.get_name
    end function exclass2_get_name

    ! const string & getName2() +deref(allocatable)
    ! arg_to_buffer
    function exclass2_get_name2(obj) &
            result(SHT_rv)
        class(exclass2) :: obj
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass2.method.get_name2
        call c_exclass2_get_name2_bufferify(obj%cxxmem, DSHF_rv)
        ! splicer end namespace.example::nested.class.ExClass2.method.get_name2
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function exclass2_get_name2

    ! string & getName3() const +deref(allocatable)
    ! arg_to_buffer
    function exclass2_get_name3(obj) &
            result(SHT_rv)
        class(exclass2) :: obj
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass2.method.get_name3
        call c_exclass2_get_name3_bufferify(obj%cxxmem, DSHF_rv)
        ! splicer end namespace.example::nested.class.ExClass2.method.get_name3
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function exclass2_get_name3

    ! string & getName4() +deref(allocatable)
    ! arg_to_buffer
    function exclass2_get_name4(obj) &
            result(SHT_rv)
        class(exclass2) :: obj
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass2.method.get_name4
        call c_exclass2_get_name4_bufferify(obj%cxxmem, DSHF_rv)
        ! splicer end namespace.example::nested.class.ExClass2.method.get_name4
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function exclass2_get_name4

    ! int GetNameLength() const
    !>
    !! \brief helper function for Fortran
    !!
    !<
    function exclass2_get_name_length(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass2.method.get_name_length
        SHT_rv = c_exclass2_get_name_length(obj%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass2.method.get_name_length
    end function exclass2_get_name_length

    ! ExClass1 * get_class1(const ExClass1 * in +intent(in))
    function exclass2_get_class1(obj, in) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(exclass2) :: obj
        type(exclass1), intent(IN) :: in
        type(C_PTR) :: SHT_prv
        type(exclass1) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass2.method.get_class1
        SHT_prv = c_exclass2_get_class1(obj%cxxmem, in%cxxmem, &
            SHT_rv%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass2.method.get_class1
    end function exclass2_get_class1

    ! void * declare(TypeID type +intent(in)+value, int len=1 +intent(in)+value)
    ! fortran_generic - has_default_arg
    subroutine exclass2_declare_0_int(obj, type)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT), value, intent(IN) :: type
        ! splicer begin namespace.example::nested.class.ExClass2.method.declare_0_int
        call c_exclass2_declare_0(obj%cxxmem, type)
        ! splicer end namespace.example::nested.class.ExClass2.method.declare_0_int
    end subroutine exclass2_declare_0_int

    ! void * declare(TypeID type +intent(in)+value, long len=1 +intent(in)+value)
    ! fortran_generic - has_default_arg
    subroutine exclass2_declare_0_long(obj, type)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT), value, intent(IN) :: type
        ! splicer begin namespace.example::nested.class.ExClass2.method.declare_0_long
        call c_exclass2_declare_0(obj%cxxmem, type)
        ! splicer end namespace.example::nested.class.ExClass2.method.declare_0_long
    end subroutine exclass2_declare_0_long

    ! void * declare(TypeID type +intent(in)+value, int len=1 +intent(in)+value)
    ! fortran_generic
    subroutine exclass2_declare_1_int(obj, type, len)
        use iso_c_binding, only : C_INT, C_LONG
        class(exclass2) :: obj
        integer(C_INT), value, intent(IN) :: type
        integer(C_INT), value, intent(IN) :: len
        ! splicer begin namespace.example::nested.class.ExClass2.method.declare_1_int
        call c_exclass2_declare_1(obj%cxxmem, type, int(len, C_LONG))
        ! splicer end namespace.example::nested.class.ExClass2.method.declare_1_int
    end subroutine exclass2_declare_1_int

    ! void * declare(TypeID type +intent(in)+value, long len=1 +intent(in)+value)
    ! fortran_generic
    subroutine exclass2_declare_1_long(obj, type, len)
        use iso_c_binding, only : C_INT, C_LONG
        class(exclass2) :: obj
        integer(C_INT), value, intent(IN) :: type
        integer(C_LONG), value, intent(IN) :: len
        ! splicer begin namespace.example::nested.class.ExClass2.method.declare_1_long
        call c_exclass2_declare_1(obj%cxxmem, type, int(len, C_LONG))
        ! splicer end namespace.example::nested.class.ExClass2.method.declare_1_long
    end subroutine exclass2_declare_1_long

    ! void destroyall()
    subroutine exclass2_destroyall(obj)
        class(exclass2) :: obj
        ! splicer begin namespace.example::nested.class.ExClass2.method.destroyall
        call c_exclass2_destroyall(obj%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass2.method.destroyall
    end subroutine exclass2_destroyall

    ! TypeID getTypeID() const
    function exclass2_get_type_id(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass2.method.get_type_id
        SHT_rv = c_exclass2_get_type_id(obj%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass2.method.get_type_id
    end function exclass2_get_type_id

    ! void setValue(int value +intent(in)+value)
    ! cxx_template
    subroutine exclass2_set_value_int(obj, value)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT), value, intent(IN) :: value
        ! splicer begin namespace.example::nested.class.ExClass2.method.set_value_int
        call c_exclass2_set_value_int(obj%cxxmem, value)
        ! splicer end namespace.example::nested.class.ExClass2.method.set_value_int
    end subroutine exclass2_set_value_int

    ! void setValue(long value +intent(in)+value)
    ! cxx_template
    subroutine exclass2_set_value_long(obj, value)
        use iso_c_binding, only : C_LONG
        class(exclass2) :: obj
        integer(C_LONG), value, intent(IN) :: value
        ! splicer begin namespace.example::nested.class.ExClass2.method.set_value_long
        call c_exclass2_set_value_long(obj%cxxmem, value)
        ! splicer end namespace.example::nested.class.ExClass2.method.set_value_long
    end subroutine exclass2_set_value_long

    ! void setValue(float value +intent(in)+value)
    ! cxx_template
    subroutine exclass2_set_value_float(obj, value)
        use iso_c_binding, only : C_FLOAT
        class(exclass2) :: obj
        real(C_FLOAT), value, intent(IN) :: value
        ! splicer begin namespace.example::nested.class.ExClass2.method.set_value_float
        call c_exclass2_set_value_float(obj%cxxmem, value)
        ! splicer end namespace.example::nested.class.ExClass2.method.set_value_float
    end subroutine exclass2_set_value_float

    ! void setValue(double value +intent(in)+value)
    ! cxx_template
    subroutine exclass2_set_value_double(obj, value)
        use iso_c_binding, only : C_DOUBLE
        class(exclass2) :: obj
        real(C_DOUBLE), value, intent(IN) :: value
        ! splicer begin namespace.example::nested.class.ExClass2.method.set_value_double
        call c_exclass2_set_value_double(obj%cxxmem, value)
        ! splicer end namespace.example::nested.class.ExClass2.method.set_value_double
    end subroutine exclass2_set_value_double

    ! int getValue()
    ! cxx_template
    function exclass2_get_value_int(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass2.method.get_value_int
        SHT_rv = c_exclass2_get_value_int(obj%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass2.method.get_value_int
    end function exclass2_get_value_int

    ! double getValue()
    ! cxx_template
    function exclass2_get_value_double(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        class(exclass2) :: obj
        real(C_DOUBLE) :: SHT_rv
        ! splicer begin namespace.example::nested.class.ExClass2.method.get_value_double
        SHT_rv = c_exclass2_get_value_double(obj%cxxmem)
        ! splicer end namespace.example::nested.class.ExClass2.method.get_value_double
    end function exclass2_get_value_double

    ! Return pointer to C++ memory.
    function exclass2_yadda(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(exclass2), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function exclass2_yadda

    function exclass2_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(exclass2), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function exclass2_associated

    ! splicer begin namespace.example::nested.class.ExClass2.additional_functions
    ! splicer end namespace.example::nested.class.ExClass2.additional_functions

    ! bool isNameValid(const std::string & name +intent(in))
    ! arg_to_buffer
    function is_name_valid(name) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_INT
        character(len=*), intent(IN) :: name
        logical :: SHT_rv
        ! splicer begin namespace.example::nested.function.is_name_valid
        rv = name .ne. " "
        ! splicer end namespace.example::nested.function.is_name_valid
    end function is_name_valid

    ! bool isInitialized()
    function is_initialized() &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        logical :: SHT_rv
        ! splicer begin namespace.example::nested.function.is_initialized
        SHT_rv = c_is_initialized()
        ! splicer end namespace.example::nested.function.is_initialized
    end function is_initialized

    ! void test_names(const std::string & name +intent(in))
    ! arg_to_buffer
    subroutine test_names(name)
        use iso_c_binding, only : C_INT
        character(len=*), intent(IN) :: name
        ! splicer begin namespace.example::nested.function.test_names
        call c_test_names_bufferify(name, len_trim(name, kind=C_INT))
        ! splicer end namespace.example::nested.function.test_names
    end subroutine test_names

    ! void test_names(const std::string & name +intent(in), int flag +intent(in)+value)
    ! arg_to_buffer
    subroutine test_names_flag(name, flag)
        use iso_c_binding, only : C_INT
        character(len=*), intent(IN) :: name
        integer(C_INT), value, intent(IN) :: flag
        ! splicer begin namespace.example::nested.function.test_names_flag
        call c_test_names_flag_bufferify(name, &
            len_trim(name, kind=C_INT), flag)
        ! splicer end namespace.example::nested.function.test_names_flag
    end subroutine test_names_flag

    ! void testoptional()
    ! has_default_arg
    subroutine testoptional_0()
        ! splicer begin namespace.example::nested.function.testoptional_0
        call c_testoptional_0()
        ! splicer end namespace.example::nested.function.testoptional_0
    end subroutine testoptional_0

    ! void testoptional(int i=1 +intent(in)+value)
    ! has_default_arg
    subroutine testoptional_1(i)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: i
        ! splicer begin namespace.example::nested.function.testoptional_1
        call c_testoptional_1(i)
        ! splicer end namespace.example::nested.function.testoptional_1
    end subroutine testoptional_1

    ! void testoptional(int i=1 +intent(in)+value, long j=2 +intent(in)+value)
    subroutine testoptional_2(i, j)
        use iso_c_binding, only : C_INT, C_LONG
        integer(C_INT), value, intent(IN) :: i
        integer(C_LONG), value, intent(IN) :: j
        ! splicer begin namespace.example::nested.function.testoptional_2
        call c_testoptional_2(i, j)
        ! splicer end namespace.example::nested.function.testoptional_2
    end subroutine testoptional_2

#ifdef HAVE_MPI
    ! void testmpi(MPI_Comm comm +intent(in)+value)
    subroutine testmpi_mpi(comm)
        integer, value, intent(IN) :: comm
        ! splicer begin namespace.example::nested.function.testmpi_mpi
        call c_testmpi_mpi(comm)
        ! splicer end namespace.example::nested.function.testmpi_mpi
    end subroutine testmpi_mpi
#endif

#ifndef HAVE_MPI
    ! void testmpi()
    subroutine testmpi_serial()
        ! splicer begin namespace.example::nested.function.testmpi_serial
        call c_testmpi_serial()
        ! splicer end namespace.example::nested.function.testmpi_serial
    end subroutine testmpi_serial
#endif

    ! void testgroup1(axom::sidre::Group * grp +intent(in))
    subroutine testgroup1(grp)
        use sidre_mod, only : datagroup
        type(datagroup), intent(IN) :: grp
        ! splicer begin namespace.example::nested.function.testgroup1
        call c_testgroup1(grp%cxxmem)
        ! splicer end namespace.example::nested.function.testgroup1
    end subroutine testgroup1

    ! void testgroup2(const axom::sidre::Group * grp +intent(in))
    subroutine testgroup2(grp)
        use sidre_mod, only : datagroup
        type(datagroup), intent(IN) :: grp
        ! splicer begin namespace.example::nested.function.testgroup2
        call c_testgroup2(grp%cxxmem)
        ! splicer end namespace.example::nested.function.testgroup2
    end subroutine testgroup2

    ! void FuncPtr3(double ( * get)(int i +value, int +value) +intent(in)+value)
    !>
    !! \brief abstract argument
    !!
    !<
    subroutine func_ptr3(get)
        procedure(func_ptr3_get) :: get
        ! splicer begin namespace.example::nested.function.func_ptr3
        call c_func_ptr3(get)
        ! splicer end namespace.example::nested.function.func_ptr3
    end subroutine func_ptr3

    ! void FuncPtr4(double ( * get)(double +value, int +value) +intent(in)+value)
    !>
    !! \brief abstract argument
    !!
    !<
    subroutine func_ptr4(get)
        procedure(custom_funptr) :: get
        ! splicer begin namespace.example::nested.function.func_ptr4
        call c_func_ptr4(get)
        ! splicer end namespace.example::nested.function.func_ptr4
    end subroutine func_ptr4

    ! void verylongfunctionname1(int * verylongname1 +intent(inout), int * verylongname2 +intent(inout), int * verylongname3 +intent(inout), int * verylongname4 +intent(inout), int * verylongname5 +intent(inout), int * verylongname6 +intent(inout), int * verylongname7 +intent(inout), int * verylongname8 +intent(inout), int * verylongname9 +intent(inout), int * verylongname10 +intent(inout))
    subroutine verylongfunctionname1(verylongname1, verylongname2, &
            verylongname3, verylongname4, verylongname5, verylongname6, &
            verylongname7, verylongname8, verylongname9, verylongname10)
        use iso_c_binding, only : C_INT
        integer(C_INT), intent(INOUT) :: verylongname1
        integer(C_INT), intent(INOUT) :: verylongname2
        integer(C_INT), intent(INOUT) :: verylongname3
        integer(C_INT), intent(INOUT) :: verylongname4
        integer(C_INT), intent(INOUT) :: verylongname5
        integer(C_INT), intent(INOUT) :: verylongname6
        integer(C_INT), intent(INOUT) :: verylongname7
        integer(C_INT), intent(INOUT) :: verylongname8
        integer(C_INT), intent(INOUT) :: verylongname9
        integer(C_INT), intent(INOUT) :: verylongname10
        ! splicer begin namespace.example::nested.function.verylongfunctionname1
        call c_verylongfunctionname1(verylongname1, verylongname2, &
            verylongname3, verylongname4, verylongname5, verylongname6, &
            verylongname7, verylongname8, verylongname9, verylongname10)
        ! splicer end namespace.example::nested.function.verylongfunctionname1
    end subroutine verylongfunctionname1

    ! int verylongfunctionname2(int verylongname1 +intent(in)+value, int verylongname2 +intent(in)+value, int verylongname3 +intent(in)+value, int verylongname4 +intent(in)+value, int verylongname5 +intent(in)+value, int verylongname6 +intent(in)+value, int verylongname7 +intent(in)+value, int verylongname8 +intent(in)+value, int verylongname9 +intent(in)+value, int verylongname10 +intent(in)+value)
    function verylongfunctionname2(verylongname1, verylongname2, &
            verylongname3, verylongname4, verylongname5, verylongname6, &
            verylongname7, verylongname8, verylongname9, verylongname10) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: verylongname1
        integer(C_INT), value, intent(IN) :: verylongname2
        integer(C_INT), value, intent(IN) :: verylongname3
        integer(C_INT), value, intent(IN) :: verylongname4
        integer(C_INT), value, intent(IN) :: verylongname5
        integer(C_INT), value, intent(IN) :: verylongname6
        integer(C_INT), value, intent(IN) :: verylongname7
        integer(C_INT), value, intent(IN) :: verylongname8
        integer(C_INT), value, intent(IN) :: verylongname9
        integer(C_INT), value, intent(IN) :: verylongname10
        integer(C_INT) :: SHT_rv
        ! splicer begin namespace.example::nested.function.verylongfunctionname2
        SHT_rv = c_verylongfunctionname2(verylongname1, verylongname2, &
            verylongname3, verylongname4, verylongname5, verylongname6, &
            verylongname7, verylongname8, verylongname9, verylongname10)
        ! splicer end namespace.example::nested.function.verylongfunctionname2
    end function verylongfunctionname2

    ! void cos_doubles(double * in +dimension(:,:)+intent(in), double * out +allocatable(mold=in)+dimension(:,:)+intent(out), int sizein +implied(size(in))+intent(in)+value)
    !>
    !! \brief Test multidimensional arrays with allocatable
    !!
    !<
    subroutine cos_doubles(in, out)
        use iso_c_binding, only : C_DOUBLE, C_INT
        real(C_DOUBLE), intent(IN) :: in(:,:)
        real(C_DOUBLE), intent(OUT), allocatable :: out(:)
        integer(C_INT) :: SH_sizein
        allocate(out, mold=in)
        SH_sizein = size(in,kind=C_INT)
        ! splicer begin namespace.example::nested.function.cos_doubles
        call c_cos_doubles(in, out, SH_sizein)
        ! splicer end namespace.example::nested.function.cos_doubles
    end subroutine cos_doubles

    ! splicer begin namespace.example::nested.additional_functions
    ! splicer end namespace.example::nested.additional_functions

    function exclass1_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(exclass1), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function exclass1_eq

    function exclass1_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(exclass1), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function exclass1_ne

    function exclass2_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(exclass2), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function exclass2_eq

    function exclass2_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(exclass2), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function exclass2_ne

end module userlibrary_example_nested_mod
