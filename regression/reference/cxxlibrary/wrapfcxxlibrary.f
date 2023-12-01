! wrapfcxxlibrary.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfcxxlibrary.f
!! \brief Shroud generated wrapper for cxxlibrary library
!<
! splicer begin file_top
! splicer end file_top
module cxxlibrary_mod
    use iso_c_binding, only : C_INT, C_LONG, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! helper type_defines
    ! Shroud type defines from helper type_defines
    integer, parameter, private :: &
        SH_TYPE_SIGNED_CHAR= 1, &
        SH_TYPE_SHORT      = 2, &
        SH_TYPE_INT        = 3, &
        SH_TYPE_LONG       = 4, &
        SH_TYPE_LONG_LONG  = 5, &
        SH_TYPE_SIZE_T     = 6, &
        SH_TYPE_UNSIGNED_SHORT      = SH_TYPE_SHORT + 100, &
        SH_TYPE_UNSIGNED_INT        = SH_TYPE_INT + 100, &
        SH_TYPE_UNSIGNED_LONG       = SH_TYPE_LONG + 100, &
        SH_TYPE_UNSIGNED_LONG_LONG  = SH_TYPE_LONG_LONG + 100, &
        SH_TYPE_INT8_T    =  7, &
        SH_TYPE_INT16_T   =  8, &
        SH_TYPE_INT32_T   =  9, &
        SH_TYPE_INT64_T   = 10, &
        SH_TYPE_UINT8_T  =  SH_TYPE_INT8_T + 100, &
        SH_TYPE_UINT16_T =  SH_TYPE_INT16_T + 100, &
        SH_TYPE_UINT32_T =  SH_TYPE_INT32_T + 100, &
        SH_TYPE_UINT64_T =  SH_TYPE_INT64_T + 100, &
        SH_TYPE_FLOAT       = 22, &
        SH_TYPE_DOUBLE      = 23, &
        SH_TYPE_LONG_DOUBLE = 24, &
        SH_TYPE_FLOAT_COMPLEX      = 25, &
        SH_TYPE_DOUBLE_COMPLEX     = 26, &
        SH_TYPE_LONG_DOUBLE_COMPLEX= 27, &
        SH_TYPE_BOOL      = 28, &
        SH_TYPE_CHAR      = 29, &
        SH_TYPE_CPTR      = 30, &
        SH_TYPE_STRUCT    = 31, &
        SH_TYPE_OTHER     = 32

    ! helper array_context
    type, bind(C) :: CXX_SHROUD_array
        ! address of data
        type(C_PTR) :: base_addr = C_NULL_PTR
        ! type of element
        integer(C_INT) :: type
        ! bytes-per-item or character len of data in cxx
        integer(C_SIZE_T) :: elem_len = 0_C_SIZE_T
        ! size of data in cxx
        integer(C_SIZE_T) :: size = 0_C_SIZE_T
        ! number of dimensions
        integer(C_INT) :: rank = -1
        integer(C_LONG) :: shape(7) = 0
    end type CXX_SHROUD_array

    ! helper capsule_data_helper
    type, bind(C) :: CXX_SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type CXX_SHROUD_capsule_data


    type, bind(C) :: nested
        integer(C_INT) :: index
        integer(C_INT) :: sublevels
        type(C_PTR) :: parent
        type(C_PTR) :: child
    end type nested

    type class1
        type(CXX_SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.Class1.component_part
        ! splicer end class.Class1.component_part
    contains
        procedure :: check_length_0 => class1_check_length_0
        procedure :: check_length_1_int => class1_check_length_1_int
        procedure :: check_length_1_long => class1_check_length_1_long
        procedure :: declare_0 => class1_declare_0
        procedure :: declare_1_int => class1_declare_1_int
        procedure :: declare_1_long => class1_declare_1_long
        procedure :: get_length => class1_get_length
        procedure :: get_instance => class1_get_instance
        procedure :: set_instance => class1_set_instance
        procedure :: associated => class1_associated
        generic :: check_length => check_length_0, check_length_1_int,  &
            check_length_1_long
        generic :: declare => declare_0, declare_1_int, declare_1_long
        ! splicer begin class.Class1.type_bound_procedure_part
        ! splicer end class.Class1.type_bound_procedure_part
    end type class1

    interface operator (.eq.)
        module procedure class1_eq
    end interface

    interface operator (.ne.)
        module procedure class1_ne
    end interface

    interface

        ! ----------------------------------------
        ! Function:  Class1
        ! Statement: f_ctor_shadow_scalar_capptr
        function c_class1_ctor(SHT_rv) &
                result(SHT_prv) &
                bind(C, name="CXX_Class1_ctor")
            use iso_c_binding, only : C_PTR
            import :: CXX_SHROUD_capsule_data
            implicit none
            type(CXX_SHROUD_capsule_data), intent(OUT) :: SHT_rv
            type(C_PTR) :: SHT_prv
        end function c_class1_ctor

        ! Generated by has_default_arg
        ! ----------------------------------------
        ! Function:  int check_length
        ! Statement: f_function_native_scalar
        function c_class1_check_length_0(self) &
                result(SHT_rv) &
                bind(C, name="CXX_Class1_check_length_0")
            use iso_c_binding, only : C_INT
            import :: CXX_SHROUD_capsule_data
            implicit none
            type(CXX_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_class1_check_length_0

        ! ----------------------------------------
        ! Function:  int check_length
        ! Statement: c_function_native_scalar
        ! ----------------------------------------
        ! Argument:  int length=1
        ! Statement: c_in_native_scalar
        function c_class1_check_length_1(self, length) &
                result(SHT_rv) &
                bind(C, name="CXX_Class1_check_length_1")
            use iso_c_binding, only : C_INT
            import :: CXX_SHROUD_capsule_data
            implicit none
            type(CXX_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: length
            integer(C_INT) :: SHT_rv
        end function c_class1_check_length_1

        ! Generated by has_default_arg
        ! ----------------------------------------
        ! Function:  Class1 * declare
        ! Statement: f_function_shadow_*_this
        ! ----------------------------------------
        ! Argument:  int flag
        ! Statement: f_in_native_scalar
        subroutine c_class1_declare_0(self, flag) &
                bind(C, name="CXX_Class1_declare_0")
            use iso_c_binding, only : C_INT
            import :: CXX_SHROUD_capsule_data
            implicit none
            type(CXX_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: flag
        end subroutine c_class1_declare_0

        ! ----------------------------------------
        ! Function:  Class1 * declare
        ! Statement: c_function_shadow_*_this
        ! ----------------------------------------
        ! Argument:  int flag
        ! Statement: c_in_native_scalar
        ! ----------------------------------------
        ! Argument:  int length=1
        ! Statement: c_in_native_scalar
        subroutine c_class1_declare_1(self, flag, length) &
                bind(C, name="CXX_Class1_declare_1")
            use iso_c_binding, only : C_INT
            import :: CXX_SHROUD_capsule_data
            implicit none
            type(CXX_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: flag
            integer(C_INT), value, intent(IN) :: length
        end subroutine c_class1_declare_1

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  int get_length +intent(getter)
        ! Statement: f_getter_native_scalar
        function c_class1_get_length(self) &
                result(SHT_rv) &
                bind(C, name="CXX_Class1_get_length")
            use iso_c_binding, only : C_INT
            import :: CXX_SHROUD_capsule_data
            implicit none
            type(CXX_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_class1_get_length

        ! Generated by has_default_arg
        ! ----------------------------------------
        ! Function:  bool defaultPtrIsNULL
        ! Statement: f_function_bool_scalar
        function c_default_ptr_is_null_0() &
                result(SHT_rv) &
                bind(C, name="CXX_defaultPtrIsNULL_0")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL) :: SHT_rv
        end function c_default_ptr_is_null_0

        ! ----------------------------------------
        ! Function:  bool defaultPtrIsNULL
        ! Statement: f_function_bool_scalar
        ! ----------------------------------------
        ! Argument:  double * data=nullptr +intent(IN)+rank(1)
        ! Statement: f_in_native_*
        function c_default_ptr_is_null_1(data) &
                result(SHT_rv) &
                bind(C, name="CXX_defaultPtrIsNULL_1")
            use iso_c_binding, only : C_BOOL, C_DOUBLE
            implicit none
            real(C_DOUBLE), intent(IN) :: data(*)
            logical(C_BOOL) :: SHT_rv
        end function c_default_ptr_is_null_1

        ! Generated by has_default_arg
        ! ----------------------------------------
        ! Function:  void defaultArgsInOut
        ! Statement: f_subroutine
        ! ----------------------------------------
        ! Argument:  int in1
        ! Statement: f_in_native_scalar
        ! ----------------------------------------
        ! Argument:  int * out1 +intent(out)
        ! Statement: f_out_native_*
        ! ----------------------------------------
        ! Argument:  int * out2 +intent(out)
        ! Statement: f_out_native_*
        subroutine c_default_args_in_out_0(in1, out1, out2) &
                bind(C, name="CXX_defaultArgsInOut_0")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: in1
            integer(C_INT), intent(OUT) :: out1
            integer(C_INT), intent(OUT) :: out2
        end subroutine c_default_args_in_out_0

        ! ----------------------------------------
        ! Function:  void defaultArgsInOut
        ! Statement: f_subroutine
        ! ----------------------------------------
        ! Argument:  int in1
        ! Statement: f_in_native_scalar
        ! ----------------------------------------
        ! Argument:  int * out1 +intent(out)
        ! Statement: f_out_native_*
        ! ----------------------------------------
        ! Argument:  int * out2 +intent(out)
        ! Statement: f_out_native_*
        ! ----------------------------------------
        ! Argument:  bool flag=false
        ! Statement: f_in_bool_scalar
        subroutine c_default_args_in_out_1(in1, out1, out2, flag) &
                bind(C, name="CXX_defaultArgsInOut_1")
            use iso_c_binding, only : C_BOOL, C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: in1
            integer(C_INT), intent(OUT) :: out1
            integer(C_INT), intent(OUT) :: out2
            logical(C_BOOL), value, intent(IN) :: flag
        end subroutine c_default_args_in_out_1

        ! ----------------------------------------
        ! Function:  const std::string & getGroupName +len(30)
        ! Statement: c_function_string_&
        ! ----------------------------------------
        ! Argument:  long idx
        ! Statement: c_in_native_scalar
        function c_get_group_name(idx) &
                result(SHT_rv) &
                bind(C, name="CXX_getGroupName")
            use iso_c_binding, only : C_LONG, C_PTR
            implicit none
            integer(C_LONG), value, intent(IN) :: idx
            type(C_PTR) SHT_rv
        end function c_get_group_name

        ! Generated by fortran_generic
        ! ----------------------------------------
        ! Function:  const std::string & getGroupName +len(30)
        ! Statement: f_function_string_&_buf_copy
        ! ----------------------------------------
        ! Argument:  int32_t idx
        ! Statement: f_in_native_scalar
        subroutine c_get_group_name_int32_t_bufferify(idx, SHT_rv, &
                SHT_rv_len) &
                bind(C, name="CXX_getGroupName_int32_t_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_INT32_T
            implicit none
            integer(C_INT32_T), value, intent(IN) :: idx
            character(kind=C_CHAR), intent(OUT) :: SHT_rv(*)
            integer(C_INT), value, intent(IN) :: SHT_rv_len
        end subroutine c_get_group_name_int32_t_bufferify

        ! Generated by fortran_generic
        ! ----------------------------------------
        ! Function:  const std::string & getGroupName +len(30)
        ! Statement: f_function_string_&_buf_copy
        ! ----------------------------------------
        ! Argument:  int64_t idx
        ! Statement: f_in_native_scalar
        subroutine c_get_group_name_int64_t_bufferify(idx, SHT_rv, &
                SHT_rv_len) &
                bind(C, name="CXX_getGroupName_int64_t_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_INT64_T
            implicit none
            integer(C_INT64_T), value, intent(IN) :: idx
            character(kind=C_CHAR), intent(OUT) :: SHT_rv(*)
            integer(C_INT), value, intent(IN) :: SHT_rv_len
        end subroutine c_get_group_name_int64_t_bufferify

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  nested * nested_get_parent +intent(getter)
        ! Statement: f_getter_struct_*_cdesc_pointer
        ! ----------------------------------------
        ! Argument:  nested * SH_this +intent(in)
        ! Statement: f_in_struct_*
        subroutine c_nested_get_parent(SH_this, SHT_rv_cdesc) &
                bind(C, name="CXX_nested_get_parent")
            import :: CXX_SHROUD_array, nested
            implicit none
            type(nested), intent(IN) :: SH_this
            type(CXX_SHROUD_array), intent(OUT) :: SHT_rv_cdesc
        end subroutine c_nested_get_parent

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  void nested_set_parent +intent(setter)
        ! Statement: f_setter
        ! ----------------------------------------
        ! Argument:  nested * SH_this
        ! Statement: f_inout_struct_*
        ! ----------------------------------------
        ! Argument:  nested * val +intent(setter)
        ! Statement: f_setter_struct_*
        subroutine nested_set_parent(SH_this, val) &
                bind(C, name="CXX_nested_set_parent")
            import :: nested
            implicit none
            type(nested), intent(INOUT) :: SH_this
            type(nested), intent(IN) :: val
        end subroutine nested_set_parent

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  nested * * nested_get_child +dimension(sublevels)+intent(getter)
        ! Statement: f_getter_struct_**_cdesc_raw
        ! ----------------------------------------
        ! Argument:  nested * SH_this +intent(in)
        ! Statement: f_in_struct_*
        subroutine c_nested_get_child(SH_this, SHT_rv_cdesc) &
                bind(C, name="CXX_nested_get_child")
            import :: CXX_SHROUD_array, nested
            implicit none
            type(nested), intent(IN) :: SH_this
            type(CXX_SHROUD_array), intent(OUT) :: SHT_rv_cdesc
        end subroutine c_nested_get_child

        ! Generated by getter/setter
        ! ----------------------------------------
        ! Function:  void nested_set_child +intent(setter)
        ! Statement: f_setter
        ! ----------------------------------------
        ! Argument:  nested * SH_this
        ! Statement: f_inout_struct_*
        ! ----------------------------------------
        ! Argument:  nested * * val +intent(setter)+rank(1)
        ! Statement: f_setter_struct_**
        subroutine nested_set_child(SH_this, val) &
                bind(C, name="CXX_nested_set_child")
            import :: nested
            implicit none
            type(nested), intent(INOUT) :: SH_this
            type(nested), intent(IN) :: val(*)
        end subroutine nested_set_child
    end interface

    interface class1
        module procedure class1_ctor
    end interface class1

    interface class1_check_length
        module procedure class1_check_length_0
        module procedure class1_check_length_1_int
        module procedure class1_check_length_1_long
    end interface class1_check_length

    interface class1_declare
        module procedure class1_declare_0
        module procedure class1_declare_1_int
        module procedure class1_declare_1_long
    end interface class1_declare

    interface default_args_in_out
        module procedure default_args_in_out_0
        module procedure default_args_in_out_1
    end interface default_args_in_out

    interface default_ptr_is_null
        module procedure default_ptr_is_null_0
        module procedure default_ptr_is_null_1
    end interface default_ptr_is_null

    interface get_group_name
        module procedure get_group_name_int32_t
        module procedure get_group_name_int64_t
    end interface get_group_name

    ! splicer begin additional_declarations
    ! splicer end additional_declarations

contains

    ! ----------------------------------------
    ! Function:  Class1
    ! Statement: f_ctor_shadow_scalar_capptr
    function class1_ctor() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(class1) :: SHT_rv
        type(C_PTR) :: SHT_prv
        ! splicer begin class.Class1.method.ctor
        SHT_prv = c_class1_ctor(SHT_rv%cxxmem)
        ! splicer end class.Class1.method.ctor
    end function class1_ctor

    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  int check_length
    ! Statement: f_function_native_scalar
    !>
    !! \brief Test fortran_generic with default arguments.
    !!
    !<
    function class1_check_length_0(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.check_length_0
        SHT_rv = c_class1_check_length_0(obj%cxxmem)
        ! splicer end class.Class1.method.check_length_0
    end function class1_check_length_0

    ! Generated by fortran_generic
    ! ----------------------------------------
    ! Function:  int check_length
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  int length=1
    ! Statement: f_in_native_scalar
    !>
    !! \brief Test fortran_generic with default arguments.
    !!
    !<
    function class1_check_length_1_int(obj, length) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT), value, intent(IN) :: length
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.check_length_1_int
        SHT_rv = c_class1_check_length_1(obj%cxxmem, length)
        ! splicer end class.Class1.method.check_length_1_int
    end function class1_check_length_1_int

    ! Generated by fortran_generic
    ! ----------------------------------------
    ! Function:  int check_length
    ! Statement: f_function_native_scalar
    ! ----------------------------------------
    ! Argument:  long length=1
    ! Statement: f_in_native_scalar
    ! Argument:  int length=1
    !>
    !! \brief Test fortran_generic with default arguments.
    !!
    !<
    function class1_check_length_1_long(obj, length) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_LONG
        class(class1) :: obj
        integer(C_LONG), value, intent(IN) :: length
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.check_length_1_long
        SHT_rv = c_class1_check_length_1(obj%cxxmem, int(length, C_INT))
        ! splicer end class.Class1.method.check_length_1_long
    end function class1_check_length_1_long

    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  Class1 * declare
    ! Statement: f_function_shadow_*_this
    ! ----------------------------------------
    ! Argument:  int flag
    ! Statement: f_in_native_scalar
    subroutine class1_declare_0(obj, flag)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT), value, intent(IN) :: flag
        ! splicer begin class.Class1.method.declare_0
        call c_class1_declare_0(obj%cxxmem, flag)
        ! splicer end class.Class1.method.declare_0
    end subroutine class1_declare_0

    ! Generated by fortran_generic
    ! ----------------------------------------
    ! Function:  Class1 * declare
    ! Statement: f_function_shadow_*_this
    ! ----------------------------------------
    ! Argument:  int flag
    ! Statement: f_in_native_scalar
    ! ----------------------------------------
    ! Argument:  int length=1
    ! Statement: f_in_native_scalar
    subroutine class1_declare_1_int(obj, flag, length)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT), value, intent(IN) :: flag
        integer(C_INT), value, intent(IN) :: length
        ! splicer begin class.Class1.method.declare_1_int
        call c_class1_declare_1(obj%cxxmem, flag, length)
        ! splicer end class.Class1.method.declare_1_int
    end subroutine class1_declare_1_int

    ! Generated by fortran_generic
    ! ----------------------------------------
    ! Function:  Class1 * declare
    ! Statement: f_function_shadow_*_this
    ! ----------------------------------------
    ! Argument:  int flag
    ! Statement: f_in_native_scalar
    ! ----------------------------------------
    ! Argument:  long length=1
    ! Statement: f_in_native_scalar
    ! Argument:  int length=1
    subroutine class1_declare_1_long(obj, flag, length)
        use iso_c_binding, only : C_INT, C_LONG
        class(class1) :: obj
        integer(C_INT), value, intent(IN) :: flag
        integer(C_LONG), value, intent(IN) :: length
        ! splicer begin class.Class1.method.declare_1_long
        call c_class1_declare_1(obj%cxxmem, flag, int(length, C_INT))
        ! splicer end class.Class1.method.declare_1_long
    end subroutine class1_declare_1_long

    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  int get_length +intent(getter)
    ! Statement: f_getter_native_scalar
    function class1_get_length(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(class1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.Class1.method.get_length
        SHT_rv = c_class1_get_length(obj%cxxmem)
        ! splicer end class.Class1.method.get_length
    end function class1_get_length

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

    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  bool defaultPtrIsNULL
    ! Statement: f_function_bool_scalar
    function default_ptr_is_null_0() &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        logical :: SHT_rv
        ! splicer begin function.default_ptr_is_null_0
        SHT_rv = c_default_ptr_is_null_0()
        ! splicer end function.default_ptr_is_null_0
    end function default_ptr_is_null_0

    ! ----------------------------------------
    ! Function:  bool defaultPtrIsNULL
    ! Statement: f_function_bool_scalar
    ! ----------------------------------------
    ! Argument:  double * data=nullptr +intent(IN)+rank(1)
    ! Statement: f_in_native_*
    function default_ptr_is_null_1(data) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_DOUBLE
        real(C_DOUBLE), intent(IN) :: data(:)
        logical :: SHT_rv
        ! splicer begin function.default_ptr_is_null_1
        SHT_rv = c_default_ptr_is_null_1(data)
        ! splicer end function.default_ptr_is_null_1
    end function default_ptr_is_null_1

    ! Generated by has_default_arg
    ! ----------------------------------------
    ! Function:  void defaultArgsInOut
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  int in1
    ! Statement: f_in_native_scalar
    ! ----------------------------------------
    ! Argument:  int * out1 +intent(out)
    ! Statement: f_out_native_*
    ! ----------------------------------------
    ! Argument:  int * out2 +intent(out)
    ! Statement: f_out_native_*
    subroutine default_args_in_out_0(in1, out1, out2)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: in1
        integer(C_INT), intent(OUT) :: out1
        integer(C_INT), intent(OUT) :: out2
        ! splicer begin function.default_args_in_out_0
        call c_default_args_in_out_0(in1, out1, out2)
        ! splicer end function.default_args_in_out_0
    end subroutine default_args_in_out_0

    ! ----------------------------------------
    ! Function:  void defaultArgsInOut
    ! Statement: f_subroutine
    ! ----------------------------------------
    ! Argument:  int in1
    ! Statement: f_in_native_scalar
    ! ----------------------------------------
    ! Argument:  int * out1 +intent(out)
    ! Statement: f_out_native_*
    ! ----------------------------------------
    ! Argument:  int * out2 +intent(out)
    ! Statement: f_out_native_*
    ! ----------------------------------------
    ! Argument:  bool flag=false
    ! Statement: f_in_bool_scalar
    subroutine default_args_in_out_1(in1, out1, out2, flag)
        use iso_c_binding, only : C_BOOL, C_INT
        integer(C_INT), value, intent(IN) :: in1
        integer(C_INT), intent(OUT) :: out1
        integer(C_INT), intent(OUT) :: out2
        logical, value, intent(IN) :: flag
        ! splicer begin function.default_args_in_out_1
        logical(C_BOOL) :: SHT_flag_cxx
        SHT_flag_cxx = flag  ! coerce to C_BOOL
        call c_default_args_in_out_1(in1, out1, out2, SHT_flag_cxx)
        ! splicer end function.default_args_in_out_1
    end subroutine default_args_in_out_1

    ! Generated by fortran_generic
    ! ----------------------------------------
    ! Function:  const std::string & getGroupName +len(30)
    ! Statement: f_function_string_&_buf_copy
    ! ----------------------------------------
    ! Argument:  int32_t idx
    ! Statement: f_in_native_scalar
    !>
    !! \brief String reference function with scalar generic args
    !!
    !<
    function get_group_name_int32_t(idx) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_INT32_T
        integer(C_INT32_T), value, intent(IN) :: idx
        character(len=30) :: SHT_rv
        ! splicer begin function.get_group_name_int32_t
        integer(C_INT) SHT_rv_len
        SHT_rv_len = len(SHT_rv, kind=C_INT)
        call c_get_group_name_int32_t_bufferify(idx, SHT_rv, SHT_rv_len)
        ! splicer end function.get_group_name_int32_t
    end function get_group_name_int32_t

    ! Generated by fortran_generic
    ! ----------------------------------------
    ! Function:  const std::string & getGroupName +len(30)
    ! Statement: f_function_string_&_buf_copy
    ! ----------------------------------------
    ! Argument:  int64_t idx
    ! Statement: f_in_native_scalar
    !>
    !! \brief String reference function with scalar generic args
    !!
    !<
    function get_group_name_int64_t(idx) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_INT64_T
        integer(C_INT64_T), value, intent(IN) :: idx
        character(len=30) :: SHT_rv
        ! splicer begin function.get_group_name_int64_t
        integer(C_INT) SHT_rv_len
        SHT_rv_len = len(SHT_rv, kind=C_INT)
        call c_get_group_name_int64_t_bufferify(idx, SHT_rv, SHT_rv_len)
        ! splicer end function.get_group_name_int64_t
    end function get_group_name_int64_t

    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  nested * nested_get_parent +intent(getter)
    ! Statement: f_getter_struct_*_cdesc_pointer
    ! ----------------------------------------
    ! Argument:  nested * SH_this +intent(in)
    ! Statement: f_in_struct_*
    function nested_get_parent(SH_this) &
            result(SHT_rv)
        use iso_c_binding, only : c_f_pointer
        type(nested), intent(IN) :: SH_this
        type(nested), pointer :: SHT_rv
        ! splicer begin function.nested_get_parent
        type(CXX_SHROUD_array) :: SHT_rv_cdesc
        call c_nested_get_parent(SH_this, SHT_rv_cdesc)
        call c_f_pointer(SHT_rv_cdesc%base_addr, SHT_rv)
        ! splicer end function.nested_get_parent
    end function nested_get_parent

#if 0
    ! Only the interface is needed
    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  void nested_set_parent +intent(setter)
    ! Statement: f_setter
    ! ----------------------------------------
    ! Argument:  nested * SH_this
    ! Statement: f_inout_struct_*
    ! ----------------------------------------
    ! Argument:  nested * val +intent(setter)
    ! Statement: f_setter_struct_*
    subroutine nested_set_parent(SH_this, val)
        type(nested), intent(INOUT) :: SH_this
        type(nested), intent(IN) :: val
        ! splicer begin function.nested_set_parent
        call c_nested_set_parent(SH_this, val)
        ! splicer end function.nested_set_parent
    end subroutine nested_set_parent
#endif

    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  nested * * nested_get_child +dimension(sublevels)+intent(getter)
    ! Statement: f_getter_struct_**_cdesc_raw
    ! ----------------------------------------
    ! Argument:  nested * SH_this +intent(in)
    ! Statement: f_in_struct_*
    function nested_get_child(SH_this) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR, c_f_pointer
        type(nested), intent(IN) :: SH_this
        type(C_PTR), pointer :: SHT_rv(:)
        ! splicer begin function.nested_get_child
        type(CXX_SHROUD_array) :: SHT_rv_cdesc
        call c_nested_get_child(SH_this, SHT_rv_cdesc)
        call c_f_pointer(SHT_rv_cdesc%base_addr, SHT_rv, &
            SHT_rv_cdesc%shape(1:1))
        ! splicer end function.nested_get_child
    end function nested_get_child

#if 0
    ! Only the interface is needed
    ! Generated by getter/setter
    ! ----------------------------------------
    ! Function:  void nested_set_child +intent(setter)
    ! Statement: f_setter
    ! ----------------------------------------
    ! Argument:  nested * SH_this
    ! Statement: f_inout_struct_*
    ! ----------------------------------------
    ! Argument:  nested * * val +intent(setter)+rank(1)
    ! Statement: f_setter_struct_**
    subroutine nested_set_child(SH_this, val)
        type(nested), intent(INOUT) :: SH_this
        type(nested), intent(IN) :: val(:)
        ! splicer begin function.nested_set_child
        call c_nested_set_child(SH_this, val)
        ! splicer end function.nested_set_child
    end subroutine nested_set_child
#endif

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

end module cxxlibrary_mod
