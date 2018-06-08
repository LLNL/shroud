! wrapfExClass2.f
! This is generated code, do not edit
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
!! \file wrapfExClass2.f
!! \brief Shroud generated wrapper for ExClass2 class
!<
! splicer begin file_top
! splicer end file_top
module exclass2_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin class.ExClass2.module_use
    ! splicer end class.ExClass2.module_use
    implicit none


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

    ! splicer begin class.ExClass2.module_top
    top of module splicer  2
    ! splicer end class.ExClass2.module_top

    type exclass2
        type(SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.ExClass2.component_part
        ! splicer end class.ExClass2.component_part
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
        ! splicer begin class.ExClass2.type_bound_procedure_part
        ! splicer end class.ExClass2.type_bound_procedure_part
    end type exclass2

    interface operator (.eq.)
        module procedure exclass2_eq
    end interface

    interface operator (.ne.)
        module procedure exclass2_ne
    end interface

    interface

        function c_exclass2_ctor(name) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_ctor")
            use iso_c_binding, only : C_CHAR
            import :: SHROUD_capsule_data
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_exclass2_ctor

        function c_exclass2_ctor_bufferify(name, trim_name) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_ctor_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_capsule_data
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: trim_name
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_exclass2_ctor_bufferify

        subroutine c_exclass2_dtor(self) &
                bind(C, name="AA_exclass2_dtor")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_exclass2_dtor

        pure function c_exclass2_get_name(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_get_name")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass2_get_name

        subroutine c_exclass2_get_name_bufferify(self, SHF_rv, NSHF_rv) &
                bind(C, name="AA_exclass2_get_name_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            character(kind=C_CHAR), intent(OUT) :: SHF_rv(*)
            integer(C_INT), value, intent(IN) :: NSHF_rv
        end subroutine c_exclass2_get_name_bufferify

        function c_exclass2_get_name2(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_get_name2")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass2_get_name2

        subroutine c_exclass2_get_name2_bufferify(self, DSHF_rv) &
                bind(C, name="AA_exclass2_get_name2_bufferify")
            import :: SHROUD_array, SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_exclass2_get_name2_bufferify

        pure function c_exclass2_get_name3(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_get_name3")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass2_get_name3

        subroutine c_exclass2_get_name3_bufferify(self, DSHF_rv) &
                bind(C, name="AA_exclass2_get_name3_bufferify")
            import :: SHROUD_array, SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_exclass2_get_name3_bufferify

        function c_exclass2_get_name4(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_get_name4")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass2_get_name4

        subroutine c_exclass2_get_name4_bufferify(self, DSHF_rv) &
                bind(C, name="AA_exclass2_get_name4_bufferify")
            import :: SHROUD_array, SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_exclass2_get_name4_bufferify

        pure function c_exclass2_get_name_length(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_get_name_length")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_exclass2_get_name_length

        function c_exclass2_get_class1(self, in) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_get_class1")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            type(SHROUD_capsule_data), intent(IN) :: in
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_exclass2_get_class1

        subroutine c_exclass2_declare_0(self, type) &
                bind(C, name="AA_exclass2_declare_0")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: type
        end subroutine c_exclass2_declare_0

        subroutine c_exclass2_declare_1(self, type, len) &
                bind(C, name="AA_exclass2_declare_1")
            use iso_c_binding, only : C_INT, C_LONG
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: type
            integer(C_LONG), value, intent(IN) :: len
        end subroutine c_exclass2_declare_1

        subroutine c_exclass2_destroyall(self) &
                bind(C, name="AA_exclass2_destroyall")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_exclass2_destroyall

        pure function c_exclass2_get_type_id(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_get_type_id")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_exclass2_get_type_id

        subroutine c_exclass2_set_value_int(self, value) &
                bind(C, name="AA_exclass2_set_value_int")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: value
        end subroutine c_exclass2_set_value_int

        subroutine c_exclass2_set_value_long(self, value) &
                bind(C, name="AA_exclass2_set_value_long")
            use iso_c_binding, only : C_LONG
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_LONG), value, intent(IN) :: value
        end subroutine c_exclass2_set_value_long

        subroutine c_exclass2_set_value_float(self, value) &
                bind(C, name="AA_exclass2_set_value_float")
            use iso_c_binding, only : C_FLOAT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            real(C_FLOAT), value, intent(IN) :: value
        end subroutine c_exclass2_set_value_float

        subroutine c_exclass2_set_value_double(self, value) &
                bind(C, name="AA_exclass2_set_value_double")
            use iso_c_binding, only : C_DOUBLE
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            real(C_DOUBLE), value, intent(IN) :: value
        end subroutine c_exclass2_set_value_double

        function c_exclass2_get_value_int(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_get_value_int")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_exclass2_get_value_int

        function c_exclass2_get_value_double(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass2_get_value_double")
            use iso_c_binding, only : C_DOUBLE
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            real(C_DOUBLE) :: SHT_rv
        end function c_exclass2_get_value_double

        ! splicer begin class.ExClass2.additional_interfaces
        ! splicer end class.ExClass2.additional_interfaces
    end interface

    interface
        ! helper function
        ! Copy the std::string in context into c_var.
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

    ! ExClass2(const string * name +intent(in)+len_trim(trim_name))
    ! arg_to_buffer
    ! function_index=19
    !>
    !! \brief constructor
    !!
    !<
    function exclass2_ctor(name) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        character(len=*), intent(IN) :: name
        type(exclass2) :: SHT_rv
        ! splicer begin class.ExClass2.method.ctor
        SHT_rv%cxxmem = c_exclass2_ctor_bufferify(name, &
            len_trim(name, kind=C_INT))
        ! splicer end class.ExClass2.method.ctor
    end function exclass2_ctor

    ! ~ExClass2()
    ! function_index=20
    !>
    !! \brief destructor
    !!
    !<
    subroutine exclass2_dtor(obj)
        use iso_c_binding, only : C_NULL_PTR
        class(exclass2) :: obj
        ! splicer begin class.ExClass2.method.delete
        call c_exclass2_dtor(obj%cxxmem)
        obj%cxxmem%addr = C_NULL_PTR
        ! splicer end class.ExClass2.method.delete
    end subroutine exclass2_dtor

    ! const string & getName() const +deref(result_as_arg)+len(aa_exclass2_get_name_length({F_this}%{F_derived_member}))
    ! arg_to_buffer
    ! function_index=21
    function exclass2_get_name(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        character(len=aa_exclass2_get_name_length({F_this}%{F_derived_member})) :: SHT_rv
        ! splicer begin class.ExClass2.method.get_name
        call c_exclass2_get_name_bufferify(obj%cxxmem, SHT_rv, &
            len(SHT_rv, kind=C_INT))
        ! splicer end class.ExClass2.method.get_name
    end function exclass2_get_name

    ! const string & getName2() +deref(allocatable)
    ! arg_to_buffer
    ! function_index=22
    function exclass2_get_name2(obj) &
            result(SHT_rv)
        class(exclass2) :: obj
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin class.ExClass2.method.get_name2
        call c_exclass2_get_name2_bufferify(obj%cxxmem, DSHF_rv)
        ! splicer end class.ExClass2.method.get_name2
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function exclass2_get_name2

    ! string & getName3() const +deref(allocatable)
    ! arg_to_buffer
    ! function_index=23
    function exclass2_get_name3(obj) &
            result(SHT_rv)
        class(exclass2) :: obj
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin class.ExClass2.method.get_name3
        call c_exclass2_get_name3_bufferify(obj%cxxmem, DSHF_rv)
        ! splicer end class.ExClass2.method.get_name3
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function exclass2_get_name3

    ! string & getName4() +deref(allocatable)
    ! arg_to_buffer
    ! function_index=24
    function exclass2_get_name4(obj) &
            result(SHT_rv)
        class(exclass2) :: obj
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin class.ExClass2.method.get_name4
        call c_exclass2_get_name4_bufferify(obj%cxxmem, DSHF_rv)
        ! splicer end class.ExClass2.method.get_name4
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function exclass2_get_name4

    ! int GetNameLength() const
    ! function_index=25
    !>
    !! \brief helper function for Fortran
    !!
    !<
    function exclass2_get_name_length(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.ExClass2.method.get_name_length
        SHT_rv = c_exclass2_get_name_length(obj%cxxmem)
        ! splicer end class.ExClass2.method.get_name_length
    end function exclass2_get_name_length

    ! ExClass1 * get_class1(const ExClass1 * in +intent(in))
    ! function_index=26
    function exclass2_get_class1(obj, in) &
            result(SHT_rv)
        use exclass1_mod, only : exclass1
        class(exclass2) :: obj
        type(exclass1), intent(IN) :: in
        type(exclass1) :: SHT_rv
        ! splicer begin class.ExClass2.method.get_class1
        SHT_rv%cxxmem = c_exclass2_get_class1(obj%cxxmem, in%cxxmem)
        ! splicer end class.ExClass2.method.get_class1
    end function exclass2_get_class1

    ! void * declare(TypeID type +intent(in)+value)
    ! fortran_generic - has_default_arg
    ! function_index=44
    subroutine exclass2_declare_0_int(obj, type)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT), value, intent(IN) :: type
        ! splicer begin class.ExClass2.method.declare_0_int
        call c_exclass2_declare_0(obj%cxxmem, type)
        ! splicer end class.ExClass2.method.declare_0_int
    end subroutine exclass2_declare_0_int

    ! void * declare(TypeID type +intent(in)+value)
    ! fortran_generic - has_default_arg
    ! function_index=45
    subroutine exclass2_declare_0_long(obj, type)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT), value, intent(IN) :: type
        ! splicer begin class.ExClass2.method.declare_0_long
        call c_exclass2_declare_0(obj%cxxmem, type)
        ! splicer end class.ExClass2.method.declare_0_long
    end subroutine exclass2_declare_0_long

    ! void * declare(TypeID type +intent(in)+value, int len=1 +intent(in)+value)
    ! fortran_generic
    ! function_index=46
    subroutine exclass2_declare_1_int(obj, type, len)
        use iso_c_binding, only : C_INT, C_LONG
        class(exclass2) :: obj
        integer(C_INT), value, intent(IN) :: type
        integer(C_INT), value, intent(IN) :: len
        ! splicer begin class.ExClass2.method.declare_1_int
        call c_exclass2_declare_1(obj%cxxmem, type, int(len, C_LONG))
        ! splicer end class.ExClass2.method.declare_1_int
    end subroutine exclass2_declare_1_int

    ! void * declare(TypeID type +intent(in)+value, long len=1 +intent(in)+value)
    ! fortran_generic
    ! function_index=47
    subroutine exclass2_declare_1_long(obj, type, len)
        use iso_c_binding, only : C_INT, C_LONG
        class(exclass2) :: obj
        integer(C_INT), value, intent(IN) :: type
        integer(C_LONG), value, intent(IN) :: len
        ! splicer begin class.ExClass2.method.declare_1_long
        call c_exclass2_declare_1(obj%cxxmem, type, int(len, C_LONG))
        ! splicer end class.ExClass2.method.declare_1_long
    end subroutine exclass2_declare_1_long

    ! void destroyall()
    ! function_index=28
    subroutine exclass2_destroyall(obj)
        class(exclass2) :: obj
        ! splicer begin class.ExClass2.method.destroyall
        call c_exclass2_destroyall(obj%cxxmem)
        ! splicer end class.ExClass2.method.destroyall
    end subroutine exclass2_destroyall

    ! TypeID getTypeID() const
    ! function_index=29
    function exclass2_get_type_id(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.ExClass2.method.get_type_id
        SHT_rv = c_exclass2_get_type_id(obj%cxxmem)
        ! splicer end class.ExClass2.method.get_type_id
    end function exclass2_get_type_id

    ! void setValue(int value +intent(in)+value)
    ! cxx_template
    ! function_index=33
    subroutine exclass2_set_value_int(obj, value)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT), value, intent(IN) :: value
        ! splicer begin class.ExClass2.method.set_value_int
        call c_exclass2_set_value_int(obj%cxxmem, value)
        ! splicer end class.ExClass2.method.set_value_int
    end subroutine exclass2_set_value_int

    ! void setValue(long value +intent(in)+value)
    ! cxx_template
    ! function_index=34
    subroutine exclass2_set_value_long(obj, value)
        use iso_c_binding, only : C_LONG
        class(exclass2) :: obj
        integer(C_LONG), value, intent(IN) :: value
        ! splicer begin class.ExClass2.method.set_value_long
        call c_exclass2_set_value_long(obj%cxxmem, value)
        ! splicer end class.ExClass2.method.set_value_long
    end subroutine exclass2_set_value_long

    ! void setValue(float value +intent(in)+value)
    ! cxx_template
    ! function_index=35
    subroutine exclass2_set_value_float(obj, value)
        use iso_c_binding, only : C_FLOAT
        class(exclass2) :: obj
        real(C_FLOAT), value, intent(IN) :: value
        ! splicer begin class.ExClass2.method.set_value_float
        call c_exclass2_set_value_float(obj%cxxmem, value)
        ! splicer end class.ExClass2.method.set_value_float
    end subroutine exclass2_set_value_float

    ! void setValue(double value +intent(in)+value)
    ! cxx_template
    ! function_index=36
    subroutine exclass2_set_value_double(obj, value)
        use iso_c_binding, only : C_DOUBLE
        class(exclass2) :: obj
        real(C_DOUBLE), value, intent(IN) :: value
        ! splicer begin class.ExClass2.method.set_value_double
        call c_exclass2_set_value_double(obj%cxxmem, value)
        ! splicer end class.ExClass2.method.set_value_double
    end subroutine exclass2_set_value_double

    ! int getValue()
    ! cxx_template
    ! function_index=37
    function exclass2_get_value_int(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass2) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.ExClass2.method.get_value_int
        SHT_rv = c_exclass2_get_value_int(obj%cxxmem)
        ! splicer end class.ExClass2.method.get_value_int
    end function exclass2_get_value_int

    ! double getValue()
    ! cxx_template
    ! function_index=38
    function exclass2_get_value_double(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        class(exclass2) :: obj
        real(C_DOUBLE) :: SHT_rv
        ! splicer begin class.ExClass2.method.get_value_double
        SHT_rv = c_exclass2_get_value_double(obj%cxxmem)
        ! splicer end class.ExClass2.method.get_value_double
    end function exclass2_get_value_double

    ! Return pointer to C++ memory.
    function exclass2_yadda(obj) result (cxxptr)
        use iso_c_binding, only: c_associated, C_NULL_PTR, C_PTR
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

    ! splicer begin class.ExClass2.additional_functions
    ! splicer end class.ExClass2.additional_functions

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

end module exclass2_mod
