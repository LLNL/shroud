! wrapfExClass1.f
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
!! \file wrapfExClass1.f
!! \brief Shroud generated wrapper for ExClass1 class
!<
! splicer begin file_top
! splicer end file_top
module exclass1_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.ExClass1.module_use
    ! splicer end class.ExClass1.module_use
    implicit none


    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
        integer(C_INT) :: refcount = 0    ! reference count
    end type SHROUD_capsule_data

    ! splicer begin class.ExClass1.module_top
    top of module splicer  1
    ! splicer end class.ExClass1.module_top

    type exclass1
        type(C_PTR), private :: cxxptr = C_NULL_PTR
        type(SHROUD_capsule_data), pointer :: cxxmem => null()
        ! splicer begin class.ExClass1.component_part
          component part 1a
          component part 1b
        ! splicer end class.ExClass1.component_part
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
        procedure :: exclass1_assign
        generic :: assignment(=) => exclass1_assign
        final :: exclass1_final
        generic :: get_value => get_value_from_int, get_value_1
        ! splicer begin class.ExClass1.type_bound_procedure_part
          type bound procedure part 1
        ! splicer end class.ExClass1.type_bound_procedure_part
    end type exclass1

    interface operator (.eq.)
        module procedure exclass1_eq
    end interface

    interface operator (.ne.)
        module procedure exclass1_ne
    end interface

    interface

        function c_exclass1_ctor_0() &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_ctor_0")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR) :: SHT_rv
        end function c_exclass1_ctor_0

        function c_exclass1_ctor_1(name) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_ctor_1")
            use iso_c_binding, only : C_CHAR, C_PTR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(C_PTR) :: SHT_rv
        end function c_exclass1_ctor_1

        function c_exclass1_ctor_1_bufferify(name, Lname) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_ctor_1_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(C_PTR) :: SHT_rv
        end function c_exclass1_ctor_1_bufferify

        subroutine c_exclass1_dtor(self) &
                bind(C, name="AA_exclass1_dtor")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
        end subroutine c_exclass1_dtor

        function c_exclass1_increment_count(self, incr) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_increment_count")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            integer(C_INT), value, intent(IN) :: incr
            integer(C_INT) :: SHT_rv
        end function c_exclass1_increment_count

        pure function c_exclass1_get_name_error_pattern(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_name_error_pattern")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass1_get_name_error_pattern

        subroutine c_exclass1_get_name_error_pattern_bufferify(self, &
                SHF_rv, NSHF_rv) &
                bind(C, name="AA_exclass1_get_name_error_pattern_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            character(kind=C_CHAR), intent(OUT) :: SHF_rv(*)
            integer(C_INT), value, intent(IN) :: NSHF_rv
        end subroutine c_exclass1_get_name_error_pattern_bufferify

        pure function c_exclass1_get_name_length(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_name_length")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_exclass1_get_name_length

        pure function c_exclass1_get_name_error_check(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_name_error_check")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass1_get_name_error_check

        subroutine c_exclass1_get_name_error_check_bufferify(self, &
                SHF_rv, NSHF_rv) &
                bind(C, name="AA_exclass1_get_name_error_check_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            character(kind=C_CHAR), intent(OUT) :: SHF_rv(*)
            integer(C_INT), value, intent(IN) :: NSHF_rv
        end subroutine c_exclass1_get_name_error_check_bufferify

        pure function c_exclass1_get_name_arg(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_name_arg")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass1_get_name_arg

        subroutine c_exclass1_get_name_arg_bufferify(self, name, Nname) &
                bind(C, name="AA_exclass1_get_name_arg_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            character(kind=C_CHAR), intent(OUT) :: name(*)
            integer(C_INT), value, intent(IN) :: Nname
        end subroutine c_exclass1_get_name_arg_bufferify

        function c_exclass1_get_root(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_root")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR) :: SHT_rv
        end function c_exclass1_get_root

        function c_exclass1_get_value_from_int(self, value) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_value_from_int")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            integer(C_INT), value, intent(IN) :: value
            integer(C_INT) :: SHT_rv
        end function c_exclass1_get_value_from_int

        function c_exclass1_get_value_1(self, value) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_value_1")
            use iso_c_binding, only : C_LONG, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            integer(C_LONG), value, intent(IN) :: value
            integer(C_LONG) :: SHT_rv
        end function c_exclass1_get_value_1

        function c_exclass1_get_addr(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_addr")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            type(C_PTR) :: SHT_rv
        end function c_exclass1_get_addr

        function c_exclass1_has_addr(self, in) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_has_addr")
            use iso_c_binding, only : C_BOOL, C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
            logical(C_BOOL), value, intent(IN) :: in
            logical(C_BOOL) :: SHT_rv
        end function c_exclass1_has_addr

        subroutine c_exclass1_splicer_special(self) &
                bind(C, name="AA_exclass1_splicer_special")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: self
        end subroutine c_exclass1_splicer_special

        ! splicer begin class.ExClass1.additional_interfaces
        ! splicer end class.ExClass1.additional_interfaces
    end interface

    interface exclass1_ctor
        module procedure exclass1_ctor_0
        module procedure exclass1_ctor_1
    end interface exclass1_ctor

contains

    ! ExClass1()
    ! function_index=0
    function exclass1_ctor_0() &
            result(SHT_rv)
        use iso_c_binding, only : c_f_pointer
        type(exclass1) :: SHT_rv
        ! splicer begin class.ExClass1.method.ctor_0
        SHT_rv%cxxptr = c_exclass1_ctor_0()
        call c_f_pointer(SHT_rv%cxxptr, SHT_rv%cxxmem)
        ! splicer end class.ExClass1.method.ctor_0
    end function exclass1_ctor_0

    ! ExClass1(const string * name +intent(in))
    ! arg_to_buffer
    ! function_index=1
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
        use iso_c_binding, only : C_INT, c_f_pointer
        character(*), intent(IN) :: name
        type(exclass1) :: SHT_rv
        ! splicer begin class.ExClass1.method.ctor_1
        SHT_rv%cxxptr = c_exclass1_ctor_1_bufferify(name, &
            len_trim(name, kind=C_INT))
        call c_f_pointer(SHT_rv%cxxptr, SHT_rv%cxxmem)
        ! splicer end class.ExClass1.method.ctor_1
    end function exclass1_ctor_1

    ! ~ExClass1()
    ! function_index=2
    !>
    !! \brief destructor
    !!
    !! longer description joined with previous line
    !<
    subroutine exclass1_dtor(obj)
        use iso_c_binding, only : C_NULL_PTR
        class(exclass1) :: obj
        ! splicer begin class.ExClass1.method.delete
        call c_exclass1_dtor(obj%cxxptr)
        obj%cxxptr = C_NULL_PTR
        nullify(obj%cxxmem)
        ! splicer end class.ExClass1.method.delete
    end subroutine exclass1_dtor

    ! int incrementCount(int incr +intent(in)+value)
    ! function_index=3
    function exclass1_increment_count(obj, incr) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        integer(C_INT), value, intent(IN) :: incr
        integer(C_INT) :: SHT_rv
        ! splicer begin class.ExClass1.method.increment_count
        SHT_rv = c_exclass1_increment_count(obj%cxxptr, incr)
        ! splicer end class.ExClass1.method.increment_count
    end function exclass1_increment_count

    ! const string & getNameErrorPattern() const +len(aa_exclass1_get_name_length({F_this}%{F_derived_member}))
    ! arg_to_buffer
    ! function_index=4
    function exclass1_get_name_error_pattern(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_CHAR, C_INT
        class(exclass1) :: obj
        character(kind=C_CHAR, &
            len=aa_exclass1_get_name_length(obj%cxxmem)) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_name_error_pattern
        call c_exclass1_get_name_error_pattern_bufferify(obj%cxxptr, &
            SHT_rv, len(SHT_rv, kind=C_INT))
        ! splicer end class.ExClass1.method.get_name_error_pattern
    end function exclass1_get_name_error_pattern

    ! int GetNameLength() const
    ! function_index=5
    !>
    !! \brief helper function for Fortran to get length of name.
    !!
    !<
    function exclass1_get_name_length(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_name_length
        SHT_rv = c_exclass1_get_name_length(obj%cxxptr)
        ! splicer end class.ExClass1.method.get_name_length
    end function exclass1_get_name_length

    ! const string & getNameErrorCheck() const
    ! arg_to_buffer
    ! function_index=6
    function exclass1_get_name_error_check(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_CHAR, C_INT
        class(exclass1) :: obj
        character(kind=C_CHAR, len=strlen_ptr( &
            c_exclass1_get_name_error_check_bufferify(obj%cxxptr, &
            SHT_rv, len(SHT_rv, kind=C_INT)))) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_name_error_check
        call c_exclass1_get_name_error_check_bufferify(obj%cxxptr, &
            SHT_rv, len(SHT_rv, kind=C_INT))
        ! splicer end class.ExClass1.method.get_name_error_check
    end function exclass1_get_name_error_check

    ! void getNameArg(string & name +intent(out)+len(Nname)) const
    ! arg_to_buffer - arg_to_buffer
    ! function_index=18
    subroutine exclass1_get_name_arg(obj, name)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        character(*), intent(OUT) :: name
        ! splicer begin class.ExClass1.method.get_name_arg
        call c_exclass1_get_name_arg_bufferify(obj%cxxptr, name, &
            len(name, kind=C_INT))
        ! splicer end class.ExClass1.method.get_name_arg
    end subroutine exclass1_get_name_arg

    ! void * getRoot()
    ! function_index=8
    function exclass1_get_root(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(exclass1) :: obj
        type(C_PTR) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_root
        SHT_rv = c_exclass1_get_root(obj%cxxptr)
        ! splicer end class.ExClass1.method.get_root
    end function exclass1_get_root

    ! int getValue(int value +intent(in)+value)
    ! function_index=9
    function exclass1_get_value_from_int(obj, value) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        integer(C_INT), value, intent(IN) :: value
        integer(C_INT) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_value_from_int
        SHT_rv = c_exclass1_get_value_from_int(obj%cxxptr, value)
        ! splicer end class.ExClass1.method.get_value_from_int
    end function exclass1_get_value_from_int

    ! long getValue(long value +intent(in)+value)
    ! function_index=10
    function exclass1_get_value_1(obj, value) &
            result(SHT_rv)
        use iso_c_binding, only : C_LONG
        class(exclass1) :: obj
        integer(C_LONG), value, intent(IN) :: value
        integer(C_LONG) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_value_1
        SHT_rv = c_exclass1_get_value_1(obj%cxxptr, value)
        ! splicer end class.ExClass1.method.get_value_1
    end function exclass1_get_value_1

    ! void * getAddr()
    ! function_index=11
    function exclass1_get_addr(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(exclass1) :: obj
        type(C_PTR) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_addr
        SHT_rv = c_exclass1_get_addr(obj%cxxptr)
        ! splicer end class.ExClass1.method.get_addr
    end function exclass1_get_addr

    ! bool hasAddr(bool in +intent(in)+value)
    ! function_index=12
    function exclass1_has_addr(obj, in) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        class(exclass1) :: obj
        logical, value, intent(IN) :: in
        logical(C_BOOL) SH_in
        logical :: SHT_rv
        SH_in = in  ! coerce to C_BOOL
        ! splicer begin class.ExClass1.method.has_addr
        SHT_rv = c_exclass1_has_addr(obj%cxxptr, SH_in)
        ! splicer end class.ExClass1.method.has_addr
    end function exclass1_has_addr

    ! void SplicerSpecial()
    ! function_index=13
    subroutine exclass1_splicer_special(obj)
        class(exclass1) :: obj
        ! splicer begin class.ExClass1.method.splicer_special
        blah blah blah
        ! splicer end class.ExClass1.method.splicer_special
    end subroutine exclass1_splicer_special

    ! Return pointer to C++ memory if allocated, else C_NULL_PTR.
    function exclass1_yadda(obj) result (cxxptr)
        use iso_c_binding, only: c_associated, C_NULL_PTR, C_PTR
        class(exclass1), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        if (c_associated(obj%cxxptr)) then
            cxxptr = obj%cxxmem%addr
        else
            cxxptr = C_NULL_PTR
        endif
    end function exclass1_yadda

    function exclass1_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(exclass1), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function exclass1_associated

    subroutine exclass1_assign(lhs, rhs)
        use iso_c_binding, only : c_associated, c_f_pointer
        class(exclass1), intent(INOUT) :: lhs
        class(exclass1), intent(IN) :: rhs

        lhs%cxxptr = rhs%cxxptr
        if (c_associated(lhs%cxxptr)) then
            call c_f_pointer(lhs%cxxptr, lhs%cxxmem)
            lhs%cxxmem%refcount = lhs%cxxmem%refcount + 1
        else
            nullify(lhs%cxxmem)
        endif
    end subroutine exclass1_assign

    subroutine exclass1_final(obj)
        use iso_c_binding, only : c_associated, C_BOOL, C_NULL_PTR
        type(exclass1), intent(INOUT) :: obj
        interface
            subroutine array_destructor(ptr, gc) &
                bind(C, name="AA_SHROUD_array_destructor_function")
                use iso_c_binding, only : C_BOOL, C_INT, C_PTR
                implicit none
                type(C_PTR), value, intent(IN) :: ptr
                logical(C_BOOL), value, intent(IN) :: gc
            end subroutine array_destructor
        end interface
        if (c_associated(obj%cxxptr)) then
            call array_destructor(obj%cxxptr, .true._C_BOOL)
            obj%cxxptr = C_NULL_PTR
            nullify(obj%cxxmem)
        endif
    end subroutine exclass1_final

    ! splicer begin class.ExClass1.additional_functions
    ! splicer end class.ExClass1.additional_functions

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

end module exclass1_mod
