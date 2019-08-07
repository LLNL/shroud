! wrapfExClass1.f
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
!! \file wrapfExClass1.f
!! \brief Shroud generated wrapper for ExClass1 class
!<
! splicer begin file_top
! splicer end file_top
module exclass1_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin class.ExClass1.module_use
    ! splicer end class.ExClass1.module_use
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

    ! splicer begin class.ExClass1.module_top
    top of module splicer  1
    ! splicer end class.ExClass1.module_top

    type, bind(C) :: SHROUD_exclass1_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_exclass1_capsule

    type exclass1
        type(SHROUD_exclass1_capsule) :: cxxmem
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

        function c_exclass1_ctor_0(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_ctor_0")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_exclass1_ctor_0

        function c_exclass1_ctor_1(name, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_ctor_1")
            use iso_c_binding, only : C_CHAR, C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            type(SHROUD_exclass1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_exclass1_ctor_1

        function c_exclass1_ctor_1_bufferify(name, Lname, SHT_crv) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_ctor_1_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
            integer(C_INT), value, intent(IN) :: Lname
            type(SHROUD_exclass1_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_exclass1_ctor_1_bufferify

        subroutine c_exclass1_dtor(self) &
                bind(C, name="AA_exclass1_dtor")
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
        end subroutine c_exclass1_dtor

        function c_exclass1_increment_count(self, incr) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_increment_count")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: incr
            integer(C_INT) :: SHT_rv
        end function c_exclass1_increment_count

        pure function c_exclass1_get_name_error_pattern(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_name_error_pattern")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass1_get_name_error_pattern

        subroutine c_exclass1_get_name_error_pattern_bufferify(self, &
                SHF_rv, NSHF_rv) &
                bind(C, name="AA_exclass1_get_name_error_pattern_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(OUT) :: SHF_rv(*)
            integer(C_INT), value, intent(IN) :: NSHF_rv
        end subroutine c_exclass1_get_name_error_pattern_bufferify

        pure function c_exclass1_get_name_length(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_name_length")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_exclass1_get_name_length

        pure function c_exclass1_get_name_error_check(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_name_error_check")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass1_get_name_error_check

        subroutine c_exclass1_get_name_error_check_bufferify(self, &
                DSHF_rv) &
                bind(C, name="AA_exclass1_get_name_error_check_bufferify")
            import :: SHROUD_array, SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_exclass1_get_name_error_check_bufferify

        pure function c_exclass1_get_name_arg(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_name_arg")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_exclass1_get_name_arg

        subroutine c_exclass1_get_name_arg_bufferify(self, name, Nname) &
                bind(C, name="AA_exclass1_get_name_arg_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            character(kind=C_CHAR), intent(OUT) :: name(*)
            integer(C_INT), value, intent(IN) :: Nname
        end subroutine c_exclass1_get_name_arg_bufferify

        function c_exclass1_get_root(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_root")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(C_PTR) :: SHT_rv
        end function c_exclass1_get_root

        function c_exclass1_get_value_from_int(self, value) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_value_from_int")
            use iso_c_binding, only : C_INT
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: value
            integer(C_INT) :: SHT_rv
        end function c_exclass1_get_value_from_int

        function c_exclass1_get_value_1(self, value) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_value_1")
            use iso_c_binding, only : C_LONG
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            integer(C_LONG), value, intent(IN) :: value
            integer(C_LONG) :: SHT_rv
        end function c_exclass1_get_value_1

        function c_exclass1_get_addr(self) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_get_addr")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            type(C_PTR) :: SHT_rv
        end function c_exclass1_get_addr

        function c_exclass1_has_addr(self, in) &
                result(SHT_rv) &
                bind(C, name="AA_exclass1_has_addr")
            use iso_c_binding, only : C_BOOL
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
            logical(C_BOOL), value, intent(IN) :: in
            logical(C_BOOL) :: SHT_rv
        end function c_exclass1_has_addr

        subroutine c_exclass1_splicer_special(self) &
                bind(C, name="AA_exclass1_splicer_special")
            import :: SHROUD_exclass1_capsule
            implicit none
            type(SHROUD_exclass1_capsule), intent(IN) :: self
        end subroutine c_exclass1_splicer_special

        ! splicer begin class.ExClass1.additional_interfaces
        ! splicer end class.ExClass1.additional_interfaces
    end interface

    interface exclass1_ctor
        module procedure exclass1_ctor_0
        module procedure exclass1_ctor_1
    end interface exclass1_ctor

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
        ! splicer begin class.ExClass1.method.ctor_0
        SHT_prv = c_exclass1_ctor_0(SHT_rv%cxxmem)
        ! splicer end class.ExClass1.method.ctor_0
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
        ! splicer begin class.ExClass1.method.ctor_1
        SHT_prv = c_exclass1_ctor_1_bufferify(name, &
            len_trim(name, kind=C_INT), SHT_rv%cxxmem)
        ! splicer end class.ExClass1.method.ctor_1
    end function exclass1_ctor_1

    ! ~ExClass1()
    !>
    !! \brief destructor
    !!
    !! longer description joined with previous line
    !<
    subroutine exclass1_dtor(obj)
        class(exclass1) :: obj
        ! splicer begin class.ExClass1.method.delete
        call c_exclass1_dtor(obj%cxxmem)
        ! splicer end class.ExClass1.method.delete
    end subroutine exclass1_dtor

    ! int incrementCount(int incr +intent(in)+value)
    function exclass1_increment_count(obj, incr) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        integer(C_INT), value, intent(IN) :: incr
        integer(C_INT) :: SHT_rv
        ! splicer begin class.ExClass1.method.increment_count
        SHT_rv = c_exclass1_increment_count(obj%cxxmem, incr)
        ! splicer end class.ExClass1.method.increment_count
    end function exclass1_increment_count

    ! const string & getNameErrorPattern() const +deref(result_as_arg)+len(aa_exclass1_get_name_length({F_this}%{F_derived_member}))
    ! arg_to_buffer
    function exclass1_get_name_error_pattern(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        character(len=aa_exclass1_get_name_length({F_this}%{F_derived_member})) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_name_error_pattern
        call c_exclass1_get_name_error_pattern_bufferify(obj%cxxmem, &
            SHT_rv, len(SHT_rv, kind=C_INT))
        ! splicer end class.ExClass1.method.get_name_error_pattern
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
        ! splicer begin class.ExClass1.method.get_name_length
        SHT_rv = c_exclass1_get_name_length(obj%cxxmem)
        ! splicer end class.ExClass1.method.get_name_length
    end function exclass1_get_name_length

    ! const string & getNameErrorCheck() const +deref(allocatable)
    ! arg_to_buffer
    function exclass1_get_name_error_check(obj) &
            result(SHT_rv)
        class(exclass1) :: obj
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin class.ExClass1.method.get_name_error_check
        call c_exclass1_get_name_error_check_bufferify(obj%cxxmem, &
            DSHF_rv)
        ! splicer end class.ExClass1.method.get_name_error_check
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function exclass1_get_name_error_check

    ! void getNameArg(string & name +intent(out)+len(Nname)) const
    ! arg_to_buffer - arg_to_buffer
    subroutine exclass1_get_name_arg(obj, name)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        character(len=*), intent(OUT) :: name
        ! splicer begin class.ExClass1.method.get_name_arg
        call c_exclass1_get_name_arg_bufferify(obj%cxxmem, name, &
            len(name, kind=C_INT))
        ! splicer end class.ExClass1.method.get_name_arg
    end subroutine exclass1_get_name_arg

    ! void * getRoot()
    function exclass1_get_root(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(exclass1) :: obj
        type(C_PTR) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_root
        SHT_rv = c_exclass1_get_root(obj%cxxmem)
        ! splicer end class.ExClass1.method.get_root
    end function exclass1_get_root

    ! int getValue(int value +intent(in)+value)
    function exclass1_get_value_from_int(obj, value) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(exclass1) :: obj
        integer(C_INT), value, intent(IN) :: value
        integer(C_INT) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_value_from_int
        SHT_rv = c_exclass1_get_value_from_int(obj%cxxmem, value)
        ! splicer end class.ExClass1.method.get_value_from_int
    end function exclass1_get_value_from_int

    ! long getValue(long value +intent(in)+value)
    function exclass1_get_value_1(obj, value) &
            result(SHT_rv)
        use iso_c_binding, only : C_LONG
        class(exclass1) :: obj
        integer(C_LONG), value, intent(IN) :: value
        integer(C_LONG) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_value_1
        SHT_rv = c_exclass1_get_value_1(obj%cxxmem, value)
        ! splicer end class.ExClass1.method.get_value_1
    end function exclass1_get_value_1

    ! void * getAddr()
    function exclass1_get_addr(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        class(exclass1) :: obj
        type(C_PTR) :: SHT_rv
        ! splicer begin class.ExClass1.method.get_addr
        SHT_rv = c_exclass1_get_addr(obj%cxxmem)
        ! splicer end class.ExClass1.method.get_addr
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
        ! splicer begin class.ExClass1.method.has_addr
        SHT_rv = c_exclass1_has_addr(obj%cxxmem, SH_in)
        ! splicer end class.ExClass1.method.has_addr
    end function exclass1_has_addr

    ! void SplicerSpecial()
    subroutine exclass1_splicer_special(obj)
        class(exclass1) :: obj
        ! splicer begin class.ExClass1.method.splicer_special
        blah blah blah
        ! splicer end class.ExClass1.method.splicer_special
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
