! wrapfclibrary.f
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
!! \file wrapfclibrary.f
!! \brief Shroud generated wrapper for Clibrary library
!<
! splicer begin file_top
! splicer end file_top
module clibrary_mod
    use iso_c_binding, only : C_INT
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    integer, parameter :: MAXNAME = 20
    ! splicer end module_top


    type, bind(C) :: cstruct1
        integer(C_INT) :: ifield
    end type cstruct1

    abstract interface

        subroutine callback2_incr(arg0) bind(C)
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT) :: arg0
        end subroutine callback2_incr

        subroutine callback3_incr(arg0) bind(C)
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT) :: arg0
        end subroutine callback3_incr

    end interface

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

        subroutine fill_int_array(out) &
                bind(C, name="fillIntArray")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), intent(OUT) :: out(*)
        end subroutine fill_int_array

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

        subroutine c_function4a_bufferify(arg1, arg2, SHF_rv, NSHF_rv) &
                bind(C, name="CLI_function4a_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg1(*)
            character(kind=C_CHAR), intent(IN) :: arg2(*)
            character(kind=C_CHAR), intent(OUT) :: SHF_rv(*)
            integer(C_INT), value, intent(IN) :: NSHF_rv
        end subroutine c_function4a_bufferify

        subroutine c_accept_name(name) &
                bind(C, name="acceptName")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(IN) :: name(*)
        end subroutine c_accept_name

        subroutine c_return_one_name(name1) &
                bind(C, name="returnOneName")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(OUT) :: name1(*)
        end subroutine c_return_one_name

        subroutine c_return_one_name_bufferify(name1, Nname1) &
                bind(C, name="CLI_return_one_name_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(OUT) :: name1(*)
            integer(C_INT), value, intent(IN) :: Nname1
        end subroutine c_return_one_name_bufferify

        subroutine c_return_two_names(name1, name2) &
                bind(C, name="returnTwoNames")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(OUT) :: name1(*)
            character(kind=C_CHAR), intent(OUT) :: name2(*)
        end subroutine c_return_two_names

        subroutine c_return_two_names_bufferify(name1, Nname1, name2, &
                Nname2) &
                bind(C, name="CLI_return_two_names_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(OUT) :: name1(*)
            integer(C_INT), value, intent(IN) :: Nname1
            character(kind=C_CHAR), intent(OUT) :: name2(*)
            integer(C_INT), value, intent(IN) :: Nname2
        end subroutine c_return_two_names_bufferify

        subroutine c_implied_text_len(text, ltext) &
                bind(C, name="ImpliedTextLen")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(OUT) :: text(*)
            integer(C_INT), value, intent(IN) :: ltext
        end subroutine c_implied_text_len

        subroutine c_implied_text_len_bufferify(text, Ntext, ltext) &
                bind(C, name="CLI_implied_text_len_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(OUT) :: text(*)
            integer(C_INT), value, intent(IN) :: Ntext
            integer(C_INT), value, intent(IN) :: ltext
        end subroutine c_implied_text_len_bufferify

        function c_implied_len(text, ltext, flag) &
                result(SHT_rv) &
                bind(C, name="ImpliedLen")
            use iso_c_binding, only : C_BOOL, C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: text(*)
            integer(C_INT), value, intent(IN) :: ltext
            logical(C_BOOL), value, intent(IN) :: flag
            integer(C_INT) :: SHT_rv
        end function c_implied_len

        function c_implied_len_trim(text, ltext, flag) &
                result(SHT_rv) &
                bind(C, name="ImpliedLenTrim")
            use iso_c_binding, only : C_BOOL, C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(IN) :: text(*)
            integer(C_INT), value, intent(IN) :: ltext
            logical(C_BOOL), value, intent(IN) :: flag
            integer(C_INT) :: SHT_rv
        end function c_implied_len_trim

        function c_implied_bool_true(flag) &
                result(SHT_rv) &
                bind(C, name="ImpliedBoolTrue")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL), value, intent(IN) :: flag
            logical(C_BOOL) :: SHT_rv
        end function c_implied_bool_true

        function c_implied_bool_false(flag) &
                result(SHT_rv) &
                bind(C, name="ImpliedBoolFalse")
            use iso_c_binding, only : C_BOOL
            implicit none
            logical(C_BOOL), value, intent(IN) :: flag
            logical(C_BOOL) :: SHT_rv
        end function c_implied_bool_false

        subroutine Fortran_bindC1a() &
                bind(C, name="bindC1")
            implicit none
        end subroutine Fortran_bindC1a

        subroutine c_bind_c2(outbuf) &
                bind(C, name="bindC2")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(OUT) :: outbuf(*)
        end subroutine c_bind_c2

        subroutine c_bind_c2_bufferify(outbuf, Noutbuf) &
                bind(C, name="CLI_bind_c2_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(OUT) :: outbuf(*)
            integer(C_INT), value, intent(IN) :: Noutbuf
        end subroutine c_bind_c2_bufferify

        subroutine pass_void_star_star(in, out) &
                bind(C, name="passVoidStarStar")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR), value, intent(IN) :: in
            type(C_PTR), intent(OUT) :: out
        end subroutine pass_void_star_star

        function pass_assumed_type(arg) &
                result(SHT_rv) &
                bind(C, name="passAssumedType")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(*) :: arg
            integer(C_INT) :: SHT_rv
        end function pass_assumed_type

        function c_pass_assumed_type_buf(arg, outbuf) &
                result(SHT_rv) &
                bind(C, name="passAssumedTypeBuf")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            implicit none
            type(*) :: arg
            character(kind=C_CHAR), intent(OUT) :: outbuf(*)
            integer(C_INT) :: SHT_rv
        end function c_pass_assumed_type_buf

        function c_pass_assumed_type_buf_bufferify(arg, outbuf, Noutbuf) &
                result(SHT_rv) &
                bind(C, name="CLI_pass_assumed_type_buf_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            implicit none
            type(*) :: arg
            character(kind=C_CHAR), intent(OUT) :: outbuf(*)
            integer(C_INT), value, intent(IN) :: Noutbuf
            integer(C_INT) :: SHT_rv
        end function c_pass_assumed_type_buf_bufferify

        subroutine c_callback2(type, in, incr) &
                bind(C, name="callback2")
            use iso_c_binding, only : C_INT, C_PTR
            import :: callback2_incr
            implicit none
            integer(C_INT), value, intent(IN) :: type
            type(*) :: in
            procedure(callback2_incr) :: incr
        end subroutine c_callback2

        subroutine c_callback3(type, in, incr, outbuf) &
                bind(C, name="callback3")
            use iso_c_binding, only : C_CHAR, C_PTR
            import :: callback3_incr
            implicit none
            character(kind=C_CHAR), intent(IN) :: type(*)
            type(*) :: in
            procedure(callback3_incr) :: incr
            character(kind=C_CHAR), intent(OUT) :: outbuf(*)
        end subroutine c_callback3

        subroutine c_callback3_bufferify(type, in, incr, outbuf, &
                Noutbuf) &
                bind(C, name="CLI_callback3_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            import :: callback3_incr
            implicit none
            character(kind=C_CHAR), intent(IN) :: type(*)
            type(*) :: in
            procedure(callback3_incr) :: incr
            character(kind=C_CHAR), intent(OUT) :: outbuf(*)
            integer(C_INT), value, intent(IN) :: Noutbuf
        end subroutine c_callback3_bufferify

        function pass_struct1(s1) &
                result(SHT_rv) &
                bind(C, name="passStruct1")
            use iso_c_binding, only : C_INT
            import :: cstruct1
            implicit none
            type(cstruct1), intent(IN) :: s1
            integer(C_INT) :: SHT_rv
        end function pass_struct1

        function c_pass_struct2(s1, outbuf) &
                result(SHT_rv) &
                bind(C, name="passStruct2")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: cstruct1
            implicit none
            type(cstruct1), intent(IN) :: s1
            character(kind=C_CHAR), intent(OUT) :: outbuf(*)
            integer(C_INT) :: SHT_rv
        end function c_pass_struct2

        function c_pass_struct2_bufferify(s1, outbuf, Noutbuf) &
                result(SHT_rv) &
                bind(C, name="CLI_pass_struct2_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            import :: cstruct1
            implicit none
            type(cstruct1), intent(IN) :: s1
            character(kind=C_CHAR), intent(OUT) :: outbuf(*)
            integer(C_INT), value, intent(IN) :: Noutbuf
            integer(C_INT) :: SHT_rv
        end function c_pass_struct2_bufferify

        function c_return_struct_ptr1(ifield) &
                result(SHT_rv) &
                bind(C, name="returnStructPtr1")
            use iso_c_binding, only : C_INT, C_PTR
            import :: cstruct1
            implicit none
            integer(C_INT), value, intent(IN) :: ifield
            type(C_PTR) SHT_rv
        end function c_return_struct_ptr1

        function c_return_struct_ptr2(ifield, outbuf) &
                result(SHT_rv) &
                bind(C, name="returnStructPtr2")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            import :: cstruct1
            implicit none
            integer(C_INT), value, intent(IN) :: ifield
            character(kind=C_CHAR), intent(OUT) :: outbuf(*)
            type(C_PTR) SHT_rv
        end function c_return_struct_ptr2

        function c_return_struct_ptr2_bufferify(ifield, outbuf, Noutbuf) &
                result(SHT_rv) &
                bind(C, name="CLI_return_struct_ptr2_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_PTR
            import :: cstruct1
            implicit none
            integer(C_INT), value, intent(IN) :: ifield
            character(kind=C_CHAR), intent(OUT) :: outbuf(*)
            integer(C_INT), value, intent(IN) :: Noutbuf
            type(C_PTR) SHT_rv
        end function c_return_struct_ptr2_bufferify

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! void Sum(int len +implied(size(values))+intent(in)+value, int * values +dimension(:)+intent(in), int * result +intent(out))
    subroutine sum(values, result)
        use iso_c_binding, only : C_INT
        integer(C_INT) :: len
        integer(C_INT), intent(IN) :: values(:)
        integer(C_INT), intent(OUT) :: result
        len = size(values,kind=C_INT)
        ! splicer begin function.sum
        call c_sum(len, values, result)
        ! splicer end function.sum
    end subroutine sum

    ! bool Function3(bool arg +intent(in)+value)
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

    ! char * Function4a(const char * arg1 +intent(in), const char * arg2 +intent(in)) +deref(result_as_arg)+len(30)
    ! arg_to_buffer
    function function4a(arg1, arg2) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_NULL_CHAR
        character(len=*), intent(IN) :: arg1
        character(len=*), intent(IN) :: arg2
        character(len=30) :: SHT_rv
        ! splicer begin function.function4a
        call c_function4a_bufferify(trim(arg1)//C_NULL_CHAR, &
            trim(arg2)//C_NULL_CHAR, SHT_rv, len(SHT_rv, kind=C_INT))
        ! splicer end function.function4a
    end function function4a

    ! void acceptName(const char * name +intent(in))
    subroutine accept_name(name)
        use iso_c_binding, only : C_NULL_CHAR
        character(len=*), intent(IN) :: name
        ! splicer begin function.accept_name
        call c_accept_name(trim(name)//C_NULL_CHAR)
        ! splicer end function.accept_name
    end subroutine accept_name

    ! void returnOneName(char * name1 +charlen(MAXNAME)+intent(out))
    ! arg_to_buffer
    !>
    !! \brief Test charlen attribute
    !!
    !! Each argument is assumed to be at least MAXNAME long.
    !! This define is provided by the user.
    !! The function will copy into the user provided buffer.
    !<
    subroutine return_one_name(name1)
        use iso_c_binding, only : C_INT
        character(len=*), intent(OUT) :: name1
        ! splicer begin function.return_one_name
        call c_return_one_name_bufferify(name1, len(name1, kind=C_INT))
        ! splicer end function.return_one_name
    end subroutine return_one_name

    ! void returnTwoNames(char * name1 +charlen(MAXNAME)+intent(out), char * name2 +charlen(MAXNAME)+intent(out))
    ! arg_to_buffer
    !>
    !! \brief Test charlen attribute
    !!
    !! Each argument is assumed to be at least MAXNAME long.
    !! This define is provided by the user.
    !! The function will copy into the user provided buffer.
    !<
    subroutine return_two_names(name1, name2)
        use iso_c_binding, only : C_INT
        character(len=*), intent(OUT) :: name1
        character(len=*), intent(OUT) :: name2
        ! splicer begin function.return_two_names
        call c_return_two_names_bufferify(name1, len(name1, kind=C_INT), &
            name2, len(name2, kind=C_INT))
        ! splicer end function.return_two_names
    end subroutine return_two_names

    ! void ImpliedTextLen(char * text +charlen(MAXNAME)+intent(out), int ltext +implied(len(text))+intent(in)+value)
    ! arg_to_buffer
    !>
    !! \brief Fill text, at most ltext characters.
    !!
    !<
    subroutine implied_text_len(text)
        use iso_c_binding, only : C_INT
        character(len=*), intent(OUT) :: text
        integer(C_INT) :: ltext
        ltext = len(text,kind=C_INT)
        ! splicer begin function.implied_text_len
        call c_implied_text_len_bufferify(text, len(text, kind=C_INT), &
            ltext)
        ! splicer end function.implied_text_len
    end subroutine implied_text_len

    ! int ImpliedLen(const char * text +intent(in), int ltext +implied(len(text))+intent(in)+value, bool flag +implied(false)+intent(in)+value)
    !>
    !! \brief Return the implied argument - text length
    !!
    !! Pass the Fortran length of the char argument directy to the C function.
    !! No need for the bufferify version which will needlessly copy the string.
    !<
    function implied_len(text) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_INT, C_NULL_CHAR
        character(len=*), intent(IN) :: text
        integer(C_INT) :: ltext
        logical(C_BOOL) :: flag
        integer(C_INT) :: SHT_rv
        ltext = len(text,kind=C_INT)
        flag = .FALSE._C_BOOL
        ! splicer begin function.implied_len
        SHT_rv = c_implied_len(trim(text)//C_NULL_CHAR, ltext, flag)
        ! splicer end function.implied_len
    end function implied_len

    ! int ImpliedLenTrim(const char * text +intent(in), int ltext +implied(len_trim(text))+intent(in)+value, bool flag +implied(true)+intent(in)+value)
    !>
    !! \brief Return the implied argument - text length
    !!
    !! Pass the Fortran length of the char argument directy to the C function.
    !! No need for the bufferify version which will needlessly copy the string.
    !<
    function implied_len_trim(text) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_INT, C_NULL_CHAR
        character(len=*), intent(IN) :: text
        integer(C_INT) :: ltext
        logical(C_BOOL) :: flag
        integer(C_INT) :: SHT_rv
        ltext = len_trim(text,kind=C_INT)
        flag = .TRUE._C_BOOL
        ! splicer begin function.implied_len_trim
        SHT_rv = c_implied_len_trim(trim(text)//C_NULL_CHAR, ltext, &
            flag)
        ! splicer end function.implied_len_trim
    end function implied_len_trim

    ! bool ImpliedBoolTrue(bool flag +implied(true)+intent(in)+value)
    !>
    !! \brief Single, implied bool argument
    !!
    !<
    function implied_bool_true() &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        logical(C_BOOL) :: flag
        logical :: SHT_rv
        flag = .TRUE._C_BOOL
        ! splicer begin function.implied_bool_true
        SHT_rv = c_implied_bool_true(flag)
        ! splicer end function.implied_bool_true
    end function implied_bool_true

    ! bool ImpliedBoolFalse(bool flag +implied(false)+intent(in)+value)
    !>
    !! \brief Single, implied bool argument
    !!
    !<
    function implied_bool_false() &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        logical(C_BOOL) :: flag
        logical :: SHT_rv
        flag = .FALSE._C_BOOL
        ! splicer begin function.implied_bool_false
        SHT_rv = c_implied_bool_false(flag)
        ! splicer end function.implied_bool_false
    end function implied_bool_false

    ! void bindC2(char * outbuf +intent(out))
    ! arg_to_buffer
    !>
    !! \brief Rename Fortran name for interface only function
    !!
    !! This creates a Fortran bufferify function and an interface.
    !<
    subroutine Fortran_bindC2a(outbuf)
        use iso_c_binding, only : C_INT
        character(len=*), intent(OUT) :: outbuf
        ! splicer begin function.bind_c2
        call c_bind_c2_bufferify(outbuf, len(outbuf, kind=C_INT))
        ! splicer end function.bind_c2
    end subroutine Fortran_bindC2a

    ! int passAssumedTypeBuf(void * arg +assumedtype+intent(in), char * outbuf +intent(out))
    ! arg_to_buffer
    !>
    !! \brief Test assumed-type
    !!
    !! A bufferify function is created.
    !! Should only be call with an C_INT argument, and will
    !! return the value passed in.
    !<
    function pass_assumed_type_buf(arg, outbuf) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        type(*) :: arg
        character(len=*), intent(OUT) :: outbuf
        integer(C_INT) :: SHT_rv
        ! splicer begin function.pass_assumed_type_buf
        SHT_rv = c_pass_assumed_type_buf_bufferify(arg, outbuf, &
            len(outbuf, kind=C_INT))
        ! splicer end function.pass_assumed_type_buf
    end function pass_assumed_type_buf

    ! void callback2(int type +intent(in)+value, void * in +assumedtype+intent(in), void ( * incr)(int *) +external+intent(in)+value)
    !>
    !! \brief Test function pointer
    !!
    !<
    subroutine callback2(type, in, incr)
        use iso_c_binding, only : C_INT
        integer(C_INT), value, intent(IN) :: type
        type(*) :: in
        external :: incr
        ! splicer begin function.callback2
        call c_callback2(type, in, incr)
        ! splicer end function.callback2
    end subroutine callback2

    ! void callback3(const char * type +intent(in), void * in +assumedtype+intent(in), void ( * incr)(int *) +external+intent(in)+value, char * outbuf +intent(out))
    ! arg_to_buffer
    !>
    !! \brief Test function pointer
    !!
    !! A bufferify function will be created.
    !<
    subroutine callback3(type, in, incr, outbuf)
        use iso_c_binding, only : C_INT, C_NULL_CHAR
        character(len=*), intent(IN) :: type
        type(*) :: in
        external :: incr
        character(len=*), intent(OUT) :: outbuf
        ! splicer begin function.callback3
        call c_callback3_bufferify(trim(type)//C_NULL_CHAR, in, incr, &
            outbuf, len(outbuf, kind=C_INT))
        ! splicer end function.callback3
    end subroutine callback3

    ! int passStruct2(Cstruct1 * s1 +intent(in), char * outbuf +intent(out))
    ! arg_to_buffer
    !>
    !! Pass name argument which will build a bufferify function.
    !<
    function pass_struct2(s1, outbuf) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        type(cstruct1), intent(IN) :: s1
        character(len=*), intent(OUT) :: outbuf
        integer(C_INT) :: SHT_rv
        ! splicer begin function.pass_struct2
        SHT_rv = c_pass_struct2_bufferify(s1, outbuf, &
            len(outbuf, kind=C_INT))
        ! splicer end function.pass_struct2
    end function pass_struct2

    ! Cstruct1 * returnStructPtr1(int ifield +intent(in)+value)
    !>
    !! \brief Return a pointer to a struct
    !!
    !! Does not generate a bufferify C wrapper.
    !<
    function return_struct_ptr1(ifield) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, c_f_pointer
        integer(C_INT), value, intent(IN) :: ifield
        type(cstruct1), pointer :: SHT_rv
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_struct_ptr1
        SHT_ptr = c_return_struct_ptr1(ifield)
        call c_f_pointer(SHT_ptr, SHT_rv)
        ! splicer end function.return_struct_ptr1
    end function return_struct_ptr1

    ! Cstruct1 * returnStructPtr2(int ifield +intent(in)+value, char * outbuf +intent(out))
    ! arg_to_buffer
    !>
    !! \brief Return a pointer to a struct
    !!
    !! Generates a bufferify C wrapper function.
    !<
    function return_struct_ptr2(ifield, outbuf) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, c_f_pointer
        integer(C_INT), value, intent(IN) :: ifield
        character(len=*), intent(OUT) :: outbuf
        type(cstruct1), pointer :: SHT_rv
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_struct_ptr2
        SHT_ptr = c_return_struct_ptr2_bufferify(ifield, outbuf, &
            len(outbuf, kind=C_INT))
        call c_f_pointer(SHT_ptr, SHT_rv)
        ! splicer end function.return_struct_ptr2
    end function return_struct_ptr2

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module clibrary_mod