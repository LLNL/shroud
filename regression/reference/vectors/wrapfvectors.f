! wrapfvectors.f
! This is generated code, do not edit
! Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfvectors.f
!! \brief Shroud generated wrapper for vectors library
!<
! splicer begin file_top
! splicer end file_top
module vectors_mod
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

    ! start array_context
    type, bind(C) :: SHROUD_array
        ! address of C++ memory
        type(SHROUD_capsule_data) :: cxx
        ! address of data in cxx
        type(C_PTR) :: base_addr = C_NULL_PTR
        ! type of element
        integer(C_INT) :: type
        ! bytes-per-item or character len of data in cxx
        integer(C_SIZE_T) :: elem_len = 0_C_SIZE_T
        ! size of data in cxx
        integer(C_SIZE_T) :: size = 0_C_SIZE_T
    end type SHROUD_array
    ! end array_context

    ! start c_vector_sum_bufferify
    interface
        function c_vector_sum_bufferify(arg, Sarg) &
                result(SHT_rv) &
                bind(C, name="VEC_vector_sum_bufferify")
            use iso_c_binding, only : C_INT, C_LONG
            implicit none
            integer(C_INT), intent(IN) :: arg(*)
            integer(C_LONG), value, intent(IN) :: Sarg
            integer(C_INT) :: SHT_rv
        end function c_vector_sum_bufferify
    end interface
    ! end c_vector_sum_bufferify

    ! start c_vector_iota_out_bufferify
    interface
        subroutine c_vector_iota_out_bufferify(Darg) &
                bind(C, name="VEC_vector_iota_out_bufferify")
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_iota_out_bufferify
    end interface
    ! end c_vector_iota_out_bufferify

    ! start c_vector_iota_out_with_num_bufferify
    interface
        function c_vector_iota_out_with_num_bufferify(Darg) &
                result(SHT_rv) &
                bind(C, name="VEC_vector_iota_out_with_num_bufferify")
            use iso_c_binding, only : C_LONG
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: Darg
            integer(C_LONG) SHT_rv
        end function c_vector_iota_out_with_num_bufferify
    end interface
    ! end c_vector_iota_out_with_num_bufferify

    ! start c_vector_iota_out_with_num2_bufferify
    interface
        subroutine c_vector_iota_out_with_num2_bufferify(Darg) &
                bind(C, name="VEC_vector_iota_out_with_num2_bufferify")
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_iota_out_with_num2_bufferify
    end interface
    ! end c_vector_iota_out_with_num2_bufferify

    ! start c_vector_iota_out_alloc_bufferify
    interface
        subroutine c_vector_iota_out_alloc_bufferify(Darg) &
                bind(C, name="VEC_vector_iota_out_alloc_bufferify")
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_iota_out_alloc_bufferify
    end interface
    ! end c_vector_iota_out_alloc_bufferify

    ! start c_vector_iota_inout_alloc_bufferify
    interface
        subroutine c_vector_iota_inout_alloc_bufferify(arg, Sarg, Darg) &
                bind(C, name="VEC_vector_iota_inout_alloc_bufferify")
            use iso_c_binding, only : C_INT, C_LONG
            import :: SHROUD_array
            implicit none
            integer(C_INT), intent(INOUT) :: arg(*)
            integer(C_LONG), value, intent(IN) :: Sarg
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_iota_inout_alloc_bufferify
    end interface
    ! end c_vector_iota_inout_alloc_bufferify

    interface
        subroutine c_vector_increment_bufferify(arg, Sarg, Darg) &
                bind(C, name="VEC_vector_increment_bufferify")
            use iso_c_binding, only : C_INT, C_LONG
            import :: SHROUD_array
            implicit none
            integer(C_INT), intent(INOUT) :: arg(*)
            integer(C_LONG), value, intent(IN) :: Sarg
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_increment_bufferify
    end interface

    interface
        subroutine c_vector_iota_out_d_bufferify(Darg) &
                bind(C, name="VEC_vector_iota_out_d_bufferify")
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_iota_out_d_bufferify
    end interface

    interface
        function c_vector_string_count_bufferify(arg, Sarg, Narg) &
                result(SHT_rv) &
                bind(C, name="VEC_vector_string_count_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_LONG
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg(*)
            integer(C_LONG), value, intent(IN) :: Sarg
            integer(C_INT), value, intent(IN) :: Narg
            integer(C_INT) :: SHT_rv
        end function c_vector_string_count_bufferify
    end interface

    interface
        subroutine c_return_vector_alloc_bufferify(n, DSHF_rv) &
                bind(C, name="VEC_return_vector_alloc_bufferify")
            use iso_c_binding, only : C_INT
            import :: SHROUD_array
            implicit none
            integer(C_INT), value, intent(IN) :: n
            type(SHROUD_array), intent(OUT) :: DSHF_rv
        end subroutine c_return_vector_alloc_bufferify
    end interface

    interface
        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface
        ! helper function
        ! Copy contents of context into c_var.
        subroutine SHROUD_copy_array_double(context, c_var, c_var_size) &
            bind(C, name="VEC_ShroudCopyArray")
            use iso_c_binding, only : C_DOUBLE, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            real(C_DOUBLE), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_array_double
    end interface

    interface
        ! helper function
        ! Copy contents of context into c_var.
        subroutine SHROUD_copy_array_int(context, c_var, c_var_size) &
            bind(C, name="VEC_ShroudCopyArray")
            use iso_c_binding, only : C_INT, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            integer(C_INT), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_array_int
    end interface

contains

    ! int vector_sum(const std::vector<int> & arg +dimension(:)+intent(in))
    ! arg_to_buffer
    ! start vector_sum
    function vector_sum(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_LONG
        integer(C_INT), intent(IN) :: arg(:)
        integer(C_INT) :: SHT_rv
        ! splicer begin function.vector_sum
        SHT_rv = c_vector_sum_bufferify(arg, size(arg, kind=C_LONG))
        ! splicer end function.vector_sum
    end function vector_sum
    ! end vector_sum

    ! void vector_iota_out(std::vector<int> & arg +dimension(:)+intent(out))
    ! arg_to_buffer
    !>
    !! \brief Copy vector into Fortran input array
    !!
    !<
    ! start vector_iota_out
    subroutine vector_iota_out(arg)
        use iso_c_binding, only : C_INT, C_SIZE_T
        integer(C_INT), intent(OUT) :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_iota_out
        call c_vector_iota_out_bufferify(Darg)
        call SHROUD_copy_array_int(Darg, arg, size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_iota_out
    end subroutine vector_iota_out
    ! end vector_iota_out

    ! void vector_iota_out_with_num(std::vector<int> & arg +dimension(:)+intent(out))
    ! arg_to_buffer
    !>
    !! \brief Copy vector into Fortran input array
    !!
    !! Return the number of items copied into argument
    !! by setting fstatements for both C and Fortran.
    !<
    ! start vector_iota_out_with_num
    function vector_iota_out_with_num(arg) &
            result(num)
        use iso_c_binding, only : C_INT, C_LONG, C_SIZE_T
        integer(C_INT), intent(OUT) :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_iota_out_with_num
        integer(C_LONG) :: num
        num = c_vector_iota_out_with_num_bufferify(Darg)
        call SHROUD_copy_array_int(Darg, arg, size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_iota_out_with_num
    end function vector_iota_out_with_num
    ! end vector_iota_out_with_num

    ! void vector_iota_out_with_num2(std::vector<int> & arg +dimension(:)+intent(out))
    ! arg_to_buffer
    !>
    !! \brief Copy vector into Fortran input array
    !!
    !! Return the number of items copied into argument
    !! by setting fstatements for the Fortran wrapper only.
    !<
    ! start vector_iota_out_with_num2
    function vector_iota_out_with_num2(arg) &
            result(num)
        use iso_c_binding, only : C_INT, C_LONG, C_SIZE_T
        integer(C_INT), intent(OUT) :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_iota_out_with_num2
        integer(C_LONG) :: num
        call c_vector_iota_out_with_num2_bufferify(Darg)
        call SHROUD_copy_array_int(Darg, arg, size(arg,kind=C_SIZE_T))
        num = Darg%size
        ! splicer end function.vector_iota_out_with_num2
    end function vector_iota_out_with_num2
    ! end vector_iota_out_with_num2

    ! void vector_iota_out_alloc(std::vector<int> & arg +deref(allocatable)+dimension(:)+intent(out))
    ! arg_to_buffer
    !>
    !! \brief Copy vector into Fortran allocatable array
    !!
    !<
    ! start vector_iota_out_alloc
    subroutine vector_iota_out_alloc(arg)
        use iso_c_binding, only : C_INT, C_SIZE_T
        integer(C_INT), intent(OUT), allocatable :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_iota_out_alloc
        call c_vector_iota_out_alloc_bufferify(Darg)
        allocate(arg(Darg%size))
        call SHROUD_copy_array_int(Darg, arg, size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_iota_out_alloc
    end subroutine vector_iota_out_alloc
    ! end vector_iota_out_alloc

    ! void vector_iota_inout_alloc(std::vector<int> & arg +deref(allocatable)+dimension(:)+intent(inout))
    ! arg_to_buffer
    !>
    !! \brief Copy vector into Fortran allocatable array
    !!
    !<
    ! start vector_iota_inout_alloc
    subroutine vector_iota_inout_alloc(arg)
        use iso_c_binding, only : C_INT, C_LONG, C_SIZE_T
        integer(C_INT), intent(INOUT), allocatable :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_iota_inout_alloc
        call c_vector_iota_inout_alloc_bufferify(arg, &
            size(arg, kind=C_LONG), Darg)
        if (allocated(arg)) deallocate(arg)
        allocate(arg(Darg%size))
        call SHROUD_copy_array_int(Darg, arg, size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_iota_inout_alloc
    end subroutine vector_iota_inout_alloc
    ! end vector_iota_inout_alloc

    ! void vector_increment(std::vector<int> & arg +dimension(:)+intent(inout))
    ! arg_to_buffer
    subroutine vector_increment(arg)
        use iso_c_binding, only : C_INT, C_LONG, C_SIZE_T
        integer(C_INT), intent(INOUT) :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_increment
        call c_vector_increment_bufferify(arg, size(arg, kind=C_LONG), &
            Darg)
        call SHROUD_copy_array_int(Darg, arg, size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_increment
    end subroutine vector_increment

    ! void vector_iota_out_d(std::vector<double> & arg +dimension(:)+intent(out))
    ! arg_to_buffer
    !>
    !! \brief Copy vector into Fortran input array
    !!
    !<
    subroutine vector_iota_out_d(arg)
        use iso_c_binding, only : C_DOUBLE, C_SIZE_T
        real(C_DOUBLE), intent(OUT) :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_iota_out_d
        call c_vector_iota_out_d_bufferify(Darg)
        call SHROUD_copy_array_double(Darg, arg, size(arg,kind=C_SIZE_T))
        ! splicer end function.vector_iota_out_d
    end subroutine vector_iota_out_d

    ! int vector_string_count(const std::vector<std::string> & arg +dimension(:)+intent(in))
    ! arg_to_buffer
    !>
    !! \brief count number of underscore in vector of strings
    !!
    !<
    function vector_string_count(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_LONG
        character(len=*), intent(IN) :: arg(:)
        integer(C_INT) :: SHT_rv
        ! splicer begin function.vector_string_count
        SHT_rv = c_vector_string_count_bufferify(arg, &
            size(arg, kind=C_LONG), len(arg, kind=C_INT))
        ! splicer end function.vector_string_count
    end function vector_string_count

    ! std::vector<int> ReturnVectorAlloc(int n +intent(in)+value) +deref(allocatable)+dimension(:)
    ! arg_to_buffer
    !>
    !! Implement iota function.
    !! Return a vector as an ALLOCATABLE array.
    !! Copy results into the new array.
    !<
    function return_vector_alloc(n) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_SIZE_T
        integer(C_INT), value, intent(IN) :: n
        type(SHROUD_array) :: DSHF_rv
        integer(C_INT), allocatable :: SHT_rv(:)
        ! splicer begin function.return_vector_alloc
        call c_return_vector_alloc_bufferify(n, DSHF_rv)
        allocate(SHT_rv(DSHF_rv%size))
        call SHROUD_copy_array_int(DSHF_rv, SHT_rv, size(SHT_rv,kind=C_SIZE_T))
        ! splicer end function.return_vector_alloc
    end function return_vector_alloc

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module vectors_mod
