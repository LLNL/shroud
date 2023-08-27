! wrapfarrayclass.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfarrayclass.f
!! \brief Shroud generated wrapper for arrayclass library
!<
! splicer begin file_top
! splicer end file_top
module arrayclass_mod
    use iso_c_binding, only : C_INT, C_LONG, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    ! helper capsule_data_helper
    type, bind(C) :: ARR_SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type ARR_SHROUD_capsule_data

    ! helper array_context
    type, bind(C) :: ARR_SHROUD_array
        ! address of C++ memory
        type(ARR_SHROUD_capsule_data) :: cxx
        ! address of data in cxx
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
    end type ARR_SHROUD_array

    type ArrayWrapper
        type(ARR_SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.ArrayWrapper.component_part
        ! splicer end class.ArrayWrapper.component_part
    contains
        procedure :: setSize => ArrayWrapper_setSize
        procedure :: getSize => ArrayWrapper_getSize
        procedure :: fillSize => ArrayWrapper_fillSize
        procedure :: allocate => ArrayWrapper_allocate
        procedure :: getArray => ArrayWrapper_getArray
        procedure :: getArrayConst => ArrayWrapper_getArrayConst
        procedure :: getArrayC => ArrayWrapper_getArrayC
        procedure :: getArrayConstC => ArrayWrapper_getArrayConstC
        procedure :: fetchArrayPtr => ArrayWrapper_fetchArrayPtr
        procedure :: fetchArrayRef => ArrayWrapper_fetchArrayRef
        procedure :: fetchArrayPtrConst => ArrayWrapper_fetchArrayPtrConst
        procedure :: fetchArrayRefConst => ArrayWrapper_fetchArrayRefConst
        procedure :: fetchVoidPtr => ArrayWrapper_fetchVoidPtr
        procedure :: fetchVoidRef => ArrayWrapper_fetchVoidRef
        procedure :: checkPtr => ArrayWrapper_checkPtr
        procedure :: sumArray => ArrayWrapper_sumArray
        procedure :: get_instance => ArrayWrapper_get_instance
        procedure :: set_instance => ArrayWrapper_set_instance
        procedure :: associated => ArrayWrapper_associated
        ! splicer begin class.ArrayWrapper.type_bound_procedure_part
        ! splicer end class.ArrayWrapper.type_bound_procedure_part
    end type ArrayWrapper

    interface operator (.eq.)
        module procedure ArrayWrapper_eq
    end interface

    interface operator (.ne.)
        module procedure ArrayWrapper_ne
    end interface

    interface

        ! ----------------------------------------
        ! Function:  ArrayWrapper
        ! Attrs:     +api(capptr)+intent(ctor)
        ! Exact:     c_ctor_shadow_scalar_capptr
        function c_ArrayWrapper_ctor(SHT_rv) &
                result(SHT_prv) &
                bind(C, name="ARR_ArrayWrapper_ctor")
            use iso_c_binding, only : C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(OUT) :: SHT_rv
            type(C_PTR) SHT_prv
        end function c_ArrayWrapper_ctor

        ! ----------------------------------------
        ! Function:  void setSize
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  int size +value
        ! Attrs:     +intent(in)
        ! Exact:     c_in_native_scalar
        subroutine c_ArrayWrapper_setSize(self, size) &
                bind(C, name="ARR_ArrayWrapper_setSize")
            use iso_c_binding, only : C_INT
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: size
        end subroutine c_ArrayWrapper_setSize

        ! ----------------------------------------
        ! Function:  int getSize
        ! Attrs:     +intent(function)
        ! Exact:     c_function_native_scalar
        pure function c_ArrayWrapper_getSize(self) &
                result(SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_getSize")
            use iso_c_binding, only : C_INT
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT) :: SHT_rv
        end function c_ArrayWrapper_getSize

        ! ----------------------------------------
        ! Function:  void fillSize
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  int & size +intent(out)
        ! Attrs:     +intent(out)
        ! Exact:     c_out_native_&
        subroutine c_ArrayWrapper_fillSize(self, size) &
                bind(C, name="ARR_ArrayWrapper_fillSize")
            use iso_c_binding, only : C_INT
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), intent(OUT) :: size
        end subroutine c_ArrayWrapper_fillSize

        ! ----------------------------------------
        ! Function:  void allocate
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        subroutine c_ArrayWrapper_allocate(self) &
                bind(C, name="ARR_ArrayWrapper_allocate")
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_ArrayWrapper_allocate

        ! ----------------------------------------
        ! Function:  double * getArray +dimension(getSize())
        ! Attrs:     +deref(pointer)+intent(function)
        ! Exact:     c_function_native_*_pointer
        function c_ArrayWrapper_getArray(self) &
                result(SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_getArray")
            use iso_c_binding, only : C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_ArrayWrapper_getArray

        ! ----------------------------------------
        ! Function:  double * getArray +dimension(getSize())
        ! Attrs:     +api(cdesc)+deref(pointer)+intent(function)
        ! Exact:     c_function_native_*_cdesc_pointer
        subroutine c_ArrayWrapper_getArray_bufferify(self, SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_getArray_bufferify")
            import :: ARR_SHROUD_array, ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(ARR_SHROUD_array), intent(OUT) :: SHT_rv
        end subroutine c_ArrayWrapper_getArray_bufferify

        ! ----------------------------------------
        ! Function:  double * getArrayConst +dimension(getSize())
        ! Attrs:     +deref(pointer)+intent(function)
        ! Exact:     c_function_native_*_pointer
        pure function c_ArrayWrapper_getArrayConst(self) &
                result(SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_getArrayConst")
            use iso_c_binding, only : C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_ArrayWrapper_getArrayConst

        ! ----------------------------------------
        ! Function:  double * getArrayConst +dimension(getSize())
        ! Attrs:     +api(cdesc)+deref(pointer)+intent(function)
        ! Exact:     c_function_native_*_cdesc_pointer
        subroutine c_ArrayWrapper_getArrayConst_bufferify(self, SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_getArrayConst_bufferify")
            import :: ARR_SHROUD_array, ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(ARR_SHROUD_array), intent(OUT) :: SHT_rv
        end subroutine c_ArrayWrapper_getArrayConst_bufferify

        ! ----------------------------------------
        ! Function:  const double * getArrayC +dimension(getSize())
        ! Attrs:     +deref(pointer)+intent(function)
        ! Exact:     c_function_native_*_pointer
        function c_ArrayWrapper_getArrayC(self) &
                result(SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_getArrayC")
            use iso_c_binding, only : C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_ArrayWrapper_getArrayC

        ! ----------------------------------------
        ! Function:  const double * getArrayC +dimension(getSize())
        ! Attrs:     +api(cdesc)+deref(pointer)+intent(function)
        ! Exact:     c_function_native_*_cdesc_pointer
        subroutine c_ArrayWrapper_getArrayC_bufferify(self, SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_getArrayC_bufferify")
            import :: ARR_SHROUD_array, ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(ARR_SHROUD_array), intent(OUT) :: SHT_rv
        end subroutine c_ArrayWrapper_getArrayC_bufferify

        ! ----------------------------------------
        ! Function:  const double * getArrayConstC +dimension(getSize())
        ! Attrs:     +deref(pointer)+intent(function)
        ! Exact:     c_function_native_*_pointer
        pure function c_ArrayWrapper_getArrayConstC(self) &
                result(SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_getArrayConstC")
            use iso_c_binding, only : C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR) SHT_rv
        end function c_ArrayWrapper_getArrayConstC

        ! ----------------------------------------
        ! Function:  const double * getArrayConstC +dimension(getSize())
        ! Attrs:     +api(cdesc)+deref(pointer)+intent(function)
        ! Exact:     c_function_native_*_cdesc_pointer
        subroutine c_ArrayWrapper_getArrayConstC_bufferify(self, SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_getArrayConstC_bufferify")
            import :: ARR_SHROUD_array, ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(ARR_SHROUD_array), intent(OUT) :: SHT_rv
        end subroutine c_ArrayWrapper_getArrayConstC_bufferify

        ! ----------------------------------------
        ! Function:  void fetchArrayPtr
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  double * * array +dimension(isize)+intent(out)
        ! Attrs:     +deref(pointer)+intent(out)
        ! Exact:     c_out_native_**_pointer
        ! ----------------------------------------
        ! Argument:  int * isize +hidden
        ! Attrs:     +intent(inout)
        ! Exact:     c_inout_native_*
        subroutine c_ArrayWrapper_fetchArrayPtr(self, array, isize) &
                bind(C, name="ARR_ArrayWrapper_fetchArrayPtr")
            use iso_c_binding, only : C_INT, C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR), intent(OUT) :: array
            integer(C_INT), intent(INOUT) :: isize
        end subroutine c_ArrayWrapper_fetchArrayPtr

        ! ----------------------------------------
        ! Function:  void fetchArrayPtr
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  double * * array +dimension(isize)+intent(out)
        ! Attrs:     +api(cdesc)+deref(pointer)+intent(out)
        ! Exact:     c_out_native_**_cdesc_pointer
        subroutine c_ArrayWrapper_fetchArrayPtr_bufferify(self, &
                SHT_array_cdesc) &
                bind(C, name="ARR_ArrayWrapper_fetchArrayPtr_bufferify")
            import :: ARR_SHROUD_array, ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(ARR_SHROUD_array), intent(OUT) :: SHT_array_cdesc
        end subroutine c_ArrayWrapper_fetchArrayPtr_bufferify

        ! ----------------------------------------
        ! Function:  void fetchArrayRef
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  double * & array +dimension(isize)+intent(out)
        ! Attrs:     +deref(pointer)+intent(out)
        ! Exact:     c_out_native_*&_pointer
        ! ----------------------------------------
        ! Argument:  int & isize +hidden
        ! Attrs:     +intent(inout)
        ! Exact:     c_inout_native_&
        subroutine c_ArrayWrapper_fetchArrayRef(self, array, isize) &
                bind(C, name="ARR_ArrayWrapper_fetchArrayRef")
            use iso_c_binding, only : C_INT, C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR), intent(OUT) :: array
            integer(C_INT), intent(INOUT) :: isize
        end subroutine c_ArrayWrapper_fetchArrayRef

        ! ----------------------------------------
        ! Function:  void fetchArrayRef
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  double * & array +dimension(isize)+intent(out)
        ! Attrs:     +api(cdesc)+deref(pointer)+intent(out)
        ! Exact:     c_out_native_*&_cdesc_pointer
        subroutine c_ArrayWrapper_fetchArrayRef_bufferify(self, &
                SHT_array_cdesc) &
                bind(C, name="ARR_ArrayWrapper_fetchArrayRef_bufferify")
            import :: ARR_SHROUD_array, ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(ARR_SHROUD_array), intent(OUT) :: SHT_array_cdesc
        end subroutine c_ArrayWrapper_fetchArrayRef_bufferify

        ! ----------------------------------------
        ! Function:  void fetchArrayPtrConst
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  const double * * array +dimension(isize)+intent(out)
        ! Attrs:     +deref(pointer)+intent(out)
        ! Exact:     c_out_native_**_pointer
        ! ----------------------------------------
        ! Argument:  int * isize +hidden
        ! Attrs:     +intent(inout)
        ! Exact:     c_inout_native_*
        subroutine c_ArrayWrapper_fetchArrayPtrConst(self, array, isize) &
                bind(C, name="ARR_ArrayWrapper_fetchArrayPtrConst")
            use iso_c_binding, only : C_INT, C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR), intent(OUT) :: array
            integer(C_INT), intent(INOUT) :: isize
        end subroutine c_ArrayWrapper_fetchArrayPtrConst

        ! ----------------------------------------
        ! Function:  void fetchArrayPtrConst
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  const double * * array +dimension(isize)+intent(out)
        ! Attrs:     +api(cdesc)+deref(pointer)+intent(out)
        ! Exact:     c_out_native_**_cdesc_pointer
        subroutine c_ArrayWrapper_fetchArrayPtrConst_bufferify(self, &
                SHT_array_cdesc) &
                bind(C, name="ARR_ArrayWrapper_fetchArrayPtrConst_bufferify")
            import :: ARR_SHROUD_array, ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(ARR_SHROUD_array), intent(OUT) :: SHT_array_cdesc
        end subroutine c_ArrayWrapper_fetchArrayPtrConst_bufferify

        ! ----------------------------------------
        ! Function:  void fetchArrayRefConst
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  const double * & array +dimension(isize)+intent(out)
        ! Attrs:     +deref(pointer)+intent(out)
        ! Exact:     c_out_native_*&_pointer
        ! ----------------------------------------
        ! Argument:  int & isize +hidden
        ! Attrs:     +intent(inout)
        ! Exact:     c_inout_native_&
        subroutine c_ArrayWrapper_fetchArrayRefConst(self, array, isize) &
                bind(C, name="ARR_ArrayWrapper_fetchArrayRefConst")
            use iso_c_binding, only : C_INT, C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR), intent(OUT) :: array
            integer(C_INT), intent(INOUT) :: isize
        end subroutine c_ArrayWrapper_fetchArrayRefConst

        ! ----------------------------------------
        ! Function:  void fetchArrayRefConst
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  const double * & array +dimension(isize)+intent(out)
        ! Attrs:     +api(cdesc)+deref(pointer)+intent(out)
        ! Exact:     c_out_native_*&_cdesc_pointer
        subroutine c_ArrayWrapper_fetchArrayRefConst_bufferify(self, &
                SHT_array_cdesc) &
                bind(C, name="ARR_ArrayWrapper_fetchArrayRefConst_bufferify")
            import :: ARR_SHROUD_array, ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(ARR_SHROUD_array), intent(OUT) :: SHT_array_cdesc
        end subroutine c_ArrayWrapper_fetchArrayRefConst_bufferify

        ! ----------------------------------------
        ! Function:  void fetchVoidPtr
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  void * * array +intent(out)
        ! Attrs:     +intent(out)
        ! Exact:     c_out_void_**
        subroutine c_ArrayWrapper_fetchVoidPtr(self, array) &
                bind(C, name="ARR_ArrayWrapper_fetchVoidPtr")
            use iso_c_binding, only : C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR), intent(OUT) :: array
        end subroutine c_ArrayWrapper_fetchVoidPtr

        ! ----------------------------------------
        ! Function:  void fetchVoidRef
        ! Attrs:     +intent(subroutine)
        ! Exact:     c_subroutine_void_scalar
        ! ----------------------------------------
        ! Argument:  void * & array +intent(out)
        ! Attrs:     +intent(out)
        ! Exact:     c_out_void_*&
        subroutine c_ArrayWrapper_fetchVoidRef(self, array) &
                bind(C, name="ARR_ArrayWrapper_fetchVoidRef")
            use iso_c_binding, only : C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR), intent(OUT) :: array
        end subroutine c_ArrayWrapper_fetchVoidRef

        ! ----------------------------------------
        ! Function:  bool checkPtr
        ! Attrs:     +intent(function)
        ! Exact:     c_function_bool_scalar
        ! ----------------------------------------
        ! Argument:  void * array +value
        ! Attrs:     +intent(in)
        ! Exact:     c_in_void_*
        function c_ArrayWrapper_checkPtr(self, array) &
                result(SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_checkPtr")
            use iso_c_binding, only : C_BOOL, C_PTR
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            type(C_PTR), value, intent(IN) :: array
            logical(C_BOOL) :: SHT_rv
        end function c_ArrayWrapper_checkPtr

        ! ----------------------------------------
        ! Function:  double sumArray
        ! Attrs:     +intent(function)
        ! Exact:     c_function_native_scalar
        function c_ArrayWrapper_sumArray(self) &
                result(SHT_rv) &
                bind(C, name="ARR_ArrayWrapper_sumArray")
            use iso_c_binding, only : C_DOUBLE
            import :: ARR_SHROUD_capsule_data
            implicit none
            type(ARR_SHROUD_capsule_data), intent(IN) :: self
            real(C_DOUBLE) :: SHT_rv
        end function c_ArrayWrapper_sumArray
    end interface

    interface ArrayWrapper
        module procedure ArrayWrapper_ctor
    end interface ArrayWrapper

    ! splicer begin additional_declarations
    ! splicer end additional_declarations

contains

    ! ----------------------------------------
    ! Function:  ArrayWrapper
    ! Attrs:     +api(capptr)+intent(ctor)
    ! Exact:     f_ctor_shadow_scalar_capptr
    ! Attrs:     +api(capptr)+intent(ctor)
    ! Exact:     c_ctor_shadow_scalar_capptr
    function ArrayWrapper_ctor() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(ArrayWrapper) :: SHT_rv
        type(C_PTR) :: SHT_prv
        ! splicer begin class.ArrayWrapper.method.ctor
        SHT_prv = c_ArrayWrapper_ctor(SHT_rv%cxxmem)
        ! splicer end class.ArrayWrapper.method.ctor
    end function ArrayWrapper_ctor

    ! ----------------------------------------
    ! Function:  void setSize
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  int size +value
    ! Attrs:     +intent(in)
    ! Exact:     f_in_native_scalar
    ! Attrs:     +intent(in)
    ! Exact:     c_in_native_scalar
    subroutine ArrayWrapper_setSize(obj, size)
        use iso_c_binding, only : C_INT
        class(ArrayWrapper) :: obj
        integer(C_INT), value, intent(IN) :: size
        ! splicer begin class.ArrayWrapper.method.setSize
        call c_ArrayWrapper_setSize(obj%cxxmem, size)
        ! splicer end class.ArrayWrapper.method.setSize
    end subroutine ArrayWrapper_setSize

    ! ----------------------------------------
    ! Function:  int getSize
    ! Attrs:     +intent(function)
    ! Exact:     f_function_native_scalar
    ! Attrs:     +intent(function)
    ! Exact:     c_function_native_scalar
    function ArrayWrapper_getSize(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT
        class(ArrayWrapper) :: obj
        integer(C_INT) :: SHT_rv
        ! splicer begin class.ArrayWrapper.method.getSize
        SHT_rv = c_ArrayWrapper_getSize(obj%cxxmem)
        ! splicer end class.ArrayWrapper.method.getSize
    end function ArrayWrapper_getSize

    ! ----------------------------------------
    ! Function:  void fillSize
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  int & size +intent(out)
    ! Attrs:     +intent(out)
    ! Exact:     f_out_native_&
    ! Attrs:     +intent(out)
    ! Exact:     c_out_native_&
    subroutine ArrayWrapper_fillSize(obj, size)
        use iso_c_binding, only : C_INT
        class(ArrayWrapper) :: obj
        integer(C_INT), intent(OUT) :: size
        ! splicer begin class.ArrayWrapper.method.fillSize
        call c_ArrayWrapper_fillSize(obj%cxxmem, size)
        ! splicer end class.ArrayWrapper.method.fillSize
    end subroutine ArrayWrapper_fillSize

    ! ----------------------------------------
    ! Function:  void allocate
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    subroutine ArrayWrapper_allocate(obj)
        class(ArrayWrapper) :: obj
        ! splicer begin class.ArrayWrapper.method.allocate
        call c_ArrayWrapper_allocate(obj%cxxmem)
        ! splicer end class.ArrayWrapper.method.allocate
    end subroutine ArrayWrapper_allocate

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  double * getArray +dimension(getSize())
    ! Attrs:     +deref(pointer)+intent(function)
    ! Exact:     f_function_native_*_cdesc_pointer
    ! Attrs:     +api(cdesc)+deref(pointer)+intent(function)
    ! Exact:     c_function_native_*_cdesc_pointer
    function ArrayWrapper_getArray(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, c_f_pointer
        class(ArrayWrapper) :: obj
        real(C_DOUBLE), pointer :: SHT_rv(:)
        ! splicer begin class.ArrayWrapper.method.getArray
        type(ARR_SHROUD_array) :: SHT_rv_cdesc
        call c_ArrayWrapper_getArray_bufferify(obj%cxxmem, SHT_rv_cdesc)
        call c_f_pointer(SHT_rv_cdesc%base_addr, SHT_rv, &
            SHT_rv_cdesc%shape(1:1))
        ! splicer end class.ArrayWrapper.method.getArray
    end function ArrayWrapper_getArray

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  double * getArrayConst +dimension(getSize())
    ! Attrs:     +deref(pointer)+intent(function)
    ! Exact:     f_function_native_*_cdesc_pointer
    ! Attrs:     +api(cdesc)+deref(pointer)+intent(function)
    ! Exact:     c_function_native_*_cdesc_pointer
    function ArrayWrapper_getArrayConst(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, c_f_pointer
        class(ArrayWrapper) :: obj
        real(C_DOUBLE), pointer :: SHT_rv(:)
        ! splicer begin class.ArrayWrapper.method.getArrayConst
        type(ARR_SHROUD_array) :: SHT_rv_cdesc
        call c_ArrayWrapper_getArrayConst_bufferify(obj%cxxmem, &
            SHT_rv_cdesc)
        call c_f_pointer(SHT_rv_cdesc%base_addr, SHT_rv, &
            SHT_rv_cdesc%shape(1:1))
        ! splicer end class.ArrayWrapper.method.getArrayConst
    end function ArrayWrapper_getArrayConst

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  const double * getArrayC +dimension(getSize())
    ! Attrs:     +deref(pointer)+intent(function)
    ! Exact:     f_function_native_*_cdesc_pointer
    ! Attrs:     +api(cdesc)+deref(pointer)+intent(function)
    ! Exact:     c_function_native_*_cdesc_pointer
    function ArrayWrapper_getArrayC(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, c_f_pointer
        class(ArrayWrapper) :: obj
        real(C_DOUBLE), pointer :: SHT_rv(:)
        ! splicer begin class.ArrayWrapper.method.getArrayC
        type(ARR_SHROUD_array) :: SHT_rv_cdesc
        call c_ArrayWrapper_getArrayC_bufferify(obj%cxxmem, &
            SHT_rv_cdesc)
        call c_f_pointer(SHT_rv_cdesc%base_addr, SHT_rv, &
            SHT_rv_cdesc%shape(1:1))
        ! splicer end class.ArrayWrapper.method.getArrayC
    end function ArrayWrapper_getArrayC

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  const double * getArrayConstC +dimension(getSize())
    ! Attrs:     +deref(pointer)+intent(function)
    ! Exact:     f_function_native_*_cdesc_pointer
    ! Attrs:     +api(cdesc)+deref(pointer)+intent(function)
    ! Exact:     c_function_native_*_cdesc_pointer
    function ArrayWrapper_getArrayConstC(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, c_f_pointer
        class(ArrayWrapper) :: obj
        real(C_DOUBLE), pointer :: SHT_rv(:)
        ! splicer begin class.ArrayWrapper.method.getArrayConstC
        type(ARR_SHROUD_array) :: SHT_rv_cdesc
        call c_ArrayWrapper_getArrayConstC_bufferify(obj%cxxmem, &
            SHT_rv_cdesc)
        call c_f_pointer(SHT_rv_cdesc%base_addr, SHT_rv, &
            SHT_rv_cdesc%shape(1:1))
        ! splicer end class.ArrayWrapper.method.getArrayConstC
    end function ArrayWrapper_getArrayConstC

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void fetchArrayPtr
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  double * * array +dimension(isize)+intent(out)
    ! Attrs:     +deref(pointer)+intent(out)
    ! Exact:     f_out_native_**_cdesc_pointer
    ! Attrs:     +api(cdesc)+deref(pointer)+intent(out)
    ! Exact:     c_out_native_**_cdesc_pointer
    subroutine ArrayWrapper_fetchArrayPtr(obj, array)
        use iso_c_binding, only : C_DOUBLE, c_f_pointer
        class(ArrayWrapper) :: obj
        real(C_DOUBLE), intent(OUT), pointer :: array(:)
        ! splicer begin class.ArrayWrapper.method.fetchArrayPtr
        type(ARR_SHROUD_array) :: SHT_array_cdesc
        call c_ArrayWrapper_fetchArrayPtr_bufferify(obj%cxxmem, &
            SHT_array_cdesc)
        call c_f_pointer(SHT_array_cdesc%base_addr, array, &
            SHT_array_cdesc%shape(1:1))
        ! splicer end class.ArrayWrapper.method.fetchArrayPtr
    end subroutine ArrayWrapper_fetchArrayPtr

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void fetchArrayRef
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  double * & array +dimension(isize)+intent(out)
    ! Attrs:     +deref(pointer)+intent(out)
    ! Exact:     f_out_native_*&_cdesc_pointer
    ! Attrs:     +api(cdesc)+deref(pointer)+intent(out)
    ! Exact:     c_out_native_*&_cdesc_pointer
    subroutine ArrayWrapper_fetchArrayRef(obj, array)
        use iso_c_binding, only : C_DOUBLE, c_f_pointer
        class(ArrayWrapper) :: obj
        real(C_DOUBLE), intent(OUT), pointer :: array(:)
        ! splicer begin class.ArrayWrapper.method.fetchArrayRef
        type(ARR_SHROUD_array) :: SHT_array_cdesc
        call c_ArrayWrapper_fetchArrayRef_bufferify(obj%cxxmem, &
            SHT_array_cdesc)
        call c_f_pointer(SHT_array_cdesc%base_addr, array, &
            SHT_array_cdesc%shape(1:1))
        ! splicer end class.ArrayWrapper.method.fetchArrayRef
    end subroutine ArrayWrapper_fetchArrayRef

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void fetchArrayPtrConst
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  const double * * array +dimension(isize)+intent(out)
    ! Attrs:     +deref(pointer)+intent(out)
    ! Exact:     f_out_native_**_cdesc_pointer
    ! Attrs:     +api(cdesc)+deref(pointer)+intent(out)
    ! Exact:     c_out_native_**_cdesc_pointer
    subroutine ArrayWrapper_fetchArrayPtrConst(obj, array)
        use iso_c_binding, only : C_DOUBLE, c_f_pointer
        class(ArrayWrapper) :: obj
        real(C_DOUBLE), intent(OUT), pointer :: array(:)
        ! splicer begin class.ArrayWrapper.method.fetchArrayPtrConst
        type(ARR_SHROUD_array) :: SHT_array_cdesc
        call c_ArrayWrapper_fetchArrayPtrConst_bufferify(obj%cxxmem, &
            SHT_array_cdesc)
        call c_f_pointer(SHT_array_cdesc%base_addr, array, &
            SHT_array_cdesc%shape(1:1))
        ! splicer end class.ArrayWrapper.method.fetchArrayPtrConst
    end subroutine ArrayWrapper_fetchArrayPtrConst

    ! Generated by arg_to_buffer
    ! ----------------------------------------
    ! Function:  void fetchArrayRefConst
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  const double * & array +dimension(isize)+intent(out)
    ! Attrs:     +deref(pointer)+intent(out)
    ! Exact:     f_out_native_*&_cdesc_pointer
    ! Attrs:     +api(cdesc)+deref(pointer)+intent(out)
    ! Exact:     c_out_native_*&_cdesc_pointer
    subroutine ArrayWrapper_fetchArrayRefConst(obj, array)
        use iso_c_binding, only : C_DOUBLE, c_f_pointer
        class(ArrayWrapper) :: obj
        real(C_DOUBLE), intent(OUT), pointer :: array(:)
        ! splicer begin class.ArrayWrapper.method.fetchArrayRefConst
        type(ARR_SHROUD_array) :: SHT_array_cdesc
        call c_ArrayWrapper_fetchArrayRefConst_bufferify(obj%cxxmem, &
            SHT_array_cdesc)
        call c_f_pointer(SHT_array_cdesc%base_addr, array, &
            SHT_array_cdesc%shape(1:1))
        ! splicer end class.ArrayWrapper.method.fetchArrayRefConst
    end subroutine ArrayWrapper_fetchArrayRefConst

    ! ----------------------------------------
    ! Function:  void fetchVoidPtr
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  void * * array +intent(out)
    ! Attrs:     +intent(out)
    ! Exact:     f_out_void_**
    ! Attrs:     +intent(out)
    ! Exact:     c_out_void_**
    subroutine ArrayWrapper_fetchVoidPtr(obj, array)
        use iso_c_binding, only : C_PTR
        class(ArrayWrapper) :: obj
        type(C_PTR), intent(OUT) :: array
        ! splicer begin class.ArrayWrapper.method.fetchVoidPtr
        call c_ArrayWrapper_fetchVoidPtr(obj%cxxmem, array)
        ! splicer end class.ArrayWrapper.method.fetchVoidPtr
    end subroutine ArrayWrapper_fetchVoidPtr

    ! ----------------------------------------
    ! Function:  void fetchVoidRef
    ! Attrs:     +intent(subroutine)
    ! Exact:     f_subroutine
    ! Attrs:     +intent(subroutine)
    ! Exact:     c_subroutine
    ! ----------------------------------------
    ! Argument:  void * & array +intent(out)
    ! Attrs:     +intent(out)
    ! Exact:     f_out_void_*&
    ! Attrs:     +intent(out)
    ! Exact:     c_out_void_*&
    subroutine ArrayWrapper_fetchVoidRef(obj, array)
        class(ArrayWrapper) :: obj
        type(C_PTR), intent(OUT) :: array
        ! splicer begin class.ArrayWrapper.method.fetchVoidRef
        call c_ArrayWrapper_fetchVoidRef(obj%cxxmem, array)
        ! splicer end class.ArrayWrapper.method.fetchVoidRef
    end subroutine ArrayWrapper_fetchVoidRef

    ! ----------------------------------------
    ! Function:  bool checkPtr
    ! Attrs:     +intent(function)
    ! Exact:     f_function_bool_scalar
    ! Attrs:     +intent(function)
    ! Exact:     c_function_bool_scalar
    ! ----------------------------------------
    ! Argument:  void * array +value
    ! Attrs:     +intent(in)
    ! Exact:     f_in_void_*
    ! Attrs:     +intent(in)
    ! Exact:     c_in_void_*
    function ArrayWrapper_checkPtr(obj, array) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL, C_PTR
        class(ArrayWrapper) :: obj
        type(C_PTR), intent(IN) :: array
        logical :: SHT_rv
        ! splicer begin class.ArrayWrapper.method.checkPtr
        SHT_rv = c_ArrayWrapper_checkPtr(obj%cxxmem, array)
        ! splicer end class.ArrayWrapper.method.checkPtr
    end function ArrayWrapper_checkPtr

    ! ----------------------------------------
    ! Function:  double sumArray
    ! Attrs:     +intent(function)
    ! Exact:     f_function_native_scalar
    ! Attrs:     +intent(function)
    ! Exact:     c_function_native_scalar
    function ArrayWrapper_sumArray(obj) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE
        class(ArrayWrapper) :: obj
        real(C_DOUBLE) :: SHT_rv
        ! splicer begin class.ArrayWrapper.method.sumArray
        SHT_rv = c_ArrayWrapper_sumArray(obj%cxxmem)
        ! splicer end class.ArrayWrapper.method.sumArray
    end function ArrayWrapper_sumArray

    ! Return pointer to C++ memory.
    function ArrayWrapper_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(ArrayWrapper), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function ArrayWrapper_get_instance

    subroutine ArrayWrapper_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(ArrayWrapper), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine ArrayWrapper_set_instance

    function ArrayWrapper_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(ArrayWrapper), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function ArrayWrapper_associated

    ! splicer begin class.ArrayWrapper.additional_functions
    ! splicer end class.ArrayWrapper.additional_functions

    ! splicer begin additional_functions
    ! splicer end additional_functions

    function ArrayWrapper_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(ArrayWrapper), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function ArrayWrapper_eq

    function ArrayWrapper_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(ArrayWrapper), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function ArrayWrapper_ne

end module arrayclass_mod
