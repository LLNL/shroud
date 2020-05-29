
! splicer begin additional_interfaces

subroutine c_get_const_string_ptr_alloc_bufferify_tr(SHT_rv) &
     bind(C, name="TR2_get_const_string_ptr_alloc_tr_bufferify")
    implicit none
    character(len=:), allocatable :: SHT_rv
end subroutine c_get_const_string_ptr_alloc_bufferify_tr

! splicer end additional_interfaces


! splicer begin additional_functions
function get_const_string_ptr_alloc_tr() &
        result(SHT_rv)
!    type(SHROUD_array) :: DSHF_rv
    character(len=:), allocatable :: SHT_rv
    call c_get_const_string_ptr_alloc_bufferify_tr(SHT_rv)
!    allocate(character(len=DSHF_rv%elem_len):: SHT_rv)
!    call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%elem_len)
end function get_const_string_ptr_alloc_tr

! splicer end additional_functions
