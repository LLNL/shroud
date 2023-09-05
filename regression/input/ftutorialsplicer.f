! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!########################################################################

! splicer begin additional_declarations
interface
  subroutine all_test1(array)
    implicit none
    integer, dimension(:), allocatable :: array
  end subroutine all_test1
end interface
! splicer end additional_declarations
