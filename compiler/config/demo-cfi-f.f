! Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)

program cfi1

  implicit none

  interface
     subroutine demo_cfi(arg) bind(C, name="Demo_CFI")
       implicit none
       type(*), dimension(..) :: arg
     end subroutine demo_cfi
  end interface

  integer iscalar
  integer idata(10)
  integer rdata(2,5)

  call demo_cfi(iscalar)
  call demo_cfi(idata)
  call demo_cfi(rdata)

end program cfi1

  
