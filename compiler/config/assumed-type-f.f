! Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)

      program tester
        use, intrinsic :: iso_c_binding
        implicit none
        integer(C_SHORT) :: i_short
        integer(C_INT)   :: i_int
        integer(C_LONG)  :: i_long
        integer(C_INT) :: rv1, rv2, rv3

        interface
           function c_func(a, flag) bind(C) result(rv)
             use, intrinsic :: iso_c_binding
             implicit none
             type(*) :: a
             integer(C_INT), value :: flag
             integer(C_INT) :: rv
           end function c_func
        end interface

        i_short = 2
        i_int   = 4
        i_long  = 8

        rv1 = c_func(i_short, 1)
        rv2 = c_func(i_int, 2)
        rv3 = c_func(i_long, 3)
        if (rv1 + rv2 + rv3 == 3) then
           print *, "have_assumed_type"
        else
           print *, "no_assumed_type"
        endif
      end program tester
