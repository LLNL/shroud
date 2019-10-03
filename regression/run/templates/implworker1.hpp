// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Put internal class in its own header to test cxx_header for a
// class.

namespace internal
{
    class ImplWorker1
    {
    public:
        static int getValue() {
            return 1;
        }
    };
}  // namespace internal
