// Copyright Shroud Project Developers. See LICENSE file for details.
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
