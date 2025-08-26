-- Copyright Shroud Project Developers. See LICENSE file for details.
--
-- SPDX-License-Identifier: (BSD-3-Clause)
--
-- #######################################################################
-- test types module

local types = require "types"
local rv_logical

rv_logical = types.bool_func(false)
print(tutorial.LastFunctionCalled(), rv_logical)

