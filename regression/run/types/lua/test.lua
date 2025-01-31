-- Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
-- other Shroud Project Developers.
-- See the top-level COPYRIGHT file for details.
--
-- SPDX-License-Identifier: (BSD-3-Clause)
--
-- #######################################################################
-- test types module

local types = require "types"
local rv_logical

rv_logical = types.bool_func(false)
print(tutorial.LastFunctionCalled(), rv_logical)

