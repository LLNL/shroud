-- test tutorial module

local tutorial = require "tutorial"
local rv_int, rv_double

tutorial.Function1()
print(tutorial.LastFunctionCalled())

rv_double = tutorial.Function2(1.0, 4)
print(tutorial.LastFunctionCalled(), rv_double)


tutorial.Function10()
print(tutorial.LastFunctionCalled())
tutorial.Function10("foo", 1.0)
print(tutorial.LastFunctionCalled())


rv_int = tutorial.overload1(10)
print(tutorial.LastFunctionCalled(), rv_int)
-- This should call overload (double type, int num)
-- but instead calls (int num, int offset)
-- since there is only one number type
rv_int = tutorial.overload1(1.0, 10)
print(tutorial.LastFunctionCalled(), rv_int)

rv_int = tutorial.overload1(10, 11, 12)
print(tutorial.LastFunctionCalled(), rv_int)
rv_int = tutorial.overload1(1.0, 10, 11, 12)
print(tutorial.LastFunctionCalled(), rv_int)

-- rv_int = tutorial.overload1("no such overload")


-- call a class
local obj = tutorial.Class1()
obj:Method1()
print(tutorial.LastFunctionCalled())

