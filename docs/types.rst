.. Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
.. Produced at the Lawrence Livermore National Laboratory 
..
.. LLNL-CODE-738041.
.. All rights reserved. 
..
.. This file is part of Shroud.  For details, see
.. https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions are
.. met:
..
.. * Redistributions of source code must retain the above copyright
..   notice, this list of conditions and the disclaimer below.
.. 
.. * Redistributions in binary form must reproduce the above copyright
..   notice, this list of conditions and the disclaimer (as noted below)
..   in the documentation and/or other materials provided with the
..   distribution.
..
.. * Neither the name of the LLNS/LLNL nor the names of its contributors
..   may be used to endorse or promote products derived from this
..   software without specific prior written permission.
..
.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
.. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
.. LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
.. A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
.. LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
.. CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
.. EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
.. PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
.. PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
.. LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
.. NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
.. SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
..
.. #######################################################################

.. _TypesAnchor:

Types
=====

A typemap is created for each type to describe to Shroud how it should
convert a type between languages for each wrapper.  Native types are
predefined and a Shroud typemap is created for each ``struct`` and
``class`` declarations.

The general form is::

    declarations:
    - type: type-name
      fields:
         field1:
         field2:

*type-name* is the name used by C++.  There are some fields which are
used by all wrappers and other fields which are used by language
specific wrappers.

type fields
-----------

These fields are common to all wrapper languages.

base
^^^^

The base type of *type-name*.
This is used to generalize operations for several types.
The base types that Shroud uses are **string**, **vector**, 
or **shadow**.

cpp_if
^^^^^^

A c preprocessor test which is used to conditionally use
other fields of the type such as *c_header* and *cxx_header*::

  - type: MPI_Comm
    fields:
      cpp_if: ifdef USE_MPI


idtor
^^^^^

Index of ``capsule_data`` destructor in the function
*C_memory_dtor_function*.
This value is computed by Shroud and should not be set.
It can be used when formatting statements as ``{idtor}``.
Defaults to *0* indicating no destructor.

.. format field

result_as_arg
^^^^^^^^^^^^^

Override fields when result should be treated as an argument.
Defaults to *None*.

Statements
----------

Each language also provides a section that is used 
to insert language specific statements into the wrapper.
These are named **c_statements**, **f_statements**, and
**py_statements**.

The are broken down into several resolutions.  The first is the
intent of the argument.  *result* is used as the intent for 
function results.

intent_in
    Code to add for argument with ``intent(IN)``.
    Can be used to convert types or copy-in semantics.
    For example, ``char *`` to ``std::string``.

intent_out
    Code to add after call when ``intent(OUT)``.
    Used to implement copy-out semantics.

intent_inout
    Code to add after call when ``intent(INOUT)``.
    Used to implement copy-out semantics.

result
    Result of function.
    Including when it is passed as an argument, *F_string_result_as_arg*.


Each intent is then broken down into code to be added into
specific sections of the wrapper.  For example, **declaration**,
**pre_call** and **post_call**.

Each statement is formatted using the format dictionary for the argument.
This will define several variables.

c_var
    The C name of the argument.

cxx_var
    Name of the C++ variable.

f_var
    Fortran variable name for argument.

For example::

    f_statements:
      intent_in:
      - '{c_var} = {f_var}  ! coerce to C_BOOL'
      intent_out:
      - '{f_var} = {c_var}  ! coerce to logical'

Note that the code lines are quoted since they begin with a curly brace.
Otherwise YAML would interpret them as a dictionary.

See the language specific sections for details.



Types
-----

.. Shroud predefines many of the native types.

  * void
  * int
  * long
  * size_t
  * bool
  * float
  * double
  * std::string
  * std::vector

  Fortran has no support for unsigned types.
          ``size_t`` will be the correct number of bytes, but
          will be signed.



Integer and Real
----------------

The numeric types usually require no conversion.
In this case the type map is mainly used to generate declaration code 
for wrappers::

    type: int
    fields:
        c_type: int 
        cxx_type: int
        f_type: integer(C_INT)
        f_kind: C_INT
        f_module:
            iso_c_binding:
            - C_INT
        f_cast: int({f_var}, C_INT)

One case where a conversion is required is when the Fortran argument
is one type and the C++ argument is another. This may happen when an
overloaded function is generated so that a ``C_INT`` or ``C_LONG``
argument may be passed to a C++ function function expecting a
``long``.  The **f_cast** field is used to convert the argument to the
type expected by the C++ function.


Bool
----

C++ functions with a ``bool`` argument generate a Fortran wrapper with
a ``logical`` argument.  One of the goals of Shroud is to produce an
idiomatic interface.  Converting the types in the wrapper avoids the
awkwardness of requiring the Fortran user to passing in
``.true._c_bool`` instead of just ``.true.``.

The type map is defined as::

    type: bool
    fields:
        c_type: bool 
        cxx_type: bool 
        f_type: logical 
        f_kind: C_BOOL
        f_c_type: logical(C_BOOL) 
        f_module:
            iso_c_binding:
            -  C_BOOL
        f_statements:
           intent_in:
              c_local_var: true 
              pre_call:
              -  {c_var} = {f_var}  ! coerce to C_BOOL
           intent_out:
              c_local_var: true 
              post_call:
              -  {f_var} = {c_var}  ! coerce to logical
           intent_inout:
              c_local_var: true 
              pre_call:
              -  {c_var} = {f_var}  ! coerce to C_BOOL
              post_call:
              -  {f_var} = {c_var}  ! coerce to logical
           result:
              need_wrapper: true

The first thing to notice is that **f_c_type** is defined.  This is
the type used in the Fortran interface for the C wrapper.  The type
is ``logical(C_BOOL)`` while **f_type**, the type of the Fortran
wrapper argument, is ``logical``.

The **f_statements** section describes code to add into the Fortran
wrapper to perform the conversion.  *c_var* and *f_var* default to
the same value as the argument name.  By setting **c_local_var**, a
local variable is generated for the call to the C wrapper.  It will be
named ``SH_{f_var}``.

There is no Fortran intrinsic function to convert between default
``logical`` and ``logical(C_BOOL)``. The **pre_call** and
**post_call** sections will insert an assignment statement to allow
the compiler to do the conversion.

Example of using intent with ``bool`` arguments::

    decl: void checkBool(bool arg1, bool * arg2+intent(out), bool * arg3+intent(inout))

The resulting wrappers are::

    module userlibrary_mod
        interface
            subroutine c_check_bool(arg1, arg2, arg3) &
                    bind(C, name="AA_check_bool")
                use iso_c_binding
                implicit none
                logical(C_BOOL), value, intent(IN) :: arg1
                logical(C_BOOL), intent(OUT) :: arg2
                logical(C_BOOL), intent(INOUT) :: arg3
            end subroutine c_check_bool
        end interface
    contains
        subroutine check_bool(arg1, arg2, arg3)
            use iso_c_binding, only : C_BOOL
            implicit none
            logical, value, intent(IN) :: arg1
            logical(C_BOOL) SH_arg1
            logical, intent(OUT) :: arg2
            logical(C_BOOL) SH_arg2
            logical, intent(INOUT) :: arg3
            logical(C_BOOL) SH_arg3
            SH_arg1 = arg1  ! coerce to C_BOOL
            SH_arg3 = arg3  ! coerce to C_BOOL
            ! splicer begin check_bool
            call c_check_bool(SH_arg1, SH_arg2, SH_arg3)
            ! splicer end check_bool
            arg2 = SH_arg2  ! coerce to logical
            arg3 = SH_arg3  ! coerce to logical
        end subroutine check_bool
    end module userlibrary_mod

Since ``arg1`` in the YAML declaration is not a pointer it defaults to
``intent(IN)``.  The intent of the other two arguments are explicitly
annotated.

If a function returns a ``bool`` result then a wrapper is always needed
to convert the result.  The **result** section sets **need_wrapper**
to force the wrapper to be created.  By default a function with no
argument would not need a wrapper since there will be no **pre_call**
or **post_call** code blocks.  Only the C interface would be required
since Fortran could call the C function directly.


Character
---------

Fortran, C, and C++ each have their own semantics for character variables.

  * Fortran ``character`` variables know their length and are blank filled
  * C ``char *`` variables are assumed to be ``NULL`` terminated.
  * C++ ``std::string`` know their own length and can provide a ``NULL`` terminated pointer.

It is not sufficient to pass an address between Fortran and C++ like
it is with other native types.  In order to get idiomatic behavior in
the Fortran wrappers it is often necessary to copy the values.  This
is to account for blank filled vs ``NULL`` terminated.

..  It also helps support ``const`` vs non-``const`` strings.

Any C++ function which has ``char`` or ``std::string`` arguments or
result will create an additional C function which include additional
arguments for the length of the strings.  Most Fortran compiler use
this convention when passing ``CHARACTER`` arguments. Shroud makes
this convention explicit for three reasons:

* It allows an interface to be used.  Functions with an interface may
  not pass the hidden, non-standard length argument, depending on compiler.
* It may pass the result of ``len`` and/or ``len_trim``.
  The convention just passes the length.
* Returning character argument from C to Fortran is non-portable.

Arguments with the *intent(in)* annotation are given the *len_trim*
annotation.  The assumption is that the trailing blanks are not part
of the data but only padding.  Return values and *intent(out)*
arguments add a *len* annotation with the assumption that the wrapper
will copy the result and blank fill the argument so it need to know
the declared length.

The additional function will be named the same as the original
function with the option **C_bufferify_suffix** appended to the end.
The Fortran wrapper will use the original function name, but call the
C function which accepts the length arguments.

The character type maps use the **c_statements** section to define
code which will be inserted into the C wrapper. *intent_in*,
*intent_out*, and *result* subsections add actions for the C wrapper.
*intent_in_buf*, *intent_out_buf*, and *result_buf* are used for
arguments with the *len* and *len_trim* annotations in the additional
C wrapper.


Char
^^^^

The type map::

        type: char
        fields:
            base: string
            cxx_type: char
            c_type: char
            c_statements:
                intent_in_buf:
                    buf_args:
                    - arg
                    - len_trim
                    cxx_local_var: pointer
                    c_header: <stdlib.h> <string.h>
                    cxx_header: <stdlib.h> <cstring>
                    pre_call:
                    -  char * {cxx_var} = (char *) malloc({c_var_trim} + 1);
                    -  {stdlib}memcpy({cxx_var}, {c_var}, {c_var_trim});
                    -  {cxx_var}[{c_var_trim}] = '\0'
                    post_call=[
                    -  free({cxx_var});
                intent_out_buf:
                    buf_args:
                    - arg
                    - len
                    cxx_local_var: pointer
                    c_header: <stdlib.h> <string.h>
                    cxx_header: <cstdlib> <cstring>
                    c_helper: ShroudStrCopy
                    pre_call:
                    -  char * {cxx_var} = (char *) {stdlib}malloc({c_var_len} + 1);
                    post_call:
                    -  ShroudStrCopy({c_var}, {c_var_len},\t {cxx_var},\t {stdlib}strlen({cxx_var}));
                    -  free({cxx_var});
                intent_inout_buf:
                    buf_args:
                    - arg
                    - len_trim
                    - len
                    cxx_local_var: pointer
                    c_helper: ShroudStrCopy
                    c_header: <stdlib.h> <string.h>
                    cxx_header: <stdlib.h> <cstring>
                    pre_call:
                    -  char * {cxx_var} = (char *) malloc({c_var_len} + 1);
                    -  {stdlib}memcpy({cxx_var}, {c_var}, {c_var_trim});
                    -  {cxx_var}[{c_var_trim}] = '\0';
                    post_call:
                    -  ShroudStrCopy({c_var}, {c_var_len}, \t {cxx_var},\t {stdlib}strlen({cxx_var}));
                    -  free({cxx_var});
                result_buf:
                    buf_args:
                    - arg
                    - len
                    c_header: <string.h>
                    cxx_header: <cstring>
                    c_helper: ShroudStrCopy
                    post_call:
                    - if ({cxx_var} == NULL) {{+
                    - {stdlib}memset({c_var}, ' ', {c_var_len});
                    - -}} else {{+
                    - ShroudStrCopy({c_var}, {c_var_len}, \t {cxx_var},\t {stdlib}strlen({cxx_var}));
                    - -}}

            f_type: character(*)
            f_kind: C_CHAR
            f_c_type: character(kind=C_CHAR)
            f_c_module:
                iso_c_binding:
                  - C_CHAR

            f_statements:
                result_pure:
                    need_wrapper: True
                    f_helper: fstr_ptr
                    call:
                      - {F_result} = fstr_ptr({F_C_call}({F_arg_c_call_tab}))


The function ``passCharPtr(dest, src)`` is equivalent to the Fortran
statement ``dest = src``::

    - decl: void passCharPtr(char *dest+intent(out), const char *src)

.. from tests/strings.cpp

The intent of ``dest`` must be explicit.  It defaults to *intent(inout)*
since it is a pointer.
``src`` is implied to be *intent(in)* since it is ``const``.

This single line will create five different wrappers.  The first is the 
pure C version.  The only feature this provides to Fortran is the ability
to call a C++ function by wrapping it in an ``extern "C"`` function::

    void STR_pass_char_ptr(char * dest, const char * src)
    {
        passCharPtr(dest, src);
        return;
    }

A Fortran interface for the routine is generated which will allow the
function to be called directly::

        subroutine c_pass_char_ptr(dest, src) &
                bind(C, name="STR_pass_char_ptr")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(OUT) :: dest(*)
            character(kind=C_CHAR), intent(IN) :: src(*)
        end subroutine c_pass_char_ptr

The user is responsible for providing the ``NULL`` termination.
The result in ``str`` will also be ``NULL`` terminated instead of 
blank filled.::

    character(30) str
    call c_pass_char_ptr(dest=str, src="mouse" // C_NULL_CHAR)

An additional C function is automatically declared which is summarized as::

    - decl: void passCharPtr(char * dest+intent(out)+len(Ndest),
                             const char * src+intent(in)+len_trim(Lsrc))

And generates::

    void STR_pass_char_ptr_bufferify(char * dest, int Ndest,
                                     const char * src, int Lsrc)
    {
        char * SH_dest = (char *) std::malloc(Ndest + 1);
        char * SH_src = (char *) malloc(Lsrc + 1);
        std::memcpy(SH_src, src, Lsrc);
        SH_src[Lsrc] = '\0';
        passCharPtr(SH_dest, SH_src);
        ShroudStrCopy(dest, Ndest, SH_dest, std::strlen(SH_dest));
        free(SH_dest);
        free(SH_src);
        return;
    }

``Ndest`` is the declared length of argument ``dest`` and ``Lsrc`` is
the trimmed length of argument ``src``.  These generated names must
not conflict with any other arguments.  There are two ways to set the
names.  First by using the options **C_var_len_template** and
**C_var_trim_template**. This can be used to control how the names are
generated for all functions if set globally or just a single function
if set in the function's options.  The other is by explicitly setting
the *len* and *len_trim* annotations which only effect a single
declaration.

The pre_call code creates space for the C strings by allocating
buffers with space for an additional character (the ``NULL``).  The
*intent(in)* string copies the data and adds an explicit terminating
``NULL``.  The function is called then the post_call section copies
the result back into the ``dest`` argument and deletes the scratch
space.  ``ShroudStrCopy`` is a function provided by Shroud which
copies character into the destination up to ``Ndest`` characters, then
blank fills any remaining space.

The Fortran interface is generated::

        subroutine c_pass_char_ptr_bufferify(dest, Ndest, src, Lsrc) &
                bind(C, name="STR_pass_char_ptr_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(OUT) :: dest(*)
            integer(C_INT), value, intent(IN) :: Ndest
            character(kind=C_CHAR), intent(IN) :: src(*)
            integer(C_INT), value, intent(IN) :: Lsrc
        end subroutine c_pass_char_ptr_bufferify

And finally, the Fortran wrapper with calls to ``len`` and ``len_trim``::

    subroutine pass_char_ptr(dest, src)
        use iso_c_binding, only : C_INT
        character(*), intent(OUT) :: dest
        character(*), intent(IN) :: src
        call c_pass_char_ptr_bufferify(dest, len(dest, kind=C_INT), src,  &
            len_trim(src, kind=C_INT))
    end subroutine pass_char_ptr

Now the function can be called without the user aware that it is written in C++::

    character(30) str
    call pass_char_ptr(dest=str, src="mouse")


std::string
^^^^^^^^^^^

The ``std::string`` type map is very similar to ``char`` but provides some
additional sections to convert between ``char *`` and ``std::string``::

        type: string
        fields:
            base: string
            cxx_type: std::string
            cxx_header: <string>
            cxx_to_c: {cxx_var}{cxx_member}c_str()
            c_type: char
    
            c_statements:
                intent_in:
                    cxx_local_var: object
                    pre_call:
                      - {c_const}std::string {cxx_var}({c_var});
                intent_out:
                    cxx_header: <cstring>
                    post_call:
                      - strcpy({c_var}, {cxx_val});
                intent_inout:
                    cxx_header: <cstring>
                    pre_call:
                      - {c_const}std::string {cxx_var}({c_var});
                    post_call:
                      - strcpy({c_var}, {cxx_val});

                intent_in_buf: dict(
                    buf_args:
                    - arg
                    - len_trim
                    cxx_local_var: scalar
                    pre_call:
                    -  {c_const}std::string {cxx_var}({c_var}, {c_var_trim});
                intent_out_buf:
                    buf_args:
                    - arg
                    - len
                    c_helper: ShroudStrCopy
                    cxx_local_var: scalar
                    pre_call:
                    -   std::string {cxx_var};
                    post_call:
                    -  ShroudStrCopy({c_var}, {c_var_len},\t {cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size());
                intent_inout_buf:
                    buf_args:
                    - arg
                    - len_trim
                    - len
                    c_helper: ShroudStrCopy
                    cxx_local_var: scalar
                    pre_call:
                    -  std::string {cxx_var}({c_var}, {c_var_trim});
                    post_call:
                    -  ShroudStrCopy({c_var}, {c_var_len},\t {cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size());
                result_buf:
                    buf_args:
                    - arg
                    - len
                    cxx_header: <cstring>
                    c_helper: ShroudStrCopy
                    post_call:
                    -  if ({cxx_var}{cxx_member}empty()) {{+
                    -  {stdlib}memset({c_var}, ' ', {c_var_len});
                    -  -}} else {{+
                    -  ShroudStrCopy({c_var}, {c_var_len},\t {cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size());
                    -  -}}
    
            f_type: character(*)
            f_kind: C_CHAR
            f_c_type: character(kind=C_CHAR)
            f_c_module:
                iso_c_binding:
                  - C_CHAR

            f_statements:
                result_pure:
                    need_wrapper: True
                    f_helper: fstr_ptr
                    call:
                      - {F_result} = fstr_ptr({F_C_call}({F_arg_c_call_tab}))


To demonstrate this type map, ``acceptStringReference`` is a function which
will accept and modify a string reference::

    - decl: void acceptStringReference(std::string & arg1)

A reference defaults to *intent(inout)* and will add both the *len*
and *len_trim* annotations.

Both generated functions will convert ``arg`` into a ``std::string``,
call the function, then copy the results back into the argument. The
important thing to notice is that the pure C version could do very bad
things since it does not know how much space it has to copy into.  The
bufferify version knows the allocated length of the argument.
However, since the input argument is a fixed length it may be too
short for the new string value::

    void STR_accept_string_reference(char * arg1)
    {
        std::string SH_arg1(arg1);
        acceptStringReference(SH_arg1);
        strcpy(arg1, SH_arg1.c_str());
        return;
    }

    void STR_accept_string_reference_bufferify(char * arg1,
                                               int Larg1, int Narg1)
    {
        std::string SH_arg1(arg1, Larg1);
        acceptStringReference(SH_arg1);
        ShroudStrCopy(arg1, Narg1, SH_arg1.data(), SH_arg1.size());
        return;
    }

Each interface matches the C wrapper::

        subroutine c_accept_string_reference(arg1) &
                bind(C, name="STR_accept_string_reference")
            use iso_c_binding, only : C_CHAR
            implicit none
            character(kind=C_CHAR), intent(INOUT) :: arg1(*)
        end subroutine c_accept_string_reference

        subroutine c_accept_string_reference_bufferify(arg1, Larg1, Narg1) &
                bind(C, name="STR_accept_string_reference_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT
            implicit none
            character(kind=C_CHAR), intent(INOUT) :: arg1(*)
            integer(C_INT), value, intent(IN) :: Larg1
            integer(C_INT), value, intent(IN) :: Narg1
        end subroutine c_accept_string_reference_bufferify

And the Fortran wrapper provides the correct values for the *len* and
*len_trim* arguments::

    subroutine accept_string_reference(arg1)
        use iso_c_binding, only : C_INT
        character(*), intent(INOUT) :: arg1
        ! splicer begin accept_string_reference
        call c_accept_string_reference_bufferify(arg1,  &
            len_trim(arg1, kind=C_INT), len(arg1, kind=C_INT))
        ! splicer end accept_string_reference
    end subroutine accept_string_reference

char functions
^^^^^^^^^^^^^^

Functions which return a ``char *`` provide an additional challenge.
Taken literally they should return a ``type(C_PTR)``.  And if you call
the function via the interface, that's what you get.  However,
Shroud provides several options to provide a more idiomatic usage.

Each of these declaration call identical C++ functions but they are
wrapped differently::

    - decl: const char * getCharPtr1()
    - decl: const char * getCharPtr2() +len(30)
    - decl: const char * getCharPtr3()
      format:
         F_string_result_as_arg: output

All of the generated C wrappers are very similar.
The first C wrapper will copy the metadata into a ``SHROUD_array`` struct::

    const char * STR_get_char_ptr1()
    {
        const char * SHC_rv = getChar1();
        return SHC_rv;
    }

    void STR_get_char_ptr1_bufferify(STR_SHROUD_array *DSHF_rv)
    {
        const char * SHC_rv = getCharPtr1();
        DSHF_rv->cxx.addr = static_cast<void *>(const_cast<char *>(SHC_rv));
        DSHF_rv->cxx.idtor = 0;
        DSHF_rv->addr.ccharp = SHC_rv;
        DSHF_rv->len = SHC_rv == NULL ? 0 : strlen(SHC_rv);
        DSHF_rv->size = 1;
        return;
    }

The Fortran wrapper uses the metadata in ``DSHF_rv`` to allocate
a ``CHARACTER`` variable of the correct length.
The helper function ``SHROUD_copy_string_and_free`` will copy 
the results of the C++ function into the return variable::

    function get_char_ptr1() &
            result(SHT_rv)
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin function.get_char_ptr1
        call c_get_char_ptr1_bufferify(DSHF_rv)
        ! splicer end function.get_char_ptr1
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function get_char_ptr1

If you know the maximum size of string that you expect the function to
return, then the *len* attribute is used to declare the length.  The
explicit ``ALLOCATE`` is avoided but any result which is longer than
the length will be silently truncated::

    function get_char_ptr2() &
            result(SHT_rv)
        use iso_c_binding, only : C_CHAR, C_INT
        character(kind=C_CHAR, len=30) :: SHT_rv
        call c_get_char_ptr2_bufferify(SHT_rv, len(SHT_rv, kind=C_INT))
    end function get_char_ptr2

The third option also avoids the ``ALLOCATE`` but allows any length
result to be returned.  The result of the C function will be returned
in the Fortran argument named by format string
**F_string_result_as_arg**.  The potential downside is that a Fortran
subroutine is generated instead of a function::

    subroutine get_char_ptr3(output)
        use iso_c_binding, only : C_INT
        character(len=*), intent(OUT) :: output
        call c_get_char_ptr3_bufferify(output, len(output, kind=C_INT))
    end subroutine get_char_ptr3

.. char ** not supported

string functions
^^^^^^^^^^^^^^^^

Functions which return ``std::string`` values are similar but must provide the
extra step of converting the result into a ``char *``::

    - decl: const string& getConstStringRefPure()

The generated wrappers are::

    const char * STR_get_const_string_ref_pure()
    {
        const std::string & SHCXX_rv = getConstStringRefPure();
        const char * SHC_rv = SHCXX_rv.c_str();
        return SHC_rv;
    }
    
    void STR_get_const_string_ref_pure_bufferify(STR_SHROUD_array *DSHF_rv)
    {
        const std::string & SHCXX_rv = getConstStringRefPure();
        DSHF_rv->cxx.addr = static_cast<void *>(const_cast<std::string *>
            (&SHCXX_rv));
        DSHF_rv->cxx.idtor = 0;
        DSHF_rv->addr.ccharp = SHCXX_rv.data();
        DSHF_rv->len = SHCXX_rv.size();
        DSHF_rv->size = 1;
        return;
    }


std::vector
-----------

A ``std::vector`` argument for a C++ function can be created from a Fortran array.
The address and size of the array is extracted and passed to the C wrapper to create
the ``std::vector``::

    int vector_sum(const std::vector<int> &arg);
    void vector_iota(std::vector<int> &arg);

Are wrapped with the YAML input::

    - decl: int vector_sum(const std::vector<int> &arg)
    - decl: void vector_iota(std::vector<int> &arg+intent(out))

``intent(in)`` is implied for the *vector_sum* argument since it is ``const``.
The Fortran wrapper passes the array and the size to C::

    function vector_sum(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_LONG
        integer(C_INT), intent(IN) :: arg(:)
        integer(C_INT) :: SHT_rv
        SHT_rv = c_vector_sum_bufferify(arg, size(arg, kind=C_LONG))
    end function vector_sum

    subroutine vector_iota(arg)
        use iso_c_binding, only : C_INT, C_LONG
        integer(C_INT), intent(OUT) :: arg(:)
        call c_vector_iota_bufferify(arg, size(arg, kind=C_LONG))
    end subroutine vector_iota

The C wrapper then creates a ``std::vector``::

    int TUT_vector_sum_bufferify(const int * arg, long Sarg)
    {
        const std::vector<int> SH_arg(arg, arg + Sarg);
        int SHC_rv = tutorial::vector_sum(SH_arg);
        return SHC_rv;
    }
    
    void TUT_vector_iota_bufferify(int * arg, long Sarg)
    {
        std::vector<int> SH_arg(Sarg);
        tutorial::vector_iota(SH_arg);
        {
            std::vector<int>::size_type
                SHT_i = 0,
                SHT_n = Sarg;
            SHT_n = std::min(SH_arg.size(), SHT_n);
            for(; SHT_i < SHT_n; SHT_i++) {
                arg[SHT_i] = SH_arg[SHT_i];
            }
        }
        return;
    }

On ``intent(in)``, the ``std::vector`` constructor copies the values
from the input pointer.  With ``intent(out)``, the values are copied
after calling the function.

.. note:: With ``intent(out)``, if *vector_iota* changes the size of ``arg`` to be longer than
          the original size of the Fortran argument, the additional values will not be copied. 

MPI_Comm
--------

MPI_Comm is provided by Shroud and serves as an example of how to wrap
a non-native type.  MPI provides a Fortran interface and the ability
to convert MPI_comm between Fortran and C. The type map tells Shroud
how to use these routines::

        type: MPI_Comm
        fields:
            cxx_type: MPI_Comm
            c_header: mpi.h
            c_type: MPI_Fint
            f_type: integer
            f_kind: C_INT
            f_c_type: integer(C_INT)
            f_c_module:
                iso_c_binding:
                  - C_INT
            cxx_to_c: MPI_Comm_c2f({cxx_var})
            c_to_cxx: MPI_Comm_f2c({c_var})


This mapping makes the assumption that ``integer`` and
``integer(C_INT)`` are the same type.


.. Complex Type
   ------------


.. Derived Types
   -------------

.. _TypesAnchor_Function_Pointers:

Function Pointers
-----------------

C or C++ arguments which are pointers to functions are supported.
The function pointer type is wrapped using a Fortran ``abstract interface``.
Only C compatible arguments in the function pointer are supported since
no wrapper for the function pointer is created.  It must be callable 
directly from Fortran.

The function is wrapped as usual::

    declarations:
    - decl: int callback1(int in, int (*incr)(int));

The main addition is the creation of an abstract interface in Fortran::

    abstract interface
        function callback1_incr(arg0) bind(C)
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value :: arg0
            integer(C_INT) :: callback1_incr
        end function callback1_incr
    end interface

    interface
        function callback1(in, incr) &
                result(SHT_rv) &
                bind(C, name="TUT_callback1")
            use iso_c_binding, only : C_INT
            import :: callback1_incr
            implicit none
            integer(C_INT), value, intent(IN) :: in
            procedure(callback1_incr) :: incr
            integer(C_INT) :: SHT_rv
        end function callback1
    end interface

The ``abstract interface`` is named from option
**F_abstract_interface_subprogram_template** which defaults to
``{underscore_name}_{argname}`` where *argname* is the name of the
function argument.

If the function pointer uses an abstract declarator
(no argument name), the argument name is created from option
**F_abstract_interface_argument_template** which defaults to
``arg{index}`` where *index* is the 0-based argument index.
When a name is given to a function pointer argument,
it is always used in the ``abstract interface``.

To change the name of the subprogram or argument, change the option.
There are no format fields **F_abstract_interface_subprogram** or
**F_abstract_interface_argument** since they vary by argument (or
argument to an argument)::

    options:
      F_abstract_interface_subprogram_template: custom_funptr
      F_abstract_interface_argument_template: XX{index}arg


Class Type
----------

Each class in the input file will create a C struct to save
information about the C++ class.
.. XXX

 
Each class in the input file will create a Fortran derived type which
acts as a shadow class for the C++ class.  A pointer to an instance is
saved as a ``type(C_PTR)`` value.  The *f_to_c* field uses the
generated ``get_instance`` function to return the pointer which will
be passed to C.

In C an opaque typedef for a struct is created as the type for the C++
instance pointer.  The *c_to_cxx* and *cxx_to_c* fields casts this
pointer to C++ and back to C.

The class example from the tutorial is::

    declarations:
    - decl: class Class1

Shroud will generate a type map for this class as::

    type: Class1
    fields:
        base: shadow
        c_type: TUT_class1
        cxx_type: Class1
        c_to_cxx: \tstatic_cast<{c_const}Class1 *>(\tstatic_cast<{c_const}void *>(\t{c_var}))
        cxx_to_c: \tstatic_cast<{c_const}TUT_class1 *>(\tstatic_cast<{c_const}void *>(\t{cxx_var}))

        f_type: type(class1)
        f_derived_type: class1
        f_c_type: type(C_PTR)
        f_c_module:
            iso_c_binding:
              - C_PTR
        f_module:
            tutorial_mod:
              - class1
        f_return_code: {F_result}%{F_derived_member} = {F_C_call}({F_arg_c_call_tab})
        f_to_c: {f_var}%get_instance()
        forward: Class1

Methods are added to a class with a ``declarations`` field::

    declarations:
    - decl: class Class1
      declarations:
      - decl: void func()

corresponds to the C++ code::

    class Class1
    {
       void func();
    }

A class will be forward declared when the ``declarations`` field is
not provided.  When the class is not defined later in the file, it may
be necessary to provide the conversion fields to complete the type::

    declarations:
    - decl: class Class1
      fields:
        c_type: TUT_class1
        f_derived_type: class1
        f_to_c: "{f_var}%get_instance()"
        f_module:
          tutorial_mod:
          - class1


The type map will be written to a file to allow its used by other
wrapped libraries.  The file is named by the global field
**YAML_type_filename**. This file will only list some of the fields
show above with the remainder set to default values by Shroud.

The default name of the constructor is ``ctor``.  The name can 
be specified with the **name** attribute.
If the constructor is overloaded, each constructor must be given the
same **name** attribute.
The *function_suffix* must not be explicitly set to blank since the name
is used by the ``generic`` interface.

The constructor and destructor will only be wrapped if explicitly added
to the YAML file to avoid wrapping ``private`` constructors and destructors.

..  chained function calls

Member Variables
^^^^^^^^^^^^^^^^

For each member variable of a C++ class a C and Fortran wrapper
function will be created to get or set the value.  The Python wrapper
will create a descriptor::

    class Class1
    {
    public:
       int m_flag;
       int m_test;
    }

It is added to the YAML file as::

    - decl: class Class1
      declarations:
      - decl: int m_flag +readonly;
      - decl: int m_test +name(test);

The *readonly* attribute will not write the setter function or descriptor.
Python will report::

    >>> obj = tutorial.Class1()
    >>> obj.m_flag =1
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    AttributeError: attribute 'm_flag' of 'tutorial.Class1' objects is not writable

The *name* attribute will change the name of generated functions and
descriptors.  This is helpful when using a naming convention like
``m_test`` and you do not want ``m_`` to be used in the wrappers.

.. _MemoryManagementAnchor:

Memory Management
=================

Shroud will maintain ownership of memory via the **owner** attribute.
It uses the value of the attribute to decided when to release memory.

Use **owner(library)** when the library owns the memory and the user
should not release it.  For example, this is used when a function
returns ``const std::string &`` for a reference to a string which is
maintained by the library.  Fortran and Python will both get the
reference, copy the contents into their own variable (Fortran
``CHARACTER`` or Python ``str``), then return without releasing any
memory.  This is the default behavior.

Use **owner(caller)** when the library allocates new memory which is
returned to the caller.  The caller is then responsible to release the
memory.  Fortran and Python can both hold on the to memory and then
provide ways to release it using a C++ callback when it is no longer
needed.

For shadow classes with a destructor defined, the destructor will 
be used to release the memory.

The *c_statements* may also define a way to destroy memory.
For example, ``std::vector`` provides the lines::

    destructor_name: std_vector_{cxx_T}
    destructor:
    -  std::vector<{cxx_T}> *cxx_ptr = reinterpret_cast<std::vector<{cxx_T}> *>(ptr);
    -  delete cxx_ptr;

Patterns can be used to provide code to free memory for a wrapped
function.  The address of the memory to free will be in the variable
``void *ptr``, which should be referenced in the pattern::

    declarations:
    - decl: char * getName() +free_pattern(free_getName)

    patterns:
       free_getName: |
          decref(ptr);

Without any explicit *destructor_name* or pattern, ``free`` will be
used to release POD pointers; otherwise, ``delete`` will be used.

.. When to use ``delete[] ptr``?

C and Fortran
-------------

Fortran keeps track of C++ objects with the struct
**C_capsule_data_type** and the ``bind(C)`` equivalent
**F_capsule_data_type**. Their names default to
``{C_prefix}SHROUD_capsule_data`` and ``SHROUD_capsule_data``. In the
Tutorial these types are defined in C as::

    struct s_TUT_class1 {
        void *addr;     /* address of C++ memory */
        int idtor;      /* index of destructor */
    };
    typedef struct s_TUT_class1 TUT_class1;

And Fortran::

    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

*addr* is the address of the C or C++ variable, such as a ``char *``
or ``std::string *``.  *idtor* is a Shroud generated index of the
destructor code defined by *destructor_name* or the *free_pattern* attribute.
These code segments are collected and written to function
*C_memory_dtor_function*.  A value of 0 indicated the memory will not
be released and is used with the **owner(library)** attribute. A
typical function would look like::

    // Release C++ allocated memory.
    void TUT_SHROUD_memory_destructor(TUT_SHROUD_capsule_data *cap)
    {
        void *ptr = cap->addr;
        switch (cap->idtor) {
        case 0:   // --none--
        {
            // Nothing to delete
            break;
        }
        case 1:   // tutorial::Class1
        {
            tutorial::Class1 *cxx_ptr = reinterpret_cast<tutorial::Class1 *>(ptr);
            delete cxx_ptr;
            break;
        }
        case 2:   // std::string
        {
            std::string *cxx_ptr = reinterpret_cast<std::string *>(ptr);
            delete cxx_ptr;
            break;
        }
        default:
        {
            // Unexpected case in destructor
            break;
        }
        }
        cap->addr = NULL;
        cap->idtor = 0;  // avoid deleting again
    }


Character and Arrays
^^^^^^^^^^^^^^^^^^^^

In order to create an allocatable copy of a C++ pointer, an additional structure
is involved.  For example, ``Function4d`` returns a pointer to a new string::

    declarations:
    - decl: const std::string * Function4d()

The C wrapper calls the function and saves the result along with
metadata consisting of the address of the data within the
``std::string`` and its length.  The Fortran wrappers allocates its
return value to the proper length, then copies the data from the C++
variable and deletes it.

The metadata for variables are saved in the C struct **C_array_type**
and the ``bind(C)`` equivalent **F_array_type**.::

    struct s_TUT_SHROUD_array {
        TUT_SHROUD_capsule_data cxx;      /* address of C++ memory */
        union {
            const void * cvoidp;
            const char * ccharp;
        } addr;
        size_t len;     /* bytes-per-item or character len of data in cxx */
        size_t size;    /* size of data in cxx */
    };
    typedef struct s_TUT_SHROUD_array TUT_SHROUD_array;

The union for ``addr`` makes some assignments easier and also aids debugging.
The union is replaced with a single ``type(C_PTR)`` for Fortran::

    type, bind(C) :: SHROUD_array
        type(SHROUD_capsule_data) :: cxx       ! address of C++ memory
        type(C_PTR) :: addr = C_NULL_PTR       ! address of data in cxx
        integer(C_SIZE_T) :: len = 0_C_SIZE_T  ! bytes-per-item or character len of data in cxx
        integer(C_SIZE_T) :: size = 0_C_SIZE_T ! size of data in cxx
    end type SHROUD_array

The C wrapper does not return a ``std::string`` pointer.  
Instead it passes in a **C_array_type** pointer as an argument.
It calls ``Function4d``, saves the results and metadata into the argument.
This allows it to be easily accessed from Fortran::

    void TUT_function4d_bufferify(TUT_SHROUD_array *DSHF_rv)
    {
        const std::string * SHCXX_rv = tutorial::Function4d();
        DSHF_rv->cxx.addr = static_cast<void *>(const_cast<std::string *>(SHCXX_rv));
        DSHF_rv->cxx.idtor = 2;
        DSHF_rv->addr.ccharp = SHCXX_rv->data();
        DSHF_rv->len = SHCXX_rv->size();
        DSHF_rv->size = 1;
        return;
    }

The Fortran wrapper uses the metadata to allocate the return argument
to the correct length::

    function function4d() &
            result(SHT_rv)
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        call c_function4d_bufferify(DSHF_rv)
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function function4d

Finally, the helper function ``SHROUD_copy_string_and_free`` is called
to set the value of the result and possible free memory for
**owner(caller)** or intermediate values::

    // Copy the std::string in context into c_var.
    // Called by Fortran to deal with allocatable character.
    void TUT_ShroudCopyStringAndFree(TUT_SHROUD_array *data, char *c_var, size_t c_var_len) {
        const char *cxx_var = data->addr.ccharp;
        size_t n = c_var_len;
        if (data->len < n) n = data->len;
        strncpy(c_var, cxx_var, n);
        TUT_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
    }

.. note:: The three steps of call, allocate, copy could be replaced
          with a single call by using the *futher interoperability
          with C* features of Fortran 2018 (a.k.a TS 29113).  This
          feature allows Fortran ``ALLOCATABLE`` variables to be
          allocated by C. However, not all compilers currently support
          that feature.  The current Shroud implementation works with
          Fortran 2003.


Python
------

NumPy arrays control garbage collection of C++ memory by creating 
a ``PyCapsule`` as the base object of NumPy objects.
Once the final reference to the NumPy array is removed, the reference
count on the ``PyCapsule`` is decremented.
When 0, the *destructor* for the capsule is called and releases the C++ memory.
This technique is discussed at [blog1]_ and [blog2]_


Old
---


Shroud generated C wrappers do not explicitly delete any memory.
However a destructor may be automatically called for some C++ stl
classes.  For example, a function which returns a ``std::string``
will have its value copied into Fortran memory since the function's
returned object will be destroyed when the C++ wrapper returns.  If a
function returns a ``char *`` value, it will also be copied into Fortran
memory. But if the caller of the C++ function wants to transfer
ownership of the pointer to its caller, the C++ wrapper will leak the
memory.

The **C_finalize** variable may be used to insert code before
returning from the wrapper.  Use **C_finalize_buf** for the buffer
version of wrapped functions.

For example, a function which returns a new string will have to 
``delete`` it before the C wrapper returns::

    std::string * getConstStringPtrLen()
    {
        std::string * rv = new std::string("getConstStringPtrLen");
        return rv;
    }

Wrapped as::

    - decl: const string * getConstStringPtrLen+len=30()
      format:
        C_finalize_buf: delete {cxx_var};

The C buffer version of the wrapper is::

    void STR_get_const_string_ptr_len_bufferify(char * SHF_rv, int NSHF_rv)
    {
        const std::string * SHCXX_rv = getConstStringPtrLen();
        if (SHCXX_rv->empty()) {
            std::memset(SHF_rv, ' ', NSHF_rv);
        } else {
            ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv->c_str());
        }
        {
            // C_finalize
            delete SHCXX_rv;
        }
        return;
    }

The unbuffer version of the function cannot ``destroy`` the string since
only a pointer to the contents of the string is returned.  It would
leak memory when called::

    const char * STR_get_const_string_ptr_len()
    {
        const std::string * SHCXX_rv = getConstStringPtrLen();
        const char * SHC_rv = SHCXX_rv->c_str();
        return SHC_rv;
    }

.. note:: Reference counting and garbage collection are still a work in progress




.. rubric:: Footnotes

.. [blog1] `<http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory>`_

.. [blog2] `<http://blog.enthought.com/python/numpy/simplified-creation-of-numpy-arrays-from-pre-allocated-memory>`_
