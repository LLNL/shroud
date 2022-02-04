# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Create and manage typemaps used to convert between languages.
"""
from __future__ import print_function

from . import util

# translation table to convert type name to flat name
# unsigned int -> unsigned_int
# vector<int>  -> vector_int
try:
    # Python 2
    from string import maketrans
    def flatten_name(name, flat_trantab = maketrans("< ", "__")):
        return name.replace("::","_").translate(flat_trantab, ">")
except:
    def flatten_name(name, flat_trantab="".maketrans("< ", "__", ">")):
        return name.replace("::","_").translate(flat_trantab)

def flatten_modules_to_line(modules):
    """Flatten modules dictionary into a line.

    This line can then be used in fc_statements to add module info.
    The flattend line looks like:
        module ":" symbol [ "," symbol ]*
        [ ";" module ":" symbol [ "," symbol ]* ]

    Parameters
    ----------
    modules : dictionary of dictionaries:
        modules['iso_c_bindings'] = ['C_INT', ...]
    """
    if modules is None:
        return None
    line = []
    for mname, symbols in modules.items():
        if mname == "__line__":
            continue
        symbolslst = ",".join(symbols)
        line.append("{}:{}".format(mname, symbolslst))
    return ";".join(line)

class Typemap(object):
    """Collect fields for an argument.
    This used to be a dict but a class has better access semantics:
       i.attr vs d['attr']
    It also initializes default values to avoid  d.get('attr', default)

    c_header and cxx_header are used for interface and are written into 
    a header file for use by C or C++.
    For example, size_t uses <stddef.h> and <cstddef>.

    wrap_header is used for generated wrappers for shadow classes.
    Contains struct definitions for capsules from Fortran.

    impl_header is used for implementation, i.e. the wrap.cpp file.
    For example, std::string uses <string>. <string> should not be in
    the interface since the wrapper is a C API.

    A new typemap is created for each class and struct

    A new typemap is created for each templated class/struct
    instantiation:
        - decl: template<typename T> class A
          cxx_template:
          - instantiation: <int>
    """

    # Array of known keys with default values
    _order = (
        ("flat_name", None),  # Name when used by wrapper identifiers
        ("template_suffix", None),  # Name when used by wrapper identifiers
                                    # when added to class/struct format.
                    # Set from format.template_suffix in YAML for class.
        ("base", "unknown"),  # Base type: 'string'
        ("forward", None),  # Forward declaration
        ("typedef", None),  # Initialize from existing type
        ("cpp_if", None),  # C preprocessor test for c_header
        ("idtor", "0"),  # index of capsule_data destructor
        ("cxx_type", None),  # Name of type in C++, including namespace
        ("cxx_to_c", None),  # Expression to convert from C++ to C
        # None implies {cxx_var} i.e. no conversion
        (
            "cxx_header",
            [],
        ),  # Name of C++ header file required for implementation
        ("cxx_instantiation", None), # Dict of instantiated template types.
            # None = non-templated type.
            # index example ["<int>"] = Typemap for instantiated class.
        # For example, if cxx_to_c was a function
        ("c_type", None),  # Name of type in C
        ("c_header", []),  # Name of C header file required for type
        ("c_to_cxx", None),  # Expression to convert from C to C++
        # None implies {c_var}  i.e. no conversion
        ("c_return_code", None),
        ("f_c_module", None), # Fortran modules needed for interface  (dictionary)
        ("f_c_module_line", None),
        ("f_class", None),  # Used with type-bound procedures
        ("f_type", None),  # Name of type in Fortran -- integer(C_INT)
        ("f_kind", None),  # Fortran kind            -- C_INT
        ("f_c_type", None),  # Type for C interface    -- int
        ("f_to_c", None),  # Expression to convert from Fortran to C
        (
            "f_module_name",
            None,
        ),  # Name of module which contains f_derived_type and f_capsule_data_type
        ("f_derived_type", None),  # Fortran derived type name
        ("f_capsule_data_type", None),  # Fortran derived type to match C struct
        ("f_module", None),  # Fortran modules needed for type  (dictionary)
        ("f_module_line", None),
        ("f_cast", "{f_var}"),  # Expression to convert to type
                                # e.g. intrinsics such as INT and REAL.
        ("impl_header", []), # implementation header
        ("wrap_header", []), # generated wrapper header
        # Python
        ("PY_format", "O"),  # 'format unit' for PyArg_Parse
        ("PY_PyTypeObject", None),  # variable name of PyTypeObject instance
        ("PY_PyObject", None),  # typedef name of PyObject instance
        ("PY_ctor", None),  # expression to create object.
        # ex. PyFloat_FromDouble({c_deref}{c_var})
        ("PY_get", None),  # expression to create type from PyObject.
        # ex. PyFloat_AsDouble({py_var})
        ("py_ctype", None),        # returned by Py_get ex. "Py_complex"
        ("pytype_to_pyctor", None),  # Used with py_ctype, passed to PY_ctor
        ("pytype_to_cxx", None),  # Used with py_ctype
        ("cxx_to_pytype", None),  # Used with py_ctype
        # Name of converter function with prototype (PyObject *, void *).
        ("PY_to_object", None),  # PyBuild - object=converter(address)
        (
            "PY_from_object",
            None,
        ),  # PyArg_Parse - status=converter(object, address);
        ("PY_to_object_idtor", None),  # object=converter(address, idtor)
        ("PY_build_arg", None),  # argument for Py_BuildValue
        ("PY_build_format", None),  # 'format unit' for Py_BuildValue
        ("PY_struct_as", None),  # For struct - "class" or "list"
        ("PYN_typenum", None),  # NumPy typenum enumeration
        (
            "PYN_descr",
            None,
        ),  # Name of PyArray_Descr variable to describe type (for structs)
        # Lua
        ("LUA_type", "LUA_TNONE"),
        ("LUA_pop", "POP"),
        ("LUA_push", "PUSH"),
        ("LUA_statements", {}),
        ("sgroup", "unknown"),  # statement group. ex. native, string, vector
        ("sh_type", "SH_TYPE_OTHER"),
        ("cfi_type", "CFI_type_other"),
        ("export", False),      # If True, export to YAML file.
        ("__line__", None),
    )

    _keyorder, _valueorder = zip(*_order)

    # valid fields
    defaults = dict(_order)

    def __init__(self, name, **kw):
        """
        Args:
            name - name of typemap.
        """
        self.name = name
        #        for key, defvalue in self.defaults.items():
        #            setattr(self, key, defvalue)
        self.__dict__.update(self.defaults)  # set all default values
        self.update(kw)
        if self.cxx_type and not self.flat_name:
            # Do not override an explicitly set value.
            self.compute_flat_name()

    def update(self, dct):
        """Add options from dictionary to self.

        Args:
            dct - dictionary-like object.
        """
        for key, value in dct.items():
            if key in ["c_header", "cxx_header", "impl_header", "wrap_header"]:
                # Blank delimited strings to list
                if isinstance(value,list):
                    setattr(self, key, value)
                else:
                    setattr(self, key, value.split())
            elif key in self.defaults:
                setattr(self, key, value)
            else:
                raise RuntimeError("Unknown key for Typemap %s", key)

    def finalize(self):
        """Compute some fields based on other fields."""
        self.f_c_module_line = flatten_modules_to_line(self.f_c_module or self.f_module)
        self.f_module_line = flatten_modules_to_line(self.f_module)

    def XXXcopy(self):
        ntypemap = Typemap(self.name)
        ntypemap.update(self._to_dict())
        return ntypemap

    def clone_as(self, name):
        """
        Args:
            name - name of new instance.
        """
        ntypemap = Typemap(name, **self._to_dict())
        return ntypemap

    def compute_flat_name(self):
        """Compute the flat_name.
        Can be called after cxx_type is set
        such as after clone_as.

        cxx_type will not be set for template arguments.
        Only set if None. Complex is set explicitly since
        C and C++ have totally different names  (double complex vs complex<double>)
        """
        if self.flat_name is None:
            self.flat_name = flatten_name(self.cxx_type)

    def _to_dict(self):
        """Convert instance to a dictionary for json.
        """
        # only export non-default values
        dct = {}
        for key, defvalue in self.defaults.items():
            value = getattr(self, key)
            if value is not defvalue:
                dct[key] = value
        return dct

    def __repr__(self):
        # only print non-default values
        args = []
        for key, defvalue in self.defaults.items():
            value = getattr(self, key)
            if value is not defvalue:
                if isinstance(value, str):
                    args.append("{0}='{1}'".format(key, value))
                else:
                    args.append("{0}={1}".format(key, value))
        return "Typemap('%s', " % self.name + ",".join(args) + ")"

    def __as_yaml__(self, indent, output):
        """Write out entire typemap as YAML.

        Args:
            indent -
            output -
        """
        util.as_yaml(self, self._keyorder, indent, output)

    def __export_yaml__(self, output, mode="all"):
        """Write out a subset of a wrapped type.
        Other fields are set with fill_shadow_typemap_defaults.

        Args:
            output -
        """
        # Temporary dictionary to allow convert on header fields.
        if mode == "all":
            order = self._keyorder
        else: # class
            # To be used by other libraries which import shadow types.
            if self.base == "shadow":
                order = [
                    "base",
                    "wrap_header",
                ]
            else:
                order = [
                    "base",
                    "cxx_header",
                    "c_header",
                    "f_kind",
                ]
            order.extend([
#                "cxx_type",  # same as the dict key
                "c_type",
                "f_module_name",
                "f_derived_type",
                "f_capsule_data_type",
                "f_to_c",
            ])
                
        util.as_yaml(self, order, output)


# Manage collection of typemaps
shared_typemaps = {}  # dictionary of registered types


def set_global_typemaps(typemaps):
    global shared_typemaps
    shared_typemaps = typemaps


def get_global_typemaps():
    return shared_typemaps


def register_typemap(name, ntypemap):
    """Register a typemap"""
    shared_typemaps[name] = ntypemap


def lookup_typemap(name):
    """Lookup name in registered types."""
    ntypemap = shared_typemaps.get(name)
    return ntypemap


def initialize():
    set_global_typemaps({})
    def_types = dict(
        void=Typemap(
            "void",
            c_type="void",
            cxx_type="void",
            # fortran='subroutine',
            f_type="type(C_PTR)",
            f_c_module=dict(iso_c_binding=["C_PTR"]),
            PY_ctor="PyCapsule_New({ctor_expr}, NULL, NULL)",
            sh_type="SH_TYPE_CPTR",
            cfi_type="CFI_type_intptr_t",
            base="void",
            sgroup="void",
        ),
        short=Typemap(
            "short",
            c_type="short",
            cxx_type="short",
            f_cast="int({f_var}, C_SHORT)",
            f_type="integer(C_SHORT)",
            f_kind="C_SHORT",
            f_module=dict(iso_c_binding=["C_SHORT"]),
            PY_format="h",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_SHORT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_SHORT",
            cfi_type="CFI_type_short",
        ),
        int=Typemap(
            "int",
            c_type="int",
            cxx_type="int",
            f_cast="int({f_var}, C_INT)",
            f_type="integer(C_INT)",
            f_kind="C_INT",
            f_module=dict(iso_c_binding=["C_INT"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_INT",
            cfi_type="CFI_type_int",
        ),
        long=Typemap(
            "long",
            c_type="long",
            cxx_type="long",
            f_cast="int({f_var}, C_LONG)",
            f_type="integer(C_LONG)",
            f_kind="C_LONG",
            f_module=dict(iso_c_binding=["C_LONG"]),
            PY_format="l",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_LONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_LONG",
            cfi_type="CFI_type_long",
        ),
        long_long=Typemap(
            "long_long",
            c_type="long long",
            cxx_type="long long",
            f_cast="int({f_var}, C_LONG_LONG)",
            f_type="integer(C_LONG_LONG)",
            f_kind="C_LONG_LONG",
            f_module=dict(iso_c_binding=["C_LONG_LONG"]),
            PY_format="L",
            # #- PY_ctor='PyInt_FromLong({ctor_expr})',
            PYN_typenum="NPY_LONGLONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_LONG_LONG",
            cfi_type="CFI_type_long_long",
        ),
        unsigned_short=Typemap(
            "unsigned_short",
            c_type="unsigned short",
            cxx_type="unsigned short",
            f_cast="int({f_var}, C_SHORT)",
            f_type="integer(C_SHORT)",
            f_kind="C_SHORT",
            f_module=dict(iso_c_binding=["C_SHORT"]),
            PY_format="h",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_SHORT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_UNSIGNED_SHORT",
            cfi_type="CFI_type_short",
        ),
        unsigned_int=Typemap(
            "unsigned_int",
            c_type="unsigned int",
            cxx_type="unsigned int",
            f_cast="int({f_var}, C_INT)",
            f_type="integer(C_INT)",
            f_kind="C_INT",
            f_module=dict(iso_c_binding=["C_INT"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_UNSIGNED_INT",
            cfi_type="CFI_type_int",
        ),
        unsigned_long=Typemap(
            "unsigned_long",
            c_type="unsigned long",
            cxx_type="unsigned long",
            f_cast="int({f_var}, C_LONG)",
            f_type="integer(C_LONG)",
            f_kind="C_LONG",
            f_module=dict(iso_c_binding=["C_LONG"]),
            PY_format="l",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_LONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_UNSIGNED_LONG",
            cfi_type="CFI_type_long",
        ),
        unsigned_long_long=Typemap(
            "unsigned_long_long",
            c_type="unsigned long long",
            cxx_type="unsigned long long",
            f_cast="int({f_var}, C_LONG_LONG)",
            f_type="integer(C_LONG_LONG)",
            f_kind="C_LONG_LONG",
            f_module=dict(iso_c_binding=["C_LONG_LONG"]),
            PY_format="L",
            # #- PY_ctor='PyInt_FromLong({ctor_expr})',
            PYN_typenum="NPY_LONGLONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_UNSIGNED_LONG_LONG",
            cfi_type="CFI_type_long_long",
        ),
        size_t=Typemap(
            "size_t",
            c_type="size_t",
            cxx_type="size_t",
            c_header="<stddef.h>",
            cxx_header="<cstddef>",
            f_cast="int({f_var}, C_SIZE_T)",
            f_type="integer(C_SIZE_T)",
            f_kind="C_SIZE_T",
            f_module=dict(iso_c_binding=["C_SIZE_T"]),
            PY_format="n",
            PY_ctor="PyInt_FromSize_t({ctor_expr})",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_SIZE_T",
            cfi_type="CFI_type_size_t",
        ),
        # XXX- sized based types for Python
        int8_t=Typemap(
            "int8_t",
            c_type="int8_t",
            cxx_type="int8_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT8_t)",
            f_type="integer(C_INT8_T)",
            f_kind="C_INT8_T",
            f_module=dict(iso_c_binding=["C_INT8_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT8",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_INT8_T",
            cfi_type="CFI_type_int8_t",
        ),
        int16_t=Typemap(
            "int16_t",
            c_type="int16_t",
            cxx_type="int16_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT16_t)",
            f_type="integer(C_INT16_T)",
            f_kind="C_INT16_T",
            f_module=dict(iso_c_binding=["C_INT16_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT16",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_INT16_T",
            cfi_type="CFI_type_int16_t",
        ),
        int32_t=Typemap(
            "int32_t",
            c_type="int32_t",
            cxx_type="int32_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT32_t)",
            f_type="integer(C_INT32_T)",
            f_kind="C_INT32_T",
            f_module=dict(iso_c_binding=["C_INT32_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT32",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_INT32_T",
            cfi_type="CFI_type_int32_t",
        ),
        int64_t=Typemap(
            "int64_t",
            c_type="int64_t",
            cxx_type="int64_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT64_t)",
            f_type="integer(C_INT64_T)",
            f_kind="C_INT64_T",
            f_module=dict(iso_c_binding=["C_INT64_T"]),
            PY_format="L",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT64",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_INT64_T",
            cfi_type="CFI_type_int64_t",
        ),
        # XXX- sized based types for Python
        uint8_t=Typemap(
            "uint8_t",
            c_type="uint8_t",
            cxx_type="uint8_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT8_t)",
            f_type="integer(C_INT8_T)",
            f_kind="C_INT8_T",
            f_module=dict(iso_c_binding=["C_INT8_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT8",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_UINT8_T",
            cfi_type="CFI_type_int8_t",
        ),
        uint16_t=Typemap(
            "uint16_t",
            c_type="uint16_t",
            cxx_type="uint16_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT16_t)",
            f_type="integer(C_INT16_T)",
            f_kind="C_INT16_T",
            f_module=dict(iso_c_binding=["C_INT16_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT16",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_UINT16_T",
            cfi_type="CFI_type_int16_t",
        ),
        uint32_t=Typemap(
            "uint32_t",
            c_type="uint32_t",
            cxx_type="uint32_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT32_t)",
            f_type="integer(C_INT32_T)",
            f_kind="C_INT32_T",
            f_module=dict(iso_c_binding=["C_INT32_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT32",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_UINT32_T",
            cfi_type="CFI_type_int32_t",
        ),
        uint64_t=Typemap(
            "uint64_t",
            c_type="uint64_t",
            cxx_type="uint64_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT64_t)",
            f_type="integer(C_INT64_T)",
            f_kind="C_INT64_T",
            f_module=dict(iso_c_binding=["C_INT64_T"]),
            PY_format="L",
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT64",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_UINT64_T",
            cfi_type="CFI_type_int64_t",
        ),
        float=Typemap(
            "float",
            c_type="float",
            cxx_type="float",
            f_cast="real({f_var}, C_FLOAT)",
            f_type="real(C_FLOAT)",
            f_kind="C_FLOAT",
            f_module=dict(iso_c_binding=["C_FLOAT"]),
            PY_format="f",
            PY_ctor="PyFloat_FromDouble({ctor_expr})",
            PY_get="PyFloat_AsDouble({py_var})",
            PYN_typenum="NPY_FLOAT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_FLOAT",
            cfi_type="CFI_type_float",
        ),
        double=Typemap(
            "double",
            c_type="double",
            cxx_type="double",
            f_cast="real({f_var}, C_DOUBLE)",
            f_type="real(C_DOUBLE)",
            f_kind="C_DOUBLE",
            f_module=dict(iso_c_binding=["C_DOUBLE"]),
            PY_format="d",
            PY_ctor="PyFloat_FromDouble({ctor_expr})",
            PY_get="PyFloat_AsDouble({py_var})",
            PYN_typenum="NPY_DOUBLE",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_DOUBLE",
            cfi_type="CFI_type_double",
        ),
        float_complex=Typemap(   # _Complex
            "float_complex",
            c_type="float complex",
            cxx_type="std::complex<float>",
            flat_name="float_complex",
            c_header="<complex.h>",
            cxx_header="<complex>",
            f_cast="cmplx({f_var}, C_FLOAT_COMPLEX)",
            f_type="complex(C_FLOAT_COMPLEX)",
            f_kind="C_FLOAT_COMPLEX",
            f_module=dict(iso_c_binding=["C_FLOAT_COMPLEX"]),
            PY_format="D",
            py_ctype="Py_complex",
            pytype_to_pyctor="creal({ctor_expr}), cimag({ctor_expr})",
            pytype_to_cxx="{work_var}.real + {work_var}.imag * I",
            cxx_to_pytype="{py_var}.real = creal({cxx_var});\n{py_var}.imag = cimag({cxx_var});",
            PY_ctor="PyComplex_FromDoubles(\t{ctor_expr})",
            PY_get="PyComplex_AsCComplex({py_var})",
            PY_build_arg="&{ctype_var}",
            PYN_typenum="NPY_DOUBLE",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_FLOAT_COMPLEX",
            cfi_type="CFI_type_float_Complex",
        ),
        double_complex=Typemap(   # _Complex
            "double_complex",
            c_type="double complex",
            cxx_type="std::complex<double>",
            flat_name="double_complex",
            c_header="<complex.h>",
            cxx_header="<complex>",
            f_cast="cmplx({f_var}, C_DOUBLE_COMPLEX)",
            f_type="complex(C_DOUBLE_COMPLEX)",
            f_kind="C_DOUBLE_COMPLEX",
            f_module=dict(iso_c_binding=["C_DOUBLE_COMPLEX"]),
            PY_format="D",
            PY_get="PyComplex_AsCComplex({py_var})",
            py_ctype="Py_complex",
            pytype_to_pyctor="creal({ctor_expr}), cimag({ctor_expr})",
            pytype_to_cxx="{work_var}.real + {work_var}.imag * I",
            cxx_to_pytype="{ctype_var}.real = creal({cxx_var});\n{ctype_var}.imag = cimag({cxx_var});",
            # fmt.work_ctor = "std::complex(\tcvalue.real, cvalue.imag)"
            # creal(), cimag()
            # std::real(), std::imag()
            # xx.real(), xx.imag()
            PY_ctor="PyComplex_FromDoubles(\t{ctor_expr})", # double real, double imag
            PY_build_arg="&{ctype_var}",
            PYN_typenum="NPY_DOUBLE",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {push_arg})",
            sgroup="native",
            sh_type="SH_TYPE_DOUBLE_COMPLEX",
            cfi_type="CFI_type_double_Complex",
        ),
        bool=Typemap(
            "bool",
            c_type="bool",
            cxx_type="bool",
            c_header="<stdbool.h>",
            f_type="logical",
            f_kind="C_BOOL",
            f_c_type="logical(C_BOOL)",
            f_module=dict(iso_c_binding=["C_BOOL"]),
            # XXX PY_format='p',  # Python 3.3 or greater
            # Use py_statements.x.ctor instead of PY_ctor. This code will always be
            # added.  Older version of Python can not create a bool directly from
            # from Py_BuildValue.
            # #- PY_ctor='PyBool_FromLong({c_var})',
            PY_PyTypeObject="PyBool_Type",
            PYN_typenum="NPY_BOOL",
            LUA_type="LUA_TBOOLEAN",
            LUA_pop="lua_toboolean({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushboolean({LUA_state_var}, {push_arg})",
            base="bool",
            sgroup="bool",
            sh_type="SH_TYPE_BOOL",
            cfi_type="CFI_type_Bool",
        ),
        # implies null terminated string
        char=Typemap(
            "char",
            cxx_type="char",
            c_type="char",  # XXX - char *
            f_type="character(*)",
            f_kind="C_CHAR",
            f_c_type="character(kind=C_CHAR)",
            f_c_module=dict(iso_c_binding=["C_CHAR"]),
            PY_format="s",
            PY_ctor="PyString_FromString({ctor_expr})",
#            PY_get="PyString_AsString({py_var})",
            PYN_typenum="NPY_INTP",  # void *    # XXX - 
            LUA_type="LUA_TSTRING",
            LUA_pop="lua_tostring({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushstring({LUA_state_var}, {push_arg})",
            base="string",
            sgroup="char",
        ),
        # C++ std::string
        string=Typemap(
            "std::string",
            cxx_type="std::string",
            cxx_to_c="{cxx_var}{cxx_member}c_str()",  # cxx_member is . or ->
            c_type="char",  # XXX - char *
            impl_header=["<string>"],
            f_type="character(*)",
            f_kind="C_CHAR",
            f_c_type="character(kind=C_CHAR)",
            f_c_module=dict(iso_c_binding=["C_CHAR"]),
            PY_format="s",
            PY_ctor="PyString_FromStringAndSize({ctor_expr})",
            PY_build_format="s#",
            # XXX need cast after PY_SSIZE_T_CLEAN
            PY_build_arg="{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size()",
            LUA_type="LUA_TSTRING",
            LUA_pop="lua_tostring({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushstring({LUA_state_var}, {push_arg})",
            base="string",
            sgroup="string",
        ),
        # C++ std::vector
        # No c_type or f_type, use attr[template]
        # C++03 "The elements of a vector are stored contiguously" (23.2.4/1).
        vector=Typemap(
            "std::vector",
            cxx_type="std::vector<{cxx_T}>",
            cxx_header="<vector>",
            # #- cxx_to_c='{cxx_var}.data()',  # C++11
            # #- cxx_to_c='{cxx_var}{cxx_member}empty() ? NULL : &{cxx_var}[0]', # C++03)
            # custom code for templates
            #            PY_format='s',
            #            PY_ctor='PyString_FromString({c_var})',
            #            LUA_type='LUA_TSTRING',
            #            LUA_pop='lua_tostring({LUA_state_var}, {LUA_index})',
            #            LUA_push='lua_pushstring({LUA_state_var}, {push_arg})',
            impl_header=["<vector>"],
            base="vector",
            sgroup="vector",
        ),
        MPI_Comm=Typemap(
            "MPI_Comm",
            cxx_type="MPI_Comm",
            c_header="mpi.h",
            cxx_header="mpi.h",
            c_type="MPI_Fint",
            # usually, MPI_Fint will be equivalent to int
            f_type="integer",
            f_kind="C_INT",
            f_c_type="integer(C_INT)",
            f_c_module=dict(iso_c_binding=["C_INT"]),
            cxx_to_c="MPI_Comm_c2f({cxx_var})",
            c_to_cxx="MPI_Comm_f2c({c_var})",
        ),
    )

    # Rename to actual types.
    # It is not possible to do dict(std::string=...)
    def_types["std::string"] = def_types["string"]
    del def_types["string"]
    def_types["std::vector"] = def_types["vector"]
    del def_types["vector"]

    set_global_typemaps(def_types)

    return def_types


def create_native_typemap_from_fields(cxx_name, fields, library):
    """Create a typemap from fields.
    Used when creating typemap from YAML. (from regression/forward.yaml)

        typemap:
        - type: indextype
          fields:
            base: integer
            f_kind: INDEXTYPE
            f_module_name: typemap_mod

    Parameters:
    -----------
    cxx_name : str
    fields : dictionary object.
    library : ast.LibraryNode.
    """
    fmt = library.fmtdict
    ntypemap = Typemap(
        cxx_name,
        sgroup="native",
        cxx_type=cxx_name,
        c_type=cxx_name,
        f_cast=None,  # Override Typemap default
    )
    ntypemap.update(fields)
    missing = []
    if ntypemap.f_kind is None:
        missing.append("f_kind")
    if ntypemap.f_module_name is None:
        missing.append("f_module_name")
    if missing:
        raise RuntimeError(
            "typemap {} requires field(s) {}".format(cxx_name, ", ".join(missing))
        )
    fill_native_typemap_defaults(ntypemap, fmt)
    ntypemap.finalize()
    register_typemap(cxx_name, ntypemap)
    library.add_typedef_by_name(cxx_name, ntypemap)
    return ntypemap


def fill_native_typemap_defaults(ntypemap, fmt):
    """Add some defaults to integer or real typemap.
    When dumping typemaps to a file, only a subset is written
    since the rest are boilerplate.  This function restores
    the boilerplate.

    Some fields can be derived from f_kind.

    Parameters
    ----------
    ntypemap : typemap.Typemap.
    fmt : util.Scope
    """
    if ntypemap.f_type is None:
        ntypemap.f_type = "{}({})".format(ntypemap.base, ntypemap.f_kind)
    if ntypemap.f_cast is None:
        ntypemap.f_cast = "{}({{f_var}}, {})".format(ntypemap.base, ntypemap.f_kind)
    if ntypemap.f_module is None:
        ntypemap.f_module = {ntypemap.f_module_name: [ntypemap.f_kind]}


def create_enum_typemap(node):
    """Create a typemap similar to an int.
    C++ enums are converted to a C int.

    Args:
        node - EnumNode instance.
    """
    fmt_enum = node.fmtdict
    type_name = util.wformat("{namespace_scope}{enum_name}", fmt_enum)

    ntypemap = lookup_typemap(type_name)
    if ntypemap is None:
        inttypemap = lookup_typemap("int")
        ntypemap = inttypemap.clone_as(type_name)
        ntypemap.cxx_type = util.wformat(
            "{namespace_scope}{enum_name}", fmt_enum
        )
        ntypemap.c_to_cxx = util.wformat(
            "static_cast<{namespace_scope}{enum_name}>({{c_var}})", fmt_enum
        )
        ntypemap.cxx_to_c = "static_cast<int>({cxx_var})"
        ntypemap.compute_flat_name()
        register_typemap(type_name, ntypemap)
    return ntypemap


def create_class_typemap_from_fields(cxx_name, fields, library):
    """Create a typemap from fields.
    Used when creating typemap from YAML. (from regression/forward.yaml)

        typemap:
        - type: tutorial::Class1
          fields:
            base: shadow

    Parameters:
    -----------
    cxx_name : str
    fields : dictionary object.
    library : ast.LibraryNode.
    """
    fmt_class = library.fmtdict
    ntypemap = Typemap(
        cxx_name,
        base="shadow", sgroup="shadow",
        cxx_type=cxx_name,
        f_capsule_data_type="missing-f_capsule_data_type",
        f_derived_type=cxx_name,
    )
    ntypemap.update(fields)
    if ntypemap.f_module_name is None:
        raise RuntimeError(
            "typemap {} requires field f_module_name".format(cxx_name)
        )
    ntypemap.f_module = {ntypemap.f_module_name: [ntypemap.f_derived_type]}
    ntypemap.f_c_module = {
        ntypemap.f_module_name: [ntypemap.f_capsule_data_type]
    }
    fill_shadow_typemap_defaults(ntypemap, fmt_class)
    ntypemap.finalize()
    register_typemap(cxx_name, ntypemap)
    library.add_shadow_typemap(ntypemap)
    return ntypemap


def create_class_typemap(node, fields=None):
    """Create a typemap from a ClassNode.
    Use fields to override defaults.

    The c_type and f_capsule_data_type are a struct which contains
    a pointer to the C++ memory and information on how to delete the memory.

    Args:
        node - ast.ClassNode.
        fields - dictionary-like object.
    """
    fmt_class = node.fmtdict
    cxx_name = util.wformat("{namespace_scope}{cxx_class}", fmt_class)
    cxx_type = util.wformat("{namespace_scope}{cxx_type}", fmt_class)

    ntypemap = lookup_typemap(cxx_name)
    # unname = util.un_camel(name)
    f_name = fmt_class.cxx_class.lower()
    c_name = fmt_class.C_prefix + fmt_class.C_name_scope[:-1]
    ntypemap = Typemap(
        cxx_name,
        base="shadow",
        sgroup="shadow",
        cxx_type=cxx_type,
        impl_header=node.find_header(),
        wrap_header=fmt_class.C_header_utility,
        c_type=c_name,
        f_module_name=fmt_class.F_module_name,
        f_derived_type=fmt_class.F_derived_name,
        f_capsule_data_type=fmt_class.F_capsule_data_type,
        f_module={fmt_class.F_module_name: [fmt_class.F_derived_name]},
        # #- f_to_c='{f_var}%%%s()' % fmt_class.F_name_instance_get, # XXX - develop test
        f_to_c="{f_var}%%%s" % fmt_class.F_derived_member,
        sh_type="SH_TYPE_OTHER",
        cfi_type="CFI_type_other",
    )
    # import classes which are wrapped by this module
    # XXX - deal with namespaces vs modules
    ntypemap.f_c_module = {"--import--": [ntypemap.f_capsule_data_type]}
    if fields is not None:
        ntypemap.update(fields)
    fill_shadow_typemap_defaults(ntypemap, fmt_class)
    ntypemap.finalize()
    register_typemap(cxx_name, ntypemap)

    fmt_class.C_type_name = ntypemap.c_type
    return ntypemap


def fill_shadow_typemap_defaults(ntypemap, fmt):
    """Add some defaults to shadow typemap.
    When dumping typemaps to a file, only a subset is written
    since the rest are boilerplate.  This function restores
    the boilerplate.

    Parameters
    ----------
    ntypemap : typemap.Typemap.
    fmt : util.Scope
    """
    if ntypemap.base != "shadow":
        return

    # Convert to void * to add to context struct
    ntypemap.cxx_to_c = "static_cast<{c_const}void *>(\t{cxx_addr}{cxx_var})"

    # void pointer in struct -> class instance pointer
    ntypemap.c_to_cxx = (
        "static_cast<{c_const}%s *>\t({c_var}->addr)" % ntypemap.cxx_type
    )

    # some default for ntypemap.f_capsule_data_type
    ntypemap.f_class = "class(%s)" % ntypemap.f_derived_type
    ntypemap.f_type = "type(%s)" % ntypemap.f_derived_type
    ntypemap.f_c_type = "type(%s)" % ntypemap.f_capsule_data_type

    # XXX module name may not conflict with type name
    #    ntypemap.f_module={fmt_class.F_module_name:[unname]}

    # return from C function
    # f_c_return_decl='type(C_PTR)' % unname,

    # The import is added in wrapf.py
    #    ntypemap.f_c_module={ '--import--': ['F_capsule_data_type']}

    # #-    if not ntypemap.PY_PyTypeObject:
    # #-        ntypemap.PY_PyTypeObject = 'UUU'
    # ntypemap.PY_ctor = 'PyObject_New({PyObject}, &{PyTypeObject})'

    ntypemap.LUA_type = "LUA_TUSERDATA"
    ntypemap.LUA_pop = (
        "\t({LUA_userdata_type} *)\t luaL_checkudata"
        '(\t{LUA_state_var}, 1, "{LUA_metadata}")'
    )
    # ntypemap.LUA_push = None  # XXX create a userdata object with metatable
    # ntypemap.LUA_statements = {}

    # allow forward declarations to avoid recursive headers
    ntypemap.forward = ntypemap.cxx_type


def create_struct_typemap_from_fields(cxx_name, fields, library):
    """Create a struct typemap from fields.
    Used when creating typemap from YAML. (from regression/forward.yaml)

        typemap:
        - type: tutorial::Struct1
          fields:
            base: struct

    Parameters:
    -----------
    cxx_name : str
    fields : dictionary object.
    library : ast.LibraryNode.
    """
    fmt_class = library.fmtdict
    ntypemap = Typemap(
        cxx_name,
        base="struct", sgroup="struct",
        c_type=cxx_name,
        cxx_type=cxx_name,
        f_type = "type(%s)" % cxx_name,
        f_to_c="{f_var}",
    )
    ntypemap.update(fields)
    if ntypemap.f_module_name is None:
        raise RuntimeError(
            "typemap {} requires field f_module_name".format(cxx_name)
        )
    if ntypemap.f_derived_type is None:
        ntypemap.f_derived_type  = ntypemap.name
    ntypemap.f_module = {ntypemap.f_module_name: [ntypemap.f_derived_type]}
# XXX - Set defaults while being created above.
#    fill_struct_typemap_defaults(node, ntypemap)

    register_typemap(cxx_name, ntypemap)
    library.add_shadow_typemap(ntypemap)
    return ntypemap


def create_struct_typemap(node, fields=None):
    """Create a typemap for a struct from a ClassNode.
    Use fields to override defaults.

    Parameters:
    -----------
    node : ast.ClassNode
    fields : dictionary-like object.
    """
    fmt_class = node.fmtdict
    cxx_name = util.wformat("{namespace_scope}{cxx_class}", fmt_class)
    cxx_type = util.wformat("{namespace_scope}{cxx_type}", fmt_class)

    # unname = util.un_camel(name)
    f_name = fmt_class.cxx_class.lower()
    c_name = fmt_class.C_prefix + f_name
    ntypemap = Typemap(
        cxx_name,
        base="struct",
        sgroup="struct",
        cxx_type=cxx_type,
        c_type=c_name,
        f_derived_type=fmt_class.F_derived_name,
        f_module={fmt_class.F_module_name: [fmt_class.F_derived_name]},
        f_c_module={"--import--": [fmt_class.F_derived_name]},
        PYN_descr=fmt_class.PY_struct_array_descr_variable,
        sh_type="SH_TYPE_STRUCT",
        cfi_type="CFI_type_struct",
    )
    if fields is not None:
        ntypemap.update(fields)
    fill_struct_typemap_defaults(node, ntypemap)
    register_typemap(cxx_name, ntypemap)

    fmt_class.C_type_name = ntypemap.c_type
    return ntypemap


def fill_struct_typemap_defaults(node, ntypemap):
    """Add some defaults to struct typemap.
    When dumping typemaps to a file, only a subset is written
    since the rest are boilerplate.  This function restores
    the boilerplate.

    Parameters:
    -----------
        node : ast.ClassNode
        ntypemap : typemap.Typemap.
    """
    if ntypemap.base != "struct":
        return

    libnode = node.get_LibraryNode()
    language = libnode.language
    if language == "c":
        # The struct from the user's library is used.
        # XXX - if struct in class, uses class.cxx_header?
        ntypemap.c_header = libnode.cxx_header
        ntypemap.c_type = ntypemap.cxx_type

    # To convert, extract correct field from union
    # #-    ntypemap.cxx_to_c = '{cxx_addr}{cxx_var}.cxx'
    # #-    ntypemap.c_to_cxx = '{cxx_addr}{cxx_var}.c'

    ntypemap.PY_struct_as = node.options.PY_struct_arg
    ntypemap.f_type = "type(%s)" % ntypemap.f_derived_type

    # XXX module name may not conflict with type name
    # #-    ntypemap.f_module = {fmt_class.F_module_name:[unname]}

    # #-    ntypemap.PYN_typenum = 'NPY_VOID'
    # #-    if not ntypemap.PY_PyTypeObject:
    # #-        ntypemap.PY_PyTypeObject = 'UUU'
    # ntypemap.PY_ctor = 'PyObject_New({PyObject}, &{PyTypeObject})'

    ntypemap.LUA_type = "LUA_TUSERDATA"
    ntypemap.LUA_pop = (
        "\t({LUA_userdata_type} *)\t luaL_checkudata"
        '(\t{LUA_state_var}, 1, "{LUA_metadata}")'
    )
    # ntypemap.LUA_push = None  # XXX create a userdata object with metatable
    # ntypemap.LUA_statements = {}


def create_fcnptr_typemap(node, fields=None):
    """Create a typemap for a function pointer
    The typemap contains a the declaration.

    Args:
        node - ast.TypedefNode
        fields - dictionary
    """
#    print("XXXXXXXX", dir(node))
#    print("  name", node.name)   # return type of function pointer
#    print("  typemap", node.typemap)   # return type of function pointer
#    raise NotImplementedError(
#        "Function pointers not supported in typedef"
#    )
    fmt = node.fmtdict
    cxx_name = node.ast.name
    fmt.typedef_name = cxx_name
    cxx_name = util.wformat("{namespace_scope}{typedef_name}", fmt)
    cxx_type = cxx_name
#    cxx_type = util.wformat("{namespace_scope}{cxx_type}", fmt)
    c_type = fmt.C_prefix + cxx_name
    ntypemap = Typemap(
        cxx_name,
        base="fcnptr",
        sgroup="fcnptr",
        c_type="c_type",
        cxx_type="cxx_type",
        f_type="XXXf_type",
    )
    # Check if all fields are C compatible
#            ntypemap.compute_flat_name()
    
    if fields is not None:
        ntypemap.update(fields)
    register_typemap(cxx_name, ntypemap)
    return ntypemap


def return_shadow_types():
    """Return a dictionary of user defined types."""
    dct = {}
    for key, ntypemap in shared_typemaps.items():
        if ntypemap.sgroup in ["shadow", "struct", "template"]:
            dct[key] = ntypemap
    return dct
