
from . import util
from . import declast

class AstNode(object):
    def update_options_from_dict(self, node):
        """Update options from node.
        """
        if 'options' in node and \
                node['options'] is not None:
            if not isinstance(node['options'], dict):
                raise TypeError("options must be a dictionary")
            self.options.update(node['options'], replace=True)

    def option_to_fmt(self):
        """Set fmt based on options dictionary.
        """
        for name in ['C_prefix', 'F_C_prefix', 
                     'C_this', 'C_result', 'CPP_this',
                     'F_this', 'F_result', 'F_derived_member',
                     'C_string_result_as_arg', 'F_string_result_as_arg',
                     'C_header_filename_suffix',
                     'C_impl_filename_suffix',
                     'F_filename_suffix',
                     'PY_header_filename_suffix',
                     'PY_impl_filename_suffix',
                     'PY_result',
                     'LUA_header_filename_suffix',
                     'LUA_impl_filename_suffix',
                     'LUA_result']:
            if self.options.inlocal(name):
                setattr(self._fmt, name, self.options[name])

    def eval_template(self, name, tname='', fmt=None):
        """fmt[name] = self.name or option[name + tname + '_template']
        """
        if fmt is None:
            fmt = self._fmt
        value = getattr(self, name)
        if value is not None:
            setattr(fmt, name, value)
        else:
            tname = name + tname + '_template'
            setattr(fmt, name, util.wformat(self.options[tname], fmt))

    def check_options_only(self, node, parent):
        """Process an options only entry in a list.

        functions:
        - options:
             a = b
        - decl:
          options:

        Return True if node only has options.
        Return Options instance to use.
        node is assumed to be a dictionary.
        Update current set of options from node['options'].
        """
        if len(node) != 1:
            return False, parent
        if 'options' not in node:
            return False, parent
        options = node['options']
        if not options:
            return False, parent
        if not isinstance(options, dict):
            raise TypeError("options must be a dictionary")

        new = util.Options(parent=parent)
        new.update(node['options'])
        return True, new

    def add_functions(self, node, cls_name, member):
        """ Add functions from dictionary 'node'.

        Used with class methods and functions.
        """
        if member not in node:
            return
        functions = node[member]
        if not isinstance(functions, list):
            raise TypeError("functions must be a list")

        options = self.options
        for func in functions:
            only, options = self.check_options_only(func, options)
            if not only:
                self.functions.append(FunctionNode(func, self, cls_name, options))

    def _to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        return dict()

######################################################################

class LibraryNode(AstNode):
    def __init__(self, node=None):
        """Populate LibraryNode from a dictionary.

        fields = value
        options:
        classes:
        functions:

        """

        self.classes = []
        self.cpp_header = ''
        self.functions = []
        # Each is given a _function_index when created.
        self.function_index = []
        self.language = 'c++'     # input language: c or c++
        self.namespace = ''
        self.options = self.default_options()

        if node is None:
            node = dict()
        self.node = node

        self.library = node.get('library', 'default_library')

        for n in ['C_header_filename', 'C_impl_filename',
                  'F_module_name', 'F_impl_filename',
                  'LUA_module_name', 'LUA_module_reg', 'LUA_module_filename', 'LUA_header_filename',
                  'PY_module_filename', 'PY_header_filename', 'PY_helper_filename',
                  'YAML_type_filename']:
            setattr(self, n, node.get(n, None))

        if 'language' in node:
            language = node['language'].lower()
            if language not in ['c', 'c++']:
                raise RuntimeError("language must be 'c' or 'c++'")
            self.language = node['language']

        self.update_options_from_dict(node)

        self.default_format(node)
        self.option_to_fmt()

#        self.fmt_stack.append(fmt_library)

        # default some options based on other options
        self.eval_template('C_header_filename', '_library')
        self.eval_template('C_impl_filename', '_library')
        # All class/methods and functions may go into this file or
        # just functions.
        self.eval_template('F_module_name', '_library')
        self.eval_template('F_impl_filename', '_library')

        # default cpp_header to blank
        if 'cpp_header' in node and node['cpp_header']:
            # YAML treats blank string as None
            self.cpp_header = node['cpp_header']

        if 'namespace' in node and node['namespace']:
            # YAML treats blank string as None
            self.namespace = node['namespace']

        self.add_classes(node)
        self.add_functions(node, None, 'functions')

    def default_options(self):
        """default options."""
        def_options = util.Options(
            parent=None,
            debug=False,   # print additional debug info

            F_module_per_class=True,
            F_string_len_trim=True,
            F_force_wrapper=False,

            wrap_c=True,
            wrap_fortran=True,
            wrap_python=False,
            wrap_lua=False,

            doxygen=True,       # create doxygen comments
            show_splicer_comments=True,

            # blank for functions, set in classes.
            class_prefix_template='{class_lower}_',

            YAML_type_filename_template='{library_lower}_types.yaml',

            C_header_filename_library_template='wrap{library}.{C_header_filename_suffix}',
            C_impl_filename_library_template='wrap{library}.{C_impl_filename_suffix}',

            C_header_filename_class_template='wrap{cpp_class}.{C_header_filename_suffix}',
            C_impl_filename_class_template='wrap{cpp_class}.{C_impl_filename_suffix}',

            C_name_template=(
                '{C_prefix}{class_prefix}{underscore_name}{function_suffix}'),

            C_bufferify_suffix='_bufferify',
            C_var_len_template = 'N{c_var}',         # argument for result of len(arg)
            C_var_trim_template = 'L{c_var}',        # argument for result of len_trim(arg)
            C_var_size_template = 'S{c_var}',        # argument for result of size(arg)

            # Fortran's names for C functions
            F_C_prefix='c_',
            F_C_name_template=(
                '{F_C_prefix}{class_prefix}{underscore_name}{function_suffix}'),

            F_name_impl_template=(
                '{class_prefix}{underscore_name}{function_suffix}'),

            F_name_function_template='{underscore_name}{function_suffix}',
            F_name_generic_template='{underscore_name}',

            F_module_name_library_template='{library_lower}_mod',
            F_impl_filename_library_template='wrapf{library_lower}.{F_filename_suffix}',

            F_module_name_class_template='{class_lower}_mod',
            F_impl_filename_class_template='wrapf{cpp_class}.{F_filename_suffix}',

            F_name_instance_get='get_instance',
            F_name_instance_set='set_instance',
            F_name_associated='associated',

            LUA_module_name_template='{library_lower}',
            LUA_module_filename_template=(
                'lua{library}module.{LUA_impl_filename_suffix}'),
            LUA_header_filename_template=(
                'lua{library}module.{LUA_header_filename_suffix}'),
            LUA_userdata_type_template='{LUA_prefix}{cpp_class}_Type',
            LUA_userdata_member_template='self',
            LUA_module_reg_template='{LUA_prefix}{library}_Reg',
            LUA_class_reg_template='{LUA_prefix}{cpp_class}_Reg',
            LUA_metadata_template='{cpp_class}.metatable',
            LUA_ctor_name_template='{cpp_class}',
            LUA_name_template='{function_name}',
            LUA_name_impl_template='{LUA_prefix}{class_prefix}{underscore_name}',

            PY_module_filename_template=(
                'py{library}module.{PY_impl_filename_suffix}'),
            PY_header_filename_template=(
                'py{library}module.{PY_header_filename_suffix}'),
            PY_helper_filename_template=(
                'py{library}helper.{PY_impl_filename_suffix}'),
            PY_PyTypeObject_template='{PY_prefix}{cpp_class}_Type',
            PY_PyObject_template='{PY_prefix}{cpp_class}',
            PY_type_filename_template=(
                'py{cpp_class}type.{PY_impl_filename_suffix}'),
            PY_name_impl_template=(
                '{PY_prefix}{class_prefix}{underscore_name}{function_suffix}'),
            )
        return def_options

    def default_format(self, node):
        """Set format dictionary.
        """

        self._fmt = util.Options(None)
        fmt_library = self._fmt

        if 'library' in node:
            fmt_library.library = node['library']
        else:
            fmt_library.library = 'default_library'
        fmt_library.library_lower = fmt_library.library.lower()
        fmt_library.library_upper = fmt_library.library.upper()
        fmt_library.function_suffix = ''   # assume no suffix
        fmt_library.C_prefix = self.options.get(
            'C_prefix', fmt_library.library_upper[:3] + '_')
        fmt_library.F_C_prefix = self.options['F_C_prefix']
        if 'namespace' in node and node['namespace']:
            fmt_library.namespace_scope = (
                '::'.join(node['namespace'].split()) + '::')
        else:
            fmt_library.namespace_scope = ''

        # set default values for fields which may be unset.
        fmt_library.class_prefix = ''
#        fmt_library.c_ptr = ''
#        fmt_library.c_const = ''
        fmt_library.CPP_this_call = ''
        fmt_library.CPP_template = ''
        fmt_library.C_pre_call = ''
        fmt_library.C_post_call = ''

        fmt_library.C_this = 'self'
        fmt_library.C_result = 'SHT_rv'
        fmt_library.c_temp = 'SHT_'

        fmt_library.CPP_this = 'SH_this'

        fmt_library.F_this = 'obj'
        fmt_library.F_result = 'SHT_rv'
        fmt_library.F_derived_member = 'voidptr'

        fmt_library.C_string_result_as_arg = 'SHF_rv'
        fmt_library.F_string_result_as_arg = ''

        fmt_library.F_filename_suffix = 'f'

        # don't have to worry about argument names in Python wrappers
        # so skip the SH_ prefix by default.
        fmt_library.PY_result = 'rv'
        fmt_library.LUA_result = 'rv'

        if self.language == 'c':
            fmt_library.C_header_filename_suffix = 'h'
            fmt_library.C_impl_filename_suffix = 'c'

            fmt_library.PY_header_filename_suffix = 'h'
            fmt_library.PY_impl_filename_suffix = 'c'

            fmt_library.LUA_header_filename_suffix = 'h'
            fmt_library.LUA_impl_filename_suffix = 'c'

            fmt_library.stdlib  = ''
        else:
            fmt_library.C_header_filename_suffix = 'h'
            fmt_library.C_impl_filename_suffix = 'cpp'

            fmt_library.PY_header_filename_suffix = 'hpp'
            fmt_library.PY_impl_filename_suffix = 'cpp'

            fmt_library.LUA_header_filename_suffix = 'hpp'
            fmt_library.LUA_impl_filename_suffix = 'cpp'

            fmt_library.stdlib  = 'std::'

    def add_classes(self, node):
        """Add classes from a dictionary.

        classes:
        - name: Class1
        - name: Class2
        """
        if 'classes' not in node:
            return
        classes = node['classes']
        if not isinstance(classes, list):
            raise TypeError("classes must be a list")

        # Add all class types to parser first
        # Emulate forward declarations of classes.
        for cls in classes:
            if not isinstance(cls, dict):
                raise TypeError("classes[n] must be a dictionary")
            if 'name' not in cls:
                raise TypeError("class does not define name")
            declast.add_type(cls['name'])

        for cls in classes:
            self.classes.append(ClassNode(cls['name'], self, cls))

    def XX_to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        d = dict(
            _fmt=self._fmt
        )
        return d

######################################################################

class ClassNode(AstNode):
    def __init__(self, name, parent, node=None):
        self.name = name
        self.functions = []
        self.cpp_header = ''

        if node is None:
            node = {}

        # default cpp_header to blank
        if 'cpp_header' in node and node['cpp_header']:
            # YAML treats blank string as None
            self.cpp_header = node['cpp_header']

        self.namespace = ''
        if 'namespace' in node and node['namespace']:
            # YAML treats blank string as None
            self.namespace = node['namespace']
        self.python = node.get('python', {})

        for n in ['C_header_filename', 'C_impl_filename',
                  'F_derived_name', 'F_impl_filename', 'F_module_name',
                  'LUA_userdata_type', 'LUA_userdata_member', 'LUA_class_reg',
                  'LUA_metadata', 'LUA_ctor_name',
                  'PY_PyTypeObject', 'PY_PyObject', 'PY_type_filename',
                  'class_prefix']:
            setattr(self, n, node.get(n, None))

        self.options = util.Options(parent=parent.options)
        self.update_options_from_dict(node)
        options = self.options

        self._fmt = util.Options(parent._fmt)
        fmt_class = self._fmt
        fmt_class.cpp_class = name
        fmt_class.class_lower = name.lower()
        fmt_class.class_upper = name.upper()
        self.eval_template('class_prefix')

        # Only one file per class for C.
        self.eval_template('C_header_filename', '_class')
        self.eval_template('C_impl_filename', '_class')

        if options.F_module_per_class:
            self.eval_template('F_module_name', '_class')
            self.eval_template('F_impl_filename', '_class')

        self.add_functions(node, name, 'methods')

    def _to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        d = dict(
            _fmt = self._fmt,
            cpp_header=self.cpp_header,
            methods=self.functions,
            name=self.name,
            options=self.options,
        )
        for key in ['namespace', 'python']:
            value = getattr(self,key)
            if value:
                d[key] = value
        for key in ['C_header_filename', 'C_impl_filename',
                    'F_derived_name', 'F_impl_filename', 'F_module_name']:
            value = getattr(self,key)
            if value is not None:
                d[key] = value
        return d


######################################################################

class FunctionNode(AstNode):
    """

    - decl:
      cpp_template:
        ArgType:
        - int
        - double
    """



    def __init__(self, node, parent, cls_name, options):
        self.options = util.Options(parent=options)
        self.update_options_from_dict(node)
        options = self.options

        self._fmt = util.Options(parent._fmt)
        self.option_to_fmt()
        fmt_func = self._fmt

        # working variables
        self._PTR_C_CPP_index = None
        self._PTR_F_C_index = None
        self._CPP_return_templated = False
        self._cpp_overload = None
        self._function_index = None
        self._error_pattern_suffix = ''
        self._function_index = None
        self._generated = False
        self._has_default_arg = False
        self._nargs = None
        self._overloaded = False
        self._subprogram = 'XXX-subprogram'

#        self.function_index = []

        # Only needed for json diff
        self.attrs = node.get('attrs', None)

        # Move fields from node into instance
        for n in [
                'C_error_pattern', 'C_name',
                'C_post_call', 'C_post_call_buf',
                'F_name_function',
                'LUA_name', 'LUA_name_impl',
                'PY_name_impl' ]:
            setattr(self, n, node.get(n, None))

        self.default_arg_suffix = node.get('default_arg_suffix', [])
        self.docs = node.get('docs', '')
        self.cpp_template = node.get('cpp_template', {})
        self.doxygen = node.get('doxygen', {})
        self.fortran_generic = node.get('fortran_generic', {})
        self.return_this = node.get('return_this', False)

        self.F_C_name = node.get('F_C_name', None)
        self.F_name_generic = node.get('F_name_generic', None)
        self.F_name_impl = node.get('F_name_impl', None)
        self.PY_error_pattern = node.get('PY_error_pattern', '')

        # referenced explicity (not via fmt)
        self.C_code = node.get('C_code', None)
        self.C_return_code = node.get('C_return_code', None)
        self.C_return_type = node.get('C_return_type', None)
        self.F_code = node.get('F_code', None)
        
#        self.function_suffix = node.get('function_suffix', None)  # '' is legal value, None=unset
        if 'function_suffix' in node:
            self.function_suffix = node['function_suffix']
            if self.function_suffix is None:
                # YAML turns blanks strings into None
                # mark as explicitly set to empty
                self.function_suffix = ''
        else:
            # Mark as unset
            self.function_suffix = None

        if 'cpp_template' in node:
            template_types = node['cpp_template'].keys()
        else:
            template_types = []

        if 'decl' in node:
            # parse decl and add to dictionary
            self.decl = node['decl']
            ast = declast.check_decl(node['decl'],
                                     current_class=cls_name,
                                     template_types=template_types)
            self._ast = ast

            # add any attributes from YAML files to the ast
            if 'attrs' in node:
                attrs = node['attrs']
                if 'result' in attrs:
                    ast.attrs.update(attrs['result'])
                for arg in ast.params:
                    name = arg.name
                    if name in attrs:
                        arg.attrs.update(attrs[name])
            # XXX - waring about unused fields in attrs
        else:
            raise RuntimeError("Missing decl")
                                        
        if ('function_suffix' in node and
                node['function_suffix'] is None):
            # YAML turns blanks strings into None
            node['function_suffix'] = ''
        if 'default_arg_suffix' in node:
            default_arg_suffix = node['default_arg_suffix']
            if not isinstance(default_arg_suffix, list):
                raise RuntimeError('default_arg_suffix must be a list')
            for i, value in enumerate(node['default_arg_suffix']):
                if value is None:
                    # YAML turns blanks strings to None
                    node['default_arg_suffix'][i] = ''

# XXX - do some error checks on ast
#        if 'name' not in result:
#            raise RuntimeError("Missing result.name")
#        if 'type' not in result:
#            raise RuntimeError("Missing result.type")

        if ast.params is None:
            raise RuntimeError("Missing arguments:", ast.gen_decl())

        fmt_func.function_name = ast.name
        fmt_func.underscore_name = util.un_camel(fmt_func.function_name)

    def _to_dict(self):
        """Convert to dictionary.
        Used by util.ExpandedEncoder.
        """
        d = dict(
            _ast=self._ast,
            _fmt=self._fmt,
            decl=self.decl,
            options=self.options,
        )
        for key in ['attrs', 'cpp_template', 'default_arg_suffix', 'docs', 'doxygen', 
                    'fortran_generic', 'return_this',
                    'C_code', 'C_error_pattern', 'C_name',
                    'C_post_call', 'C_post_call_buf', 
                    'C_return_code', 'C_return_type',
                    'F_C_name', 'F_code', 'F_name_function', 'F_name_generic', 'F_name_impl',
                    'PY_error_pattern',
                    '_error_pattern_suffix']:
            value = getattr(self,key)
            if value:
                d[key] = value

        for key in ['function_suffix']:
            value = getattr(self,key)
            if value is not None:   # '' is OK
                d[key] = value

        for key in self.genlist:
            if hasattr(self,key):
                value = getattr(self,key)
                if value is not None and value is not False:
                    d[key] = value
        return d


#####################
    genlist = ['_CPP_return_templated',
               '_PTR_C_CPP_index',
               '_PTR_F_C_index',
               '_cpp_overload', '_decl', '_default_funcs', 
#               '_error_pattern_suffix',
               '_fmtargs', '_fmtresult',
               '_function_index', '_generated',
               '_has_default_arg',
               '_nargs', '_overloaded', '_subprogram']


    def setdefault(self, name, dflt):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            setattr(self, name, dflt)
            return dflt
