
---------- PY_converter_type ----------
{
    "c_include": [
        "<stddef.h>"
    ],
    "cxx_include": [
        "<cstddef>"
    ],
    "scope": "pwrap_impl"
}

##### start PY_converter_type source

// helper PY_converter_type
// Store PyObject and pointer to the data it contains.
// name - used in error messages
// obj  - A mutable object which holds the data.
//        For example, a NumPy array, Python array.
//        But not a list or str object.
// dataobj - converter allocated memory.
//           Decrement dataobj to release memory.
//           For example, extracted from a list or str.
// data  - C accessable pointer to data which is in obj or dataobj.
// size  - number of items in data (not number of bytes).
typedef struct {
    const char *name;
    PyObject *obj;
    PyObject *dataobj;
    void *data;   // points into obj.
    size_t size;
} LIB_SHROUD_converter_value;
##### end PY_converter_type source

---------- array_context ----------
{
    "dependent_helpers": [
        "type_defines"
    ],
    "fmtname": "LIB_SHROUD_array",
    "include": [
        "<stddef.h>"
    ],
    "scope": "cwrap_include"
}

##### start array_context source

// helper array_context
struct s_LIB_SHROUD_array {
    void * base_addr;
    int type;        /* type of element */
    size_t elem_len; /* bytes-per-item or character len in c++ */
    size_t size;     /* size of data in c++ */
    int rank;        /* number of dimensions, 0=scalar */
    long shape[7];
};
typedef struct s_LIB_SHROUD_array LIB_SHROUD_array;
##### end array_context source

---------- array_string_allocatable ----------
{
    "api": "c",
    "dependent_helpers": [
        "capsule_data_helper",
        "array_context",
        "array_string_out"
    ],
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudArrayStringAllocatable",
        "cnameproto": "void {cnamefunc}({C_array_type} *dest, {C_capsule_data_type} *src)",
        "fnamefunc": "{C_prefix}SHROUD_array_string_allocatable"
    },
    "fmtname": "LIB_ShroudArrayStringAllocatable",
    "name": "array_string_allocatable",
    "proto": "void LIB_ShroudArrayStringAllocatable(LIB_SHROUD_array *dest, LIB_SHROUD_capsule_data *src);",
    "scope": "cwrap_impl"
}

##### start array_string_allocatable source

// helper array_string_allocatable
// Copy the std::string array into Fortran array.
// Called by Fortran to deal with allocatable character.
// out is already blank filled.
void LIB_ShroudArrayStringAllocatable(LIB_SHROUD_array *dest, LIB_SHROUD_capsule_data *src)
{
    std::string *cxxvec = static_cast< std::string *>(src->addr);
    LIB_ShroudArrayStringOut(dest, cxxvec, dest->size);
}

##### end array_string_allocatable source

---------- array_string_out ----------
{
    "api": "cxx",
    "cxx_include": [
        "<cstring>",
        "<cstddef>"
    ],
    "dependent_helpers": [
        "array_context"
    ],
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudArrayStringOut",
        "cnamefunc_array_string_out": "{cnamefunc}",
        "cnameproto": "void {cnamefunc}({C_array_type} *outdesc, std::string *in, size_t nsize)"
    },
    "fmtname": "LIB_ShroudArrayStringOut",
    "name": "array_string_out",
    "proto": "void LIB_ShroudArrayStringOut(LIB_SHROUD_array *outdesc, std::string *in, size_t nsize);",
    "proto_include": [
        "<string>",
        "<vector>"
    ],
    "scope": "cwrap_impl"
}

##### start array_string_out source

// helper array_string_out
// Copy the std::vector<std::string> into Fortran array argument.
// Called by C++.
void LIB_ShroudArrayStringOut(LIB_SHROUD_array *outdesc, std::string *in, size_t nsize)
{
    size_t nvect = outdesc->size;
    size_t len = outdesc->elem_len;
    char *dest = static_cast<char *>(outdesc->base_addr);
    // Clear user memory
    std::memset(dest, ' ', nvect*len);

    // Copy into user memory
    nvect = std::min(nvect, nsize);
    for (size_t i = 0; i < nvect; ++i) {
        std::memcpy(dest, in[i].data(), std::min(len, in[i].length()));
        dest += outdesc->elem_len;
    }
}

##### end array_string_out source

---------- array_string_out_len ----------
{
    "api": "cxx",
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudArrayStringOutSize",
        "cnameproto": "size_t {cnamefunc}(std::string *in, size_t nsize)"
    },
    "fmtname": "LIB_ShroudArrayStringOutSize",
    "name": "array_string_out_len",
    "proto": "size_t LIB_ShroudArrayStringOutSize(std::string *in, size_t nsize);",
    "proto_include": [
        "<string>",
        "<vector>"
    ],
    "scope": "cwrap_impl"
}

##### start array_string_out_len source

// helper array_string_out_len
// Return the maximum string length in a std::vector<std::string>.
size_t LIB_ShroudArrayStringOutSize(std::string *in, size_t nsize)
{
    size_t len = 0;
    for (size_t i = 0; i < nsize; ++i) {
        len = std::max(len, in[i].length());
    }
    return len;
}

##### end array_string_out_len source

---------- capsule_data_helper ----------
{
    "scope": "cwrap_include"
}

##### start capsule_data_helper source

// helper capsule_data_helper
struct s_LIB_SHROUD_capsule_data {
    void *addr;     /* address of C++ memory */
    int idtor;      /* index of destructor */
};
typedef struct s_LIB_SHROUD_capsule_data LIB_SHROUD_capsule_data;
##### end capsule_data_helper source

---------- capsule_dtor ----------
{
    "api": "c",
    "dependent_helpers": [
        "capsule_data_helper"
    ],
    "fmtdict": {
        "cnamefunc": "{C_memory_dtor_function}",
        "cnameproto": "void {cnamefunc}\t({C_capsule_data_type} *cap)",
        "fnamefunc": "{C_prefix}SHROUD_capsule_dtor"
    },
    "fmtname": "LIB_SHROUD_memory_destructor",
    "name": "capsule_dtor",
    "proto": "void LIB_SHROUD_memory_destructor\t(LIB_SHROUD_capsule_data *cap);"
}

---------- char_alloc ----------
{
    "c_include": [
        "<string.h>",
        "<stdlib.h>",
        "<stddef.h>"
    ],
    "cxx_include": [
        "<cstring>",
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "char_len_trim"
    ],
    "fmtname": "ShroudCharAlloc"
}

##### start char_alloc c_source

// helper char_alloc
// Copy src into new memory and null terminate.
// If ntrim is 0, return NULL pointer.
// If blanknull is 1, return NULL when string is blank.
static char *ShroudCharAlloc(const char *src, int nsrc, int blanknull)
{
   int ntrim = ShroudCharLenTrim(src, nsrc);
   if (ntrim == 0 && blanknull == 1) {
     return NULL;
   }
   char *rv = malloc(nsrc + 1);
   if (ntrim > 0) {
     memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\0';
   return rv;
}
##### end char_alloc c_source

##### start char_alloc cxx_source

// helper char_alloc
// Copy src into new memory and null terminate.
// If ntrim is 0, return NULL pointer.
// If blanknull is 1, return NULL when string is blank.
static char *ShroudCharAlloc(const char *src, int nsrc, int blanknull)
{
   int ntrim = ShroudCharLenTrim(src, nsrc);
   if (ntrim == 0 && blanknull == 1) {
     return nullptr;
   }
   char *rv = (char *) std::malloc(nsrc + 1);
   if (ntrim > 0) {
     std::memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\0';
   return rv;
}
##### end char_alloc cxx_source

---------- char_array_alloc ----------
{
    "c_include": [
        "<string.h>",
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstring>",
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "char_len_trim"
    ],
    "fmtname": "ShroudStrArrayAlloc"
}

##### start char_array_alloc c_source

// helper char_array_alloc
// Copy src into new memory and null terminate.
static char **ShroudStrArrayAlloc(const char *src, int nsrc, int len)
{
   char **rv = malloc(sizeof(char *) * nsrc);
   const char *src0 = src;
   for(int i=0; i < nsrc; ++i) {
      int ntrim = ShroudCharLenTrim(src0, len);
      char *tgt = malloc(ntrim+1);
      memcpy(tgt, src0, ntrim);
      tgt[ntrim] = '\0';
      rv[i] = tgt;
      src0 += len;
   }
   return rv;
}
##### end char_array_alloc c_source

##### start char_array_alloc cxx_source

// helper char_array_alloc
// Copy src into new memory and null terminate.
// char **src +size(nsrc) +len(len)
// CHARACTER(len) src(nsrc)
static char **ShroudStrArrayAlloc(const char *src, int nsrc, int len)
{
   char **rv = static_cast<char **>(std::malloc(sizeof(char *) * nsrc));
   const char *src0 = src;
   for(int i=0; i < nsrc; ++i) {
      int ntrim = ShroudCharLenTrim(src0, len);
      char *tgt = static_cast<char *>(std::malloc(ntrim+1));
      std::memcpy(tgt, src0, ntrim);
      tgt[ntrim] = '\0';
      rv[i] = tgt;
      src0 += len;
   }
   return rv;
}
##### end char_array_alloc cxx_source

---------- char_array_free ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "fmtname": "ShroudStrArrayFree"
}

##### start char_array_free c_source

// helper char_array_free
// Release memory allocated by ShroudStrArrayAlloc
static void ShroudStrArrayFree(char **src, int nsrc)
{
   for(int i=0; i < nsrc; ++i) {
       free(src[i]);
   }
   free(src);
}
##### end char_array_free c_source

##### start char_array_free cxx_source

// helper char_array_free
// Release memory allocated by ShroudStrArrayAlloc
static void ShroudStrArrayFree(char **src, int nsrc)
{
   for(int i=0; i < nsrc; ++i) {
       std::free(src[i]);
   }
   std::free(src);
}
##### end char_array_free cxx_source

---------- char_blank_fill ----------
{
    "c_include": [
        "<string.h>"
    ],
    "cxx_include": [
        "<cstring>"
    ],
    "fmtname": "ShroudCharBlankFill"
}

##### start char_blank_fill c_source

// helper char_blank_fill
// blank fill dest starting at trailing NULL.
static void ShroudCharBlankFill(char *dest, int ndest)
{
   int nm = strlen(dest);
   if(ndest > nm) memset(dest+nm,' ',ndest-nm);
}
##### end char_blank_fill c_source

##### start char_blank_fill cxx_source

// helper char_blank_fill
// blank fill dest starting at trailing NULL.
static void ShroudCharBlankFill(char *dest, int ndest)
{
   int nm = std::strlen(dest);
   if(ndest > nm) std::memset(dest+nm,' ',ndest-nm);
}
##### end char_blank_fill cxx_source

---------- char_copy ----------
{
    "c_include": [
        "<string.h>"
    ],
    "cxx_include": [
        "<cstring>"
    ],
    "fmtname": "ShroudCharCopy"
}

##### start char_copy c_source

// helper ShroudCharCopy
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudCharCopy(char *dest, int ndest, const char *src, int nsrc)
{
   if (src == NULL) {
     memset(dest,' ',ndest); // convert NULL pointer to blank filled string
   } else {
     if (nsrc < 0) nsrc = strlen(src);
     int nm = nsrc < ndest ? nsrc : ndest;
     memcpy(dest,src,nm);
     if(ndest > nm) memset(dest+nm,' ',ndest-nm); // blank fill
   }
}
##### end char_copy c_source

##### start char_copy cxx_source

// helper ShroudCharCopy
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudCharCopy(char *dest, int ndest, const char *src, int nsrc)
{
   if (src == NULL) {
     std::memset(dest,' ',ndest); // convert NULL pointer to blank filled string
   } else {
     if (nsrc < 0) nsrc = std::strlen(src);
     int nm = nsrc < ndest ? nsrc : ndest;
     std::memcpy(dest,src,nm);
     if(ndest > nm) std::memset(dest+nm,' ',ndest-nm); // blank fill
   }
}
##### end char_copy cxx_source

---------- char_free ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "fmtname": "ShroudCharFree"
}

##### start char_free c_source

// helper char_free
// Release memory allocated by ShroudCharAlloc
static void ShroudCharFree(char *src)
{
   if (src != NULL) {
     free(src);
   }
}
##### end char_free c_source

##### start char_free cxx_source

// helper char_free
// Release memory allocated by ShroudCharAlloc
static void ShroudCharFree(char *src)
{
   if (src != NULL) {
     std::free(src);
   }
}
##### end char_free cxx_source

---------- char_len_trim ----------
{
    "fmtname": "ShroudCharLenTrim"
}

##### start char_len_trim source

// helper char_len_trim
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
static int ShroudCharLenTrim(const char *src, int nsrc) {
    int i;

    for (i = nsrc - 1; i >= 0; i--) {
        if (src[i] != ' ') {
            break;
        }
    }

    return i + 1;
}

##### end char_len_trim source

---------- copy_array ----------
{
    "c_include": [
        "<string.h>",
        "<stddef.h>"
    ],
    "cxx_include": [
        "<cstring>",
        "<cstddef>"
    ],
    "dependent_helpers": [
        "array_context"
    ],
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudCopyArray",
        "fnamefunc": "{C_prefix}SHROUD_{hname}"
    },
    "fmtname": "LIB_ShroudCopyArray",
    "name": "copy_array",
    "scope": "cwrap_impl"
}

##### start copy_array source

// helper copy_array
// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
// Called from Fortran.
void LIB_ShroudCopyArray(LIB_SHROUD_array *data, void *c_var, 
    size_t c_var_size)
{
    const void *cxx_var = data->base_addr;
    int n = c_var_size < data->size ? c_var_size : data->size;
    n *= data->elem_len;
    std::memcpy(c_var, cxx_var, n);
}
##### end copy_array source

---------- copy_string ----------
{
    "cxx_include": [
        "<cstring>",
        "<cstddef>"
    ],
    "dependent_helpers": [
        "array_context"
    ],
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudCopyString",
        "fnamefunc": "{C_prefix}SHROUD_copy_string"
    },
    "fmtname": "LIB_ShroudCopyString",
    "name": "copy_string",
    "scope": "cwrap_impl"
}

##### start copy_string source

// helper copy_string
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void LIB_ShroudCopyString(LIB_SHROUD_array *data, char *c_var,
    size_t c_var_len) {
    const void *cxx_var = data->base_addr;
    size_t n = c_var_len;
    if (data->elem_len < n) n = data->elem_len;
    std::memcpy(c_var, cxx_var, n);
}

##### end copy_string source

---------- create_from_PyObject_vector_double ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_double\t(PyObject *obj,\t const char *name,\t std::vector<double> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_double"
}

##### start create_from_PyObject_vector_double cxx_source

// helper create_from_PyObject_vector_double
// Convert obj into an array of type double
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_double(PyObject *obj,
    const char *name, std::vector<double> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        double cvalue = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be double", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_double cxx_source

---------- create_from_PyObject_vector_double_complex ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_double_complex\t(PyObject *obj,\t const char *name,\t std::vector<std::complex<double>> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_double_complex"
}

##### start create_from_PyObject_vector_double_complex cxx_source

// helper create_from_PyObject_vector_double_complex
// Convert obj into an array of type std::complex<double>
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_double_complex
    (PyObject *obj, const char *name,
    std::vector<std::complex<double>> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        Py_complex cvalue = PyComplex_AsCComplex(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be double complex", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue.real + cvalue.imag * I);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_double_complex cxx_source

---------- create_from_PyObject_vector_float ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_float\t(PyObject *obj,\t const char *name,\t std::vector<float> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_float"
}

##### start create_from_PyObject_vector_float cxx_source

// helper create_from_PyObject_vector_float
// Convert obj into an array of type float
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_float(PyObject *obj,
    const char *name, std::vector<float> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        float cvalue = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be float", name, (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_float cxx_source

---------- create_from_PyObject_vector_float_complex ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_float_complex\t(PyObject *obj,\t const char *name,\t std::vector<std::complex<float>> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_float_complex"
}

##### start create_from_PyObject_vector_float_complex cxx_source

// helper create_from_PyObject_vector_float_complex
// Convert obj into an array of type std::complex<float>
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_float_complex
    (PyObject *obj, const char *name,
    std::vector<std::complex<float>> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        Py_complex cvalue = PyComplex_AsCComplex(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be float complex", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue.real + cvalue.imag * I);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_float_complex cxx_source

---------- create_from_PyObject_vector_int ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_int\t(PyObject *obj,\t const char *name,\t std::vector<int> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_int"
}

##### start create_from_PyObject_vector_int cxx_source

// helper create_from_PyObject_vector_int
// Convert obj into an array of type int
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_int(PyObject *obj,
    const char *name, std::vector<int> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        int cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int", name, (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_int cxx_source

---------- create_from_PyObject_vector_int16_t ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_int16_t\t(PyObject *obj,\t const char *name,\t std::vector<int16_t> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_int16_t"
}

##### start create_from_PyObject_vector_int16_t cxx_source

// helper create_from_PyObject_vector_int16_t
// Convert obj into an array of type int16_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_int16_t(PyObject *obj,
    const char *name, std::vector<int16_t> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        int16_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int16_t", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_int16_t cxx_source

---------- create_from_PyObject_vector_int32_t ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_int32_t\t(PyObject *obj,\t const char *name,\t std::vector<int32_t> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_int32_t"
}

##### start create_from_PyObject_vector_int32_t cxx_source

// helper create_from_PyObject_vector_int32_t
// Convert obj into an array of type int32_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_int32_t(PyObject *obj,
    const char *name, std::vector<int32_t> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        int32_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int32_t", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_int32_t cxx_source

---------- create_from_PyObject_vector_int64_t ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_int64_t\t(PyObject *obj,\t const char *name,\t std::vector<int64_t> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_int64_t"
}

##### start create_from_PyObject_vector_int64_t cxx_source

// helper create_from_PyObject_vector_int64_t
// Convert obj into an array of type int64_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_int64_t(PyObject *obj,
    const char *name, std::vector<int64_t> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        int64_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int64_t", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_int64_t cxx_source

---------- create_from_PyObject_vector_int8_t ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_int8_t\t(PyObject *obj,\t const char *name,\t std::vector<int8_t> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_int8_t"
}

##### start create_from_PyObject_vector_int8_t cxx_source

// helper create_from_PyObject_vector_int8_t
// Convert obj into an array of type int8_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_int8_t(PyObject *obj,
    const char *name, std::vector<int8_t> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        int8_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int8_t", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_int8_t cxx_source

---------- create_from_PyObject_vector_long ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_long\t(PyObject *obj,\t const char *name,\t std::vector<long> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_long"
}

##### start create_from_PyObject_vector_long cxx_source

// helper create_from_PyObject_vector_long
// Convert obj into an array of type long
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_long(PyObject *obj,
    const char *name, std::vector<long> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        long cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be long", name, (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_long cxx_source

---------- create_from_PyObject_vector_long_long ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_long_long\t(PyObject *obj,\t const char *name,\t std::vector<long long> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_long_long"
}

##### start create_from_PyObject_vector_long_long cxx_source

// helper create_from_PyObject_vector_long_long
// Convert obj into an array of type long long
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_long_long(PyObject *obj,
    const char *name, std::vector<long long> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        long long cvalue = XXXPy_get;
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be long long", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_long_long cxx_source

---------- create_from_PyObject_vector_short ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_short\t(PyObject *obj,\t const char *name,\t std::vector<short> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_short"
}

##### start create_from_PyObject_vector_short cxx_source

// helper create_from_PyObject_vector_short
// Convert obj into an array of type short
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_short(PyObject *obj,
    const char *name, std::vector<short> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        short cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be short", name, (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_short cxx_source

---------- create_from_PyObject_vector_size_t ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_size_t\t(PyObject *obj,\t const char *name,\t std::vector<size_t> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_size_t"
}

##### start create_from_PyObject_vector_size_t cxx_source

// helper create_from_PyObject_vector_size_t
// Convert obj into an array of type size_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_size_t(PyObject *obj,
    const char *name, std::vector<size_t> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        size_t cvalue = XXXPy_get;
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be size_t", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_size_t cxx_source

---------- create_from_PyObject_vector_uint16_t ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_uint16_t\t(PyObject *obj,\t const char *name,\t std::vector<uint16_t> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_uint16_t"
}

##### start create_from_PyObject_vector_uint16_t cxx_source

// helper create_from_PyObject_vector_uint16_t
// Convert obj into an array of type uint16_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_uint16_t(PyObject *obj,
    const char *name, std::vector<uint16_t> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        uint16_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be uint16_t", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_uint16_t cxx_source

---------- create_from_PyObject_vector_uint32_t ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_uint32_t\t(PyObject *obj,\t const char *name,\t std::vector<uint32_t> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_uint32_t"
}

##### start create_from_PyObject_vector_uint32_t cxx_source

// helper create_from_PyObject_vector_uint32_t
// Convert obj into an array of type uint32_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_uint32_t(PyObject *obj,
    const char *name, std::vector<uint32_t> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        uint32_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be uint32_t", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_uint32_t cxx_source

---------- create_from_PyObject_vector_uint64_t ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_uint64_t\t(PyObject *obj,\t const char *name,\t std::vector<uint64_t> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_uint64_t"
}

##### start create_from_PyObject_vector_uint64_t cxx_source

// helper create_from_PyObject_vector_uint64_t
// Convert obj into an array of type uint64_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_uint64_t(PyObject *obj,
    const char *name, std::vector<uint64_t> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        uint64_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be uint64_t", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_uint64_t cxx_source

---------- create_from_PyObject_vector_uint8_t ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_uint8_t\t(PyObject *obj,\t const char *name,\t std::vector<uint8_t> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_uint8_t"
}

##### start create_from_PyObject_vector_uint8_t cxx_source

// helper create_from_PyObject_vector_uint8_t
// Convert obj into an array of type uint8_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_uint8_t(PyObject *obj,
    const char *name, std::vector<uint8_t> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        uint8_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be uint8_t", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_uint8_t cxx_source

---------- create_from_PyObject_vector_unsigned_int ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_unsigned_int\t(PyObject *obj,\t const char *name,\t std::vector<unsigned int> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_unsigned_int"
}

##### start create_from_PyObject_vector_unsigned_int cxx_source

// helper create_from_PyObject_vector_unsigned_int
// Convert obj into an array of type unsigned int
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_unsigned_int
    (PyObject *obj, const char *name, std::vector<unsigned int> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        unsigned int cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be unsigned int", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_unsigned_int cxx_source

---------- create_from_PyObject_vector_unsigned_long ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_unsigned_long\t(PyObject *obj,\t const char *name,\t std::vector<unsigned long> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_unsigned_long"
}

##### start create_from_PyObject_vector_unsigned_long cxx_source

// helper create_from_PyObject_vector_unsigned_long
// Convert obj into an array of type unsigned long
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_unsigned_long
    (PyObject *obj, const char *name, std::vector<unsigned long> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        unsigned long cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be unsigned long", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_unsigned_long cxx_source

---------- create_from_PyObject_vector_unsigned_long_long ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_unsigned_long_long\t(PyObject *obj,\t const char *name,\t std::vector<unsigned long long> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_unsigned_long_long"
}

##### start create_from_PyObject_vector_unsigned_long_long cxx_source

// helper create_from_PyObject_vector_unsigned_long_long
// Convert obj into an array of type unsigned long long
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_unsigned_long_long
    (PyObject *obj, const char *name,
    std::vector<unsigned long long> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        unsigned long long cvalue = XXXPy_get;
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be unsigned long long",
                name, (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_unsigned_long_long cxx_source

---------- create_from_PyObject_vector_unsigned_short ----------
{
    "cxx_proto": "int SHROUD_create_from_PyObject_vector_unsigned_short\t(PyObject *obj,\t const char *name,\t std::vector<unsigned short> & in);",
    "fmtname": "SHROUD_create_from_PyObject_vector_unsigned_short"
}

##### start create_from_PyObject_vector_unsigned_short cxx_source

// helper create_from_PyObject_vector_unsigned_short
// Convert obj into an array of type unsigned short
// Return -1 on error.
static int SHROUD_create_from_PyObject_vector_unsigned_short
    (PyObject *obj, const char *name, std::vector<unsigned short> & in)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        unsigned short cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be unsigned short", name,
                (int) i);
            return -1;
        }
        in.push_back(cvalue);
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_unsigned_short cxx_source

---------- fill_from_PyObject_char ----------
{
    "c_include": [
        "<string.h>"
    ],
    "cxx_include": [
        "<cstring>"
    ],
    "dependent_helpers": [
        "get_from_object_char"
    ],
    "fmtname": "SHROUD_fill_from_PyObject_char",
    "proto": "int SHROUD_fill_from_PyObject_char\t(PyObject *obj,\t const char *name,\t char *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_char source

// helper fill_from_PyObject_char
// Fill existing char array from PyObject.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_char(PyObject *obj,
    const char *name, char *in, Py_ssize_t insize)
{
    LIB_SHROUD_converter_value value;
    int i = SHROUD_get_from_object_char(obj, &value);
    if (i == 0) {
        Py_DECREF(obj);
        return -1;
    }
    if (value.data == nullptr) {
        in[0] = '\0';
    } else {
        std::strncpy(in, static_cast<char *>(value.data), insize);
        Py_DECREF(value.dataobj);
    }
    return 0;
}
##### end fill_from_PyObject_char source

---------- fill_from_PyObject_double_complex_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_double_complex_list",
    "proto": "int SHROUD_fill_from_PyObject_double_complex_list\t(PyObject *obj,\t const char *name,\t double complex *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_double_complex_list source

// helper fill_from_PyObject_double_complex_list
// Fill double complex array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_double_complex_list(PyObject *obj,
    const char *name, double complex *in, Py_ssize_t insize)
{
    Py_complex cvalue = PyComplex_AsCComplex(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue.real + cvalue.imag * I;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyComplex_AsCComplex(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be double complex", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue.real + cvalue.imag * I;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_double_complex_list source

---------- fill_from_PyObject_double_complex_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_double_complex_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_double_complex_numpy\t(PyObject *obj,\t const char *name,\t double complex *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_double_complex_numpy source

// helper fill_from_PyObject_double_complex_numpy
// Fill double complex array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_double_complex_numpy(PyObject *obj,
    const char *name, double complex *in, Py_ssize_t insize)
{
    Py_complex cvalue = PyComplex_AsCComplex(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue.real + cvalue.imag * I;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of double complex",
            name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    double complex *data = static_cast<double complex *>
        (PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_double_complex_numpy source

---------- fill_from_PyObject_double_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_double_list",
    "proto": "int SHROUD_fill_from_PyObject_double_list\t(PyObject *obj,\t const char *name,\t double *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_double_list source

// helper fill_from_PyObject_double_list
// Fill double array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_double_list(PyObject *obj,
    const char *name, double *in, Py_ssize_t insize)
{
    double cvalue = PyFloat_AsDouble(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be double", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_double_list source

---------- fill_from_PyObject_double_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_double_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_double_numpy\t(PyObject *obj,\t const char *name,\t double *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_double_numpy source

// helper fill_from_PyObject_double_numpy
// Fill double array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_double_numpy(PyObject *obj,
    const char *name, double *in, Py_ssize_t insize)
{
    double cvalue = PyFloat_AsDouble(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of double", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    double *data = static_cast<double *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_double_numpy source

---------- fill_from_PyObject_float_complex_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_float_complex_list",
    "proto": "int SHROUD_fill_from_PyObject_float_complex_list\t(PyObject *obj,\t const char *name,\t float complex *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_float_complex_list source

// helper fill_from_PyObject_float_complex_list
// Fill float complex array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_float_complex_list(PyObject *obj,
    const char *name, float complex *in, Py_ssize_t insize)
{
    Py_complex cvalue = PyComplex_AsCComplex(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue.real + cvalue.imag * I;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyComplex_AsCComplex(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be float complex", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue.real + cvalue.imag * I;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_float_complex_list source

---------- fill_from_PyObject_float_complex_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_float_complex_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_float_complex_numpy\t(PyObject *obj,\t const char *name,\t float complex *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_float_complex_numpy source

// helper fill_from_PyObject_float_complex_numpy
// Fill float complex array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_float_complex_numpy(PyObject *obj,
    const char *name, float complex *in, Py_ssize_t insize)
{
    Py_complex cvalue = PyComplex_AsCComplex(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue.real + cvalue.imag * I;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of float complex", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    float complex *data = static_cast<float complex *>
        (PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_float_complex_numpy source

---------- fill_from_PyObject_float_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_float_list",
    "proto": "int SHROUD_fill_from_PyObject_float_list\t(PyObject *obj,\t const char *name,\t float *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_float_list source

// helper fill_from_PyObject_float_list
// Fill float array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_float_list(PyObject *obj,
    const char *name, float *in, Py_ssize_t insize)
{
    float cvalue = PyFloat_AsDouble(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be float", name, (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_float_list source

---------- fill_from_PyObject_float_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_float_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_float_numpy\t(PyObject *obj,\t const char *name,\t float *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_float_numpy source

// helper fill_from_PyObject_float_numpy
// Fill float array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_float_numpy(PyObject *obj,
    const char *name, float *in, Py_ssize_t insize)
{
    float cvalue = PyFloat_AsDouble(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_FLOAT,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of float", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    float *data = static_cast<float *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_float_numpy source

---------- fill_from_PyObject_int16_t_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_int16_t_list",
    "proto": "int SHROUD_fill_from_PyObject_int16_t_list\t(PyObject *obj,\t const char *name,\t int16_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_int16_t_list source

// helper fill_from_PyObject_int16_t_list
// Fill int16_t array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int16_t_list(PyObject *obj,
    const char *name, int16_t *in, Py_ssize_t insize)
{
    int16_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int16_t", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_int16_t_list source

---------- fill_from_PyObject_int16_t_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_int16_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_int16_t_numpy\t(PyObject *obj,\t const char *name,\t int16_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_int16_t_numpy source

// helper fill_from_PyObject_int16_t_numpy
// Fill int16_t array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int16_t_numpy(PyObject *obj,
    const char *name, int16_t *in, Py_ssize_t insize)
{
    int16_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT16,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of int16_t", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    int16_t *data = static_cast<int16_t *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_int16_t_numpy source

---------- fill_from_PyObject_int32_t_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_int32_t_list",
    "proto": "int SHROUD_fill_from_PyObject_int32_t_list\t(PyObject *obj,\t const char *name,\t int32_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_int32_t_list source

// helper fill_from_PyObject_int32_t_list
// Fill int32_t array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int32_t_list(PyObject *obj,
    const char *name, int32_t *in, Py_ssize_t insize)
{
    int32_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int32_t", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_int32_t_list source

---------- fill_from_PyObject_int32_t_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_int32_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_int32_t_numpy\t(PyObject *obj,\t const char *name,\t int32_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_int32_t_numpy source

// helper fill_from_PyObject_int32_t_numpy
// Fill int32_t array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int32_t_numpy(PyObject *obj,
    const char *name, int32_t *in, Py_ssize_t insize)
{
    int32_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT32,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of int32_t", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    int32_t *data = static_cast<int32_t *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_int32_t_numpy source

---------- fill_from_PyObject_int64_t_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_int64_t_list",
    "proto": "int SHROUD_fill_from_PyObject_int64_t_list\t(PyObject *obj,\t const char *name,\t int64_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_int64_t_list source

// helper fill_from_PyObject_int64_t_list
// Fill int64_t array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int64_t_list(PyObject *obj,
    const char *name, int64_t *in, Py_ssize_t insize)
{
    int64_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int64_t", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_int64_t_list source

---------- fill_from_PyObject_int64_t_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_int64_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_int64_t_numpy\t(PyObject *obj,\t const char *name,\t int64_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_int64_t_numpy source

// helper fill_from_PyObject_int64_t_numpy
// Fill int64_t array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int64_t_numpy(PyObject *obj,
    const char *name, int64_t *in, Py_ssize_t insize)
{
    int64_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT64,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of int64_t", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    int64_t *data = static_cast<int64_t *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_int64_t_numpy source

---------- fill_from_PyObject_int8_t_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_int8_t_list",
    "proto": "int SHROUD_fill_from_PyObject_int8_t_list\t(PyObject *obj,\t const char *name,\t int8_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_int8_t_list source

// helper fill_from_PyObject_int8_t_list
// Fill int8_t array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int8_t_list(PyObject *obj,
    const char *name, int8_t *in, Py_ssize_t insize)
{
    int8_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int8_t", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_int8_t_list source

---------- fill_from_PyObject_int8_t_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_int8_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_int8_t_numpy\t(PyObject *obj,\t const char *name,\t int8_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_int8_t_numpy source

// helper fill_from_PyObject_int8_t_numpy
// Fill int8_t array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int8_t_numpy(PyObject *obj,
    const char *name, int8_t *in, Py_ssize_t insize)
{
    int8_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT8,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of int8_t", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    int8_t *data = static_cast<int8_t *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_int8_t_numpy source

---------- fill_from_PyObject_int_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_int_list",
    "proto": "int SHROUD_fill_from_PyObject_int_list\t(PyObject *obj,\t const char *name,\t int *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_int_list source

// helper fill_from_PyObject_int_list
// Fill int array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int_list(PyObject *obj,
    const char *name, int *in, Py_ssize_t insize)
{
    int cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int", name, (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_int_list source

---------- fill_from_PyObject_int_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_int_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_int_numpy\t(PyObject *obj,\t const char *name,\t int *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_int_numpy source

// helper fill_from_PyObject_int_numpy
// Fill int array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int_numpy(PyObject *obj,
    const char *name, int *in, Py_ssize_t insize)
{
    int cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of int", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    int *data = static_cast<int *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_int_numpy source

---------- fill_from_PyObject_long_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_long_list",
    "proto": "int SHROUD_fill_from_PyObject_long_list\t(PyObject *obj,\t const char *name,\t long *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_long_list source

// helper fill_from_PyObject_long_list
// Fill long array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_long_list(PyObject *obj,
    const char *name, long *in, Py_ssize_t insize)
{
    long cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be long", name, (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_long_list source

---------- fill_from_PyObject_long_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_long_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_long_numpy\t(PyObject *obj,\t const char *name,\t long *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_long_numpy source

// helper fill_from_PyObject_long_numpy
// Fill long array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_long_numpy(PyObject *obj,
    const char *name, long *in, Py_ssize_t insize)
{
    long cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_LONG,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of long", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    long *data = static_cast<long *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_long_numpy source

---------- fill_from_PyObject_short_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_short_list",
    "proto": "int SHROUD_fill_from_PyObject_short_list\t(PyObject *obj,\t const char *name,\t short *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_short_list source

// helper fill_from_PyObject_short_list
// Fill short array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_short_list(PyObject *obj,
    const char *name, short *in, Py_ssize_t insize)
{
    short cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be short", name, (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_short_list source

---------- fill_from_PyObject_short_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_short_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_short_numpy\t(PyObject *obj,\t const char *name,\t short *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_short_numpy source

// helper fill_from_PyObject_short_numpy
// Fill short array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_short_numpy(PyObject *obj,
    const char *name, short *in, Py_ssize_t insize)
{
    short cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_SHORT,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of short", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    short *data = static_cast<short *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_short_numpy source

---------- fill_from_PyObject_uint16_t_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_uint16_t_list",
    "proto": "int SHROUD_fill_from_PyObject_uint16_t_list\t(PyObject *obj,\t const char *name,\t uint16_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_uint16_t_list source

// helper fill_from_PyObject_uint16_t_list
// Fill uint16_t array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint16_t_list(PyObject *obj,
    const char *name, uint16_t *in, Py_ssize_t insize)
{
    uint16_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint16_t", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_uint16_t_list source

---------- fill_from_PyObject_uint16_t_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_uint16_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_uint16_t_numpy\t(PyObject *obj,\t const char *name,\t uint16_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_uint16_t_numpy source

// helper fill_from_PyObject_uint16_t_numpy
// Fill uint16_t array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint16_t_numpy(PyObject *obj,
    const char *name, uint16_t *in, Py_ssize_t insize)
{
    uint16_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_UINT16,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of uint16_t", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    uint16_t *data = static_cast<uint16_t *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_uint16_t_numpy source

---------- fill_from_PyObject_uint32_t_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_uint32_t_list",
    "proto": "int SHROUD_fill_from_PyObject_uint32_t_list\t(PyObject *obj,\t const char *name,\t uint32_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_uint32_t_list source

// helper fill_from_PyObject_uint32_t_list
// Fill uint32_t array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint32_t_list(PyObject *obj,
    const char *name, uint32_t *in, Py_ssize_t insize)
{
    uint32_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint32_t", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_uint32_t_list source

---------- fill_from_PyObject_uint32_t_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_uint32_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_uint32_t_numpy\t(PyObject *obj,\t const char *name,\t uint32_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_uint32_t_numpy source

// helper fill_from_PyObject_uint32_t_numpy
// Fill uint32_t array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint32_t_numpy(PyObject *obj,
    const char *name, uint32_t *in, Py_ssize_t insize)
{
    uint32_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_UINT32,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of uint32_t", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    uint32_t *data = static_cast<uint32_t *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_uint32_t_numpy source

---------- fill_from_PyObject_uint64_t_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_uint64_t_list",
    "proto": "int SHROUD_fill_from_PyObject_uint64_t_list\t(PyObject *obj,\t const char *name,\t uint64_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_uint64_t_list source

// helper fill_from_PyObject_uint64_t_list
// Fill uint64_t array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint64_t_list(PyObject *obj,
    const char *name, uint64_t *in, Py_ssize_t insize)
{
    uint64_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint64_t", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_uint64_t_list source

---------- fill_from_PyObject_uint64_t_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_uint64_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_uint64_t_numpy\t(PyObject *obj,\t const char *name,\t uint64_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_uint64_t_numpy source

// helper fill_from_PyObject_uint64_t_numpy
// Fill uint64_t array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint64_t_numpy(PyObject *obj,
    const char *name, uint64_t *in, Py_ssize_t insize)
{
    uint64_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_UINT64,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of uint64_t", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    uint64_t *data = static_cast<uint64_t *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_uint64_t_numpy source

---------- fill_from_PyObject_uint8_t_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_uint8_t_list",
    "proto": "int SHROUD_fill_from_PyObject_uint8_t_list\t(PyObject *obj,\t const char *name,\t uint8_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_uint8_t_list source

// helper fill_from_PyObject_uint8_t_list
// Fill uint8_t array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint8_t_list(PyObject *obj,
    const char *name, uint8_t *in, Py_ssize_t insize)
{
    uint8_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint8_t", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_uint8_t_list source

---------- fill_from_PyObject_uint8_t_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_uint8_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_uint8_t_numpy\t(PyObject *obj,\t const char *name,\t uint8_t *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_uint8_t_numpy source

// helper fill_from_PyObject_uint8_t_numpy
// Fill uint8_t array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint8_t_numpy(PyObject *obj,
    const char *name, uint8_t *in, Py_ssize_t insize)
{
    uint8_t cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_UINT8,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of uint8_t", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    uint8_t *data = static_cast<uint8_t *>(PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_uint8_t_numpy source

---------- fill_from_PyObject_unsigned_int_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_unsigned_int_list",
    "proto": "int SHROUD_fill_from_PyObject_unsigned_int_list\t(PyObject *obj,\t const char *name,\t unsigned int *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_unsigned_int_list source

// helper fill_from_PyObject_unsigned_int_list
// Fill unsigned int array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_int_list(PyObject *obj,
    const char *name, unsigned int *in, Py_ssize_t insize)
{
    unsigned int cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned int", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_unsigned_int_list source

---------- fill_from_PyObject_unsigned_int_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_unsigned_int_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_unsigned_int_numpy\t(PyObject *obj,\t const char *name,\t unsigned int *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_unsigned_int_numpy source

// helper fill_from_PyObject_unsigned_int_numpy
// Fill unsigned int array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_int_numpy(PyObject *obj,
    const char *name, unsigned int *in, Py_ssize_t insize)
{
    unsigned int cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of unsigned int", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    unsigned int *data = static_cast<unsigned int *>
        (PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_unsigned_int_numpy source

---------- fill_from_PyObject_unsigned_long_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_unsigned_long_list",
    "proto": "int SHROUD_fill_from_PyObject_unsigned_long_list\t(PyObject *obj,\t const char *name,\t unsigned long *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_unsigned_long_list source

// helper fill_from_PyObject_unsigned_long_list
// Fill unsigned long array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_long_list(PyObject *obj,
    const char *name, unsigned long *in, Py_ssize_t insize)
{
    unsigned long cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned long", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_unsigned_long_list source

---------- fill_from_PyObject_unsigned_long_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_unsigned_long_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_unsigned_long_numpy\t(PyObject *obj,\t const char *name,\t unsigned long *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_unsigned_long_numpy source

// helper fill_from_PyObject_unsigned_long_numpy
// Fill unsigned long array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_long_numpy(PyObject *obj,
    const char *name, unsigned long *in, Py_ssize_t insize)
{
    unsigned long cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_LONG,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of unsigned long", name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    unsigned long *data = static_cast<unsigned long *>
        (PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_unsigned_long_numpy source

---------- fill_from_PyObject_unsigned_short_list ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_unsigned_short_list",
    "proto": "int SHROUD_fill_from_PyObject_unsigned_short_list\t(PyObject *obj,\t const char *name,\t unsigned short *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_unsigned_short_list source

// helper fill_from_PyObject_unsigned_short_list
// Fill unsigned short array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_short_list(PyObject *obj,
    const char *name, unsigned short *in, Py_ssize_t insize)
{
    unsigned short cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    // Look for sequence.
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned short", name,
                (int) i);
            return -1;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_unsigned_short_list source

---------- fill_from_PyObject_unsigned_short_numpy ----------
{
    "fmtname": "SHROUD_fill_from_PyObject_unsigned_short_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_fill_from_PyObject_unsigned_short_numpy\t(PyObject *obj,\t const char *name,\t unsigned short *in,\t Py_ssize_t insize);"
}

##### start fill_from_PyObject_unsigned_short_numpy source

// helper fill_from_PyObject_unsigned_short_numpy
// Fill unsigned short array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_short_numpy(PyObject *obj,
    const char *name, unsigned short *in, Py_ssize_t insize)
{
    unsigned short cvalue = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = cvalue;
        }
        return 0;
    }
    PyErr_Clear();

    PyObject *array = PyArray_FROM_OTF(obj, NPY_SHORT,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_Format(PyExc_TypeError,
            "argument '%s' must be a 1-D array of unsigned short",
            name);
        return -1;
    }
    PyArrayObject *pyarray = reinterpret_cast<PyArrayObject *>(array);

    unsigned short *data = static_cast<unsigned short *>
        (PyArray_DATA(pyarray));
    npy_intp size = PyArray_SIZE(pyarray);
    if (size > insize) {
        size = insize;
    }
    for (Py_ssize_t i = 0; i < size; ++i) {
        in[i] = data[i];
    }
    Py_DECREF(pyarray);
    return 0;
}
##### end fill_from_PyObject_unsigned_short_numpy source

---------- get_from_object_char ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_char",
    "proto": "int SHROUD_get_from_object_char\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_char source

// helper get_from_object_char
// Converter from PyObject to char *.
// The returned status will be 1 for a successful conversion
// and 0 if the conversion has failed.
// value.obj is unused.
// value.dataobj - object which holds the data.
// If same as obj argument, its refcount is incremented.
// value.data is owned by value.dataobj and must be copied to be preserved.
// Caller must use Py_XDECREF(value.dataobj).
static int SHROUD_get_from_object_char(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    size_t size = 0;
    char *out;
    if (PyUnicode_Check(obj)) {
#if PY_MAJOR_VERSION >= 3
        PyObject *strobj = PyUnicode_AsUTF8String(obj);
        out = PyBytes_AS_STRING(strobj);
        size = PyBytes_GET_SIZE(strobj);
        value->dataobj = strobj;  // steal reference
#else
        PyObject *strobj = PyUnicode_AsUTF8String(obj);
        out = PyString_AsString(strobj);
        size = PyString_Size(obj);
        value->dataobj = strobj;  // steal reference
#endif
#if PY_MAJOR_VERSION < 3
    } else if (PyString_Check(obj)) {
        out = PyString_AsString(obj);
        size = PyString_Size(obj);
        value->dataobj = obj;
        Py_INCREF(obj);
#endif
    } else if (PyBytes_Check(obj)) {
        out = PyBytes_AS_STRING(obj);
        size = PyBytes_GET_SIZE(obj);
        value->dataobj = obj;
        Py_INCREF(obj);
    } else if (PyByteArray_Check(obj)) {
        out = PyByteArray_AS_STRING(obj);
        size = PyByteArray_GET_SIZE(obj);
        value->dataobj = obj;
        Py_INCREF(obj);
    } else if (obj == Py_None) {
        out = NULL;
        size = 0;
        value->dataobj = NULL;
    } else {
        PyErr_Format(PyExc_TypeError,
            "argument should be string or None, not %.200s",
            Py_TYPE(obj)->tp_name);
        return 0;
    }
    value->obj = nullptr;
    value->data = out;
    value->size = size;
    return 1;
}

##### end get_from_object_char source

---------- get_from_object_char_list ----------
{
    "dependent_helpers": [
        "get_from_object_char"
    ],
    "fmtname": "SHROUD_get_from_object_char"
}

---------- get_from_object_char_numpy ----------
{
    "dependent_helpers": [
        "get_from_object_char"
    ],
    "fmtname": "SHROUD_get_from_object_char"
}

---------- get_from_object_charptr ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "get_from_object_char"
    ],
    "fmtname": "SHROUD_get_from_object_charptr",
    "proto": "int SHROUD_get_from_object_charptr\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_charptr source


// helper FREE_get_from_object_charptr
static void FREE_get_from_object_charptr(PyObject *obj)
{
    char **in = static_cast<char **>
        (PyCapsule_GetPointer(obj, nullptr));
    if (in == nullptr)
        return;
    size_t *size = static_cast<size_t *>(PyCapsule_GetContext(obj));
    if (size == nullptr)
        return;
    for (size_t i=0; i < *size; ++i) {
        if (in[i] == nullptr)
            continue;
        std::free(in[i]);
    }
    std::free(in);
    std::free(size);
}

// helper get_from_object_charptr
// Convert obj into an array of char * (i.e. char **).
static int SHROUD_get_from_object_charptr(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    char **in = static_cast<char **>(std::calloc(size, sizeof(char *)));
    PyObject *dataobj = PyCapsule_New(in, nullptr, FREE_get_from_object_charptr);
    size_t *size_context = static_cast<size_t *>
        (malloc(sizeof(size_t)));
    *size_context = size;
    int ierr = PyCapsule_SetContext(dataobj, size_context);
    // XXX - check error
    LIB_SHROUD_converter_value itemvalue = {NULL, NULL, NULL, NULL, 0};
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        ierr = SHROUD_get_from_object_char(item, &itemvalue);
        if (ierr == 0) {
            Py_XDECREF(itemvalue.dataobj);
            Py_DECREF(dataobj);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be string", value->name,
                (int) i);
            return 0;
        }
        if (itemvalue.data != nullptr) {
            in[i] = strdup(static_cast<char *>(itemvalue.data));
        }
        Py_XDECREF(itemvalue.dataobj);
    }
    Py_DECREF(seq);

    value->obj = nullptr;
    value->dataobj = dataobj;
    value->data = in;
    value->size = size;
    return 1;
}
##### end get_from_object_charptr source

---------- get_from_object_charptr_list ----------
{
    "dependent_helpers": [
        "get_from_object_charptr"
    ],
    "fmtname": "SHROUD_get_from_object_charptr"
}

---------- get_from_object_charptr_numpy ----------
{
    "dependent_helpers": [
        "get_from_object_charptr"
    ],
    "fmtname": "SHROUD_get_from_object_charptr"
}

---------- get_from_object_double_complex_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_double_complex_list",
    "proto": "int SHROUD_get_from_object_double_complex_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_double_complex_list source

// helper get_from_object_double_complex_list
// Convert list of PyObject to array of double complex.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_double_complex_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    double complex *in = static_cast<double complex *>
        (std::malloc(size * sizeof(double complex)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        double complex cvalue = PyComplex_AsCComplex(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be double complex",
                value->name, (int) i);
            return 0;
        }
        in[i] = cvalue.real + cvalue.imag * I;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<double complex *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_double_complex_list source

---------- get_from_object_double_complex_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_double_complex_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_double_complex_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_double_complex_numpy source

// helper get_from_object_double_complex_numpy
// Convert PyObject to double complex pointer.
static int SHROUD_get_from_object_double_complex_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of double complex");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_double_complex_numpy source

---------- get_from_object_double_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_double_list",
    "proto": "int SHROUD_get_from_object_double_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_double_list source

// helper get_from_object_double_list
// Convert list of PyObject to array of double.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_double_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    double *in = static_cast<double *>
        (std::malloc(size * sizeof(double)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        double cvalue = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be double", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<double *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_double_list source

---------- get_from_object_double_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_double_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_double_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_double_numpy source

// helper get_from_object_double_numpy
// Convert PyObject to double pointer.
static int SHROUD_get_from_object_double_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of double");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_double_numpy source

---------- get_from_object_float_complex_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_float_complex_list",
    "proto": "int SHROUD_get_from_object_float_complex_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_float_complex_list source

// helper get_from_object_float_complex_list
// Convert list of PyObject to array of float complex.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_float_complex_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    float complex *in = static_cast<float complex *>
        (std::malloc(size * sizeof(float complex)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        float complex cvalue = PyComplex_AsCComplex(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be float complex",
                value->name, (int) i);
            return 0;
        }
        in[i] = cvalue.real + cvalue.imag * I;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<float complex *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_float_complex_list source

---------- get_from_object_float_complex_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_float_complex_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_float_complex_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_float_complex_numpy source

// helper get_from_object_float_complex_numpy
// Convert PyObject to float complex pointer.
static int SHROUD_get_from_object_float_complex_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of float complex");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_float_complex_numpy source

---------- get_from_object_float_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_float_list",
    "proto": "int SHROUD_get_from_object_float_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_float_list source

// helper get_from_object_float_list
// Convert list of PyObject to array of float.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_float_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    float *in = static_cast<float *>(std::malloc(size * sizeof(float)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        float cvalue = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be float", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<float *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_float_list source

---------- get_from_object_float_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_float_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_float_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_float_numpy source

// helper get_from_object_float_numpy
// Convert PyObject to float pointer.
static int SHROUD_get_from_object_float_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_FLOAT,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of float");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_float_numpy source

---------- get_from_object_int16_t_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_int16_t_list",
    "proto": "int SHROUD_get_from_object_int16_t_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_int16_t_list source

// helper get_from_object_int16_t_list
// Convert list of PyObject to array of int16_t.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_int16_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    int16_t *in = static_cast<int16_t *>
        (std::malloc(size * sizeof(int16_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        int16_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int16_t", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<int16_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_int16_t_list source

---------- get_from_object_int16_t_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_int16_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_int16_t_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_int16_t_numpy source

// helper get_from_object_int16_t_numpy
// Convert PyObject to int16_t pointer.
static int SHROUD_get_from_object_int16_t_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT16,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of int16_t");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_int16_t_numpy source

---------- get_from_object_int32_t_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_int32_t_list",
    "proto": "int SHROUD_get_from_object_int32_t_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_int32_t_list source

// helper get_from_object_int32_t_list
// Convert list of PyObject to array of int32_t.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_int32_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    int32_t *in = static_cast<int32_t *>
        (std::malloc(size * sizeof(int32_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        int32_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int32_t", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<int32_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_int32_t_list source

---------- get_from_object_int32_t_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_int32_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_int32_t_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_int32_t_numpy source

// helper get_from_object_int32_t_numpy
// Convert PyObject to int32_t pointer.
static int SHROUD_get_from_object_int32_t_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT32,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of int32_t");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_int32_t_numpy source

---------- get_from_object_int64_t_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_int64_t_list",
    "proto": "int SHROUD_get_from_object_int64_t_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_int64_t_list source

// helper get_from_object_int64_t_list
// Convert list of PyObject to array of int64_t.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_int64_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    int64_t *in = static_cast<int64_t *>
        (std::malloc(size * sizeof(int64_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        int64_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int64_t", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<int64_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_int64_t_list source

---------- get_from_object_int64_t_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_int64_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_int64_t_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_int64_t_numpy source

// helper get_from_object_int64_t_numpy
// Convert PyObject to int64_t pointer.
static int SHROUD_get_from_object_int64_t_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT64,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of int64_t");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_int64_t_numpy source

---------- get_from_object_int8_t_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_int8_t_list",
    "proto": "int SHROUD_get_from_object_int8_t_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_int8_t_list source

// helper get_from_object_int8_t_list
// Convert list of PyObject to array of int8_t.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_int8_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    int8_t *in = static_cast<int8_t *>
        (std::malloc(size * sizeof(int8_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        int8_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int8_t", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<int8_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_int8_t_list source

---------- get_from_object_int8_t_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_int8_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_int8_t_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_int8_t_numpy source

// helper get_from_object_int8_t_numpy
// Convert PyObject to int8_t pointer.
static int SHROUD_get_from_object_int8_t_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT8,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of int8_t");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_int8_t_numpy source

---------- get_from_object_int_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_int_list",
    "proto": "int SHROUD_get_from_object_int_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_int_list source

// helper get_from_object_int_list
// Convert list of PyObject to array of int.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_int_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    int *in = static_cast<int *>(std::malloc(size * sizeof(int)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        int cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<int *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_int_list source

---------- get_from_object_int_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_int_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_int_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_int_numpy source

// helper get_from_object_int_numpy
// Convert PyObject to int pointer.
static int SHROUD_get_from_object_int_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError, "must be a 1-D array of int");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_int_numpy source

---------- get_from_object_long_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_long_list",
    "proto": "int SHROUD_get_from_object_long_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_long_list source

// helper get_from_object_long_list
// Convert list of PyObject to array of long.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_long_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    long *in = static_cast<long *>(std::malloc(size * sizeof(long)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        long cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be long", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<long *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_long_list source

---------- get_from_object_long_long_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_long_long_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_long_long_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_long_long_numpy source

// helper get_from_object_long_long_numpy
// Convert PyObject to long long pointer.
static int SHROUD_get_from_object_long_long_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_LONGLONG,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of long long");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_long_long_numpy source

---------- get_from_object_long_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_long_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_long_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_long_numpy source

// helper get_from_object_long_numpy
// Convert PyObject to long pointer.
static int SHROUD_get_from_object_long_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_LONG,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of long");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_long_numpy source

---------- get_from_object_short_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_short_list",
    "proto": "int SHROUD_get_from_object_short_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_short_list source

// helper get_from_object_short_list
// Convert list of PyObject to array of short.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_short_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    short *in = static_cast<short *>(std::malloc(size * sizeof(short)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        short cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be short", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<short *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_short_list source

---------- get_from_object_short_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_short_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_short_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_short_numpy source

// helper get_from_object_short_numpy
// Convert PyObject to short pointer.
static int SHROUD_get_from_object_short_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_SHORT,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of short");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_short_numpy source

---------- get_from_object_size_t_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_size_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_size_t_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_size_t_numpy source

// helper get_from_object_size_t_numpy
// Convert PyObject to size_t pointer.
static int SHROUD_get_from_object_size_t_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, None, NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of size_t");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_size_t_numpy source

---------- get_from_object_uint16_t_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_uint16_t_list",
    "proto": "int SHROUD_get_from_object_uint16_t_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_uint16_t_list source

// helper get_from_object_uint16_t_list
// Convert list of PyObject to array of uint16_t.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_uint16_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    uint16_t *in = static_cast<uint16_t *>
        (std::malloc(size * sizeof(uint16_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        uint16_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint16_t", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<uint16_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_uint16_t_list source

---------- get_from_object_uint16_t_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_uint16_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_uint16_t_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_uint16_t_numpy source

// helper get_from_object_uint16_t_numpy
// Convert PyObject to uint16_t pointer.
static int SHROUD_get_from_object_uint16_t_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_UINT16,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of uint16_t");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_uint16_t_numpy source

---------- get_from_object_uint32_t_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_uint32_t_list",
    "proto": "int SHROUD_get_from_object_uint32_t_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_uint32_t_list source

// helper get_from_object_uint32_t_list
// Convert list of PyObject to array of uint32_t.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_uint32_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    uint32_t *in = static_cast<uint32_t *>
        (std::malloc(size * sizeof(uint32_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        uint32_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint32_t", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<uint32_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_uint32_t_list source

---------- get_from_object_uint32_t_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_uint32_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_uint32_t_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_uint32_t_numpy source

// helper get_from_object_uint32_t_numpy
// Convert PyObject to uint32_t pointer.
static int SHROUD_get_from_object_uint32_t_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_UINT32,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of uint32_t");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_uint32_t_numpy source

---------- get_from_object_uint64_t_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_uint64_t_list",
    "proto": "int SHROUD_get_from_object_uint64_t_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_uint64_t_list source

// helper get_from_object_uint64_t_list
// Convert list of PyObject to array of uint64_t.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_uint64_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    uint64_t *in = static_cast<uint64_t *>
        (std::malloc(size * sizeof(uint64_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        uint64_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint64_t", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<uint64_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_uint64_t_list source

---------- get_from_object_uint64_t_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_uint64_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_uint64_t_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_uint64_t_numpy source

// helper get_from_object_uint64_t_numpy
// Convert PyObject to uint64_t pointer.
static int SHROUD_get_from_object_uint64_t_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_UINT64,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of uint64_t");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_uint64_t_numpy source

---------- get_from_object_uint8_t_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_uint8_t_list",
    "proto": "int SHROUD_get_from_object_uint8_t_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_uint8_t_list source

// helper get_from_object_uint8_t_list
// Convert list of PyObject to array of uint8_t.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_uint8_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    uint8_t *in = static_cast<uint8_t *>
        (std::malloc(size * sizeof(uint8_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        uint8_t cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint8_t", value->name,
                (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<uint8_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_uint8_t_list source

---------- get_from_object_uint8_t_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_uint8_t_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_uint8_t_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_uint8_t_numpy source

// helper get_from_object_uint8_t_numpy
// Convert PyObject to uint8_t pointer.
static int SHROUD_get_from_object_uint8_t_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_UINT8,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of uint8_t");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_uint8_t_numpy source

---------- get_from_object_unsigned_int_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_unsigned_int_list",
    "proto": "int SHROUD_get_from_object_unsigned_int_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_unsigned_int_list source

// helper get_from_object_unsigned_int_list
// Convert list of PyObject to array of unsigned int.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_unsigned_int_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    unsigned int *in = static_cast<unsigned int *>
        (std::malloc(size * sizeof(unsigned int)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        unsigned int cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned int",
                value->name, (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<unsigned int *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_unsigned_int_list source

---------- get_from_object_unsigned_int_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_unsigned_int_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_unsigned_int_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_unsigned_int_numpy source

// helper get_from_object_unsigned_int_numpy
// Convert PyObject to unsigned int pointer.
static int SHROUD_get_from_object_unsigned_int_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_INT,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of unsigned int");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_unsigned_int_numpy source

---------- get_from_object_unsigned_long_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_unsigned_long_list",
    "proto": "int SHROUD_get_from_object_unsigned_long_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_unsigned_long_list source

// helper get_from_object_unsigned_long_list
// Convert list of PyObject to array of unsigned long.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_unsigned_long_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    unsigned long *in = static_cast<unsigned long *>
        (std::malloc(size * sizeof(unsigned long)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        unsigned long cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned long",
                value->name, (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<unsigned long *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_unsigned_long_list source

---------- get_from_object_unsigned_long_long_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_unsigned_long_long_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_unsigned_long_long_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_unsigned_long_long_numpy source

// helper get_from_object_unsigned_long_long_numpy
// Convert PyObject to unsigned long long pointer.
static int SHROUD_get_from_object_unsigned_long_long_numpy
    (PyObject *obj, LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_LONGLONG,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of unsigned long long");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_unsigned_long_long_numpy source

---------- get_from_object_unsigned_long_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_unsigned_long_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_unsigned_long_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_unsigned_long_numpy source

// helper get_from_object_unsigned_long_numpy
// Convert PyObject to unsigned long pointer.
static int SHROUD_get_from_object_unsigned_long_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_LONG,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of unsigned long");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_unsigned_long_numpy source

---------- get_from_object_unsigned_short_list ----------
{
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "dependent_helpers": [
        "PY_converter_type",
        "py_capsule_dtor"
    ],
    "fmtname": "SHROUD_get_from_object_unsigned_short_list",
    "proto": "int SHROUD_get_from_object_unsigned_short_list\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_unsigned_short_list source

// helper get_from_object_unsigned_short_list
// Convert list of PyObject to array of unsigned short.
// Return 0 on error, 1 on success.
// Set Python exception on error.
static int SHROUD_get_from_object_unsigned_short_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            value->name);
        return 0;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    unsigned short *in = static_cast<unsigned short *>
        (std::malloc(size * sizeof(unsigned short)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        unsigned short cvalue = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned short",
                value->name, (int) i);
            return 0;
        }
        in[i] = cvalue;
    }
    Py_DECREF(seq);

    value->obj = nullptr;  // Do not save list object.
    value->dataobj = PyCapsule_New(in, nullptr, FREE_py_capsule_dtor);
    value->data = static_cast<unsigned short *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_unsigned_short_list source

---------- get_from_object_unsigned_short_numpy ----------
{
    "dependent_helpers": [
        "PY_converter_type"
    ],
    "fmtname": "SHROUD_get_from_object_unsigned_short_numpy",
    "need_numpy": true,
    "proto": "int SHROUD_get_from_object_unsigned_short_numpy\t(PyObject *obj,\t LIB_SHROUD_converter_value *value);"
}

##### start get_from_object_unsigned_short_numpy source

// helper get_from_object_unsigned_short_numpy
// Convert PyObject to unsigned short pointer.
static int SHROUD_get_from_object_unsigned_short_numpy(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    PyObject *array = PyArray_FROM_OTF(obj, NPY_SHORT,
        NPY_ARRAY_IN_ARRAY);
    if (array == nullptr) {
        PyErr_SetString(PyExc_ValueError,
            "must be a 1-D array of unsigned short");
        return 0;
    }
    value->obj = array;
    value->dataobj = nullptr;
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_unsigned_short_numpy source

---------- py_capsule_dtor ----------
{
    "fmtname": "FREE_py_capsule_dtor"
}

##### start py_capsule_dtor source

// helper py_capsule_dtor
// Release memory in PyCapsule.
// Used with native arrays.
static void FREE_py_capsule_dtor(PyObject *obj)
{
    void *in = PyCapsule_GetPointer(obj, nullptr);
    if (in != nullptr) {
        std::free(in);
    }
}
##### end py_capsule_dtor source

---------- size_CFI ----------
{
    "c_include": [
        "<stddef.h>"
    ],
    "cxx_include": [
        "<cstddef>"
    ]
}

##### start size_CFI source

// helper size_CFI
// Compute number of items in CFI_cdesc_t
size_t ShroudSizeCFI(CFI_cdesc_t *desc)
{
    size_t nitems = 1;
    for (int i = 0; i < desc->rank; i++) {
        nitems *= desc->dim[i].extent;
    }
    return nitems;
}
##### end size_CFI source

---------- string_to_cdesc ----------
{
    "cxx_include": [
        "<cstring>",
        "<cstddef>"
    ],
    "dependent_helpers": [
        "array_context"
    ],
    "fmtdict": {
        "cnamefunc": "ShroudStringToCdesc"
    },
    "fmtname": "ShroudStringToCdesc",
    "name": "string_to_cdesc"
}

##### start string_to_cdesc source

// helper string_to_cdesc
// Save std::string metadata into array to allow Fortran to access values.
// CHARACTER(len=elem_size) src
static void ShroudStringToCdesc(LIB_SHROUD_array *cdesc,
    const std::string * src)
{
    if (src->empty()) {
        cdesc->base_addr = NULL;
        cdesc->elem_len = 0;
    } else {
        cdesc->base_addr = const_cast<char *>(src->data());
        cdesc->elem_len = src->length();
    }
    cdesc->size = 1;
    cdesc->rank = 0;  // scalar
}
##### end string_to_cdesc source

---------- to_PyList_char ----------
{
    "fmtname": "SHROUD_to_PyList_char",
    "proto": "PyObject *SHROUD_to_PyList_char\t(char * *in, size_t size);"
}

##### start to_PyList_char source

// helper to_PyList_char
// Convert char * pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_char(char * *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyString_FromString(in[i]));
    }
    return out;
}
##### end to_PyList_char source

---------- to_PyList_double ----------
{
    "fmtname": "SHROUD_to_PyList_double",
    "proto": "PyObject *SHROUD_to_PyList_double\t(const double *in, size_t size);"
}

##### start to_PyList_double source

// helper to_PyList_double
// Convert double pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_double(const double *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
    return out;
}
##### end to_PyList_double source

---------- to_PyList_double_complex ----------
{
    "fmtname": "SHROUD_to_PyList_double_complex",
    "proto": "PyObject *SHROUD_to_PyList_double_complex\t(const double complex *in, size_t size);"
}

##### start to_PyList_double_complex source

// helper to_PyList_double_complex
// Convert double complex pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_double_complex
    (const double complex *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyComplex_FromDoubles(
            creal(in[i]), cimag(in[i])));
    }
    return out;
}
##### end to_PyList_double_complex source

---------- to_PyList_float ----------
{
    "fmtname": "SHROUD_to_PyList_float",
    "proto": "PyObject *SHROUD_to_PyList_float\t(const float *in, size_t size);"
}

##### start to_PyList_float source

// helper to_PyList_float
// Convert float pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_float(const float *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
    return out;
}
##### end to_PyList_float source

---------- to_PyList_float_complex ----------
{
    "fmtname": "SHROUD_to_PyList_float_complex",
    "proto": "PyObject *SHROUD_to_PyList_float_complex\t(const float complex *in, size_t size);"
}

##### start to_PyList_float_complex source

// helper to_PyList_float_complex
// Convert float complex pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_float_complex
    (const float complex *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyComplex_FromDoubles(
            creal(in[i]), cimag(in[i])));
    }
    return out;
}
##### end to_PyList_float_complex source

---------- to_PyList_int ----------
{
    "fmtname": "SHROUD_to_PyList_int",
    "proto": "PyObject *SHROUD_to_PyList_int\t(const int *in, size_t size);"
}

##### start to_PyList_int source

// helper to_PyList_int
// Convert int pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_int(const int *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_int source

---------- to_PyList_int16_t ----------
{
    "fmtname": "SHROUD_to_PyList_int16_t",
    "proto": "PyObject *SHROUD_to_PyList_int16_t\t(const int16_t *in, size_t size);"
}

##### start to_PyList_int16_t source

// helper to_PyList_int16_t
// Convert int16_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_int16_t
    (const int16_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_int16_t source

---------- to_PyList_int32_t ----------
{
    "fmtname": "SHROUD_to_PyList_int32_t",
    "proto": "PyObject *SHROUD_to_PyList_int32_t\t(const int32_t *in, size_t size);"
}

##### start to_PyList_int32_t source

// helper to_PyList_int32_t
// Convert int32_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_int32_t
    (const int32_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_int32_t source

---------- to_PyList_int64_t ----------
{
    "fmtname": "SHROUD_to_PyList_int64_t",
    "proto": "PyObject *SHROUD_to_PyList_int64_t\t(const int64_t *in, size_t size);"
}

##### start to_PyList_int64_t source

// helper to_PyList_int64_t
// Convert int64_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_int64_t
    (const int64_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_int64_t source

---------- to_PyList_int8_t ----------
{
    "fmtname": "SHROUD_to_PyList_int8_t",
    "proto": "PyObject *SHROUD_to_PyList_int8_t\t(const int8_t *in, size_t size);"
}

##### start to_PyList_int8_t source

// helper to_PyList_int8_t
// Convert int8_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_int8_t(const int8_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_int8_t source

---------- to_PyList_long ----------
{
    "fmtname": "SHROUD_to_PyList_long",
    "proto": "PyObject *SHROUD_to_PyList_long\t(const long *in, size_t size);"
}

##### start to_PyList_long source

// helper to_PyList_long
// Convert long pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_long(const long *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_long source

---------- to_PyList_short ----------
{
    "fmtname": "SHROUD_to_PyList_short",
    "proto": "PyObject *SHROUD_to_PyList_short\t(const short *in, size_t size);"
}

##### start to_PyList_short source

// helper to_PyList_short
// Convert short pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_short(const short *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_short source

---------- to_PyList_size_t ----------
{
    "fmtname": "SHROUD_to_PyList_size_t",
    "proto": "PyObject *SHROUD_to_PyList_size_t\t(const size_t *in, size_t size);"
}

##### start to_PyList_size_t source

// helper to_PyList_size_t
// Convert size_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_size_t(const size_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromSize_t(in[i]));
    }
    return out;
}
##### end to_PyList_size_t source

---------- to_PyList_uint16_t ----------
{
    "fmtname": "SHROUD_to_PyList_uint16_t",
    "proto": "PyObject *SHROUD_to_PyList_uint16_t\t(const uint16_t *in, size_t size);"
}

##### start to_PyList_uint16_t source

// helper to_PyList_uint16_t
// Convert uint16_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_uint16_t
    (const uint16_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_uint16_t source

---------- to_PyList_uint32_t ----------
{
    "fmtname": "SHROUD_to_PyList_uint32_t",
    "proto": "PyObject *SHROUD_to_PyList_uint32_t\t(const uint32_t *in, size_t size);"
}

##### start to_PyList_uint32_t source

// helper to_PyList_uint32_t
// Convert uint32_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_uint32_t
    (const uint32_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_uint32_t source

---------- to_PyList_uint64_t ----------
{
    "fmtname": "SHROUD_to_PyList_uint64_t",
    "proto": "PyObject *SHROUD_to_PyList_uint64_t\t(const uint64_t *in, size_t size);"
}

##### start to_PyList_uint64_t source

// helper to_PyList_uint64_t
// Convert uint64_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_uint64_t
    (const uint64_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_uint64_t source

---------- to_PyList_uint8_t ----------
{
    "fmtname": "SHROUD_to_PyList_uint8_t",
    "proto": "PyObject *SHROUD_to_PyList_uint8_t\t(const uint8_t *in, size_t size);"
}

##### start to_PyList_uint8_t source

// helper to_PyList_uint8_t
// Convert uint8_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_uint8_t
    (const uint8_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_uint8_t source

---------- to_PyList_unsigned_int ----------
{
    "fmtname": "SHROUD_to_PyList_unsigned_int",
    "proto": "PyObject *SHROUD_to_PyList_unsigned_int\t(const unsigned int *in, size_t size);"
}

##### start to_PyList_unsigned_int source

// helper to_PyList_unsigned_int
// Convert unsigned int pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_unsigned_int
    (const unsigned int *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_unsigned_int source

---------- to_PyList_unsigned_long ----------
{
    "fmtname": "SHROUD_to_PyList_unsigned_long",
    "proto": "PyObject *SHROUD_to_PyList_unsigned_long\t(const unsigned long *in, size_t size);"
}

##### start to_PyList_unsigned_long source

// helper to_PyList_unsigned_long
// Convert unsigned long pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_unsigned_long
    (const unsigned long *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_unsigned_long source

---------- to_PyList_unsigned_short ----------
{
    "fmtname": "SHROUD_to_PyList_unsigned_short",
    "proto": "PyObject *SHROUD_to_PyList_unsigned_short\t(const unsigned short *in, size_t size);"
}

##### start to_PyList_unsigned_short source

// helper to_PyList_unsigned_short
// Convert unsigned short pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_unsigned_short
    (const unsigned short *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_unsigned_short source

---------- to_PyList_vector_double ----------
{
    "fmtname": "SHROUD_to_PyList_vector_double",
    "proto": "PyObject *SHROUD_to_PyList_vector_double\t(std::vector<double> & in);"
}

##### start to_PyList_vector_double source

// helper to_PyList_vector_double
static PyObject *SHROUD_to_PyList_vector_double
    (std::vector<double> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
    return out;
}
##### end to_PyList_vector_double source

---------- to_PyList_vector_double_complex ----------
{
    "fmtname": "SHROUD_to_PyList_vector_double_complex",
    "proto": "PyObject *SHROUD_to_PyList_vector_double_complex\t(std::vector<double complex> & in);"
}

##### start to_PyList_vector_double_complex source

// helper to_PyList_vector_double_complex
static PyObject *SHROUD_to_PyList_vector_double_complex
    (std::vector<double complex> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyComplex_FromDoubles(
            creal(in[i]), cimag(in[i])));
    }
    return out;
}
##### end to_PyList_vector_double_complex source

---------- to_PyList_vector_float ----------
{
    "fmtname": "SHROUD_to_PyList_vector_float",
    "proto": "PyObject *SHROUD_to_PyList_vector_float\t(std::vector<float> & in);"
}

##### start to_PyList_vector_float source

// helper to_PyList_vector_float
static PyObject *SHROUD_to_PyList_vector_float(std::vector<float> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
    return out;
}
##### end to_PyList_vector_float source

---------- to_PyList_vector_float_complex ----------
{
    "fmtname": "SHROUD_to_PyList_vector_float_complex",
    "proto": "PyObject *SHROUD_to_PyList_vector_float_complex\t(std::vector<float complex> & in);"
}

##### start to_PyList_vector_float_complex source

// helper to_PyList_vector_float_complex
static PyObject *SHROUD_to_PyList_vector_float_complex
    (std::vector<float complex> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyComplex_FromDoubles(
            creal(in[i]), cimag(in[i])));
    }
    return out;
}
##### end to_PyList_vector_float_complex source

---------- to_PyList_vector_int ----------
{
    "fmtname": "SHROUD_to_PyList_vector_int",
    "proto": "PyObject *SHROUD_to_PyList_vector_int\t(std::vector<int> & in);"
}

##### start to_PyList_vector_int source

// helper to_PyList_vector_int
static PyObject *SHROUD_to_PyList_vector_int(std::vector<int> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_int source

---------- to_PyList_vector_int16_t ----------
{
    "fmtname": "SHROUD_to_PyList_vector_int16_t",
    "proto": "PyObject *SHROUD_to_PyList_vector_int16_t\t(std::vector<int16_t> & in);"
}

##### start to_PyList_vector_int16_t source

// helper to_PyList_vector_int16_t
static PyObject *SHROUD_to_PyList_vector_int16_t
    (std::vector<int16_t> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_int16_t source

---------- to_PyList_vector_int32_t ----------
{
    "fmtname": "SHROUD_to_PyList_vector_int32_t",
    "proto": "PyObject *SHROUD_to_PyList_vector_int32_t\t(std::vector<int32_t> & in);"
}

##### start to_PyList_vector_int32_t source

// helper to_PyList_vector_int32_t
static PyObject *SHROUD_to_PyList_vector_int32_t
    (std::vector<int32_t> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_int32_t source

---------- to_PyList_vector_int64_t ----------
{
    "fmtname": "SHROUD_to_PyList_vector_int64_t",
    "proto": "PyObject *SHROUD_to_PyList_vector_int64_t\t(std::vector<int64_t> & in);"
}

##### start to_PyList_vector_int64_t source

// helper to_PyList_vector_int64_t
static PyObject *SHROUD_to_PyList_vector_int64_t
    (std::vector<int64_t> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_int64_t source

---------- to_PyList_vector_int8_t ----------
{
    "fmtname": "SHROUD_to_PyList_vector_int8_t",
    "proto": "PyObject *SHROUD_to_PyList_vector_int8_t\t(std::vector<int8_t> & in);"
}

##### start to_PyList_vector_int8_t source

// helper to_PyList_vector_int8_t
static PyObject *SHROUD_to_PyList_vector_int8_t
    (std::vector<int8_t> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_int8_t source

---------- to_PyList_vector_long ----------
{
    "fmtname": "SHROUD_to_PyList_vector_long",
    "proto": "PyObject *SHROUD_to_PyList_vector_long\t(std::vector<long> & in);"
}

##### start to_PyList_vector_long source

// helper to_PyList_vector_long
static PyObject *SHROUD_to_PyList_vector_long(std::vector<long> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_long source

---------- to_PyList_vector_long_long ----------
{
    "fmtname": "SHROUD_to_PyList_vector_long_long",
    "proto": "PyObject *SHROUD_to_PyList_vector_long_long\t(std::vector<long long> & in);"
}

##### start to_PyList_vector_long_long source

// helper to_PyList_vector_long_long
static PyObject *SHROUD_to_PyList_vector_long_long
    (std::vector<long long> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, XXXPy_ctor);
    }
    return out;
}
##### end to_PyList_vector_long_long source

---------- to_PyList_vector_short ----------
{
    "fmtname": "SHROUD_to_PyList_vector_short",
    "proto": "PyObject *SHROUD_to_PyList_vector_short\t(std::vector<short> & in);"
}

##### start to_PyList_vector_short source

// helper to_PyList_vector_short
static PyObject *SHROUD_to_PyList_vector_short(std::vector<short> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_short source

---------- to_PyList_vector_size_t ----------
{
    "fmtname": "SHROUD_to_PyList_vector_size_t",
    "proto": "PyObject *SHROUD_to_PyList_vector_size_t\t(std::vector<size_t> & in);"
}

##### start to_PyList_vector_size_t source

// helper to_PyList_vector_size_t
static PyObject *SHROUD_to_PyList_vector_size_t
    (std::vector<size_t> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromSize_t(in[i]));
    }
    return out;
}
##### end to_PyList_vector_size_t source

---------- to_PyList_vector_uint16_t ----------
{
    "fmtname": "SHROUD_to_PyList_vector_uint16_t",
    "proto": "PyObject *SHROUD_to_PyList_vector_uint16_t\t(std::vector<uint16_t> & in);"
}

##### start to_PyList_vector_uint16_t source

// helper to_PyList_vector_uint16_t
static PyObject *SHROUD_to_PyList_vector_uint16_t
    (std::vector<uint16_t> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_uint16_t source

---------- to_PyList_vector_uint32_t ----------
{
    "fmtname": "SHROUD_to_PyList_vector_uint32_t",
    "proto": "PyObject *SHROUD_to_PyList_vector_uint32_t\t(std::vector<uint32_t> & in);"
}

##### start to_PyList_vector_uint32_t source

// helper to_PyList_vector_uint32_t
static PyObject *SHROUD_to_PyList_vector_uint32_t
    (std::vector<uint32_t> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_uint32_t source

---------- to_PyList_vector_uint64_t ----------
{
    "fmtname": "SHROUD_to_PyList_vector_uint64_t",
    "proto": "PyObject *SHROUD_to_PyList_vector_uint64_t\t(std::vector<uint64_t> & in);"
}

##### start to_PyList_vector_uint64_t source

// helper to_PyList_vector_uint64_t
static PyObject *SHROUD_to_PyList_vector_uint64_t
    (std::vector<uint64_t> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_uint64_t source

---------- to_PyList_vector_uint8_t ----------
{
    "fmtname": "SHROUD_to_PyList_vector_uint8_t",
    "proto": "PyObject *SHROUD_to_PyList_vector_uint8_t\t(std::vector<uint8_t> & in);"
}

##### start to_PyList_vector_uint8_t source

// helper to_PyList_vector_uint8_t
static PyObject *SHROUD_to_PyList_vector_uint8_t
    (std::vector<uint8_t> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_uint8_t source

---------- to_PyList_vector_unsigned_int ----------
{
    "fmtname": "SHROUD_to_PyList_vector_unsigned_int",
    "proto": "PyObject *SHROUD_to_PyList_vector_unsigned_int\t(std::vector<unsigned int> & in);"
}

##### start to_PyList_vector_unsigned_int source

// helper to_PyList_vector_unsigned_int
static PyObject *SHROUD_to_PyList_vector_unsigned_int
    (std::vector<unsigned int> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_unsigned_int source

---------- to_PyList_vector_unsigned_long ----------
{
    "fmtname": "SHROUD_to_PyList_vector_unsigned_long",
    "proto": "PyObject *SHROUD_to_PyList_vector_unsigned_long\t(std::vector<unsigned long> & in);"
}

##### start to_PyList_vector_unsigned_long source

// helper to_PyList_vector_unsigned_long
static PyObject *SHROUD_to_PyList_vector_unsigned_long
    (std::vector<unsigned long> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_unsigned_long source

---------- to_PyList_vector_unsigned_long_long ----------
{
    "fmtname": "SHROUD_to_PyList_vector_unsigned_long_long",
    "proto": "PyObject *SHROUD_to_PyList_vector_unsigned_long_long\t(std::vector<unsigned long long> & in);"
}

##### start to_PyList_vector_unsigned_long_long source

// helper to_PyList_vector_unsigned_long_long
static PyObject *SHROUD_to_PyList_vector_unsigned_long_long
    (std::vector<unsigned long long> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, XXXPy_ctor);
    }
    return out;
}
##### end to_PyList_vector_unsigned_long_long source

---------- to_PyList_vector_unsigned_short ----------
{
    "fmtname": "SHROUD_to_PyList_vector_unsigned_short",
    "proto": "PyObject *SHROUD_to_PyList_vector_unsigned_short\t(std::vector<unsigned short> & in);"
}

##### start to_PyList_vector_unsigned_short source

// helper to_PyList_vector_unsigned_short
static PyObject *SHROUD_to_PyList_vector_unsigned_short
    (std::vector<unsigned short> & in)
{
    size_t size = in.size();
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_vector_unsigned_short source

---------- type_defines ----------
{
    "scope": "cwrap_include"
}

##### start type_defines source

/* helper type_defines */
/* Shroud type defines */
#define SH_TYPE_SIGNED_CHAR 1
#define SH_TYPE_SHORT       2
#define SH_TYPE_INT         3
#define SH_TYPE_LONG        4
#define SH_TYPE_LONG_LONG   5
#define SH_TYPE_SIZE_T      6

#define SH_TYPE_UNSIGNED_SHORT       SH_TYPE_SHORT + 100
#define SH_TYPE_UNSIGNED_INT         SH_TYPE_INT + 100
#define SH_TYPE_UNSIGNED_LONG        SH_TYPE_LONG + 100
#define SH_TYPE_UNSIGNED_LONG_LONG   SH_TYPE_LONG_LONG + 100

#define SH_TYPE_INT8_T      7
#define SH_TYPE_INT16_T     8
#define SH_TYPE_INT32_T     9
#define SH_TYPE_INT64_T    10

#define SH_TYPE_UINT8_T    SH_TYPE_INT8_T + 100
#define SH_TYPE_UINT16_T   SH_TYPE_INT16_T + 100
#define SH_TYPE_UINT32_T   SH_TYPE_INT32_T + 100
#define SH_TYPE_UINT64_T   SH_TYPE_INT64_T + 100

/* least8 least16 least32 least64 */
/* fast8 fast16 fast32 fast64 */
/* intmax_t intptr_t ptrdiff_t */

#define SH_TYPE_FLOAT        22
#define SH_TYPE_DOUBLE       23
#define SH_TYPE_LONG_DOUBLE  24
#define SH_TYPE_FLOAT_COMPLEX       25
#define SH_TYPE_DOUBLE_COMPLEX      26
#define SH_TYPE_LONG_DOUBLE_COMPLEX 27

#define SH_TYPE_BOOL       28
#define SH_TYPE_CHAR       29
#define SH_TYPE_CPTR       30
#define SH_TYPE_STRUCT     31
#define SH_TYPE_OTHER      32
##### end type_defines source

---------- update_PyList_double ----------
{
    "proto": "void SHROUD_update_PyList_double\t(PyObject *out, double *in, size_t size);"
}

##### start update_PyList_double source

// helper update_PyList_double
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_double
    (PyObject *out, double *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
}
##### end update_PyList_double source

---------- update_PyList_double_complex ----------
{
    "proto": "void SHROUD_update_PyList_double_complex\t(PyObject *out, double complex *in, size_t size);"
}

##### start update_PyList_double_complex source

// helper update_PyList_double_complex
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_double_complex
    (PyObject *out, double complex *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyComplex_FromDoubles(
            creal(in[i]), cimag(in[i])));
    }
}
##### end update_PyList_double_complex source

---------- update_PyList_float ----------
{
    "proto": "void SHROUD_update_PyList_float\t(PyObject *out, float *in, size_t size);"
}

##### start update_PyList_float source

// helper update_PyList_float
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_float
    (PyObject *out, float *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
}
##### end update_PyList_float source

---------- update_PyList_float_complex ----------
{
    "proto": "void SHROUD_update_PyList_float_complex\t(PyObject *out, float complex *in, size_t size);"
}

##### start update_PyList_float_complex source

// helper update_PyList_float_complex
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_float_complex
    (PyObject *out, float complex *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyComplex_FromDoubles(
            creal(in[i]), cimag(in[i])));
    }
}
##### end update_PyList_float_complex source

---------- update_PyList_int ----------
{
    "proto": "void SHROUD_update_PyList_int\t(PyObject *out, int *in, size_t size);"
}

##### start update_PyList_int source

// helper update_PyList_int
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_int
    (PyObject *out, int *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_int source

---------- update_PyList_int16_t ----------
{
    "proto": "void SHROUD_update_PyList_int16_t\t(PyObject *out, int16_t *in, size_t size);"
}

##### start update_PyList_int16_t source

// helper update_PyList_int16_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_int16_t
    (PyObject *out, int16_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_int16_t source

---------- update_PyList_int32_t ----------
{
    "proto": "void SHROUD_update_PyList_int32_t\t(PyObject *out, int32_t *in, size_t size);"
}

##### start update_PyList_int32_t source

// helper update_PyList_int32_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_int32_t
    (PyObject *out, int32_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_int32_t source

---------- update_PyList_int64_t ----------
{
    "proto": "void SHROUD_update_PyList_int64_t\t(PyObject *out, int64_t *in, size_t size);"
}

##### start update_PyList_int64_t source

// helper update_PyList_int64_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_int64_t
    (PyObject *out, int64_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_int64_t source

---------- update_PyList_int8_t ----------
{
    "proto": "void SHROUD_update_PyList_int8_t\t(PyObject *out, int8_t *in, size_t size);"
}

##### start update_PyList_int8_t source

// helper update_PyList_int8_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_int8_t
    (PyObject *out, int8_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_int8_t source

---------- update_PyList_long ----------
{
    "proto": "void SHROUD_update_PyList_long\t(PyObject *out, long *in, size_t size);"
}

##### start update_PyList_long source

// helper update_PyList_long
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_long
    (PyObject *out, long *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_long source

---------- update_PyList_short ----------
{
    "proto": "void SHROUD_update_PyList_short\t(PyObject *out, short *in, size_t size);"
}

##### start update_PyList_short source

// helper update_PyList_short
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_short
    (PyObject *out, short *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_short source

---------- update_PyList_size_t ----------
{
    "proto": "void SHROUD_update_PyList_size_t\t(PyObject *out, size_t *in, size_t size);"
}

##### start update_PyList_size_t source

// helper update_PyList_size_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_size_t
    (PyObject *out, size_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromSize_t(in[i]));
    }
}
##### end update_PyList_size_t source

---------- update_PyList_uint16_t ----------
{
    "proto": "void SHROUD_update_PyList_uint16_t\t(PyObject *out, uint16_t *in, size_t size);"
}

##### start update_PyList_uint16_t source

// helper update_PyList_uint16_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_uint16_t
    (PyObject *out, uint16_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_uint16_t source

---------- update_PyList_uint32_t ----------
{
    "proto": "void SHROUD_update_PyList_uint32_t\t(PyObject *out, uint32_t *in, size_t size);"
}

##### start update_PyList_uint32_t source

// helper update_PyList_uint32_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_uint32_t
    (PyObject *out, uint32_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_uint32_t source

---------- update_PyList_uint64_t ----------
{
    "proto": "void SHROUD_update_PyList_uint64_t\t(PyObject *out, uint64_t *in, size_t size);"
}

##### start update_PyList_uint64_t source

// helper update_PyList_uint64_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_uint64_t
    (PyObject *out, uint64_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_uint64_t source

---------- update_PyList_uint8_t ----------
{
    "proto": "void SHROUD_update_PyList_uint8_t\t(PyObject *out, uint8_t *in, size_t size);"
}

##### start update_PyList_uint8_t source

// helper update_PyList_uint8_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_uint8_t
    (PyObject *out, uint8_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_uint8_t source

---------- update_PyList_unsigned_int ----------
{
    "proto": "void SHROUD_update_PyList_unsigned_int\t(PyObject *out, unsigned int *in, size_t size);"
}

##### start update_PyList_unsigned_int source

// helper update_PyList_unsigned_int
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_unsigned_int
    (PyObject *out, unsigned int *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_unsigned_int source

---------- update_PyList_unsigned_long ----------
{
    "proto": "void SHROUD_update_PyList_unsigned_long\t(PyObject *out, unsigned long *in, size_t size);"
}

##### start update_PyList_unsigned_long source

// helper update_PyList_unsigned_long
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_unsigned_long
    (PyObject *out, unsigned long *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_unsigned_long source

---------- update_PyList_unsigned_short ----------
{
    "proto": "void SHROUD_update_PyList_unsigned_short\t(PyObject *out, unsigned short *in, size_t size);"
}

##### start update_PyList_unsigned_short source

// helper update_PyList_unsigned_short
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_unsigned_short
    (PyObject *out, unsigned short *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_unsigned_short source

---------- update_PyList_vector_double ----------
{
    "fmtname": "SHROUD_update_PyList_vector_double",
    "proto": "void SHROUD_update_PyList_vector_double\t(PyObject *out, double *in, size_t size);"
}

##### start update_PyList_vector_double source

// helper update_PyList_vector_double
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_double
    (PyObject *out, double *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
}
##### end update_PyList_vector_double source

---------- update_PyList_vector_double_complex ----------
{
    "fmtname": "SHROUD_update_PyList_vector_double_complex",
    "proto": "void SHROUD_update_PyList_vector_double_complex\t(PyObject *out, double complex *in, size_t size);"
}

##### start update_PyList_vector_double_complex source

// helper update_PyList_vector_double_complex
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_double_complex
    (PyObject *out, double complex *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyComplex_FromDoubles(
            creal(in[i]), cimag(in[i])));
    }
}
##### end update_PyList_vector_double_complex source

---------- update_PyList_vector_float ----------
{
    "fmtname": "SHROUD_update_PyList_vector_float",
    "proto": "void SHROUD_update_PyList_vector_float\t(PyObject *out, float *in, size_t size);"
}

##### start update_PyList_vector_float source

// helper update_PyList_vector_float
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_float
    (PyObject *out, float *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
}
##### end update_PyList_vector_float source

---------- update_PyList_vector_float_complex ----------
{
    "fmtname": "SHROUD_update_PyList_vector_float_complex",
    "proto": "void SHROUD_update_PyList_vector_float_complex\t(PyObject *out, float complex *in, size_t size);"
}

##### start update_PyList_vector_float_complex source

// helper update_PyList_vector_float_complex
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_float_complex
    (PyObject *out, float complex *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyComplex_FromDoubles(
            creal(in[i]), cimag(in[i])));
    }
}
##### end update_PyList_vector_float_complex source

---------- update_PyList_vector_int ----------
{
    "fmtname": "SHROUD_update_PyList_vector_int",
    "proto": "void SHROUD_update_PyList_vector_int\t(PyObject *out, int *in, size_t size);"
}

##### start update_PyList_vector_int source

// helper update_PyList_vector_int
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_int
    (PyObject *out, int *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_int source

---------- update_PyList_vector_int16_t ----------
{
    "fmtname": "SHROUD_update_PyList_vector_int16_t",
    "proto": "void SHROUD_update_PyList_vector_int16_t\t(PyObject *out, int16_t *in, size_t size);"
}

##### start update_PyList_vector_int16_t source

// helper update_PyList_vector_int16_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_int16_t
    (PyObject *out, int16_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_int16_t source

---------- update_PyList_vector_int32_t ----------
{
    "fmtname": "SHROUD_update_PyList_vector_int32_t",
    "proto": "void SHROUD_update_PyList_vector_int32_t\t(PyObject *out, int32_t *in, size_t size);"
}

##### start update_PyList_vector_int32_t source

// helper update_PyList_vector_int32_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_int32_t
    (PyObject *out, int32_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_int32_t source

---------- update_PyList_vector_int64_t ----------
{
    "fmtname": "SHROUD_update_PyList_vector_int64_t",
    "proto": "void SHROUD_update_PyList_vector_int64_t\t(PyObject *out, int64_t *in, size_t size);"
}

##### start update_PyList_vector_int64_t source

// helper update_PyList_vector_int64_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_int64_t
    (PyObject *out, int64_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_int64_t source

---------- update_PyList_vector_int8_t ----------
{
    "fmtname": "SHROUD_update_PyList_vector_int8_t",
    "proto": "void SHROUD_update_PyList_vector_int8_t\t(PyObject *out, int8_t *in, size_t size);"
}

##### start update_PyList_vector_int8_t source

// helper update_PyList_vector_int8_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_int8_t
    (PyObject *out, int8_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_int8_t source

---------- update_PyList_vector_long ----------
{
    "fmtname": "SHROUD_update_PyList_vector_long",
    "proto": "void SHROUD_update_PyList_vector_long\t(PyObject *out, long *in, size_t size);"
}

##### start update_PyList_vector_long source

// helper update_PyList_vector_long
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_long
    (PyObject *out, long *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_long source

---------- update_PyList_vector_long_long ----------
{
    "fmtname": "SHROUD_update_PyList_vector_long_long",
    "proto": "void SHROUD_update_PyList_vector_long_long\t(PyObject *out, long long *in, size_t size);"
}

##### start update_PyList_vector_long_long source

// helper update_PyList_vector_long_long
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_long_long
    (PyObject *out, long long *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, XXXPy_ctor);
    }
}
##### end update_PyList_vector_long_long source

---------- update_PyList_vector_short ----------
{
    "fmtname": "SHROUD_update_PyList_vector_short",
    "proto": "void SHROUD_update_PyList_vector_short\t(PyObject *out, short *in, size_t size);"
}

##### start update_PyList_vector_short source

// helper update_PyList_vector_short
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_short
    (PyObject *out, short *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_short source

---------- update_PyList_vector_size_t ----------
{
    "fmtname": "SHROUD_update_PyList_vector_size_t",
    "proto": "void SHROUD_update_PyList_vector_size_t\t(PyObject *out, size_t *in, size_t size);"
}

##### start update_PyList_vector_size_t source

// helper update_PyList_vector_size_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_size_t
    (PyObject *out, size_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromSize_t(in[i]));
    }
}
##### end update_PyList_vector_size_t source

---------- update_PyList_vector_uint16_t ----------
{
    "fmtname": "SHROUD_update_PyList_vector_uint16_t",
    "proto": "void SHROUD_update_PyList_vector_uint16_t\t(PyObject *out, uint16_t *in, size_t size);"
}

##### start update_PyList_vector_uint16_t source

// helper update_PyList_vector_uint16_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_uint16_t
    (PyObject *out, uint16_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_uint16_t source

---------- update_PyList_vector_uint32_t ----------
{
    "fmtname": "SHROUD_update_PyList_vector_uint32_t",
    "proto": "void SHROUD_update_PyList_vector_uint32_t\t(PyObject *out, uint32_t *in, size_t size);"
}

##### start update_PyList_vector_uint32_t source

// helper update_PyList_vector_uint32_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_uint32_t
    (PyObject *out, uint32_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_uint32_t source

---------- update_PyList_vector_uint64_t ----------
{
    "fmtname": "SHROUD_update_PyList_vector_uint64_t",
    "proto": "void SHROUD_update_PyList_vector_uint64_t\t(PyObject *out, uint64_t *in, size_t size);"
}

##### start update_PyList_vector_uint64_t source

// helper update_PyList_vector_uint64_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_uint64_t
    (PyObject *out, uint64_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_uint64_t source

---------- update_PyList_vector_uint8_t ----------
{
    "fmtname": "SHROUD_update_PyList_vector_uint8_t",
    "proto": "void SHROUD_update_PyList_vector_uint8_t\t(PyObject *out, uint8_t *in, size_t size);"
}

##### start update_PyList_vector_uint8_t source

// helper update_PyList_vector_uint8_t
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_uint8_t
    (PyObject *out, uint8_t *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_uint8_t source

---------- update_PyList_vector_unsigned_int ----------
{
    "fmtname": "SHROUD_update_PyList_vector_unsigned_int",
    "proto": "void SHROUD_update_PyList_vector_unsigned_int\t(PyObject *out, unsigned int *in, size_t size);"
}

##### start update_PyList_vector_unsigned_int source

// helper update_PyList_vector_unsigned_int
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_unsigned_int
    (PyObject *out, unsigned int *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_unsigned_int source

---------- update_PyList_vector_unsigned_long ----------
{
    "fmtname": "SHROUD_update_PyList_vector_unsigned_long",
    "proto": "void SHROUD_update_PyList_vector_unsigned_long\t(PyObject *out, unsigned long *in, size_t size);"
}

##### start update_PyList_vector_unsigned_long source

// helper update_PyList_vector_unsigned_long
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_unsigned_long
    (PyObject *out, unsigned long *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_unsigned_long source

---------- update_PyList_vector_unsigned_long_long ----------
{
    "fmtname": "SHROUD_update_PyList_vector_unsigned_long_long",
    "proto": "void SHROUD_update_PyList_vector_unsigned_long_long\t(PyObject *out, unsigned long long *in, size_t size);"
}

##### start update_PyList_vector_unsigned_long_long source

// helper update_PyList_vector_unsigned_long_long
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_unsigned_long_long
    (PyObject *out, unsigned long long *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, XXXPy_ctor);
    }
}
##### end update_PyList_vector_unsigned_long_long source

---------- update_PyList_vector_unsigned_short ----------
{
    "fmtname": "SHROUD_update_PyList_vector_unsigned_short",
    "proto": "void SHROUD_update_PyList_vector_unsigned_short\t(PyObject *out, unsigned short *in, size_t size);"
}

##### start update_PyList_vector_unsigned_short source

// helper update_PyList_vector_unsigned_short
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
static void SHROUD_update_PyList_vector_unsigned_short
    (PyObject *out, unsigned short *in, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(out, i);
        Py_DECREF(item);
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
}
##### end update_PyList_vector_unsigned_short source

---------- vector_string_allocatable ----------
{
    "api": "c",
    "dependent_helpers": [
        "capsule_data_helper",
        "array_context",
        "vector_string_out"
    ],
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudVectorStringAllocatable",
        "cnameproto": "void {cnamefunc}({C_array_type} *dest, {C_capsule_data_type} *src)",
        "fnamefunc": "{C_prefix}SHROUD_vector_string_allocatable"
    },
    "fmtname": "LIB_ShroudVectorStringAllocatable",
    "name": "vector_string_allocatable",
    "proto": "void LIB_ShroudVectorStringAllocatable(LIB_SHROUD_array *dest, LIB_SHROUD_capsule_data *src);",
    "scope": "cwrap_impl"
}

##### start vector_string_allocatable source

// helper vector_string_allocatable
// Copy the std::vector<std::string> into Fortran array.
// Called by Fortran to deal with allocatable character.
// out is already blank filled.
void LIB_ShroudVectorStringAllocatable(LIB_SHROUD_array *dest, LIB_SHROUD_capsule_data *src)
{
    std::vector<std::string> *cxxvec =
        static_cast< std::vector<std::string> * >(src->addr);
    LIB_ShroudVectorStringOut(dest, *cxxvec);
}

##### end vector_string_allocatable source

---------- vector_string_out ----------
{
    "api": "cxx",
    "cxx_include": [
        "<cstring>",
        "<cstddef>"
    ],
    "dependent_helpers": [
        "array_context"
    ],
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudVectorStringOut",
        "cnamefunc_vector_string_out": "{cnamefunc}",
        "cnameproto": "void {cnamefunc}({C_array_type} *outdesc, std::vector<std::string> &in)",
        "fnamefunc": "{C_prefix}shroud_vector_string_out"
    },
    "fmtname": "LIB_ShroudVectorStringOut",
    "name": "vector_string_out",
    "proto": "void LIB_ShroudVectorStringOut(LIB_SHROUD_array *outdesc, std::vector<std::string> &in);",
    "proto_include": [
        "<string>",
        "<vector>"
    ],
    "scope": "cwrap_impl"
}

##### start vector_string_out source

// helper vector_string_out
// Copy the std::vector<std::string> into Fortran array argument.
// Called by C++.
void LIB_ShroudVectorStringOut(LIB_SHROUD_array *outdesc, std::vector<std::string> &in)
{
    size_t nvect = outdesc->size;
    size_t len = outdesc->elem_len;
    char *dest = static_cast<char *>(outdesc->base_addr);
    // Clear user memory
    std::memset(dest, ' ', nvect*len);

    // Copy into user memory
    nvect = std::min(nvect, in.size());
    //char *dest = static_cast<char *>(outdesc->cxx.addr);
    for (size_t i = 0; i < nvect; ++i) {
        std::memcpy(dest, in[i].data(), std::min(len, in[i].length()));
        dest += outdesc->elem_len;
    }
}

##### end vector_string_out source

---------- vector_string_out_len ----------
{
    "api": "cxx",
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudVectorStringOutSize",
        "cnameproto": "size_t {cnamefunc}(std::vector<std::string> &in)"
    },
    "fmtname": "LIB_ShroudVectorStringOutSize",
    "name": "vector_string_out_len",
    "proto": "size_t LIB_ShroudVectorStringOutSize(std::vector<std::string> &in);",
    "proto_include": [
        "<string>",
        "<vector>"
    ],
    "scope": "cwrap_impl"
}

##### start vector_string_out_len source

// helper vector_string_out_len
// Return the maximum string length in a std::vector<std::string>.
size_t LIB_ShroudVectorStringOutSize(std::vector<std::string> &in)
{
    size_t nvect = in.size();
    size_t len = 0;
    for (size_t i = 0; i < nvect; ++i) {
        len = std::max(len, in[i].length());
    }
    return len;
}

##### end vector_string_out_len source
