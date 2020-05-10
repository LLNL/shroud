
##### start PY_converter_type source

// helper PY_converter_type
// Store PyObject and pointer to the data it contains.
typedef struct {
    PyObject *obj;
    void *data;   // points into obj.
    size_t size;
} LIB_SHROUD_converter_value;
##### end PY_converter_type source

##### start ShroudLenTrim source

// helper ShroudLenTrim
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
static int ShroudLenTrim(const char *src, int nsrc) {
    int i;

    for (i = nsrc - 1; i >= 0; i--) {
        if (src[i] != ' ') {
            break;
        }
    }

    return i + 1;
}

##### end ShroudLenTrim source

##### start ShroudStrAlloc c_source

// helper ShroudStrAlloc
// Copy src into new memory and null terminate.
static char *ShroudStrAlloc(const char *src, int nsrc, int ntrim)
{
   char *rv = malloc(nsrc + 1);
   if (ntrim > 0) {
     memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\0';
   return rv;
}
##### end ShroudStrAlloc c_source

##### start ShroudStrAlloc cxx_source

// helper ShroudStrAlloc
// Copy src into new memory and null terminate.
static char *ShroudStrAlloc(const char *src, int nsrc, int ntrim)
{
   char *rv = (char *) std::malloc(nsrc + 1);
   if (ntrim > 0) {
     std::memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\0';
   return rv;
}
##### end ShroudStrAlloc cxx_source

##### start ShroudStrArrayAlloc c_source

// helper ShroudStrArrayAlloc
// Copy src into new memory and null terminate.
static char **ShroudStrArrayAlloc(const char *src, int nsrc, int len)
{
   char **rv = malloc(sizeof(char *) * nsrc);
   const char *src0 = src;
   for(int i=0; i < nsrc; ++i) {
      int ntrim = ShroudLenTrim(src0, len);
      char *tgt = malloc(ntrim+1);
      memcpy(tgt, src0, ntrim);
      tgt[ntrim] = '\0';
      rv[i] = tgt;
      src0 += len;
   }
   return rv;
}
##### end ShroudStrArrayAlloc c_source

##### start ShroudStrArrayAlloc cxx_source

// helper ShroudStrArrayAlloc
// Copy src into new memory and null terminate.
// char **src +size(nsrc) +len(len)
// CHARACTER(len) src(nsrc)
static char **ShroudStrArrayAlloc(const char *src, int nsrc, int len)
{
   char **rv = static_cast<char **>(std::malloc(sizeof(char *) * nsrc));
   const char *src0 = src;
   for(int i=0; i < nsrc; ++i) {
      int ntrim = ShroudLenTrim(src0, len);
      char *tgt = static_cast<char *>(std::malloc(ntrim+1));
      std::memcpy(tgt, src0, ntrim);
      tgt[ntrim] = '\0';
      rv[i] = tgt;
      src0 += len;
   }
   return rv;
}
##### end ShroudStrArrayAlloc cxx_source

##### start ShroudStrArrayFree c_source

// helper ShroudStrArrayFree
// Release memory allocated by ShroudStrArrayAlloc
static void ShroudStrArrayFree(char **src, int nsrc)
{
   for(int i=0; i < nsrc; ++i) {
       free(src[i]);
   }
   free(src);
}
##### end ShroudStrArrayFree c_source

##### start ShroudStrArrayFree cxx_source

// helper ShroudStrArrayFree
// Release memory allocated by ShroudStrArrayAlloc
static void ShroudStrArrayFree(char **src, int nsrc)
{
   for(int i=0; i < nsrc; ++i) {
       std::free(src[i]);
   }
   std::free(src);
}
##### end ShroudStrArrayFree cxx_source

##### start ShroudStrBlankFill c_source

// helper ShroudStrBlankFill
// blank fill dest starting at trailing NULL.
static void ShroudStrBlankFill(char *dest, int ndest)
{
   int nm = strlen(dest);
   if(ndest > nm) memset(dest+nm,' ',ndest-nm);
}
##### end ShroudStrBlankFill c_source

##### start ShroudStrBlankFill cxx_source

// helper ShroudStrBlankFill
// blank fill dest starting at trailing NULL.
static void ShroudStrBlankFill(char *dest, int ndest)
{
   int nm = std::strlen(dest);
   if(ndest > nm) std::memset(dest+nm,' ',ndest-nm);
}
##### end ShroudStrBlankFill cxx_source

##### start ShroudStrCopy c_source

// helper ShroudStrCopy
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudStrCopy(char *dest, int ndest, const char *src, int nsrc)
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
##### end ShroudStrCopy c_source

##### start ShroudStrCopy cxx_source

// helper ShroudStrCopy
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudStrCopy(char *dest, int ndest, const char *src, int nsrc)
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
##### end ShroudStrCopy cxx_source

##### start ShroudStrFree c_source

// helper ShroudStrFree
// Release memory allocated by ShroudStrAlloc
static void ShroudStrFree(char *src)
{
   free(src);
}
##### end ShroudStrFree c_source

##### start ShroudStrFree cxx_source

// helper ShroudStrFree
// Release memory allocated by ShroudStrAlloc
static void ShroudStrFree(char *src)
{
   free(src);
}
##### end ShroudStrFree cxx_source

##### start ShroudStrToArray source

// helper ShroudStrToArray
// Save str metadata into array to allow Fortran to access values.
// CHARACTER(len=elem_size) src
static void ShroudStrToArray(LIB_SHROUD_array *array, const std::string * src, int idtor)
{
    array->cxx.addr.cbase = src;
    array->cxx.idtor = idtor;
    if (src->empty()) {
        array->addr.ccharp = NULL;
        array->elem_len = 0;
    } else {
        array->addr.ccharp = src->data();
        array->elem_len = src->length();
    }
    array->size = 1;
    array->rank = 0;  // scalar
}
##### end ShroudStrToArray source

##### start ShroudTypeDefines source

/* helper ShroudTypeDefines */
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
##### end ShroudTypeDefines source

##### start array_context source

// helper array_context
struct s_LIB_SHROUD_array {
    LIB_SHROUD_capsule_data cxx;      /* address of C++ memory */
    union {
        const void * base;
        const char * ccharp;
    } addr;
    int type;        /* type of element */
    size_t elem_len; /* bytes-per-item or character len in c++ */
    size_t size;     /* size of data in c++ */
    int rank;        /* number of dimensions, 0=scalar */
    long shape[7];
};
typedef struct s_LIB_SHROUD_array LIB_SHROUD_array;
##### end array_context source

##### start capsule_data_helper source

// helper capsule_data_helper
struct s_LIB_SHROUD_capsule_data {
    union {
        void *base; /* address of C++ memory */
        const void *cbase;
    } addr;
    int idtor;      /* index of destructor */
};
typedef struct s_LIB_SHROUD_capsule_data LIB_SHROUD_capsule_data;
##### end capsule_data_helper source

##### start copy_array cxx_source

// helper copy_array
// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
// Called from Fortran.
void LIB_ShroudCopyArray(LIB_SHROUD_array *data, void *c_var, 
    size_t c_var_size)
{
    const void *cxx_var = data->addr.base;
    int n = c_var_size < data->size ? c_var_size : data->size;
    n *= data->elem_len;
    std::memcpy(c_var, cxx_var, n);
    LIB_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}
##### end copy_array cxx_source

##### start copy_string source

// helper copy_string
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void LIB_ShroudCopyStringAndFree(LIB_SHROUD_array *data, char *c_var, size_t c_var_len) {
    const char *cxx_var = data->addr.ccharp;
    size_t n = c_var_len;
    if (data->elem_len < n) n = data->elem_len;
    std::strncpy(c_var, cxx_var, n);
    LIB_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}

##### end copy_string source

##### start create_from_PyObject_char source

// helper create_from_PyObject_char
// Convert obj into an array of type char *
// Return -1 on error.
static int SHROUD_create_from_PyObject_char(PyObject *obj,
    const char *name, char * **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    char * *in = static_cast<char * *>
        (std::malloc(size * sizeof(char *)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyString_AsString(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be string", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_char source

##### start create_from_PyObject_double source

// helper create_from_PyObject_double
// Convert obj into an array of type double
// Return -1 on error.
static int SHROUD_create_from_PyObject_double(PyObject *obj,
    const char *name, double **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    double *in = static_cast<double *>
        (std::malloc(size * sizeof(double)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be double", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_double source

##### start create_from_PyObject_float source

// helper create_from_PyObject_float
// Convert obj into an array of type float
// Return -1 on error.
static int SHROUD_create_from_PyObject_float(PyObject *obj,
    const char *name, float **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    float *in = static_cast<float *>(std::malloc(size * sizeof(float)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be float", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_float source

##### start create_from_PyObject_int source

// helper create_from_PyObject_int
// Convert obj into an array of type int
// Return -1 on error.
static int SHROUD_create_from_PyObject_int(PyObject *obj,
    const char *name, int **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    int *in = static_cast<int *>(std::malloc(size * sizeof(int)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_int source

##### start create_from_PyObject_int16_t source

// helper create_from_PyObject_int16_t
// Convert obj into an array of type int16_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_int16_t(PyObject *obj,
    const char *name, int16_t **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    int16_t *in = static_cast<int16_t *>
        (std::malloc(size * sizeof(int16_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int16_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_int16_t source

##### start create_from_PyObject_int32_t source

// helper create_from_PyObject_int32_t
// Convert obj into an array of type int32_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_int32_t(PyObject *obj,
    const char *name, int32_t **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    int32_t *in = static_cast<int32_t *>
        (std::malloc(size * sizeof(int32_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int32_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_int32_t source

##### start create_from_PyObject_int64_t source

// helper create_from_PyObject_int64_t
// Convert obj into an array of type int64_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_int64_t(PyObject *obj,
    const char *name, int64_t **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    int64_t *in = static_cast<int64_t *>
        (std::malloc(size * sizeof(int64_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int64_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_int64_t source

##### start create_from_PyObject_int8_t source

// helper create_from_PyObject_int8_t
// Convert obj into an array of type int8_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_int8_t(PyObject *obj,
    const char *name, int8_t **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    int8_t *in = static_cast<int8_t *>
        (std::malloc(size * sizeof(int8_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int8_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_int8_t source

##### start create_from_PyObject_long source

// helper create_from_PyObject_long
// Convert obj into an array of type long
// Return -1 on error.
static int SHROUD_create_from_PyObject_long(PyObject *obj,
    const char *name, long **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    long *in = static_cast<long *>(std::malloc(size * sizeof(long)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be long", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_long source

##### start create_from_PyObject_short source

// helper create_from_PyObject_short
// Convert obj into an array of type short
// Return -1 on error.
static int SHROUD_create_from_PyObject_short(PyObject *obj,
    const char *name, short **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    short *in = static_cast<short *>(std::malloc(size * sizeof(short)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be short", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_short source

##### start create_from_PyObject_uint16_t source

// helper create_from_PyObject_uint16_t
// Convert obj into an array of type uint16_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_uint16_t(PyObject *obj,
    const char *name, uint16_t **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    uint16_t *in = static_cast<uint16_t *>
        (std::malloc(size * sizeof(uint16_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint16_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_uint16_t source

##### start create_from_PyObject_uint32_t source

// helper create_from_PyObject_uint32_t
// Convert obj into an array of type uint32_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_uint32_t(PyObject *obj,
    const char *name, uint32_t **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    uint32_t *in = static_cast<uint32_t *>
        (std::malloc(size * sizeof(uint32_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint32_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_uint32_t source

##### start create_from_PyObject_uint64_t source

// helper create_from_PyObject_uint64_t
// Convert obj into an array of type uint64_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_uint64_t(PyObject *obj,
    const char *name, uint64_t **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    uint64_t *in = static_cast<uint64_t *>
        (std::malloc(size * sizeof(uint64_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint64_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_uint64_t source

##### start create_from_PyObject_uint8_t source

// helper create_from_PyObject_uint8_t
// Convert obj into an array of type uint8_t
// Return -1 on error.
static int SHROUD_create_from_PyObject_uint8_t(PyObject *obj,
    const char *name, uint8_t **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    uint8_t *in = static_cast<uint8_t *>
        (std::malloc(size * sizeof(uint8_t)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint8_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_uint8_t source

##### start create_from_PyObject_unsigned_int source

// helper create_from_PyObject_unsigned_int
// Convert obj into an array of type unsigned int
// Return -1 on error.
static int SHROUD_create_from_PyObject_unsigned_int(PyObject *obj,
    const char *name, unsigned int **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    unsigned int *in = static_cast<unsigned int *>
        (std::malloc(size * sizeof(unsigned int)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned int", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_unsigned_int source

##### start create_from_PyObject_unsigned_long source

// helper create_from_PyObject_unsigned_long
// Convert obj into an array of type unsigned long
// Return -1 on error.
static int SHROUD_create_from_PyObject_unsigned_long(PyObject *obj,
    const char *name, unsigned long **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    unsigned long *in = static_cast<unsigned long *>
        (std::malloc(size * sizeof(unsigned long)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned long", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_unsigned_long source

##### start create_from_PyObject_unsigned_short source

// helper create_from_PyObject_unsigned_short
// Convert obj into an array of type unsigned short
// Return -1 on error.
static int SHROUD_create_from_PyObject_unsigned_short(PyObject *obj,
    const char *name, unsigned short **pin, Py_ssize_t *psize)
{
    PyObject *seq = PySequence_Fast(obj, "holder");
    if (seq == NULL) {
        PyErr_Format(PyExc_TypeError, "argument '%s' must be iterable",
            name);
        return -1;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    unsigned short *in = static_cast<unsigned short *>
        (std::malloc(size * sizeof(unsigned short)));
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            std::free(in);
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned short", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    *pin = in;
    *psize = size;
    return 0;
}
##### end create_from_PyObject_unsigned_short source

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
        in.push_back(PyFloat_AsDouble(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be double", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_double cxx_source

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
        in.push_back(PyFloat_AsDouble(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be float", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_float cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_int cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int16_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_int16_t cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int32_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_int32_t cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int64_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_int64_t cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be int8_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_int8_t cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be long", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_long cxx_source

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
        in.push_back(XXXPy_get);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be long long", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_long_long cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be short", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_short cxx_source

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
        in.push_back(XXXPy_get);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be size_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_size_t cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be uint16_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_uint16_t cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be uint32_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_uint32_t cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be uint64_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_uint64_t cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be uint8_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_uint8_t cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be unsigned int", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_unsigned_int cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be unsigned long", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_unsigned_long cxx_source

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
        in.push_back(XXXPy_get);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be unsigned long long",
                name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_unsigned_long_long cxx_source

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
        in.push_back(PyInt_AsLong(item));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError,
                "argument '%s', index %d must be unsigned short", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end create_from_PyObject_vector_unsigned_short cxx_source

##### start fill_from_PyObject_char source

// helper fill_from_PyObject_char
// Copy PyObject to char array.
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
        Py_DECREF(value.obj);
    }
    return 0;
}
##### end fill_from_PyObject_char source

##### start fill_from_PyObject_double_list source

// helper fill_from_PyObject_double_list
// Convert obj into an array of type double
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_double_list(PyObject *obj,
    const char *name, double *in, Py_ssize_t insize)
{
    double value = PyFloat_AsDouble(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be double", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_double_list source

##### start fill_from_PyObject_double_numpy source

// helper fill_from_PyObject_double_numpy
// Convert obj into an array of type double
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_double_numpy(PyObject *obj,
    const char *name, double *in, Py_ssize_t insize)
{
    double value = PyFloat_AsDouble(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_float_list source

// helper fill_from_PyObject_float_list
// Convert obj into an array of type float
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_float_list(PyObject *obj,
    const char *name, float *in, Py_ssize_t insize)
{
    float value = PyFloat_AsDouble(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be float", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_float_list source

##### start fill_from_PyObject_float_numpy source

// helper fill_from_PyObject_float_numpy
// Convert obj into an array of type float
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_float_numpy(PyObject *obj,
    const char *name, float *in, Py_ssize_t insize)
{
    float value = PyFloat_AsDouble(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_int16_t_list source

// helper fill_from_PyObject_int16_t_list
// Convert obj into an array of type int16_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int16_t_list(PyObject *obj,
    const char *name, int16_t *in, Py_ssize_t insize)
{
    int16_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int16_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_int16_t_list source

##### start fill_from_PyObject_int16_t_numpy source

// helper fill_from_PyObject_int16_t_numpy
// Convert obj into an array of type int16_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int16_t_numpy(PyObject *obj,
    const char *name, int16_t *in, Py_ssize_t insize)
{
    int16_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_int32_t_list source

// helper fill_from_PyObject_int32_t_list
// Convert obj into an array of type int32_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int32_t_list(PyObject *obj,
    const char *name, int32_t *in, Py_ssize_t insize)
{
    int32_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int32_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_int32_t_list source

##### start fill_from_PyObject_int32_t_numpy source

// helper fill_from_PyObject_int32_t_numpy
// Convert obj into an array of type int32_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int32_t_numpy(PyObject *obj,
    const char *name, int32_t *in, Py_ssize_t insize)
{
    int32_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_int64_t_list source

// helper fill_from_PyObject_int64_t_list
// Convert obj into an array of type int64_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int64_t_list(PyObject *obj,
    const char *name, int64_t *in, Py_ssize_t insize)
{
    int64_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int64_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_int64_t_list source

##### start fill_from_PyObject_int64_t_numpy source

// helper fill_from_PyObject_int64_t_numpy
// Convert obj into an array of type int64_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int64_t_numpy(PyObject *obj,
    const char *name, int64_t *in, Py_ssize_t insize)
{
    int64_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_int8_t_list source

// helper fill_from_PyObject_int8_t_list
// Convert obj into an array of type int8_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int8_t_list(PyObject *obj,
    const char *name, int8_t *in, Py_ssize_t insize)
{
    int8_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int8_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_int8_t_list source

##### start fill_from_PyObject_int8_t_numpy source

// helper fill_from_PyObject_int8_t_numpy
// Convert obj into an array of type int8_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int8_t_numpy(PyObject *obj,
    const char *name, int8_t *in, Py_ssize_t insize)
{
    int8_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_int_list source

// helper fill_from_PyObject_int_list
// Convert obj into an array of type int
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int_list(PyObject *obj,
    const char *name, int *in, Py_ssize_t insize)
{
    int value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be int", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_int_list source

##### start fill_from_PyObject_int_numpy source

// helper fill_from_PyObject_int_numpy
// Convert obj into an array of type int
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_int_numpy(PyObject *obj,
    const char *name, int *in, Py_ssize_t insize)
{
    int value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_long_list source

// helper fill_from_PyObject_long_list
// Convert obj into an array of type long
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_long_list(PyObject *obj,
    const char *name, long *in, Py_ssize_t insize)
{
    long value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be long", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_long_list source

##### start fill_from_PyObject_long_numpy source

// helper fill_from_PyObject_long_numpy
// Convert obj into an array of type long
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_long_numpy(PyObject *obj,
    const char *name, long *in, Py_ssize_t insize)
{
    long value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_short_list source

// helper fill_from_PyObject_short_list
// Convert obj into an array of type short
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_short_list(PyObject *obj,
    const char *name, short *in, Py_ssize_t insize)
{
    short value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be short", name, (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_short_list source

##### start fill_from_PyObject_short_numpy source

// helper fill_from_PyObject_short_numpy
// Convert obj into an array of type short
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_short_numpy(PyObject *obj,
    const char *name, short *in, Py_ssize_t insize)
{
    short value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_uint16_t_list source

// helper fill_from_PyObject_uint16_t_list
// Convert obj into an array of type uint16_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint16_t_list(PyObject *obj,
    const char *name, uint16_t *in, Py_ssize_t insize)
{
    uint16_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint16_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_uint16_t_list source

##### start fill_from_PyObject_uint16_t_numpy source

// helper fill_from_PyObject_uint16_t_numpy
// Convert obj into an array of type uint16_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint16_t_numpy(PyObject *obj,
    const char *name, uint16_t *in, Py_ssize_t insize)
{
    uint16_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_uint32_t_list source

// helper fill_from_PyObject_uint32_t_list
// Convert obj into an array of type uint32_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint32_t_list(PyObject *obj,
    const char *name, uint32_t *in, Py_ssize_t insize)
{
    uint32_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint32_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_uint32_t_list source

##### start fill_from_PyObject_uint32_t_numpy source

// helper fill_from_PyObject_uint32_t_numpy
// Convert obj into an array of type uint32_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint32_t_numpy(PyObject *obj,
    const char *name, uint32_t *in, Py_ssize_t insize)
{
    uint32_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_uint64_t_list source

// helper fill_from_PyObject_uint64_t_list
// Convert obj into an array of type uint64_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint64_t_list(PyObject *obj,
    const char *name, uint64_t *in, Py_ssize_t insize)
{
    uint64_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint64_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_uint64_t_list source

##### start fill_from_PyObject_uint64_t_numpy source

// helper fill_from_PyObject_uint64_t_numpy
// Convert obj into an array of type uint64_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint64_t_numpy(PyObject *obj,
    const char *name, uint64_t *in, Py_ssize_t insize)
{
    uint64_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_uint8_t_list source

// helper fill_from_PyObject_uint8_t_list
// Convert obj into an array of type uint8_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint8_t_list(PyObject *obj,
    const char *name, uint8_t *in, Py_ssize_t insize)
{
    uint8_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be uint8_t", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_uint8_t_list source

##### start fill_from_PyObject_uint8_t_numpy source

// helper fill_from_PyObject_uint8_t_numpy
// Convert obj into an array of type uint8_t
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_uint8_t_numpy(PyObject *obj,
    const char *name, uint8_t *in, Py_ssize_t insize)
{
    uint8_t value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_unsigned_int_list source

// helper fill_from_PyObject_unsigned_int_list
// Convert obj into an array of type unsigned int
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_int_list(PyObject *obj,
    const char *name, unsigned int *in, Py_ssize_t insize)
{
    unsigned int value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned int", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_unsigned_int_list source

##### start fill_from_PyObject_unsigned_int_numpy source

// helper fill_from_PyObject_unsigned_int_numpy
// Convert obj into an array of type unsigned int
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_int_numpy(PyObject *obj,
    const char *name, unsigned int *in, Py_ssize_t insize)
{
    unsigned int value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_unsigned_long_list source

// helper fill_from_PyObject_unsigned_long_list
// Convert obj into an array of type unsigned long
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_long_list(PyObject *obj,
    const char *name, unsigned long *in, Py_ssize_t insize)
{
    unsigned long value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned long", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_unsigned_long_list source

##### start fill_from_PyObject_unsigned_long_numpy source

// helper fill_from_PyObject_unsigned_long_numpy
// Convert obj into an array of type unsigned long
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_long_numpy(PyObject *obj,
    const char *name, unsigned long *in, Py_ssize_t insize)
{
    unsigned long value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start fill_from_PyObject_unsigned_short_list source

// helper fill_from_PyObject_unsigned_short_list
// Convert obj into an array of type unsigned short
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_short_list(PyObject *obj,
    const char *name, unsigned short *in, Py_ssize_t insize)
{
    unsigned short value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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
        in[i] = PyInt_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError,
                "argument '%s', index %d must be unsigned short", name,
                (int) i);
            return -1;
        }
    }
    Py_DECREF(seq);
    return 0;
}
##### end fill_from_PyObject_unsigned_short_list source

##### start fill_from_PyObject_unsigned_short_numpy source

// helper fill_from_PyObject_unsigned_short_numpy
// Convert obj into an array of type unsigned short
// Return 0 on success, -1 on error.
static int SHROUD_fill_from_PyObject_unsigned_short_numpy(PyObject *obj,
    const char *name, unsigned short *in, Py_ssize_t insize)
{
    unsigned short value = PyInt_AsLong(obj);
    if (!PyErr_Occurred()) {
        // Broadcast scalar.
        for (Py_ssize_t i = 0; i < insize; ++i) {
            in[i] = value;
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

##### start get_from_object_char source

// helper get_from_object_char
// Converter to PyObject to char *.
// The returned status will be 1 for a successful conversion
// and 0 if the conversion has failed.
static int SHROUD_get_from_object_char(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    size_t size = 0;
    char *out;
    if (PyUnicode_Check(obj)) {
#if PY_MAJOR_VERSION >= 3
        PyObject *strobj = PyUnicode_AsUTF8String(obj);
        out = PyBytes_AS_STRING(strobj);
        size = PyString_Size(obj);
        value->obj = strobj;  // steal reference
#else
        PyObject *strobj = PyUnicode_AsUTF8String(obj);
        out = PyString_AsString(strobj);
        size = PyString_Size(obj);
        value->obj = strobj;  // steal reference
#endif
#if PY_MAJOR_VERSION >= 3
    } else if (PyByteArray_Check(obj)) {
        out = PyBytes_AS_STRING(obj);
        size = PyBytes_GET_SIZE(obj);
        value->obj = obj;
        Py_INCREF(obj);
#else
    } else if (PyString_Check(obj)) {
        out = PyString_AsString(obj);
        size = PyString_Size(obj);
        value->obj = obj;
        Py_INCREF(obj);
#endif
    } else if (obj == Py_None) {
        out = NULL;
        size = 0;
        value->obj = NULL;
    } else {
        PyErr_Format(PyExc_TypeError,
            "argument should be string or None, not %.200s",
            Py_TYPE(obj)->tp_name);
        return 0;
    }
    value->data = out;
    value->size = size;
    return 1;
}

##### end get_from_object_char source

##### start get_from_object_charptr source

// helper get_from_object_charptr
// Convert PyObject to char * pointer.
static int SHROUD_get_from_object_charptr(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    char * *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_char(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<char * *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_charptr source

##### start get_from_object_double_list source

// helper get_from_object_double_list
// Convert PyObject to double pointer.
static int SHROUD_get_from_object_double_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    double *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_double(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<double *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_double_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_double_numpy source

##### start get_from_object_float_list source

// helper get_from_object_float_list
// Convert PyObject to float pointer.
static int SHROUD_get_from_object_float_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    float *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_float(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<float *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_float_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_float_numpy source

##### start get_from_object_int16_t_list source

// helper get_from_object_int16_t_list
// Convert PyObject to int16_t pointer.
static int SHROUD_get_from_object_int16_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    int16_t *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_int16_t(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<int16_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_int16_t_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_int16_t_numpy source

##### start get_from_object_int32_t_list source

// helper get_from_object_int32_t_list
// Convert PyObject to int32_t pointer.
static int SHROUD_get_from_object_int32_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    int32_t *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_int32_t(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<int32_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_int32_t_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_int32_t_numpy source

##### start get_from_object_int64_t_list source

// helper get_from_object_int64_t_list
// Convert PyObject to int64_t pointer.
static int SHROUD_get_from_object_int64_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    int64_t *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_int64_t(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<int64_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_int64_t_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_int64_t_numpy source

##### start get_from_object_int8_t_list source

// helper get_from_object_int8_t_list
// Convert PyObject to int8_t pointer.
static int SHROUD_get_from_object_int8_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    int8_t *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_int8_t(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<int8_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_int8_t_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_int8_t_numpy source

##### start get_from_object_int_list source

// helper get_from_object_int_list
// Convert PyObject to int pointer.
static int SHROUD_get_from_object_int_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    int *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_int(obj, "in", &in,  &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<int *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_int_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_int_numpy source

##### start get_from_object_long_list source

// helper get_from_object_long_list
// Convert PyObject to long pointer.
static int SHROUD_get_from_object_long_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    long *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_long(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<long *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_long_list source

##### start get_from_object_long_long_list source

// helper get_from_object_long_long_list
// Convert PyObject to long long pointer.
static int SHROUD_get_from_object_long_long_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    long long *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_long_long(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<long long *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_long_long_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_long_long_numpy source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_long_numpy source

##### start get_from_object_short_list source

// helper get_from_object_short_list
// Convert PyObject to short pointer.
static int SHROUD_get_from_object_short_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    short *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_short(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<short *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_short_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_short_numpy source

##### start get_from_object_size_t_list source

// helper get_from_object_size_t_list
// Convert PyObject to size_t pointer.
static int SHROUD_get_from_object_size_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    size_t *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_size_t(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<size_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_size_t_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_size_t_numpy source

##### start get_from_object_uint16_t_list source

// helper get_from_object_uint16_t_list
// Convert PyObject to uint16_t pointer.
static int SHROUD_get_from_object_uint16_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    uint16_t *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_uint16_t(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<uint16_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_uint16_t_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_uint16_t_numpy source

##### start get_from_object_uint32_t_list source

// helper get_from_object_uint32_t_list
// Convert PyObject to uint32_t pointer.
static int SHROUD_get_from_object_uint32_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    uint32_t *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_uint32_t(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<uint32_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_uint32_t_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_uint32_t_numpy source

##### start get_from_object_uint64_t_list source

// helper get_from_object_uint64_t_list
// Convert PyObject to uint64_t pointer.
static int SHROUD_get_from_object_uint64_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    uint64_t *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_uint64_t(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<uint64_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_uint64_t_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_uint64_t_numpy source

##### start get_from_object_uint8_t_list source

// helper get_from_object_uint8_t_list
// Convert PyObject to uint8_t pointer.
static int SHROUD_get_from_object_uint8_t_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    uint8_t *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_uint8_t(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<uint8_t *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_uint8_t_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_uint8_t_numpy source

##### start get_from_object_unsigned_int_list source

// helper get_from_object_unsigned_int_list
// Convert PyObject to unsigned int pointer.
static int SHROUD_get_from_object_unsigned_int_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    unsigned int *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_unsigned_int(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<unsigned int *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_unsigned_int_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_unsigned_int_numpy source

##### start get_from_object_unsigned_long_list source

// helper get_from_object_unsigned_long_list
// Convert PyObject to unsigned long pointer.
static int SHROUD_get_from_object_unsigned_long_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    unsigned long *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_unsigned_long(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<unsigned long *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_unsigned_long_list source

##### start get_from_object_unsigned_long_long_list source

// helper get_from_object_unsigned_long_long_list
// Convert PyObject to unsigned long long pointer.
static int SHROUD_get_from_object_unsigned_long_long_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    unsigned long long *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_unsigned_long_long(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<unsigned long long *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_unsigned_long_long_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_unsigned_long_long_numpy source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_unsigned_long_numpy source

##### start get_from_object_unsigned_short_list source

// helper get_from_object_unsigned_short_list
// Convert PyObject to unsigned short pointer.
static int SHROUD_get_from_object_unsigned_short_list(PyObject *obj,
    LIB_SHROUD_converter_value *value)
{
    unsigned short *in;
    Py_ssize_t size;
    if (SHROUD_create_from_PyObject_unsigned_short(obj, "in", &in, 
        &size) == -1) {
        return 0;
    }
    value->obj = nullptr;
    value->data = static_cast<unsigned short *>(in);
    value->size = size;
    return 1;
}
##### end get_from_object_unsigned_short_list source

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
    value->data = PyArray_DATA(reinterpret_cast<PyArrayObject *>
        (array));
    value->size = PyArray_SIZE(reinterpret_cast<PyArrayObject *>
        (array));
    return 1;
}
##### end get_from_object_unsigned_short_numpy source

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

##### start to_PyList_double source

// helper to_PyList_double
// Convert double pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_double(double *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
    return out;
}
##### end to_PyList_double source

##### start to_PyList_float source

// helper to_PyList_float
// Convert float pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_float(float *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyFloat_FromDouble(in[i]));
    }
    return out;
}
##### end to_PyList_float source

##### start to_PyList_int source

// helper to_PyList_int
// Convert int pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_int(int *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_int source

##### start to_PyList_int16_t source

// helper to_PyList_int16_t
// Convert int16_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_int16_t(int16_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_int16_t source

##### start to_PyList_int32_t source

// helper to_PyList_int32_t
// Convert int32_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_int32_t(int32_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_int32_t source

##### start to_PyList_int64_t source

// helper to_PyList_int64_t
// Convert int64_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_int64_t(int64_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_int64_t source

##### start to_PyList_int8_t source

// helper to_PyList_int8_t
// Convert int8_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_int8_t(int8_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_int8_t source

##### start to_PyList_long source

// helper to_PyList_long
// Convert long pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_long(long *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_long source

##### start to_PyList_short source

// helper to_PyList_short
// Convert short pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_short(short *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_short source

##### start to_PyList_size_t source

// helper to_PyList_size_t
// Convert size_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_size_t(size_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromSize_t(in[i]));
    }
    return out;
}
##### end to_PyList_size_t source

##### start to_PyList_uint16_t source

// helper to_PyList_uint16_t
// Convert uint16_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_uint16_t(uint16_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_uint16_t source

##### start to_PyList_uint32_t source

// helper to_PyList_uint32_t
// Convert uint32_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_uint32_t(uint32_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_uint32_t source

##### start to_PyList_uint64_t source

// helper to_PyList_uint64_t
// Convert uint64_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_uint64_t(uint64_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_uint64_t source

##### start to_PyList_uint8_t source

// helper to_PyList_uint8_t
// Convert uint8_t pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_uint8_t(uint8_t *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_uint8_t source

##### start to_PyList_unsigned_int source

// helper to_PyList_unsigned_int
// Convert unsigned int pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_unsigned_int
    (unsigned int *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_unsigned_int source

##### start to_PyList_unsigned_long source

// helper to_PyList_unsigned_long
// Convert unsigned long pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_unsigned_long
    (unsigned long *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_unsigned_long source

##### start to_PyList_unsigned_short source

// helper to_PyList_unsigned_short
// Convert unsigned short pointer to PyList of PyObjects.
static PyObject *SHROUD_to_PyList_unsigned_short
    (unsigned short *in, size_t size)
{
    PyObject *out = PyList_New(size);
    for (size_t i = 0; i < size; ++i) {
        PyList_SET_ITEM(out, i, PyInt_FromLong(in[i]));
    }
    return out;
}
##### end to_PyList_unsigned_short source

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
