
---------- array_context ----------
{
    "c_fmtname": "LIB_SHROUD_array",
    "dependent_helpers": [
        "type_defines"
    ],
    "f_fmtname": "LIB_SHROUD_array",
    "include": [
        "<stddef.h>"
    ],
    "modules": {
        "iso_c_binding": [
            "C_NULL_PTR",
            "C_PTR",
            "C_SIZE_T",
            "C_INT",
            "C_LONG"
        ]
    },
    "name": "array_context",
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

##### start array_context derived_type

! helper array_context
type, bind(C) :: LIB_SHROUD_array
    ! address of data
    type(C_PTR) :: base_addr = C_NULL_PTR
    ! type of element
    integer(C_INT) :: type
    ! bytes-per-item or character len of data in cxx
    integer(C_SIZE_T) :: elem_len = 0_C_SIZE_T
    ! size of data in cxx
    integer(C_SIZE_T) :: size = 0_C_SIZE_T
    ! number of dimensions
    integer(C_INT) :: rank = -1
    integer(C_LONG) :: shape(7) = 0
end type LIB_SHROUD_array
##### end array_context derived_type

---------- array_string_allocatable ----------
{
    "api": "c",
    "c_fmtname": "LIB_ShroudArrayStringAllocatable",
    "dependent_helpers": [
        "capsule_data_helper",
        "array_context",
        "array_string_out"
    ],
    "f_fmtname": "LIB_SHROUD_array_string_allocatable",
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudArrayStringAllocatable",
        "cnameproto": "void {cnamefunc}({C_array_type} *dest, {C_capsule_data_type} *src)",
        "fnamefunc": "{C_prefix}SHROUD_array_string_allocatable"
    },
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

##### start array_string_allocatable interface

interface
    ! helper array_string_allocatable
    subroutine LIB_SHROUD_array_string_allocatable(dest, src) &
         bind(c,name="LIB_ShroudArrayStringAllocatable")
        import LIB_SHROUD_array, LIB_SHROUD_capsule_data
        type(LIB_SHROUD_array), intent(IN) :: dest
        type(LIB_SHROUD_capsule_data), intent(IN) :: src
    end subroutine LIB_SHROUD_array_string_allocatable
end interface
##### end array_string_allocatable interface

---------- array_string_out ----------
{
    "api": "cxx",
    "c_fmtname": "LIB_ShroudArrayStringOut",
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

---------- capsule_data_helper ----------
{
    "f_fmtname": "LIB_SHROUD_capsule_data",
    "modules": {
        "iso_c_binding": [
            "C_PTR",
            "C_INT",
            "C_NULL_PTR"
        ]
    },
    "name": "capsule_data_helper",
    "scope": "cwrap_include"
}

##### start capsule_data_helper derived_type

! helper capsule_data_helper
type, bind(C) :: LIB_SHROUD_capsule_data
    type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
    integer(C_INT) :: idtor = 0       ! index of destructor
end type LIB_SHROUD_capsule_data
##### end capsule_data_helper derived_type

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
    "c_fmtname": "LIB_SHROUD_memory_destructor",
    "dependent_helpers": [
        "capsule_data_helper"
    ],
    "f_fmtname": "LIB_SHROUD_capsule_dtor",
    "fmtdict": {
        "cnamefunc": "{C_memory_dtor_function}",
        "cnameproto": "void {cnamefunc}\t({C_capsule_data_type} *cap)",
        "fnamefunc": "{C_prefix}SHROUD_capsule_dtor"
    },
    "name": "capsule_dtor",
    "proto": "void LIB_SHROUD_memory_destructor\t(LIB_SHROUD_capsule_data *cap);"
}

##### start capsule_dtor interface

interface
    ! helper capsule_dtor
    ! Delete memory in a capsule.
    subroutine LIB_SHROUD_capsule_dtor(ptr)&
        bind(C, name="LIB_SHROUD_memory_destructor")
        import LIB_SHROUD_capsule_data
        implicit none
        type(LIB_SHROUD_capsule_data), intent(INOUT) :: ptr
    end subroutine LIB_SHROUD_capsule_dtor
end interface
##### end capsule_dtor interface

---------- capsule_helper ----------
{
    "dependent_helpers": [
        "capsule_data_helper",
        "capsule_dtor"
    ],
    "name": "capsule_helper"
}

##### start capsule_helper derived_type

! helper capsule_helper
type :: LIB_SHROUD_capsule
    private
    type(LIB_SHROUD_capsule_data) :: mem
contains
    final :: SHROUD_capsule_final
    procedure :: delete => SHROUD_capsule_delete
end type LIB_SHROUD_capsule
##### end capsule_helper derived_type

##### start capsule_helper f_source

! helper capsule_helper
! finalize a static LIB_SHROUD_capsule_data
subroutine SHROUD_capsule_final(cap)
    type(LIB_SHROUD_capsule), intent(INOUT) :: cap
    call LIB_SHROUD_capsule_dtor(cap%mem)
end subroutine SHROUD_capsule_final

subroutine SHROUD_capsule_delete(cap)
    class(LIB_SHROUD_capsule) :: cap
    call LIB_SHROUD_capsule_dtor(cap%mem)
end subroutine SHROUD_capsule_delete
##### end capsule_helper f_source

---------- char_alloc ----------
{
    "c_fmtname": "ShroudCharAlloc",
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
    "name": "char_alloc"
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
    "c_fmtname": "ShroudStrArrayAlloc",
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
    "name": "char_array_alloc"
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
    "c_fmtname": "ShroudStrArrayFree",
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "name": "char_array_free"
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
    "c_fmtname": "ShroudCharBlankFill",
    "c_include": [
        "<string.h>"
    ],
    "cxx_include": [
        "<cstring>"
    ],
    "name": "char_blank_fill"
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
    "c_fmtname": "ShroudCharCopy",
    "c_include": [
        "<string.h>"
    ],
    "cxx_include": [
        "<cstring>"
    ],
    "name": "char_copy"
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
    "c_fmtname": "ShroudCharFree",
    "c_include": [
        "<stdlib.h>"
    ],
    "cxx_include": [
        "<cstdlib>"
    ],
    "name": "char_free"
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
    "c_fmtname": "ShroudCharLenTrim",
    "name": "char_len_trim"
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
    "c_fmtname": "LIB_ShroudCopyArray",
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
    "f_fmtname": "LIB_SHROUD_copy_array",
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudCopyArray",
        "fnamefunc": "{C_prefix}SHROUD_{hname}"
    },
    "name": "copy_array",
    "scope": "cwrap_impl"
}

##### start copy_array source

// helper copy_array
// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
// Called from Fortran.
void LIB_ShroudCopyArray(LIB_SHROUD_array *data, void *c_var, &
    size_t c_var_size)
{
    const void *cxx_var = data->base_addr;
    int n = c_var_size < data->size ? c_var_size : data->size;
    n *= data->elem_len;
    std::memcpy(c_var, cxx_var, n);
}
##### end copy_array source

##### start copy_array interface

interface
    ! helper copy_array
    ! Copy contents of context into c_var.
    subroutine LIB_SHROUD_copy_array(context, c_var, c_var_size) &
        bind(C, name="LIB_ShroudCopyArray")
        use iso_c_binding, only : C_PTR, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        type(C_PTR), intent(IN), value :: c_var
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_array
end interface
##### end copy_array interface

---------- copy_string ----------
{
    "c_fmtname": "LIB_ShroudCopyString",
    "cxx_include": [
        "<cstring>",
        "<cstddef>"
    ],
    "dependent_helpers": [
        "array_context"
    ],
    "f_fmtname": "LIB_SHROUD_copy_string",
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudCopyString",
        "fnamefunc": "{C_prefix}SHROUD_copy_string"
    },
    "name": "copy_string",
    "scope": "cwrap_impl"
}

##### start copy_string source

// helper copy_string
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void LIB_ShroudCopyString(LIB_SHROUD_array *data, char *c_var,&
    size_t c_var_len) {
    const void *cxx_var = data->base_addr;
    size_t n = c_var_len;
    if (data->elem_len < n) n = data->elem_len;
    std::memcpy(c_var, cxx_var, n);
}

##### end copy_string source

##### start copy_string interface

interface
    ! helper copy_string
    ! Copy the char* or std::string in context into c_var.
    subroutine LIB_SHROUD_copy_string(context, c_var, c_var_size) &
         bind(c,name="LIB_ShroudCopyString")
        use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
        import LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: context
        character(kind=C_CHAR), intent(OUT) :: c_var(*)
        integer(C_SIZE_T), value :: c_var_size
    end subroutine LIB_SHROUD_copy_string
end interface
##### end copy_string interface

---------- pointer_string ----------
{
    "dependent_helpers": [
        "array_context"
    ],
    "f_fmtname": "LIB_SHROUD_pointer_string",
    "fmtdict": {
        "fnamefunc": "{C_prefix}SHROUD_pointer_string"
    },
    "name": "pointer_string"
}

##### start pointer_string f_source

! helper pointer_string
! Assign context to an assumed-length character pointer
subroutine LIB_SHROUD_pointer_string(context, var)
    use iso_c_binding, only : c_f_pointer, C_PTR
    implicit none
    type(LIB_SHROUD_array), intent(IN) :: context
    character(len=:), pointer, intent(OUT) :: var
    character(len=context%elem_len), pointer :: fptr
    call c_f_pointer(context%base_addr, fptr)
    var => fptr
end subroutine LIB_SHROUD_pointer_string
##### end pointer_string f_source

---------- size_CFI ----------
{
    "c_include": [
        "<stddef.h>"
    ],
    "cxx_include": [
        "<cstddef>"
    ],
    "name": "size_CFI"
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

---------- type_defines ----------
{
    "name": "type_defines",
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

##### start type_defines derived_type

! helper type_defines
! Shroud type defines from helper type_defines
integer, parameter, private :: &
    SH_TYPE_SIGNED_CHAR= 1, &
    SH_TYPE_SHORT      = 2, &
    SH_TYPE_INT        = 3, &
    SH_TYPE_LONG       = 4, &
    SH_TYPE_LONG_LONG  = 5, &
    SH_TYPE_SIZE_T     = 6, &
    SH_TYPE_UNSIGNED_SHORT      = SH_TYPE_SHORT + 100, &
    SH_TYPE_UNSIGNED_INT        = SH_TYPE_INT + 100, &
    SH_TYPE_UNSIGNED_LONG       = SH_TYPE_LONG + 100, &
    SH_TYPE_UNSIGNED_LONG_LONG  = SH_TYPE_LONG_LONG + 100, &
    SH_TYPE_INT8_T    =  7, &
    SH_TYPE_INT16_T   =  8, &
    SH_TYPE_INT32_T   =  9, &
    SH_TYPE_INT64_T   = 10, &
    SH_TYPE_UINT8_T  =  SH_TYPE_INT8_T + 100, &
    SH_TYPE_UINT16_T =  SH_TYPE_INT16_T + 100, &
    SH_TYPE_UINT32_T =  SH_TYPE_INT32_T + 100, &
    SH_TYPE_UINT64_T =  SH_TYPE_INT64_T + 100, &
    SH_TYPE_FLOAT       = 22, &
    SH_TYPE_DOUBLE      = 23, &
    SH_TYPE_LONG_DOUBLE = 24, &
    SH_TYPE_FLOAT_COMPLEX      = 25, &
    SH_TYPE_DOUBLE_COMPLEX     = 26, &
    SH_TYPE_LONG_DOUBLE_COMPLEX= 27, &
    SH_TYPE_BOOL      = 28, &
    SH_TYPE_CHAR      = 29, &
    SH_TYPE_CPTR      = 30, &
    SH_TYPE_STRUCT    = 31, &
    SH_TYPE_OTHER     = 32
##### end type_defines derived_type

---------- vector_string_allocatable ----------
{
    "api": "c",
    "c_fmtname": "LIB_ShroudVectorStringAllocatable",
    "dependent_helpers": [
        "capsule_data_helper",
        "array_context",
        "vector_string_out"
    ],
    "f_fmtname": "LIB_SHROUD_vector_string_allocatable",
    "fmtdict": {
        "cnamefunc": "{C_prefix}ShroudVectorStringAllocatable",
        "cnameproto": "void {cnamefunc}({C_array_type} *dest, {C_capsule_data_type} *src)",
        "fnamefunc": "{C_prefix}SHROUD_vector_string_allocatable"
    },
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
    std::vector<std::string> *cxxvec =&
        static_cast< std::vector<std::string> * >(src->addr);
    LIB_ShroudVectorStringOut(dest, *cxxvec);
}

##### end vector_string_allocatable source

##### start vector_string_allocatable interface

interface
    ! helper vector_string_allocatable
    ! Copy the char* or std::string in context into c_var.
    subroutine LIB_SHROUD_vector_string_allocatable(dest, src) &
         bind(c,name="LIB_ShroudVectorStringAllocatable")
        import LIB_SHROUD_capsule_data, LIB_SHROUD_array
        type(LIB_SHROUD_array), intent(IN) :: dest
        type(LIB_SHROUD_capsule_data), intent(IN) :: src
    end subroutine LIB_SHROUD_vector_string_allocatable
end interface
##### end vector_string_allocatable interface

---------- vector_string_out ----------
{
    "api": "cxx",
    "c_fmtname": "LIB_ShroudVectorStringOut",
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
