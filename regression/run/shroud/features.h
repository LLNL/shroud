#if 0
/*
 * Define compiler features that are available
 *
 * https://sourceforge.net/p/predef/wiki/Compilers/
 *
 * Used with Fortran too, must use C style comments.
 */
#endif


#if 0
/* Assume this works for all compilers. */
#endif
#define HAVE_CHARACTER_POINTER_FUNCTION 1

#ifdef __INTEL_COMPILER
/* Check intel before __GNUC__, since intel also definds __GNUC__ */
/*
 * __INTEL_COMPILER  VRP    Version Revision Path
 * 19.0.4 - 1900
 */

#elif defined(__PGI)
/* __PGIC__  __PGIC_MINOR__  __PGIC_PATCHLEVEL__ */

#elif defined(__ibmxl__)
#if 0
/* __ibmxl_vrm__   compiler version */
#endif

#elif defined(__GNUC__)
/*  __GNUC__  __GNUC_MINOR__ __GNUC_PATCHLEVEL__ */
/*
# if defined(__GNUC_PATCHLEVEL__)
#  define __GNUC_VERSION__ (__GNUC__ * 10000     \
                            + __GNUC_MINOR__ * 100 \
                            + __GNUC_PATCHLEVEL__)
# else
#  define __GNUC_VERSION__ (__GNUC__ * 10000          \
                            + __GNUC_MINOR__ * 100)
# endif
*/

#if __GNUC__ < 6
#undef HAVE_CHARACTER_POINTER_FUNCTION
#endif

#endif
