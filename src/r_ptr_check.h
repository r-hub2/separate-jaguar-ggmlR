// Checked external-pointer accessors for the R <-> C boundary.
//
// R_ExternalPtrAddr() is only meaningful for an EXTPTRSXP. Handing it any other
// SEXP (notably R's NULL, a NILSXP) does not yield a NULL pointer -- it reads a
// garbage address, so the usual `if (p == NULL) error(...)` guard never fires
// and the first dereference segfaults R.
//
// Use these helpers on every argument where the R-level contract admits NULL,
// or where a caller could plausibly pass a non-pointer:
//
//   r_ptr_or_null()  -- NULL is a legal value; returns NULL for R NULL/NILSXP,
//                       errors on a non-pointer non-NULL SEXP.
//   r_ptr_required() -- a live pointer is mandatory; errors on anything else.
//   r_ptr_freeable() -- for free()/shutdown entry points: rejects non-pointers,
//                       but returns NULL (rather than erroring) for an already
//                       cleared pointer, so double-free stays the harmless
//                       no-op that R_ClearExternalPtr() is meant to make it.

#ifndef R_PTR_CHECK_H
#define R_PTR_CHECK_H

#include <R.h>
#include <Rinternals.h>

// Optional pointer: R NULL (and NILSXP generally) maps to a real C NULL.
static inline void * r_ptr_or_null(SEXP p, const char * what) {
    if (p == R_NilValue || TYPEOF(p) == NILSXP) {
        return NULL;
    }
    if (TYPEOF(p) != EXTPTRSXP) {
        Rf_error("%s must be an external pointer or NULL", what);
    }
    return R_ExternalPtrAddr(p);
}

// Mandatory pointer: must be an external pointer holding a non-NULL address.
static inline void * r_ptr_required(SEXP p, const char * what) {
    if (TYPEOF(p) != EXTPTRSXP) {
        Rf_error("Invalid %s pointer", what);
    }
    void * addr = R_ExternalPtrAddr(p);
    if (addr == NULL) {
        Rf_error("Invalid %s pointer", what);
    }
    return addr;
}

// Free/shutdown entry points: the pointer must be a real external pointer, but
// a NULL payload is fine and means "already freed, nothing to do".
static inline void * r_ptr_freeable(SEXP p, const char * what) {
    if (TYPEOF(p) != EXTPTRSXP) {
        Rf_error("Invalid %s pointer", what);
    }
    return R_ExternalPtrAddr(p);
}

#endif // R_PTR_CHECK_H
