// Built-in custom-op kernels shipped with ggmlR.
//
// These exist to exercise and demonstrate the ggml_custom() API from R without
// requiring a downstream package: each one covers a different path through the
// API rather than a different piece of maths.
//
//   row_median   -- one input, kernel-chosen output shape (ne[0] collapses to 1)
//   row_permute  -- two inputs, the second being int32 indices read from src[1]
//   clip_inplace -- ggml_custom_inplace: writes into a view of src[0]
//
// row_median and row_permute compute things ggml has no graph op for. Clipping
// is available as ggml_clamp(); clip_inplace duplicates it deliberately, as the
// simplest correct example of the in-place path, which would otherwise ship
// with no test and no worked example.
//
// Kernel contract (see inst/include/ggmlR.h):
//   - may run on any thread, concurrently, with ith in [0, nth)
//   - must write only the slice of dst that ith owns
//   - must not call into R
//
// All three parallelise over rows: row r is handled by thread r % nth, so the
// slices are disjoint by construction and no synchronisation is needed.

#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-impl.h"

// Rows here means "all dimensions above ne[0]", i.e. the number of
// ne[0]-length runs in the tensor.
static int64_t r_kernel_n_rows(const struct ggml_tensor * t) {
    return t->ne[1] * t->ne[2] * t->ne[3];
}

// Byte offset of flat row index r. Rows are enumerated in ggml's own order
// (ne[1] fastest), and the offset is built from nb[] rather than assuming a
// contiguous layout, so views with a gap between rows address correctly.
static size_t r_kernel_row_offset(const struct ggml_tensor * t, int64_t r) {
    const int64_t i1 = r % t->ne[1];
    const int64_t i2 = (r / t->ne[1]) % t->ne[2];
    const int64_t i3 = r / (t->ne[1] * t->ne[2]);
    return (size_t) (i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
}

// ============================================================================
// row_median -- median of each row
// ============================================================================
//
// dst: [1, ne1, ne2, ne3] from src[0]: [ne0, ne1, ne2, ne3], both F32.
// Not expressible with graph ops: it needs a selection of the k-th order
// statistic, which argsort + view cannot give without materialising a full sort.

static int r_kernel_cmp_f32(const void * a, const void * b) {
    const float x = *(const float *) a;
    const float y = *(const float *) b;
    // NaN sorts last so it cannot displace a real median from the middle.
    if (x != x) return (y != y) ? 0 : 1;
    if (y != y) return -1;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

static void r_kernel_row_median(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;

    const struct ggml_tensor * src = dst->src[0];
    if (src == NULL || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return;
    }

    const int64_t ne0    = src->ne[0];
    const int64_t n_rows = r_kernel_n_rows(src);
    if (ne0 <= 0) {
        return;
    }

    // Per-thread scratch: qsort needs a mutable copy of the row.
    float * buf = (float *) malloc((size_t) ne0 * sizeof(float));
    if (buf == NULL) {
        return;
    }

    for (int64_t r = ith; r < n_rows; r += nth) {
        const float * in =
            (const float *) ((const char *) src->data + r_kernel_row_offset(src, r));
        memcpy(buf, in, (size_t) ne0 * sizeof(float));
        qsort(buf, (size_t) ne0, sizeof(float), r_kernel_cmp_f32);

        const float med = (ne0 % 2 == 1)
            ? buf[ne0 / 2]
            : 0.5f * (buf[ne0 / 2 - 1] + buf[ne0 / 2]);

        float * out = (float *) ((char *) dst->data + r_kernel_row_offset(dst, r));
        out[0] = med;
    }

    free(buf);
}

// ============================================================================
// row_permute -- reorder elements within each row
// ============================================================================
//
// dst[i, r] = src0[perm[i], r], with perm in src[1] as 0-based I32 indices.
// ggml_get_rows() permutes whole rows; there is no graph op that permutes
// elements *within* a row, which is what a permutation acting on tensor indices
// requires.
//
// Out-of-range indices yield 0 rather than reading out of bounds: a kernel must
// not fault on bad input, and it cannot raise an R error from a worker thread.

static void r_kernel_row_permute(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;

    const struct ggml_tensor * src  = dst->src[0];
    const struct ggml_tensor * perm = dst->src[1];
    if (src == NULL || perm == NULL) {
        return;
    }
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        perm->type != GGML_TYPE_I32) {
        return;
    }

    const int64_t ne0    = dst->ne[0];
    const int64_t src_n0 = src->ne[0];
    const int64_t n_rows = r_kernel_n_rows(dst);
    const int64_t n_perm = ggml_nelements(perm);

    const int32_t * idx = (const int32_t *) perm->data;

    for (int64_t r = ith; r < n_rows; r += nth) {
        const float * in  =
            (const float *) ((const char *) src->data + r_kernel_row_offset(src, r));
        float       * out =
            (float *) ((char *) dst->data + r_kernel_row_offset(dst, r));

        for (int64_t i = 0; i < ne0; i++) {
            const int32_t j = (i < n_perm) ? idx[i] : -1;
            out[i] = (j >= 0 && j < src_n0) ? in[j] : 0.0f;
        }
    }
}

// ============================================================================
// clip_inplace -- clamp values into [lo, hi], writing into src[0]
// ============================================================================
//
// dst is a view of src[0]; src[1] is a 2-element F32 tensor holding {lo, hi}.
// The bounds travel as a tensor because a custom node carries no parameters of
// its own through this binding -- userdata would have to outlive the graph, and
// nothing on the R side owns it.
//
// Duplicates ggml_clamp() by design: it is the in-place example.

static void r_kernel_clip_inplace(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;

    const struct ggml_tensor * bounds = dst->src[1];
    if (dst->type != GGML_TYPE_F32) {
        return;
    }
    if (bounds == NULL || bounds->type != GGML_TYPE_F32 ||
        ggml_nelements(bounds) < 2) {
        return;
    }

    const float * lohi = (const float *) bounds->data;
    const float lo = lohi[0];
    const float hi = lohi[1];

    const int64_t ne0    = dst->ne[0];
    const int64_t n_rows = r_kernel_n_rows(dst);

    for (int64_t r = ith; r < n_rows; r += nth) {
        float * row = (float *) ((char *) dst->data + r_kernel_row_offset(dst, r));
        for (int64_t i = 0; i < ne0; i++) {
            const float v = row[i];
            row[i] = (v < lo) ? lo : (v > hi) ? hi : v;
        }
    }
}

// ============================================================================
// Registration
// ============================================================================

extern void ggmlR_register_custom_op(const char * name, ggml_custom_op_t fun);

// Called from R_init_ggmlR, so the built-ins are present from package load and
// show up in ggml_custom_ops().
void ggmlR_register_builtin_custom_ops(void) {
    ggmlR_register_custom_op("row_median",   r_kernel_row_median);
    ggmlR_register_custom_op("row_permute",  r_kernel_row_permute);
    ggmlR_register_custom_op("clip_inplace", r_kernel_clip_inplace);
}
