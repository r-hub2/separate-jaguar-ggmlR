/* onnx_ggml.c — Map ONNX ops to ggml ops and run inference
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_ggml.h"
#include "../ggml.h"
#include "../ggml-backend.h"
#include "../ggml-cpu.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Check if Vulkan is available at compile time */
#ifdef GGML_USE_VULKAN
#include "../ggml-vulkan.h"
#endif

/* ── Tensor name map ────────────────────────────────────────────── */

static void tmap_put(onnx_ggml_ctx_t *c, const char *name, struct ggml_tensor *t) {
    if (c->tensor_map_size >= c->tensor_map_cap) {
        c->tensor_map_cap = c->tensor_map_cap ? c->tensor_map_cap * 2 : 256;
        c->tensor_map_keys = realloc(c->tensor_map_keys,
                                      c->tensor_map_cap * sizeof(*c->tensor_map_keys));
        c->tensor_map_vals = realloc(c->tensor_map_vals,
                                      c->tensor_map_cap * sizeof(*c->tensor_map_vals));
    }
    strncpy(c->tensor_map_keys[c->tensor_map_size], name, ONNX_MAX_NAME - 1);
    c->tensor_map_keys[c->tensor_map_size][ONNX_MAX_NAME - 1] = '\0';
    c->tensor_map_vals[c->tensor_map_size] = t;
    c->tensor_map_size++;
}

static struct ggml_tensor *tmap_get(onnx_ggml_ctx_t *c, const char *name) {
    for (int i = c->tensor_map_size - 1; i >= 0; i--) {
        if (strcmp(c->tensor_map_keys[i], name) == 0)
            return c->tensor_map_vals[i];
    }
    return NULL;
}

/* ── ONNX dtype → ggml type ─────────────────────────────────────── */

static enum ggml_type onnx_dtype_to_ggml(int32_t dt) {
    switch (dt) {
        case ONNX_DTYPE_FLOAT:    return GGML_TYPE_F32;
        case ONNX_DTYPE_FLOAT16:  return GGML_TYPE_F16;
        case ONNX_DTYPE_BFLOAT16: return GGML_TYPE_BF16;
        case ONNX_DTYPE_INT32:    return GGML_TYPE_I32;
        case ONNX_DTYPE_INT64:    return GGML_TYPE_I32; /* downcast to i32 */
        case ONNX_DTYPE_DOUBLE:   return GGML_TYPE_F32; /* downcast to f32 */
        default:                  return GGML_TYPE_F32;
    }
}

/* ── Size of ONNX data type in bytes ────────────────────────────── */

static size_t onnx_dtype_size(int32_t dt) {
    switch (dt) {
        case ONNX_DTYPE_FLOAT:    return 4;
        case ONNX_DTYPE_DOUBLE:   return 8;
        case ONNX_DTYPE_FLOAT16:  return 2;
        case ONNX_DTYPE_BFLOAT16: return 2;
        case ONNX_DTYPE_INT32:    return 4;
        case ONNX_DTYPE_INT64:    return 8;
        case ONNX_DTYPE_INT8:     return 1;
        case ONNX_DTYPE_UINT8:    return 1;
        case ONNX_DTYPE_INT16:    return 2;
        case ONNX_DTYPE_BOOL:     return 1;
        default:                  return 4;
    }
}

/* ── Create ggml tensors for initializers (weights) ─────────────── */

static int create_initializer_tensors(onnx_ggml_ctx_t *c) {
    for (int i = 0; i < c->onnx->n_initializers; i++) {
        onnx_initializer_t *init = &c->onnx->initializers[i];
        enum ggml_type type = onnx_dtype_to_ggml(init->data_type);

        /* Use ONNX dims directly (no reversal).
         * Data will be transposed from row-major to column-major during load. */
        int64_t ne[4] = {1, 1, 1, 1};
        int ndims = init->n_dims > 0 ? init->n_dims : 1;
        if (ndims > 4) ndims = 4;
        for (int d = 0; d < init->n_dims && d < 4; d++) {
            ne[d] = init->dims[d];
        }

        struct ggml_tensor *t;
        switch (ndims) {
            case 1: t = ggml_new_tensor_1d(c->ctx, type, ne[0]); break;
            case 2: t = ggml_new_tensor_2d(c->ctx, type, ne[0], ne[1]); break;
            case 3: t = ggml_new_tensor_3d(c->ctx, type, ne[0], ne[1], ne[2]); break;
            default: t = ggml_new_tensor_4d(c->ctx, type, ne[0], ne[1], ne[2], ne[3]); break;
        }
        if (!t) return -1;
        ggml_set_name(t, init->name);
        tmap_put(c, init->name, t);
    }
    return 0;
}

/* ── Load weight data into tensors ──────────────────────────────── */

static int load_weights(onnx_ggml_ctx_t *c) {
    for (int i = 0; i < c->onnx->n_initializers; i++) {
        onnx_initializer_t *init = &c->onnx->initializers[i];
        struct ggml_tensor *t = tmap_get(c, init->name);
        if (!t) continue;

        const void *data = NULL;
        size_t data_size = 0;

        if (init->raw_data && init->raw_size > 0) {
            data = init->raw_data;
            data_size = init->raw_size;
        } else if (init->decoded_data && init->decoded_size > 0) {
            data = init->decoded_data;
            data_size = init->decoded_size;
        }

        if (data && data_size > 0) {
            size_t tsize = ggml_nbytes(t);
            size_t elem_size = onnx_dtype_size(init->data_type);

            /* For 2D+ tensors, transpose from ONNX row-major to ggml column-major.
             * For 1D tensors, data is identical in both layouts. */
            if (init->n_dims == 2 && elem_size > 0) {
                int64_t rows = init->dims[0];
                int64_t cols = init->dims[1];
                size_t total = (size_t)(rows * cols) * elem_size;
                if (total <= tsize) {
                    uint8_t *tmp = malloc(total);
                    if (tmp) {
                        const uint8_t *src = (const uint8_t *)data;
                        /* Transpose: src[i*cols+j] → dst[j*rows+i] */
                        for (int64_t i = 0; i < rows; i++) {
                            for (int64_t j = 0; j < cols; j++) {
                                memcpy(tmp + (j * rows + i) * elem_size,
                                       src + (i * cols + j) * elem_size,
                                       elem_size);
                            }
                        }
                        ggml_backend_tensor_set(t, tmp, 0, total);
                        free(tmp);
                    }
                }
            } else {
                size_t copy_size = data_size < tsize ? data_size : tsize;
                ggml_backend_tensor_set(t, data, 0, copy_size);
            }
        }
    }
    return 0;
}

/* ── Create input placeholder tensors ───────────────────────────── */

static int create_input_tensors(onnx_ggml_ctx_t *c) {
    for (int i = 0; i < c->onnx->n_inputs; i++) {
        onnx_value_info_t *vi = &c->onnx->inputs[i];
        /* Skip if already created as initializer */
        if (tmap_get(c, vi->name)) continue;

        enum ggml_type type = onnx_dtype_to_ggml(vi->elem_type);
        int64_t ne[4] = {1, 1, 1, 1};
        int ndims = vi->n_dims > 0 ? vi->n_dims : 1;
        if (ndims > 4) ndims = 4;
        /* Use ONNX dims directly (no reversal) */
        for (int d = 0; d < vi->n_dims && d < 4; d++) {
            int64_t dim = vi->dims[d];
            if (dim <= 0) dim = 1; /* symbolic/dynamic dim → default 1 */
            ne[d] = dim;
        }

        struct ggml_tensor *t;
        switch (ndims) {
            case 1: t = ggml_new_tensor_1d(c->ctx, type, ne[0]); break;
            case 2: t = ggml_new_tensor_2d(c->ctx, type, ne[0], ne[1]); break;
            case 3: t = ggml_new_tensor_3d(c->ctx, type, ne[0], ne[1], ne[2]); break;
            default: t = ggml_new_tensor_4d(c->ctx, type, ne[0], ne[1], ne[2], ne[3]); break;
        }
        if (!t) return -1;
        ggml_set_name(t, vi->name);
        ggml_set_input(t);
        tmap_put(c, vi->name, t);
    }
    return 0;
}

/* ── Map ONNX node → ggml op ────────────────────────────────────── */

static struct ggml_tensor *get_input(onnx_ggml_ctx_t *c, const onnx_node_t *n, int idx) {
    if (idx >= n->n_inputs) return NULL;
    if (n->inputs[idx][0] == '\0') return NULL; /* optional empty input */
    return tmap_get(c, n->inputs[idx]);
}

static int map_node(onnx_ggml_ctx_t *c, const onnx_node_t *n) {
    struct ggml_tensor *out = NULL;
    struct ggml_tensor *a = get_input(c, n, 0);
    struct ggml_tensor *b = get_input(c, n, 1);

    const char *op = n->op_type;

    /* ── Elementwise binary ─────────────────────────────────────── */
    if (strcmp(op, "Add") == 0) {
        if (!a || !b) return -1;
        out = ggml_add(c->ctx, a, b);
    }
    else if (strcmp(op, "Sub") == 0) {
        if (!a || !b) return -1;
        out = ggml_sub(c->ctx, a, b);
    }
    else if (strcmp(op, "Mul") == 0) {
        if (!a || !b) return -1;
        out = ggml_mul(c->ctx, a, b);
    }
    else if (strcmp(op, "Div") == 0) {
        if (!a || !b) return -1;
        out = ggml_div(c->ctx, a, b);
    }

    /* ── MatMul / Gemm ──────────────────────────────────────────── */
    else if (strcmp(op, "MatMul") == 0) {
        if (!a || !b) return -1;
        /* ONNX MatMul(A[M,K], B[K,N]): result = A @ B.
         * Stored with ONNX dims: A.ne=[M,K], B.ne=[K,N].
         * Data transposed to column-major during load → correct values.
         * ggml_mul_mat(a,b) contracts ne[0]: a.ne[0]==b.ne[0].
         * B.ne[0]=K (contraction dim). A has K at ne[1].
         * Transpose A to move K to ne[0]: ggml_mul_mat(B, A^T).
         * Only 'a' (first arg) must be non-transposed → B is first. */
        out = ggml_mul_mat(c->ctx, b, ggml_transpose(c->ctx, a));
    }
    else if (strcmp(op, "Gemm") == 0) {
        if (!a || !b) return -1;
        int64_t transA = onnx_attr_int(n, "transA", 0);
        int64_t transB = onnx_attr_int(n, "transB", 0);
        float alpha = onnx_attr_float(n, "alpha", 1.0f);
        float beta  = onnx_attr_float(n, "beta", 1.0f);

        /* ONNX dims stored directly (no reversal), data transposed to col-major.
         * A[M,K], B[K,N]. ggml_mul_mat(first, second) needs first.ne[0]==second.ne[0].
         * Default (no trans): need to transpose A so K is at ne[0].
         * B already has ne[0]=K. transA/transB flip this. */
        struct ggml_tensor *ta = transA ? a : ggml_transpose(c->ctx, a);
        struct ggml_tensor *tb = transB ? ggml_transpose(c->ctx, b) : b;

        out = ggml_mul_mat(c->ctx, tb, ta);

        if (alpha != 1.0f)
            out = ggml_scale(c->ctx, out, alpha);

        struct ggml_tensor *bias = get_input(c, n, 2);
        if (bias) {
            if (beta != 1.0f)
                bias = ggml_scale(c->ctx, bias, beta);
            out = ggml_add(c->ctx, out, bias);
        }
    }

    /* ── Activations ────────────────────────────────────────────── */
    else if (strcmp(op, "Relu") == 0) {
        if (!a) return -1;
        out = ggml_relu(c->ctx, a);
    }
    else if (strcmp(op, "Sigmoid") == 0) {
        if (!a) return -1;
        out = ggml_sigmoid(c->ctx, a);
    }
    else if (strcmp(op, "Tanh") == 0) {
        if (!a) return -1;
        out = ggml_tanh(c->ctx, a);
    }
    else if (strcmp(op, "Gelu") == 0) {
        if (!a) return -1;
        out = ggml_gelu(c->ctx, a);
    }
    else if (strcmp(op, "Softmax") == 0) {
        if (!a) return -1;
        out = ggml_soft_max(c->ctx, a);
    }
    else if (strcmp(op, "LeakyRelu") == 0) {
        if (!a) return -1;
        float alpha = onnx_attr_float(n, "alpha", 0.01f);
        out = ggml_leaky_relu(c->ctx, a, alpha, false);
    }
    else if (strcmp(op, "Elu") == 0) {
        if (!a) return -1;
        out = ggml_elu(c->ctx, a);
    }
    else if (strcmp(op, "Silu") == 0 || strcmp(op, "SiLU") == 0) {
        if (!a) return -1;
        out = ggml_silu(c->ctx, a);
    }

    /* ── Math ───────────────────────────────────────────────────── */
    else if (strcmp(op, "Sqrt") == 0) {
        if (!a) return -1;
        out = ggml_sqrt(c->ctx, a);
    }
    else if (strcmp(op, "Exp") == 0) {
        if (!a) return -1;
        out = ggml_exp(c->ctx, a);
    }
    else if (strcmp(op, "Log") == 0) {
        if (!a) return -1;
        out = ggml_log(c->ctx, a);
    }
    else if (strcmp(op, "Abs") == 0) {
        if (!a) return -1;
        out = ggml_abs(c->ctx, a);
    }
    else if (strcmp(op, "Neg") == 0) {
        if (!a) return -1;
        out = ggml_neg(c->ctx, a);
    }
    else if (strcmp(op, "Floor") == 0) {
        if (!a) return -1;
        out = ggml_floor(c->ctx, a);
    }
    else if (strcmp(op, "Ceil") == 0) {
        if (!a) return -1;
        out = ggml_ceil(c->ctx, a);
    }
    else if (strcmp(op, "Clip") == 0) {
        if (!a) return -1;
        /* min/max can be inputs (opset 11+) or attributes */
        float min_val = -3.402823e+38f;
        float max_val =  3.402823e+38f;
        struct ggml_tensor *min_t = get_input(c, n, 1);
        struct ggml_tensor *max_t = get_input(c, n, 2);
        if (!min_t) min_val = onnx_attr_float(n, "min", min_val);
        if (!max_t) max_val = onnx_attr_float(n, "max", max_val);
        /* TODO: handle tensor min/max inputs */
        out = ggml_clamp(c->ctx, a, min_val, max_val);
    }

    /* ── Shape ops ──────────────────────────────────────────────── */
    else if (strcmp(op, "Reshape") == 0) {
        if (!a || !b) return -1;
        /* b is the shape tensor — for now we read from initializer */
        const onnx_initializer_t *shape_init = onnx_find_initializer(c->onnx, n->inputs[1]);
        if (!shape_init) return -1;

        int64_t shape[ONNX_MAX_DIMS];
        int ndims = 0;
        if (shape_init->raw_data && shape_init->data_type == ONNX_DTYPE_INT64) {
            ndims = (int)(shape_init->raw_size / sizeof(int64_t));
            if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
            memcpy(shape, shape_init->raw_data, ndims * sizeof(int64_t));
        } else if (shape_init->decoded_data) {
            ndims = (int)(shape_init->decoded_size / sizeof(int64_t));
            if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
            memcpy(shape, shape_init->decoded_data, ndims * sizeof(int64_t));
        }

        /* Resolve -1 and 0 dims */
        int64_t total = ggml_nelements(a);
        int64_t product = 1;
        int neg_idx = -1;
        for (int d = 0; d < ndims; d++) {
            if (shape[d] == 0) shape[d] = (d < ggml_n_dims(a)) ?  a->ne[d] : 1;
            if (shape[d] == -1) neg_idx = d;
            else product *= shape[d];
        }
        if (neg_idx >= 0 && product > 0)
            shape[neg_idx] = total / product;

        switch (ndims) {
            case 1: out = ggml_reshape_1d(c->ctx, a, shape[0]); break;
            case 2: out = ggml_reshape_2d(c->ctx, a, shape[1], shape[0]); break;
            case 3: out = ggml_reshape_3d(c->ctx, a, shape[2], shape[1], shape[0]); break;
            case 4: out = ggml_reshape_4d(c->ctx, a, shape[3], shape[2], shape[1], shape[0]); break;
            default: out = ggml_reshape_2d(c->ctx, a, shape[ndims-1], total / shape[ndims-1]); break;
        }
    }
    else if (strcmp(op, "Transpose") == 0) {
        if (!a) return -1;
        /* Default: reverse all dims. For 2D this is standard transpose. */
        int64_t perm[4];
        int n_perm = onnx_attr_ints(n, "perm", perm, 4);
        if (n_perm == 0 || ggml_n_dims(a) == 2) {
            out = ggml_transpose(c->ctx, a);
        } else {
            /* General permute */
            int ax[4] = {0, 1, 2, 3};
            for (int d = 0; d < n_perm && d < 4; d++)
                ax[d] = (int)perm[d];
            out = ggml_permute(c->ctx, a, ax[0], ax[1], ax[2], ax[3]);
        }
    }
    else if (strcmp(op, "Flatten") == 0) {
        if (!a) return -1;
        int64_t total = ggml_nelements(a);
        out = ggml_reshape_1d(c->ctx, a, total);
    }
    else if (strcmp(op, "Squeeze") == 0 || strcmp(op, "Unsqueeze") == 0) {
        if (!a) return -1;
        /* For now, pass through — reshape handled at boundaries */
        out = ggml_cont(c->ctx, a);
    }

    /* ── Concat ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Concat") == 0) {
        if (!a || !b) return -1;
        int64_t axis = onnx_attr_int(n, "axis", 0);
        /* ggml_concat concatenates along dim 2 by default.
         * Map ONNX axis to ggml dim. */
        int dim = (int)axis;
        if (dim < 0) dim = ggml_n_dims(a) + dim;
        /* ggml concat: dim parameter */
        out = ggml_concat(c->ctx, a, b, dim);
    }

    /* ── Gather ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Gather") == 0) {
        if (!a || !b) return -1;
        out = ggml_get_rows(c->ctx, a, b);
    }

    /* ── Normalization ──────────────────────────────────────────── */
    else if (strcmp(op, "BatchNormalization") == 0) {
        if (!a) return -1;
        float eps = onnx_attr_float(n, "epsilon", 1e-5f);
        struct ggml_tensor *scale = get_input(c, n, 1);
        struct ggml_tensor *bias  = get_input(c, n, 2);
        struct ggml_tensor *mean  = get_input(c, n, 3);
        struct ggml_tensor *var   = get_input(c, n, 4);

        /* x_norm = (x - mean) / sqrt(var + eps) */
        out = ggml_sub(c->ctx, a, mean);
        /* var + eps, then sqrt */
        struct ggml_tensor *eps_t = ggml_new_f32(c->ctx, eps);
        struct ggml_tensor *std = ggml_sqrt(c->ctx, ggml_add(c->ctx, var, eps_t));
        out = ggml_div(c->ctx, out, std);
        if (scale) out = ggml_mul(c->ctx, out, scale);
        if (bias)  out = ggml_add(c->ctx, out, bias);
    }
    else if (strcmp(op, "LayerNormalization") == 0) {
        if (!a) return -1;
        float eps = onnx_attr_float(n, "epsilon", 1e-5f);
        out = ggml_norm(c->ctx, a, eps);
        struct ggml_tensor *scale = get_input(c, n, 1);
        struct ggml_tensor *bias  = get_input(c, n, 2);
        if (scale) out = ggml_mul(c->ctx, out, scale);
        if (bias)  out = ggml_add(c->ctx, out, bias);
    }
    else if (strcmp(op, "GroupNormalization") == 0) {
        if (!a) return -1;
        float eps = onnx_attr_float(n, "epsilon", 1e-5f);
        int64_t num_groups = onnx_attr_int(n, "num_groups", 1);
        out = ggml_group_norm(c->ctx, a, (int)num_groups, eps);
        struct ggml_tensor *scale = get_input(c, n, 1);
        struct ggml_tensor *bias  = get_input(c, n, 2);
        if (scale) out = ggml_mul(c->ctx, out, scale);
        if (bias)  out = ggml_add(c->ctx, out, bias);
    }
    else if (strcmp(op, "RMSNormalization") == 0) {
        if (!a) return -1;
        float eps = onnx_attr_float(n, "epsilon", 1e-5f);
        out = ggml_rms_norm(c->ctx, a, eps);
    }

    /* ── Pooling ────────────────────────────────────────────────── */
    else if (strcmp(op, "MaxPool") == 0) {
        if (!a) return -1;
        int64_t kshape[2] = {1, 1}, strides[2] = {1, 1}, pads[4] = {0};
        onnx_attr_ints(n, "kernel_shape", kshape, 2);
        onnx_attr_ints(n, "strides", strides, 2);
        onnx_attr_ints(n, "pads", pads, 4);
        out = ggml_pool_2d(c->ctx, a, GGML_OP_POOL_MAX,
                           (int)kshape[0], (int)kshape[1],
                           (int)strides[0], (int)strides[1],
                           (int)pads[0], (int)pads[1]);
    }
    else if (strcmp(op, "AveragePool") == 0) {
        if (!a) return -1;
        int64_t kshape[2] = {1, 1}, strides[2] = {1, 1}, pads[4] = {0};
        onnx_attr_ints(n, "kernel_shape", kshape, 2);
        onnx_attr_ints(n, "strides", strides, 2);
        onnx_attr_ints(n, "pads", pads, 4);
        out = ggml_pool_2d(c->ctx, a, GGML_OP_POOL_AVG,
                           (int)kshape[0], (int)kshape[1],
                           (int)strides[0], (int)strides[1],
                           (int)pads[0], (int)pads[1]);
    }
    else if (strcmp(op, "GlobalAveragePool") == 0) {
        if (!a) return -1;
        /* Pool over entire spatial dims */
        int h = (int)a->ne[1];
        int w = (int)a->ne[0];
        out = ggml_pool_2d(c->ctx, a, GGML_OP_POOL_AVG,
                           w, h, w, h, 0, 0);
    }

    /* ── Conv ───────────────────────────────────────────────────── */
    else if (strcmp(op, "Conv") == 0) {
        if (!a || !b) return -1;
        int64_t strides[2] = {1, 1}, pads[4] = {0}, dilations[2] = {1, 1};
        onnx_attr_ints(n, "strides", strides, 2);
        onnx_attr_ints(n, "pads", pads, 4);
        onnx_attr_ints(n, "dilations", dilations, 2);

        int ndims_kernel = ggml_n_dims(b);
        if (ndims_kernel <= 3) {
            /* Conv1D */
            out = ggml_conv_1d(c->ctx, b, a,
                               (int)strides[0], (int)pads[0], (int)dilations[0]);
        } else {
            /* Conv2D */
            out = ggml_conv_2d(c->ctx, b, a,
                               (int)strides[0], (int)strides[1],
                               (int)pads[0], (int)pads[1],
                               (int)dilations[0], (int)dilations[1]);
        }
        /* Add bias if present */
        struct ggml_tensor *bias = get_input(c, n, 2);
        if (bias) out = ggml_add(c->ctx, out, bias);
    }

    /* ── Reduction ──────────────────────────────────────────────── */
    else if (strcmp(op, "ReduceMean") == 0) {
        if (!a) return -1;
        out = ggml_mean(c->ctx, a);
    }
    else if (strcmp(op, "ReduceSum") == 0) {
        if (!a) return -1;
        out = ggml_sum(c->ctx, a);
    }

    /* ── Identity / Dropout (inference mode) ────────────────────── */
    else if (strcmp(op, "Identity") == 0 || strcmp(op, "Dropout") == 0) {
        if (!a) return -1;
        out = a; /* pass-through */
    }

    /* ── Constant (skip — should be in initializers) ────────────── */
    else if (strcmp(op, "Constant") == 0) {
        /* TODO: create tensor from attribute value */
        return 0; /* skip silently */
    }

    /* ── Unknown op ─────────────────────────────────────────────── */
    else {
        fprintf(stderr, "onnx_ggml: unsupported op '%s'\n", op);
        return -1;
    }

    /* Register outputs */
    if (out) {
        for (int i = 0; i < n->n_outputs; i++) {
            if (n->outputs[i][0] != '\0') {
                ggml_set_name(out, n->outputs[i]);
                tmap_put(c, n->outputs[i], out);
            }
        }
    }

    return 0;
}

/* ── Build full graph ───────────────────────────────────────────── */

onnx_ggml_ctx_t *onnx_ggml_build(onnx_model_t *onnx, const char *device) {
    onnx_ggml_ctx_t *c = calloc(1, sizeof(onnx_ggml_ctx_t));
    if (!c) return NULL;
    c->onnx = onnx;

    /* Estimate memory: rough heuristic based on file size */
    size_t mem_size = onnx->mmap_size * 2 + 256 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    c->ctx = ggml_init(params);
    if (!c->ctx) { free(c); return NULL; }

    /* Create tensors for initializers and inputs */
    if (create_initializer_tensors(c) != 0) goto fail;
    if (create_input_tensors(c) != 0) goto fail;

    /* Map ONNX nodes to ggml ops */
    for (int i = 0; i < onnx->n_nodes; i++) {
        if (map_node(c, &onnx->nodes[i]) != 0) {
            /* Non-fatal: skip unsupported ops with warning */
        }
    }

    /* Build forward graph from output tensors */
    {
        c->graph = ggml_new_graph(c->ctx);
        for (int i = 0; i < onnx->n_outputs; i++) {
            struct ggml_tensor *t = tmap_get(c, onnx->outputs[i].name);
            if (t) {
                ggml_set_output(t);
                ggml_build_forward_expand(c->graph, t);
            }
        }
    }

    /* Choose backend */
    int use_vulkan = 0;
    if (device == NULL || strcmp(device, "vulkan") == 0) {
#ifdef GGML_USE_VULKAN
        use_vulkan = 1;
#else
        if (device && strcmp(device, "vulkan") == 0) {
            fprintf(stderr, "onnx_ggml: Vulkan not available, falling back to CPU\n");
        }
#endif
    }

    if (use_vulkan) {
#ifdef GGML_USE_VULKAN
        c->backend = ggml_backend_vk_init(0);
        if (!c->backend) {
            fprintf(stderr, "onnx_ggml: Vulkan init failed, falling back to CPU\n");
            c->backend = ggml_backend_cpu_init();
        }
#endif
    } else {
        c->backend = ggml_backend_cpu_init();
    }
    if (!c->backend) goto fail;

    /* Allocate buffer */
    c->buffer = ggml_backend_alloc_ctx_tensors(c->ctx, c->backend);
    if (!c->buffer) goto fail;

    /* Load weights into allocated tensors */
    if (load_weights(c) != 0) goto fail;

    return c;

fail:
    onnx_ggml_free(c);
    return NULL;
}

int onnx_ggml_run(onnx_ggml_ctx_t *ctx,
                  const char **input_names, const float **input_data,
                  int n_inputs) {
    /* Set input data — R passes row-major data, ggml needs column-major */
    for (int i = 0; i < n_inputs; i++) {
        struct ggml_tensor *t = tmap_get(ctx, input_names[i]);
        if (!t) {
            fprintf(stderr, "onnx_ggml: input '%s' not found\n", input_names[i]);
            return -1;
        }
        size_t nbytes = ggml_nbytes(t);
        int ndims = ggml_n_dims(t);

        if (ndims == 2 && t->ne[0] > 1 && t->ne[1] > 1) {
            /* Transpose 2D input from row-major to column-major */
            int64_t rows = t->ne[0], cols = t->ne[1];
            size_t elem_size = sizeof(float);
            float *tmp = malloc(nbytes);
            if (tmp) {
                for (int64_t r = 0; r < rows; r++)
                    for (int64_t cc = 0; cc < cols; cc++)
                        tmp[cc * rows + r] = input_data[i][r * cols + cc];
                ggml_backend_tensor_set(t, tmp, 0, nbytes);
                free(tmp);
            }
        } else {
            ggml_backend_tensor_set(t, input_data[i], 0, nbytes);
        }
    }

    /* Compute */
    if (!ctx->graph) return -1;
    enum ggml_status status = ggml_backend_graph_compute(ctx->backend, ctx->graph);
    return (status == GGML_STATUS_SUCCESS) ? 0 : -1;
}

struct ggml_tensor *onnx_ggml_output(onnx_ggml_ctx_t *ctx, int index) {
    if (!ctx->onnx || index < 0 || index >= ctx->onnx->n_outputs)
        return NULL;
    return tmap_get(ctx, ctx->onnx->outputs[index].name);
}

void onnx_ggml_free(onnx_ggml_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->buffer)  ggml_backend_buffer_free(ctx->buffer);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    if (ctx->ctx)     ggml_free(ctx->ctx);
    free(ctx->tensor_map_keys);
    free(ctx->tensor_map_vals);
    /* Note: onnx model is NOT freed here — caller manages it */
    free(ctx);
}
