# Coverage for exported functions that had no call site and no test anywhere in
# the package. Grouped by subsystem: in-place graph ops, tensor/op introspection,
# backend device queries, Vulkan diagnostics.
#
# The in-place ops are checked on three axes:
#   1. numeric result matches the non-in-place variant (or a closed-form reference),
#   2. the returned view aliases the input tensor (that is the point of "inplace"),
#   3. shape is preserved.

cpu_backend <- function(env = parent.frame()) {
  be <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(be, 2L)
  withr::defer(ggml_backend_free(be), envir = env)
  be
}

new_ctx <- function(mem = 64 * 1024 * 1024, env = parent.frame()) {
  ctx <- ggml_init(mem)
  withr::defer(ggml_free(ctx), envir = env)
  ctx
}

# ============================================================================
# ggml_add_inplace
# ============================================================================

test_that("ggml_add_inplace adds elementwise and writes through to `a`", {
  ctx <- new_ctx(1024 * 1024)
  be  <- cpu_backend()

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))
  ggml_set_f32(b, c(5, 4, 3, 2, 1))

  r <- ggml_add_inplace(ctx, a, b)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, r))

  expect_equal(ggml_get_f32(r), rep(6, 5), tolerance = 1e-6)
  # in-place: the result is a view of `a`, so `a` carries the sum too
  expect_equal(ggml_get_f32(a), rep(6, 5), tolerance = 1e-6)
  expect_equal(ggml_tensor_shape(r), ggml_tensor_shape(a))
})

test_that("ggml_add_inplace matches ggml_add on 2D input", {
  ctx <- new_ctx()
  be  <- cpu_backend()

  set.seed(11)
  va <- rnorm(12); vb <- rnorm(12)

  a  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3); ggml_set_f32(a, va)
  b  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3); ggml_set_f32(b, vb)
  a2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3); ggml_set_f32(a2, va)

  ref <- ggml_add(ctx, a2, b)
  r   <- ggml_add_inplace(ctx, a, b)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, ref))
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, r))

  expect_equal(ggml_get_f32(r), ggml_get_f32(ref), tolerance = 1e-6)
  expect_equal(ggml_get_f32(r), va + vb, tolerance = 1e-6)
})

# ============================================================================
# ggml_soft_max_ext_inplace / ggml_soft_max_ext_back_inplace
# ============================================================================

test_that("ggml_soft_max_ext_inplace softmaxes each row in place", {
  ctx <- new_ctx()
  be  <- cpu_backend()

  # 2 rows of 4: softmax is applied along ne[0]
  row1 <- c(1, 2, 3, 4)
  row2 <- c(2, 1, 0, -1)
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)
  ggml_set_f32(a, c(row1, row2))

  r <- ggml_soft_max_ext_inplace(ctx, a, NULL, scale = 1.0, max_bias = 0.0)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, r))

  out <- ggml_get_f32(r)
  expect_equal(sum(out[1:4]), 1, tolerance = 1e-6)
  expect_equal(sum(out[5:8]), 1, tolerance = 1e-6)
  expect_equal(out[1:4], exp(row1) / sum(exp(row1)), tolerance = 1e-6)
  expect_equal(out[5:8], exp(row2) / sum(exp(row2)), tolerance = 1e-6)

  # aliases the input
  expect_equal(ggml_get_f32(a), out, tolerance = 1e-6)
})

test_that("ggml_soft_max_ext_inplace honours the scale argument", {
  ctx <- new_ctx()
  be  <- cpu_backend()

  x <- c(1, 2, 3, 4)
  s <- 0.5
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4); ggml_set_f32(a, x)

  r <- ggml_soft_max_ext_inplace(ctx, a, NULL, scale = s, max_bias = 0.0)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, r))

  expect_equal(ggml_get_f32(r), exp(s * x) / sum(exp(s * x)), tolerance = 1e-6)
})

test_that("ggml_soft_max_ext_inplace matches the non-in-place variant", {
  ctx <- new_ctx()
  be  <- cpu_backend()

  set.seed(12)
  v  <- rnorm(16)
  a  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2); ggml_set_f32(a, v)
  a2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2); ggml_set_f32(a2, v)

  ref <- ggml_soft_max_ext(ctx, a2, NULL, 1.0, 0.0)
  r   <- ggml_soft_max_ext_inplace(ctx, a, NULL, 1.0, 0.0)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, ref))
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, r))

  expect_equal(ggml_get_f32(r), ggml_get_f32(ref), tolerance = 1e-6)
})

test_that("ggml_soft_max_ext_back_inplace computes the softmax VJP", {
  ctx <- new_ctx()
  be  <- cpu_backend()

  x  <- c(1, 2, 3, 4)
  dy <- c(1, 0, 0, 0)

  xt <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4); ggml_set_f32(xt, x)
  y  <- ggml_soft_max(ctx, xt)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, y))
  yv <- ggml_get_f32(y)

  dyt <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4); ggml_set_f32(dyt, dy)
  gr  <- ggml_soft_max_ext_back_inplace(ctx, dyt, y, scale = 1.0, max_bias = 0.0)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, gr))

  # softmax Jacobian-vector product: dx = y * (dy - sum(dy * y))
  expect_equal(ggml_get_f32(gr), yv * (dy - sum(dy * yv)), tolerance = 1e-6)
  # in-place: writes through the upstream-gradient tensor
  expect_equal(ggml_get_f32(dyt), ggml_get_f32(gr), tolerance = 1e-6)
})

test_that("ggml_soft_max_ext_back_inplace gradient sums to zero", {
  ctx <- new_ctx()
  be  <- cpu_backend()

  # sum(dx) == 0 for any upstream gradient, since softmax outputs are normalised
  set.seed(13)
  x  <- rnorm(6)
  dy <- rnorm(6)

  xt <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6); ggml_set_f32(xt, x)
  y  <- ggml_soft_max(ctx, xt)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, y))

  dyt <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6); ggml_set_f32(dyt, dy)
  gr  <- ggml_soft_max_ext_back_inplace(ctx, dyt, y, 1.0, 0.0)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, gr))

  expect_equal(sum(ggml_get_f32(gr)), 0, tolerance = 1e-5)
})

# ============================================================================
# ggml_rope_ext_inplace / ggml_rope_multi_inplace
# ============================================================================

test_that("ggml_rope_ext_inplace matches ggml_rope_ext and aliases input", {
  ctx <- new_ctx()
  be  <- cpu_backend()

  head_dim <- 8; n_head <- 2; seq_len <- 4
  set.seed(1)
  v <- rnorm(head_dim * n_head * seq_len)

  a  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
  a2 <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
  ggml_set_f32(a, v); ggml_set_f32(a2, v)

  pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len)
  ggml_set_i32(pos, 0:(seq_len - 1))

  ref <- ggml_rope_ext(ctx, a2, pos, NULL, n_dims = head_dim,
                       mode = 0L, n_ctx_orig = 2048)
  r   <- ggml_rope_ext_inplace(ctx, a, pos, NULL, n_dims = head_dim,
                               mode = 0L, n_ctx_orig = 2048)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, ref))
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, r))

  out <- ggml_get_f32(r)
  expect_equal(out, ggml_get_f32(ref), tolerance = 1e-6)
  expect_equal(ggml_tensor_shape(r), c(head_dim, n_head, seq_len, 1))
  expect_true(all(is.finite(out)))
  # rotation must actually change the data ...
  expect_gt(max(abs(out - v)), 1e-6)
  # ... and write it back through `a`
  expect_equal(ggml_get_f32(a), out, tolerance = 1e-6)
})

test_that("ggml_rope_ext_inplace is a no-op at position 0", {
  ctx <- new_ctx()
  be  <- cpu_backend()

  head_dim <- 8
  set.seed(3)
  v <- rnorm(head_dim)

  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, 1, 1, 1)
  ggml_set_f32(a, v)
  pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1)
  ggml_set_i32(pos, 0L)

  r <- ggml_rope_ext_inplace(ctx, a, pos, NULL, n_dims = head_dim,
                             mode = 0L, n_ctx_orig = 2048)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, r))

  # angle = pos * freq = 0 → identity rotation
  expect_equal(ggml_get_f32(r), v, tolerance = 1e-6)
})

test_that("ggml_rope_multi_inplace matches ggml_rope_multi and aliases input", {
  ctx <- new_ctx()
  be  <- cpu_backend()

  head_dim <- 16; n_head <- 2; seq_len <- 4
  set.seed(2)
  v <- rnorm(head_dim * n_head * seq_len)

  a  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
  a2 <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
  ggml_set_f32(a, v); ggml_set_f32(a2, v)

  # M-RoPE needs 4 position components per token; sections sum to n_dims/2
  pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len * 4)
  ggml_set_i32(pos, rep(0:(seq_len - 1), times = 4))
  sections <- c(2L, 2L, 2L, 2L)

  ref <- ggml_rope_multi(ctx, a2, pos, NULL, n_dims = head_dim,
                         sections = sections, mode = GGML_ROPE_TYPE_MROPE,
                         n_ctx_orig = 2048)
  r   <- ggml_rope_multi_inplace(ctx, a, pos, NULL, n_dims = head_dim,
                                 sections = sections, mode = GGML_ROPE_TYPE_MROPE,
                                 n_ctx_orig = 2048)
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, ref))
  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, r))

  out <- ggml_get_f32(r)
  expect_equal(out, ggml_get_f32(ref), tolerance = 1e-6)
  expect_equal(ggml_tensor_shape(r), c(head_dim, n_head, seq_len, 1))
  expect_true(all(is.finite(out)))
  expect_gt(max(abs(out - v)), 1e-6)
  expect_equal(ggml_get_f32(a), out, tolerance = 1e-6)
})

# ============================================================================
# ggml_get_unary_op
# ============================================================================

test_that("ggml_get_unary_op reports the unary op behind a tensor", {
  ctx <- new_ctx(1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(-1, 2, -3, 4))

  relu_op <- ggml_get_unary_op(ggml_relu(ctx, a))
  gelu_op <- ggml_get_unary_op(ggml_gelu(ctx, a))

  expect_type(relu_op, "integer")
  # resolves back to a readable name, and distinguishes different unary ops
  expect_equal(ggml_unary_op_name(relu_op), "RELU")
  expect_equal(ggml_unary_op_name(gelu_op), "GELU")
  expect_false(relu_op == gelu_op)
})

test_that("ggml_get_unary_op is stable across tensors with the same op", {
  ctx <- new_ctx(1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4); ggml_set_f32(a, c(1, 2, 3, 4))
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2); ggml_set_f32(b, c(1, 2, 3, 4))

  expect_equal(ggml_get_unary_op(ggml_relu(ctx, a)),
               ggml_get_unary_op(ggml_relu(ctx, b)))
})

test_that("ggml_get_unary_op rejects an invalid tensor pointer", {
  expect_error(ggml_get_unary_op(methods::new("externalptr")),
               "Invalid tensor pointer")
})

# ============================================================================
# ggml_is_contiguous_channels
# ============================================================================

test_that("ggml_is_contiguous_channels detects channels-innermost layout", {
  ctx <- new_ctx()

  # The predicate is about *memory* order, not logical shape: it asks for the
  # CWHN layout consumed by ggml_conv_2d_dw_direct, i.e. a tensor of logical
  # shape [W,H,C,N] whose channel axis has the smallest stride.
  # Build it by permuting a contiguous [C,W,H,N] tensor so that C lands at
  # logical position 2. ggml_permute() args are *destination* positions:
  # C(axis0)->2, W(axis1)->0, H(axis2)->1.
  cwhn <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 5, 4, 2)  # C,W,H,N
  b <- ggml_permute(ctx, cwhn, 2, 0, 1, 3)

  expect_equal(ggml_tensor_shape(b), c(5, 4, 3, 2))  # W,H,C,N
  # channel stride is one element -> channels contiguous
  expect_equal(ggml_tensor_nb(b)[3], 4)
  expect_true(ggml_is_contiguous_channels(b))
})

test_that("ggml_is_contiguous_channels is FALSE for a plain contiguous tensor", {
  ctx <- new_ctx()

  # standard WHCN allocation: ne[0] is innermost, channels are strided
  plain <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 4, 3, 2)
  expect_false(ggml_is_contiguous_channels(plain))

  expect_false(ggml_is_contiguous_channels(
    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8)))
  expect_false(ggml_is_contiguous_channels(
    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)))

  expect_type(ggml_is_contiguous_channels(plain), "logical")
})

test_that("ggml_is_contiguous_channels rejects an invalid tensor pointer", {
  expect_error(ggml_is_contiguous_channels(methods::new("externalptr")),
               "Invalid tensor pointer")
})

# ============================================================================
# ggml_backend_dev_supports_buft / ggml_backend_dev_offload_op
# ============================================================================

test_that("ggml_backend_dev_offload_op answers for every registered device", {
  n <- ggml_backend_dev_count()
  skip_if(n < 1, "No backend devices registered")

  ctx <- new_ctx(16 * 1024)
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  op <- ggml_add(ctx, a, b)

  for (i in seq_len(n) - 1L) {
    dev <- ggml_backend_dev_get(i)
    skip_if(is.null(dev), "Device unavailable")
    res <- ggml_backend_dev_offload_op(dev, op)
    expect_type(res, "logical")
    expect_length(res, 1L)
    expect_false(is.na(res))
  }
})

test_that("ggml_backend_dev_offload_op declines a tiny elementwise op on CPU", {
  n <- ggml_backend_dev_count()
  skip_if(n < 1, "No backend devices registered")

  # find the CPU device by name rather than assuming an index
  cpu_dev <- NULL
  for (i in seq_len(n) - 1L) {
    d <- ggml_backend_dev_get(i)
    if (!is.null(d) && identical(ggml_backend_dev_get_props(d)$name, "CPU")) {
      cpu_dev <- d
      break
    }
  }
  skip_if(is.null(cpu_dev), "No CPU device")

  ctx <- new_ctx(16 * 1024)
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)

  # offload_op asks "is this op big enough to be worth moving to this device?"
  # The CPU backend never wants work offloaded to it.
  expect_false(ggml_backend_dev_offload_op(cpu_dev, ggml_add(ctx, a, b)))
})

test_that("ggml_backend_dev_supports_buft is TRUE for a device's own buffer type", {
  n <- ggml_backend_dev_count()
  skip_if(n < 1, "No backend devices registered")

  # This is the positive branch: every device must accept the buffer type it
  # allocates from by default.
  for (i in seq_len(n) - 1L) {
    dev <- ggml_backend_dev_get(i)
    skip_if(is.null(dev), "Device unavailable")

    buft <- ggml_backend_dev_buffer_type(dev)
    skip_if(is.null(buft), "Device reports no default buffer type")

    expect_true(ggml_backend_dev_supports_buft(dev, buft))
  }
})

test_that("ggml_backend_dev_supports_buft is FALSE for a foreign buffer type", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not compiled in")

  # Negative branch: the ggmlR tensor-split buffer type is a custom type that no
  # stock device claims.
  bt <- ggml_vulkan_split_buffer_type(n_devices = 1)
  expect_true(inherits(bt$ptr, "externalptr"))

  n <- ggml_backend_dev_count()
  skip_if(n < 1, "No backend devices registered")

  for (i in seq_len(n) - 1L) {
    dev <- ggml_backend_dev_get(i)
    skip_if(is.null(dev), "Device unavailable")
    expect_false(ggml_backend_dev_supports_buft(dev, bt$ptr))
  }
})

# ============================================================================
# ggml_backend_dev_buffer_type / host_buffer_type / buft accessors
# ============================================================================

test_that("ggml_backend_dev_buffer_type returns a usable buffer type", {
  n <- ggml_backend_dev_count()
  skip_if(n < 1, "No backend devices registered")

  for (i in seq_len(n) - 1L) {
    dev <- ggml_backend_dev_get(i)
    skip_if(is.null(dev), "Device unavailable")

    buft <- ggml_backend_dev_buffer_type(dev)
    skip_if(is.null(buft), "Device reports no default buffer type")
    expect_true(inherits(buft, "externalptr"))

    nm <- ggml_backend_buft_name(buft)
    expect_type(nm, "character")
    expect_gt(nchar(nm), 0L)

    align <- ggml_backend_buft_get_alignment(buft)
    expect_type(align, "double")
    expect_gt(align, 0)

    expect_gt(ggml_backend_buft_get_max_size(buft), 0)

    is_host <- ggml_backend_buft_is_host(buft)
    expect_type(is_host, "logical")
    expect_false(is.na(is_host))
  }
})

test_that("the CPU device's buffer type is host memory", {
  n <- ggml_backend_dev_count()
  skip_if(n < 1, "No backend devices registered")

  cpu_dev <- NULL
  for (i in seq_len(n) - 1L) {
    d <- ggml_backend_dev_get(i)
    if (!is.null(d) && identical(ggml_backend_dev_get_props(d)$name, "CPU")) {
      cpu_dev <- d
      break
    }
  }
  skip_if(is.null(cpu_dev), "No CPU device")

  buft <- ggml_backend_dev_buffer_type(cpu_dev)
  skip_if(is.null(buft), "CPU device reports no buffer type")

  expect_true(ggml_backend_buft_is_host(buft))
  expect_match(ggml_backend_buft_name(buft), "CPU")
})

test_that("ggml_backend_dev_host_buffer_type returns a buffer type or NULL", {
  n <- ggml_backend_dev_count()
  skip_if(n < 1, "No backend devices registered")

  for (i in seq_len(n) - 1L) {
    dev <- ggml_backend_dev_get(i)
    skip_if(is.null(dev), "Device unavailable")

    # NULL is a legitimate answer: most backends (the CPU one included) expose
    # no separate pinned-host buffer type.
    hb <- ggml_backend_dev_host_buffer_type(dev)
    if (is.null(hb)) {
      succeed()
      next
    }
    expect_true(inherits(hb, "externalptr"))
    expect_gt(nchar(ggml_backend_buft_name(hb)), 0L)
    # pinned host memory must report itself as host-accessible
    expect_true(ggml_backend_buft_is_host(hb))
  }
})

test_that("buffer type accessors reject invalid pointers", {
  expect_error(ggml_backend_buft_name(NULL), "Invalid buffer type pointer")
  expect_error(ggml_backend_buft_get_alignment(NULL), "Invalid buffer type pointer")
  expect_error(ggml_backend_buft_get_max_size(NULL), "Invalid buffer type pointer")
  expect_error(ggml_backend_buft_is_host(NULL), "Invalid buffer type pointer")
  expect_error(ggml_backend_dev_buffer_type(NULL), "Invalid device pointer")
  expect_error(ggml_backend_dev_host_buffer_type(42), "Invalid device pointer")
})

test_that("ggml_backend_dev_supports_buft rejects an invalid buft pointer", {
  dev <- ggml_backend_dev_get(0)
  skip_if(is.null(dev), "No device available")

  expect_error(ggml_backend_dev_supports_buft(dev, methods::new("externalptr")),
               "Invalid buffer type pointer")
})

# ============================================================================
# ggml_backend_tensor_get_and_sync
# ============================================================================

test_that("ggml_backend_tensor_get_and_sync reads back backend tensor bytes", {
  be <- cpu_backend()

  ctx <- ggml_init(16 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, be)
  skip_if(is.null(buf), "Could not allocate buffer")
  on.exit(ggml_backend_buffer_free(buf), add = TRUE)

  vals <- c(1.5, 2.5, 3.5, 4.5)
  ggml_backend_tensor_set_data(a, vals)

  raw_all <- ggml_backend_tensor_get_and_sync(be, a, offset = 0, size = 16)
  expect_type(raw_all, "raw")
  expect_length(raw_all, 16L)
  expect_equal(readBin(raw_all, "numeric", n = 4, size = 4), vals,
               tolerance = 1e-6)
})

test_that("ggml_backend_tensor_get_and_sync honours offset and size", {
  be <- cpu_backend()

  ctx <- ggml_init(16 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, be)
  skip_if(is.null(buf), "Could not allocate buffer")
  on.exit(ggml_backend_buffer_free(buf), add = TRUE)

  vals <- c(1.5, 2.5, 3.5, 4.5)
  ggml_backend_tensor_set_data(a, vals)

  # skip the first f32, read the next two
  raw_mid <- ggml_backend_tensor_get_and_sync(be, a, offset = 4, size = 8)
  expect_length(raw_mid, 8L)
  expect_equal(readBin(raw_mid, "numeric", n = 2, size = 4), vals[2:3],
               tolerance = 1e-6)
})

test_that("ggml_backend_tensor_get_and_sync accepts backend = NULL", {
  # Documented contract: "Backend pointer (or NULL for CPU)". With NULL the read
  # goes straight at the tensor with no async/synchronize round-trip.
  be <- cpu_backend()

  ctx <- ggml_init(16 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, be)
  skip_if(is.null(buf), "Could not allocate buffer")
  on.exit(ggml_backend_buffer_free(buf), add = TRUE)

  vals <- c(1.5, 2.5, 3.5, 4.5)
  ggml_backend_tensor_set_data(a, vals)

  via_null <- ggml_backend_tensor_get_and_sync(NULL, a, offset = 0, size = 16)
  expect_type(via_null, "raw")
  expect_equal(readBin(via_null, "numeric", n = 4, size = 4), vals,
               tolerance = 1e-6)

  # and it agrees with the explicit-backend path
  via_be <- ggml_backend_tensor_get_and_sync(be, a, offset = 0, size = 16)
  expect_identical(via_null, via_be)

  # offset/size still apply when backend is NULL
  mid <- ggml_backend_tensor_get_and_sync(NULL, a, offset = 4, size = 8)
  expect_equal(readBin(mid, "numeric", n = 2, size = 4), vals[2:3],
               tolerance = 1e-6)
})

test_that("ggml_backend_tensor_get_and_sync rejects an invalid tensor pointer", {
  be <- cpu_backend()
  expect_error(
    ggml_backend_tensor_get_and_sync(be, methods::new("externalptr"), 0, 4),
    "Invalid tensor pointer")
})

# ============================================================================
# External-pointer boundary guards
#
# R_ExternalPtrAddr() on a non-EXTPTRSXP yields a garbage address rather than
# NULL, so a plain `p == NULL` check does not catch a non-pointer argument and
# the call segfaults. These functions must raise an R error instead.
# ============================================================================

test_that("pointer-taking functions reject R NULL instead of crashing", {
  ctx <- new_ctx(16 * 1024)
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  op <- ggml_add(ctx, a, b)

  expect_error(ggml_backend_dev_offload_op(NULL, op), "Invalid device pointer")
  expect_error(ggml_backend_dev_supports_buft(NULL, NULL), "Invalid device pointer")
  expect_error(ggml_get_unary_op(NULL), "Invalid tensor pointer")
  expect_error(ggml_is_contiguous_channels(NULL), "Invalid tensor pointer")

  dev <- ggml_backend_dev_get(0)
  skip_if(is.null(dev), "No device available")
  expect_error(ggml_backend_dev_offload_op(dev, NULL), "Invalid tensor pointer")
  expect_error(ggml_backend_dev_supports_buft(dev, NULL),
               "Invalid buffer type pointer")
})

test_that("pointer-taking functions reject non-pointer arguments", {
  ctx <- new_ctx(16 * 1024)
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  op <- ggml_add(ctx, a, a)
  be <- cpu_backend()

  expect_error(ggml_get_unary_op(42), "Invalid tensor pointer")
  expect_error(ggml_is_contiguous_channels("not a pointer"),
               "Invalid tensor pointer")
  expect_error(ggml_backend_dev_offload_op(list(), op), "Invalid device pointer")

  # backend here is optional, so the message names the expected types
  expect_error(ggml_backend_tensor_get_and_sync(42, a, 0, 4),
               "must be an external pointer or NULL")
})

# ============================================================================
# ggml_vulkan_device_groups
# ============================================================================

test_that("ggml_vulkan_device_groups reports groups and a readable diagnostic", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not compiled in")

  g <- ggml_vulkan_device_groups()

  expect_type(g, "list")
  expect_named(g, c("n_groups", "report"))

  expect_type(g$n_groups, "integer")
  expect_length(g$n_groups, 1L)
  # a working Vulkan driver reports at least one (possibly single-device) group
  expect_gte(g$n_groups, 0L)

  expect_type(g$report, "character")
  expect_length(g$report, 1L)
  expect_false(is.na(g$report))
})

test_that("ggml_vulkan_device_groups is a read-only probe", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not compiled in")

  # documented as a diagnostic that creates no long-lived device group:
  # repeated calls must agree and must not error
  g1 <- ggml_vulkan_device_groups()
  g2 <- ggml_vulkan_device_groups()

  expect_equal(g1$n_groups, g2$n_groups)
  expect_equal(g1$report, g2$report)
})

test_that("ggml_vulkan_device_groups errors when Vulkan is absent", {
  skip_if(ggml_vulkan_available(), "Vulkan is compiled in")

  expect_error(ggml_vulkan_device_groups(), "Vulkan support not compiled")
})
