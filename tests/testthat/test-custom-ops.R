library(ggmlR)

# Custom ops are registered from C, never from R. ggmlR ships three built-in
# kernels (registered in R_init_ggmlR), which is what makes a successful-compute
# test possible here; downstream packages register their own the same way.

test_that("ggml_custom_ops lists the built-in kernels", {
  ops <- ggml_custom_ops()
  expect_type(ops, "character")
  expect_true(all(c("row_median", "row_permute", "clip_inplace") %in% ops))
})

# ---------------------------------------------------------------------------
# row_median -- one input, caller-chosen output shape
# ---------------------------------------------------------------------------

test_that("row_median computes the median of each row (odd length)", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2)
  ggml_set_f32(x, c(5, 1, 3, 2, 4,
                    9, 7, 8, 6, 10))

  med <- ggml_custom(ctx, "row_median", args = list(x), ne = c(1, 2))
  graph <- ggml_build_forward_expand(ctx, med)
  ggml_graph_compute(ctx, graph)

  expect_equal(ggml_get_f32(med), c(3, 8), tolerance = 1e-5)
})

test_that("row_median averages the middle pair for even-length rows", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1)
  ggml_set_f32(x, c(1, 2, 3, 10))   # median = (2 + 3) / 2

  med <- ggml_custom(ctx, "row_median", args = list(x), ne = c(1, 1))
  graph <- ggml_build_forward_expand(ctx, med)
  ggml_graph_compute(ctx, graph)

  expect_equal(ggml_get_f32(med), 2.5, tolerance = 1e-5)
})

test_that("row_median agrees with stats::median on random data", {
  set.seed(1L)
  ctx <- ggml_init(32 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  n_col <- 7L
  n_row <- 4L
  vals  <- rnorm(n_col * n_row)

  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_col, n_row)
  ggml_set_f32(x, vals)

  med <- ggml_custom(ctx, "row_median", args = list(x), ne = c(1, n_row))
  graph <- ggml_build_forward_expand(ctx, med)
  ggml_graph_compute(ctx, graph)

  # ggml rows are contiguous runs of ne[0], i.e. rows of the n_col x n_row
  # matrix formed by filling columns first.
  expected <- apply(matrix(vals, nrow = n_col, ncol = n_row), 2, stats::median)
  expect_equal(ggml_get_f32(med), expected, tolerance = 1e-5)
})

# ---------------------------------------------------------------------------
# row_permute -- two inputs, the second an I32 index tensor
# ---------------------------------------------------------------------------

test_that("row_permute reorders elements within each row", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)
  ggml_set_f32(x, c(10, 20, 30, 40,
                    50, 60, 70, 80))

  # Reverse each row; indices are 0-based.
  perm <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4)
  ggml_set_i32(perm, c(3L, 2L, 1L, 0L))

  y <- ggml_custom(ctx, "row_permute", args = list(x, perm), ne = c(4, 2))
  graph <- ggml_build_forward_expand(ctx, y)
  ggml_graph_compute(ctx, graph)

  expect_equal(ggml_get_f32(y),
               c(40, 30, 20, 10,
                 80, 70, 60, 50),
               tolerance = 1e-5)
})

test_that("row_permute yields 0 for out-of-range indices rather than faulting", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1)
  ggml_set_f32(x, c(1, 2, 3))

  perm <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3)
  ggml_set_i32(perm, c(0L, 99L, -1L))   # only the first is in range

  y <- ggml_custom(ctx, "row_permute", args = list(x, perm), ne = c(3, 1))
  graph <- ggml_build_forward_expand(ctx, y)
  ggml_graph_compute(ctx, graph)

  expect_equal(ggml_get_f32(y), c(1, 0, 0), tolerance = 1e-5)
})

# ---------------------------------------------------------------------------
# clip_inplace -- ggml_custom_inplace, writing into src[0]
# ---------------------------------------------------------------------------

test_that("clip_inplace clamps into the given bounds", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(x, c(-3, -1, 0, 1, 3))

  bounds <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  ggml_set_f32(bounds, c(-1, 1))

  y <- ggml_custom_inplace(ctx, x, "clip_inplace", args = list(bounds))
  graph <- ggml_build_forward_expand(ctx, y)
  ggml_graph_compute(ctx, graph)

  expect_equal(ggml_get_f32(y), c(-1, -1, 0, 1, 1), tolerance = 1e-5)
})

test_that("clip_inplace writes through to the input tensor", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(x, c(-5, 0, 5, 10))

  bounds <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  ggml_set_f32(bounds, c(0, 5))

  y <- ggml_custom_inplace(ctx, x, "clip_inplace", args = list(bounds))
  graph <- ggml_build_forward_expand(ctx, y)
  ggml_graph_compute(ctx, graph)

  # The whole point of the in-place path: x itself is modified.
  expect_equal(ggml_get_f32(x), c(0, 0, 5, 5), tolerance = 1e-5)
})

test_that("unknown custom op name is a clean error, not a crash", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)

  err <- expect_error(
    ggml_custom(ctx, "no_such_kernel", args = list(x), ne = 4),
    "no_such_kernel"
  )
  # The built-ins are registered, so the message lists what is available.
  expect_match(conditionMessage(err), "row_median")
})

test_that("ggml_custom_inplace rejects an unknown name too", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)

  expect_error(
    ggml_custom_inplace(ctx, x, "no_such_kernel"),
    "no_such_kernel"
  )
})

test_that("custom op name must be a character scalar", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  expect_error(ggml_custom(ctx, 42, ne = 4), "single character string")
  expect_error(ggml_custom(ctx, c("a", "b"), ne = 4), "single character string")
  expect_error(ggml_custom(ctx, character(0), ne = 4), "single character string")
})

test_that("'args' must be a list when supplied", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  expect_error(ggml_custom(ctx, "k", args = x, ne = 4), "must be a list")
})

test_that("output shape is validated before reaching C", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  expect_error(ggml_custom(ctx, "k", ne = numeric(0)), "between 1 and 4")
  expect_error(ggml_custom(ctx, "k", ne = c(1, 2, 3, 4, 5)), "between 1 and 4")
  expect_error(ggml_custom(ctx, "k", ne = c(4, 0)), "positive dimensions")
  expect_error(ggml_custom(ctx, "k", ne = c(4, NA)), "positive dimensions")
})

test_that("GGML_N_TASKS_MAX is the documented sentinel", {
  expect_identical(GGML_N_TASKS_MAX, -1L)
})
