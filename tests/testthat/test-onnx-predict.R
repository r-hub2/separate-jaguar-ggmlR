# predict() for ONNX models -- the keras-compatible entry point.
#
# The reference values come from plain R arithmetic rather than onnx_run(),
# so these also pin down the row-major/column-major conversion: onnx_run()
# takes and returns flat ONNX-order buffers, predict() takes and returns
# ordinary R arrays.

test_that("predict() matches hand-computed Gemm for an exact batch", {
  B <- matrix(seq(0.1, 0.6, by = 0.1), 3, 2, byrow = TRUE)
  m <- onnx_load(.onnx_make_gemm(M = 4L, K = 3L, N = 2L,
                                 weight_data = seq(0.1, 0.6, by = 0.1)),
                 device = "cpu")
  set.seed(1)
  x <- matrix(rnorm(12), 4, 3)

  p <- predict(m, x)
  expect_true(is.matrix(p))
  expect_equal(dim(p), c(4L, 2L))
  expect_equal(p, x %*% B, tolerance = 1e-5, ignore_attr = TRUE)
})

test_that("predict() batches a sample count above the model's fixed batch", {
  B <- matrix(seq(0.1, 0.6, by = 0.1), 3, 2, byrow = TRUE)
  m <- onnx_load(.onnx_make_gemm(M = 4L, K = 3L, N = 2L,
                                 weight_data = seq(0.1, 0.6, by = 0.1)),
                 device = "cpu")
  set.seed(2)
  # 10 is not a multiple of 4: two full batches plus a padded tail.
  x <- matrix(rnorm(30), 10, 3)

  p <- predict(m, x)
  expect_equal(dim(p), c(10L, 2L))
  expect_equal(p, x %*% B, tolerance = 1e-5, ignore_attr = TRUE)
  # The padded rows must not leak into the result.
  expect_equal(as.vector(p[10, ]), as.vector(x[10, ] %*% B), tolerance = 1e-5)
})

test_that("predict() handles a single sample", {
  B <- matrix(seq(0.1, 0.6, by = 0.1), 3, 2, byrow = TRUE)
  m <- onnx_load(.onnx_make_gemm(M = 4L, K = 3L, N = 2L,
                                 weight_data = seq(0.1, 0.6, by = 0.1)),
                 device = "cpu")
  set.seed(3)
  x <- matrix(rnorm(3), 1, 3)

  p <- predict(m, x)
  expect_equal(dim(p), c(1L, 2L))
  expect_equal(p, x %*% B, tolerance = 1e-5, ignore_attr = TRUE)
})

test_that("predict() keeps non-2D output dimensions", {
  m <- onnx_load(.onnx_make_unary("Relu", input_dims = c(2L, 3L, 4L, 5L)),
                 device = "cpu")
  set.seed(7)
  a <- array(rnorm(2 * 3 * 4 * 5), c(2, 3, 4, 5))

  p <- predict(m, a)
  expect_equal(dim(p), c(2L, 3L, 4L, 5L))
  expect_equal(p, pmax(a, 0), tolerance = 1e-5, ignore_attr = TRUE)
})

test_that("predict() batches 4D input and rebinds along the sample axis", {
  m <- onnx_load(.onnx_make_unary("Relu", input_dims = c(2L, 3L, 4L, 5L)),
                 device = "cpu")
  set.seed(8)
  # 5 samples through a batch-2 model: two full chunks plus a padded one.
  a <- array(rnorm(5 * 3 * 4 * 5), c(5, 3, 4, 5))

  p <- predict(m, a)
  expect_equal(dim(p), c(5L, 3L, 4L, 5L))
  expect_equal(p, pmax(a, 0), tolerance = 1e-5, ignore_attr = TRUE)
})

test_that("predict() agrees with onnx_run() on the same batch", {
  m <- onnx_load(.onnx_make_gemm(M = 4L, K = 3L, N = 2L,
                                 weight_data = seq(0.1, 0.6, by = 0.1)),
                 device = "cpu")
  set.seed(4)
  x <- matrix(rnorm(12), 4, 3)

  # onnx_run() wants a flat row-major buffer and returns one; predict() does
  # that conversion internally.
  ref <- onnx_run(m, list(A = as.vector(t(x))))[[1L]]
  p   <- predict(m, x)
  expect_equal(as.vector(t(p)), as.vector(ref), tolerance = 1e-5)
})

test_that("predict() warns that batch_size is fixed at load time", {
  m <- onnx_load(.onnx_make_gemm(M = 4L, K = 3L, N = 2L,
                                 weight_data = seq(0.1, 0.6, by = 0.1)),
                 device = "cpu")
  set.seed(5)
  x <- matrix(rnorm(12), 4, 3)
  expect_warning(predict(m, x, batch_size = 8L), "fixed batch size")
})

test_that("predict() rejects malformed input", {
  m <- onnx_load(.onnx_make_gemm(M = 4L, K = 3L, N = 2L,
                                 weight_data = seq(0.1, 0.6, by = 0.1)),
                 device = "cpu")
  set.seed(6)
  x <- matrix(rnorm(12), 4, 3)

  expect_error(predict(m, list(x)), "must be named")
  expect_error(predict(m, list(WrongName = x)), "Missing input")
  expect_error(predict(m, matrix(numeric(0), 0, 3)), "No samples")
})
