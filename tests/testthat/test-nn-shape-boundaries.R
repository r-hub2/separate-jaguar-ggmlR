# Boundary cases in the sequential API's shape handling.
#
# Three defects found in one sitting all shared a shape: they broke on a
# boundary the tests never visited.
#
#   * nn_build_flatten() read the batch size as shape[ggml_n_dims(t)]. ggml
#     reports a trailing unit dimension as absent, so a batch of 1 makes a
#     [W, H, C, 1] input look 3-D and the batch size came off the channel axis.
#   * The same line existed in the functional API's flatten.
#   * batch_norm on conv-shaped inputs normalized with ggml_rms_norm(), and its
#     gamma/beta for a 3-D input were reshaped onto the sequence axis rather
#     than the channel axis.
#
# What they have in common is that predicting one sample at a time, or training
# a non-flat input shape, was never exercised. These tests visit those corners
# for every input rank the sequential API supports.

# A model's predictions must not depend on how samples are grouped into
# batches, so predicting one at a time has to match predicting them together.
expect_batch_invariant <- function(model, x, n_check = 4L) {
  p_all <- ggml_predict(model, x, batch_size = 8L)
  idx <- seq_len(min(n_check, dim(x)[1]))
  p_one <- do.call(rbind, lapply(idx, function(i) {
    ggml_predict(model, slice_first(x, i), batch_size = 1L)
  }))
  expect_equal(nrow(p_one), length(idx))
  expect_true(all(is.finite(p_one)))
  expect_lt(max(abs(p_one - p_all[idx, , drop = FALSE])), 1e-4)
}

# Keep the leading (sample) dimension while slicing out one sample.
slice_first <- function(x, i) {
  if (length(dim(x)) == 2L) x[i, , drop = FALSE]
  else if (length(dim(x)) == 3L) x[i, , , drop = FALSE]
  else x[i, , , , drop = FALSE]
}

shape_labels <- function(n) {
  y <- matrix(0, n, 2L)
  lab <- rep(c(1L, 2L), length.out = n)
  for (i in seq_len(n)) y[i, lab[i]] <- 1
  list(y = y, lab = lab)
}

# ── 1. batch_size = 1 for every input rank ───────────────────

test_that("predict with batch_size 1 matches batched predict (flat input)", {
  set.seed(11)
  n <- 16L
  x <- matrix(rnorm(n * 5L), n, 5L)
  d <- shape_labels(n)

  m <- ggml_model_sequential() |>
    ggml_layer_dense(8L, activation = "relu", input_shape = 5L) |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                 backend = "cpu")
  m <- ggml_fit(m, x, d$y, epochs = 5L, batch_size = 8L, verbose = 0L)

  expect_batch_invariant(m, x)
})

test_that("predict with batch_size 1 matches batched predict (conv_2d input)", {
  # The case that used to abort: [W, H, C, 1] reports ggml_n_dims() == 3, so
  # flatten read the batch size off the channel axis.
  set.seed(12)
  n <- 16L; h <- 6L; w <- 6L; ch <- 1L
  x <- array(rnorm(n * h * w * ch), dim = c(n, h, w, ch))
  d <- shape_labels(n)

  m <- ggml_model_sequential() |>
    ggml_layer_conv_2d(filters = 4L, kernel_size = c(3L, 3L),
                       activation = "relu", input_shape = c(h, w, ch)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                 backend = "cpu")
  m <- ggml_fit(m, x, d$y, epochs = 5L, batch_size = 8L, verbose = 0L)

  expect_batch_invariant(m, x)
})

test_that("predict with batch_size 1 matches batched predict (conv_1d input)", {
  set.seed(13)
  n <- 16L; sl <- 8L; feat <- 3L
  x <- array(rnorm(n * sl * feat), dim = c(n, sl, feat))
  d <- shape_labels(n)

  m <- ggml_model_sequential() |>
    ggml_layer_conv_1d(filters = 5L, kernel_size = 3L, activation = "relu",
                       input_shape = c(sl, feat)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                 backend = "cpu")
  m <- ggml_fit(m, x, d$y, epochs = 5L, batch_size = 8L, verbose = 0L)

  expect_batch_invariant(m, x)
})

# ── 2. 3-D input with batch_norm ─────────────────────────────

test_that("batch_norm on a 3-D input normalizes per channel, not per timestep", {
  # R input_shape c(seq, size) becomes ggml [size, seq, N], so the channel is
  # ne[0]. gamma/beta used to be reshaped to [1, C, 1], putting the per-channel
  # scale on the sequence axis instead.
  set.seed(14)
  n <- 32L; sl <- 6L; feat <- 4L
  x <- array(0, dim = c(n, sl, feat))
  # Distinct offset and scale per channel: batch normalization must remove both.
  for (c in seq_len(feat)) {
    x[, , c] <- rnorm(n * sl, mean = 3 * c, sd = 0.5 * c)
  }
  d <- shape_labels(n)

  # Seed the weight initialization, not just the data -- see the conv_2d test.
  set.seed(102L)
  m <- ggml_model_sequential() |>
    ggml_layer_conv_1d(filters = 5L, kernel_size = 3L, activation = "relu",
                       input_shape = c(sl, feat)) |>
    ggml_layer_batch_norm() |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                 backend = "cpu")

  # The batch_norm layer sees the conv output: input_shape c(OL, filters).
  expect_equal(length(m$layers[[2]]$input_shape), 2L)

  m <- ggml_fit(m, x, d$y, epochs = 10L, batch_size = 8L, verbose = 0L)

  tl <- m$history$train_loss
  expect_true(all(is.finite(tl)))
  expect_lt(tl[length(tl)], tl[1])
  # Inference uses the running estimates, so it cannot depend on batching.
  expect_batch_invariant(m, x)
})

test_that("batch_norm gamma is sized from the channel axis for a 3-D input", {
  set.seed(15)
  n <- 16L; sl <- 6L; feat <- 4L
  x <- array(rnorm(n * sl * feat), dim = c(n, sl, feat))
  d <- shape_labels(n)

  filters <- 5L
  m <- ggml_model_sequential() |>
    ggml_layer_conv_1d(filters = filters, kernel_size = 3L,
                       input_shape = c(sl, feat)) |>
    ggml_layer_batch_norm() |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                 backend = "cpu")
  m <- ggml_fit(m, x, d$y, epochs = 2L, batch_size = 8L, verbose = 0L)

  # One scale/shift per channel -- i.e. per conv filter, not per timestep.
  expect_equal(ggml_nelements(m$layers[[2]]$weights$gamma), filters)
  expect_equal(ggml_nelements(m$layers[[2]]$weights$running_mean), filters)
})

# ── 3. Training a conv net with batch_norm ───────────────────

test_that("conv_2d + batch_norm trains and stays batch-invariant", {
  # Exercises the conv-shaped batch_norm backward pass, which reaches the
  # channel axis through a permute.
  set.seed(16)
  n <- 64L; h <- 6L; w <- 6L; ch <- 1L
  x <- array(rnorm(n * h * w * ch, mean = 3, sd = 2), dim = c(n, h, w, ch))
  d <- shape_labels(n)
  x[d$lab == 2L, , , ] <- x[d$lab == 2L, , , ] + 1.5

  # Weights are drawn with runif() at compile time, so the seed has to be set
  # here rather than before the data: otherwise the initialization depends on
  # how many random numbers the data generation happened to consume, and the
  # accuracy this test asserts on swings between runs (0.77 to 0.98 observed).
  set.seed(100L)
  m <- ggml_model_sequential() |>
    ggml_layer_conv_2d(filters = 4L, kernel_size = c(3L, 3L),
                       activation = "relu", input_shape = c(h, w, ch)) |>
    ggml_layer_batch_norm() |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                 backend = "cpu")
  m <- ggml_fit(m, x, d$y, epochs = 40L, batch_size = 16L, verbose = 0L)

  tl <- m$history$train_loss
  expect_true(all(is.finite(tl)))
  expect_lt(tl[length(tl)], tl[1])

  p <- ggml_predict(m, x, batch_size = 32L)
  expect_gt(mean(max.col(p) == max.col(d$y)), 0.85)
  expect_batch_invariant(m, x)
})

test_that("conv_1d trains to convergence on a separable problem", {
  # conv_1d used to be unusable in the sequential API: the input axes did not
  # match what the convolution wanted, and the result was reassembled in a way
  # that only held for a batch of 1 -- training ran but the loss went up.
  set.seed(17)
  n <- 64L; sl <- 10L; feat <- 4L
  x <- array(rnorm(n * sl * feat), dim = c(n, sl, feat))
  d <- shape_labels(n)
  x[d$lab == 2L, , ] <- x[d$lab == 2L, , ] + 1.2

  # Seed the weight initialization, not just the data -- see the conv_2d test.
  set.seed(101L)
  m <- ggml_model_sequential() |>
    ggml_layer_conv_1d(filters = 7L, kernel_size = 3L, activation = "relu",
                       input_shape = c(sl, feat)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                 backend = "cpu")
  m <- ggml_fit(m, x, d$y, epochs = 20L, batch_size = 8L, verbose = 0L)

  tl <- m$history$train_loss
  expect_lt(tl[length(tl)], tl[1])

  p <- ggml_predict(m, x, batch_size = 8L)
  expect_gt(mean(max.col(p) == max.col(d$y)), 0.9)
})
