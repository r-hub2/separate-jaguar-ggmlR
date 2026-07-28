# BatchNorm on conv-shaped inputs (4-D [W, H, C, N]).
#
# Batch normalization is defined per channel: the mean and variance are taken
# over the batch AND all spatial positions, so every channel of the output has
# zero mean and unit variance across the whole feature map.
#
# Conv-shaped inputs used to take an early return to ggml_rms_norm(), which
# divides by the RMS along ne[0] (the width of a single row of a single channel
# of a single sample) and never subtracts a mean -- neither per-channel nor
# batch-wise. These tests were written against that defect and now guard the
# real implementation in nn_bn_normalize_conv().
#
# The tests below pin the semantics numerically. The network is
#   conv_2d(1x1, identity kernel, zero bias) -> batch_norm -> flatten
# so the input to batch_norm is exactly the input image and the output of
# ggml_predict() is exactly the output of batch_norm, flattened. That makes the
# expected values computable in plain R with no reference to ggml internals.

# Reference batch normalization at inference time, following the definition:
# normalize each channel by statistics pooled over batch and space.
#
# x: R array [N, H, W, C]; returns an array of the same shape.
bnconv_reference <- function(x, eps = 1e-5) {
  out <- x
  for (c in seq_len(dim(x)[4])) {
    v <- x[, , , c]
    out[, , , c] <- (v - mean(v)) / sqrt(stats::var(as.vector(v)) *
                                         (length(v) - 1) / length(v) + eps)
  }
  out
}

# conv_2d kernel is [kw, kh, ic, oc]; for a 1x1 kernel that is the ic-by-oc
# matrix in column-major order, so the identity is diag(C).
bnconv_identity_kernel <- function(channels) {
  as.vector(diag(channels))
}

# `calibrate_on` supplies the running estimates that ggml_fit() would normally
# compute after training (see nn_bn_calibrate). ggml_predict() takes the
# inference path, which normalizes with those estimates; left at their (0, 1)
# defaults the layer would be an identity transform.
#
# They are passed through weights_data rather than by calling nn_bn_calibrate()
# directly: weight tensors only exist inside nn_build_graph(), which
# ggml_predict() re-runs on every call, so anything written to layer$weights
# outside a fit is discarded. weights_data is the supported way to seed them
# (nn_model.R restores running_mean/running_var from it). Computing the
# statistics here in plain R also keeps the expectations independent of the
# implementation being tested.
bnconv_running_stats <- function(x) {
  channels <- dim(x)[4]
  mu <- vapply(seq_len(channels), function(c) mean(x[, , , c]), numeric(1))
  va <- vapply(seq_len(channels),
               function(c) mean((x[, , , c] - mu[c])^2), numeric(1))
  list(mean = mu, var = va)
}

bnconv_build <- function(h, w, channels, eps = 1e-5, calibrate_on = NULL) {
  m <- ggml_model_sequential() |>
    ggml_layer_conv_2d(filters = channels, kernel_size = c(1L, 1L),
                       padding = "same", input_shape = c(h, w, channels)) |>
    ggml_layer_batch_norm(eps = eps) |>
    ggml_layer_flatten()

  # Pin the convolution to the identity so batch_norm sees the raw input.
  m$layers[[1]]$weights_data <- list(
    kernel = bnconv_identity_kernel(channels),
    bias   = rep(0, channels)
  )
  # gamma = 1, beta = 0: the scale-and-shift is the identity too.
  bn_weights <- list(
    gamma = rep(1, channels),
    beta  = rep(0, channels)
  )
  if (!is.null(calibrate_on)) {
    stats <- bnconv_running_stats(calibrate_on)
    bn_weights$running_mean <- stats$mean
    bn_weights$running_var  <- stats$var
  }
  m$layers[[2]]$weights_data <- bn_weights

  ggml_compile(m, optimizer = "adam", loss = "mse", backend = "cpu")
}

# ggml_predict() returns [N, W*H*C] flattened from ggml order [W, H, C, N].
# Undo that to get back an R array [N, H, W, C].
bnconv_unflatten <- function(p, h, w, channels) {
  n <- nrow(p)
  a <- array(as.vector(t(p)), dim = c(w, h, channels, n))  # ggml [W, H, C, N]
  aperm(a, c(4, 2, 1, 3))                                  # -> [N, H, W, C]
}

bnconv_data <- function(seed, n, h, w, channels) {
  set.seed(seed)
  # Deliberately different scale and offset per channel: batch normalization
  # must remove both, rms_norm removes neither.
  x <- array(0, dim = c(n, h, w, channels))
  for (c in seq_len(channels)) {
    x[, , , c] <- rnorm(n * h * w, mean = 4 * c, sd = 0.5 * c)
  }
  x
}

# ── The identity-conv harness itself has to be trustworthy ───

test_that("batchnorm conv harness: identity conv reproduces its input", {
  h <- 4L; w <- 3L; channels <- 2L; n <- 8L
  x <- bnconv_data(1L, n, h, w, channels)

  # Same model without the batch_norm layer: predict() must return the input.
  m <- ggml_model_sequential() |>
    ggml_layer_conv_2d(filters = channels, kernel_size = c(1L, 1L),
                       padding = "same", input_shape = c(h, w, channels)) |>
    ggml_layer_flatten()
  m$layers[[1]]$weights_data <- list(
    kernel = bnconv_identity_kernel(channels), bias = rep(0, channels)
  )
  m <- ggml_compile(m, optimizer = "adam", loss = "mse", backend = "cpu")

  got <- bnconv_unflatten(ggml_predict(m, x, batch_size = n), h, w, channels)
  expect_lt(max(abs(got - x)), 1e-4)
})

# ── The defect ───────────────────────────────────────────────

test_that("batchnorm conv: output matches the batch-norm definition", {
  h <- 4L; w <- 3L; channels <- 2L; n <- 8L
  x <- bnconv_data(2L, n, h, w, channels)

  m <- bnconv_build(h, w, channels, calibrate_on = x)
  got <- bnconv_unflatten(ggml_predict(m, x, batch_size = n), h, w, channels)

  expect_lt(max(abs(got - bnconv_reference(x))), 1e-3)
})

test_that("batchnorm conv: each channel is centred and scaled to unit variance", {
  h <- 4L; w <- 3L; channels <- 2L; n <- 8L
  x <- bnconv_data(3L, n, h, w, channels)

  m <- bnconv_build(h, w, channels, calibrate_on = x)
  got <- bnconv_unflatten(ggml_predict(m, x, batch_size = n), h, w, channels)

  # The defining property, stated without reference to any implementation: per
  # channel, pooled over batch and space, mean 0 and variance 1.
  for (c in seq_len(channels)) {
    v <- as.vector(got[, , , c])
    expect_lt(abs(mean(v)), 1e-3)
    expect_lt(abs(stats::var(v) - 1), 5e-2)
  }
})

test_that("batchnorm conv: a channel-wise offset is removed", {
  h <- 4L; w <- 3L; channels <- 2L; n <- 8L
  x <- bnconv_data(4L, n, h, w, channels)

  m <- bnconv_build(h, w, channels, calibrate_on = x)
  p_base <- ggml_predict(m, x, batch_size = n)

  # Adding a constant to a whole channel shifts its mean by that constant, so
  # batch normalization must produce exactly the same output as before.
  x_shift <- x
  x_shift[, , , 1] <- x_shift[, , , 1] + 10
  p_shift <- ggml_predict(
    bnconv_build(h, w, channels, calibrate_on = x_shift), x_shift, batch_size = n)

  expect_lt(max(abs(p_shift - p_base)), 1e-3)
})
