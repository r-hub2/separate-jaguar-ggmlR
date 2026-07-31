# Tests for ag_layer_norm (autograd LayerNorm).
# Pure R math — no GPU required.

# Central-difference gradient of fn at p.
num_grad <- function(fn, p, h = 1e-6) {
  out <- p
  for (i in seq_along(p)) {
    pp <- p; pp[i] <- pp[i] + h
    pm <- p; pm[i] <- pm[i] - h
    out[i] <- (fn(pp) - fn(pm)) / (2 * h)
  }
  out
}

# ============================================================================
# Forward
# ============================================================================

test_that("ag_layer_norm: normalizes each column to mean 0 / var 1", {
  ln <- ag_layer_norm(4L, eps = 0)
  x  <- ag_tensor(matrix(c(1, 2, 3, 4,
                           10, 20, 30, 40), 4, 2))
  y  <- ln$forward(x)

  # default gamma=1, beta=0 -> output is the normalized input
  expect_equal(colMeans(y$data), c(0, 0), tolerance = 1e-10)
  expect_equal(apply(y$data, 2, function(c) mean(c^2)), c(1, 1),
               tolerance = 1e-10)
})

test_that("ag_layer_norm: normalizes over features, not over the batch", {
  # Columns are scaled copies of each other: LayerNorm maps both to the same
  # normalized vector, whereas BatchNorm-style row normalization would not.
  ln <- ag_layer_norm(3L, eps = 0)
  x  <- ag_tensor(matrix(c(1, 2, 3,
                           2, 4, 6), 3, 2))
  y  <- ln$forward(x)

  expect_equal(y$data[, 1], y$data[, 2], tolerance = 1e-10)
})

test_that("ag_layer_norm: applies gamma and beta", {
  ln <- ag_layer_norm(3L, eps = 0)
  ln$gamma$data <- matrix(c(2, 2, 2), 3, 1)
  ln$beta$data  <- matrix(c(5, 5, 5), 3, 1)

  x <- ag_tensor(matrix(c(1, 2, 3), 3, 1))
  y <- ln$forward(x)

  ln_plain <- ag_layer_norm(3L, eps = 0)
  base     <- ln_plain$forward(x)$data

  expect_equal(y$data, 2 * base + 5, tolerance = 1e-10)
})

test_that("ag_layer_norm: is independent of batch size", {
  ln <- ag_layer_norm(3L)
  x1 <- ag_tensor(matrix(c(1, 5, 9), 3, 1))
  x3 <- ag_tensor(matrix(c(1, 5, 9,  2, 2, 2,  0, 4, 8), 3, 3))

  y1 <- ln$forward(x1)
  y3 <- ln$forward(x3)

  # the first column is normalized identically regardless of its neighbours
  expect_equal(y1$data[, 1], y3$data[, 1], tolerance = 1e-12)
})

test_that("ag_layer_norm: constant input maps to beta (no NaN)", {
  ln <- ag_layer_norm(4L, eps = 1e-5)
  x  <- ag_tensor(matrix(7, 4, 2))
  y  <- ln$forward(x)

  expect_false(any(is.nan(y$data)))
  expect_equal(y$data, matrix(0, 4, 2), tolerance = 1e-10)
})

test_that("ag_layer_norm: elementwise_affine=FALSE has no parameters", {
  ln <- ag_layer_norm(3L, elementwise_affine = FALSE)
  expect_length(ln$parameters(), 0L)
  expect_null(ln$gamma)

  x <- ag_tensor(matrix(c(1, 2, 3), 3, 1))
  y <- ln$forward(x)
  expect_equal(as.numeric(colMeans(y$data)), 0, tolerance = 1e-10)
})

test_that("ag_layer_norm: rejects a feature-count mismatch", {
  ln <- ag_layer_norm(4L)
  x  <- ag_tensor(matrix(1:6, 3, 2))
  expect_error(ln$forward(x), "expected 4 features")
})

test_that("ag_layer_norm: validates normalized_shape", {
  expect_error(ag_layer_norm(0L), "positive")
  expect_error(ag_layer_norm(-3L), "positive")
})

test_that("ag_layer_norm: parameters() exposes gamma and beta", {
  ln <- ag_layer_norm(5L)
  p  <- ln$parameters()
  expect_named(p, c("gamma", "beta"))
  expect_equal(dim(p$gamma$data), c(5L, 1L))
  expect_true(p$gamma$requires_grad)
})

# ============================================================================
# Backward — checked against central differences
# ============================================================================

test_that("ag_layer_norm: gradient w.r.t. input matches finite differences", {
  set.seed(11)
  d <- 5L; n <- 3L
  x0    <- matrix(rnorm(d * n), d, n)
  gam0  <- matrix(rnorm(d), d, 1)
  bet0  <- matrix(rnorm(d), d, 1)
  gout  <- matrix(rnorm(d * n), d, n)

  run_loss <- function(x_val) {
    ln <- ag_layer_norm(d)
    ln$gamma$data <- gam0
    ln$beta$data  <- bet0
    sum(ln$forward(ag_tensor(x_val))$data * gout)
  }

  ln <- ag_layer_norm(d)
  ln$gamma$data <- gam0
  ln$beta$data  <- bet0
  x <- ag_param(x0)
  with_grad_tape({
    y    <- ln$forward(x)
    loss <- ag_sum(ag_mul(y, ag_tensor(gout)))
  })
  grads <- backward(loss)
  g_x   <- get0(as.character(x$id), envir = grads)

  expect_equal(g_x, num_grad(run_loss, x0), tolerance = 1e-6)
})

test_that("ag_layer_norm: gradients w.r.t. gamma and beta match finite differences", {
  set.seed(12)
  d <- 4L; n <- 3L
  x0   <- matrix(rnorm(d * n), d, n)
  gam0 <- matrix(rnorm(d), d, 1)
  bet0 <- matrix(rnorm(d), d, 1)
  gout <- matrix(rnorm(d * n), d, n)

  loss_gamma <- function(g_val) {
    ln <- ag_layer_norm(d)
    ln$gamma$data <- matrix(g_val, d, 1)
    ln$beta$data  <- bet0
    sum(ln$forward(ag_tensor(x0))$data * gout)
  }
  loss_beta <- function(b_val) {
    ln <- ag_layer_norm(d)
    ln$gamma$data <- gam0
    ln$beta$data  <- matrix(b_val, d, 1)
    sum(ln$forward(ag_tensor(x0))$data * gout)
  }

  ln <- ag_layer_norm(d)
  ln$gamma$data <- gam0
  ln$beta$data  <- bet0
  with_grad_tape({
    y    <- ln$forward(ag_tensor(x0))
    loss <- ag_sum(ag_mul(y, ag_tensor(gout)))
  })
  grads <- backward(loss)

  g_gam <- get0(as.character(ln$gamma$id), envir = grads)
  g_bet <- get0(as.character(ln$beta$id),  envir = grads)

  expect_equal(as.numeric(g_gam), as.numeric(num_grad(loss_gamma, gam0)),
               tolerance = 1e-6)
  expect_equal(as.numeric(g_bet), as.numeric(num_grad(loss_beta, bet0)),
               tolerance = 1e-6)
})

test_that("ag_layer_norm: input gradient is not the naive grad_out/sd", {
  # The exact LayerNorm gradient subtracts the mean terms; a constant grad_out
  # must therefore produce (almost) zero input gradient, whereas the naive
  # 'treat mu/sd as constants' shortcut would not.
  set.seed(13)
  d <- 6L
  x <- ag_param(matrix(rnorm(d), d, 1))
  ln <- ag_layer_norm(d)
  with_grad_tape({
    y    <- ln$forward(x)
    loss <- ag_sum(y)
  })
  grads <- backward(loss)
  g_x   <- get0(as.character(x$id), envir = grads)

  expect_true(max(abs(g_x)) < 1e-8)
})

test_that("ag_layer_norm: no gradient recorded for a non-param input", {
  ln <- ag_layer_norm(3L, elementwise_affine = FALSE)
  x  <- ag_tensor(matrix(c(1, 2, 3), 3, 1))
  with_grad_tape({
    y <- ln$forward(x)
  })
  expect_false(y$requires_grad)
})

test_that("ag_layer_norm: gradients flow through to an upstream layer", {
  set.seed(14)
  lin <- ag_linear(3L, 4L)
  ln  <- ag_layer_norm(4L)
  x   <- ag_tensor(matrix(rnorm(3 * 2), 3, 2))

  with_grad_tape({
    h    <- lin$forward(x)
    y    <- ln$forward(h)
    loss <- ag_mse_loss(y, matrix(0, 4, 2))
  })
  grads <- backward(loss)

  g_w <- get0(as.character(lin$W$id), envir = grads)
  expect_false(is.null(g_w))
  expect_true(any(abs(g_w) > 0))
})

# ============================================================================
# Training-mode semantics and integration
# ============================================================================

test_that("ag_layer_norm: eval mode gives the same result as training mode", {
  set.seed(15)
  ln <- ag_layer_norm(4L)
  x  <- ag_tensor(matrix(rnorm(8), 4, 2))

  ag_train(ln); y_train <- ln$forward(x)$data
  ag_eval(ln);  y_eval  <- ln$forward(x)$data

  expect_equal(y_train, y_eval, tolerance = 1e-12)
})

test_that("ag_layer_norm: works inside ag_sequential and trains", {
  set.seed(16)
  model <- ag_sequential(
    ag_linear(4L, 8L),
    ag_layer_norm(8L),
    ag_linear(8L, 2L)
  )
  x <- ag_tensor(matrix(rnorm(4 * 5), 4, 5))
  target <- matrix(rnorm(2 * 5), 2, 5)

  params <- model$parameters()
  expect_true(length(params) > 0L)

  opt <- optimizer_sgd(params, lr = 0.05)

  first <- NA_real_; last <- NA_real_
  for (i in 1:30) {
    opt$zero_grad()
    with_grad_tape({
      out  <- model$forward(x)
      loss <- ag_mse_loss(out, target)
    })
    if (i == 1L)  first <- as.numeric(loss$data)
    if (i == 30L) last  <- as.numeric(loss$data)
    grads <- backward(loss)
    opt$step(grads)
  }

  expect_true(is.finite(last))
  expect_lt(last, first)
})

test_that("ag_layer_norm: print method reports the configuration", {
  ln <- ag_layer_norm(16L)
  expect_output(print(ln), "ag_layer_norm\\(16\\)")
  expect_output(print(ln), "affine=TRUE")
})
