# Multi-batch predict must agree with single-batch predict on every backend.
#
# ggml_predict() splits the input into batches and runs the same graph once per
# batch. It used to call ggml_backend_sched_reset() + alloc_graph() inside that
# loop, which lets the scheduler re-lay the intermediate buffers on every pass.
# On Vulkan the second and later passes then read stale data: the first batch was
# exact while every batch after it was wrong by ~0.4 on softmax outputs. The
# graph is now allocated once, before the loop.
#
# What made this hard to see is that it is not tied to a layer or an input shape
# -- it is the Nth pass that breaks, whichever batch that happens to be. Running
# the batches in the opposite order moved the error to the other batch. So these
# tests compare per batch and, where it is cheap, across execution orders.
#
# They run on CPU too: the CPU backend never showed the fault, but the assertions
# are backend-agnostic and cost little, so a future scheduler change that breaks
# CPU the same way gets caught as well.
#
# Tolerance: 5e-3. Repeating a prediction at the same batch size is bit-exact on
# both backends, but changing the batch size lets Vulkan pick different shader
# paths for the new geometry, which measures up to ~2e-4 here. The defect these
# tests guard against was ~0.4, more than two orders of magnitude above that, so
# the threshold separates the two without turning legitimate GPU arithmetic into
# a failure.

pmb_backends <- function() {
  bes <- "cpu"
  has_vk <- tryCatch(
    ggml_vulkan_available() && ggml_vulkan_device_count() > 0,
    error = function(e) FALSE
  )
  if (has_vk) bes <- c(bes, "vulkan")
  bes
}

pmb_labels <- function(n) {
  y <- matrix(0, n, 2L)
  lab <- rep(c(1L, 2L), length.out = n)
  for (i in seq_len(n)) y[i, lab[i]] <- 1
  list(y = y, lab = lab)
}

# Train on CPU once and reuse the weights on every backend, so a comparison
# reflects execution and not two independent initializations.
pmb_shared_model <- function(build, x, y, epochs = 3L) {
  m <- build("cpu")
  m <- ggml_fit(m, x, y, epochs = epochs, batch_size = 8L, verbose = 0L)
  f <- tempfile(fileext = ".rds")
  ggml_save_model(m, f)
  f
}

# ── Batch splitting must not change the answer ───────────────

test_that("predict is identical whether run in one batch or several", {
  set.seed(41)
  n <- 32L; sl <- 10L; feat <- 4L
  x <- array(rnorm(n * sl * feat), dim = c(n, sl, feat))
  d <- pmb_labels(n)

  build <- function(be) {
    set.seed(103L)
    ggml_model_sequential() |>
      ggml_layer_conv_1d(filters = 7L, kernel_size = 3L, activation = "relu",
                         input_shape = c(sl, feat)) |>
      ggml_layer_batch_norm() |>
      ggml_layer_flatten() |>
      ggml_layer_dense(2L, activation = "softmax") |>
      ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                   backend = be)
  }
  f <- pmb_shared_model(build, x, d$y)
  on.exit(unlink(f), add = TRUE)

  for (be in pmb_backends()) {
    m <- ggml_load_model(f, backend = be)
    p_one <- ggml_predict(m, x, batch_size = n)    # a single pass
    p_many <- ggml_predict(m, x, batch_size = 8L)  # four passes

    expect_equal(dim(p_many), dim(p_one))
    # Per batch, so a report names which pass went wrong rather than just "some
    # rows differ".
    for (ib in seq_len(4L)) {
      rows <- ((ib - 1L) * 8L + 1L):(ib * 8L)
      expect_lt(max(abs(p_many[rows, , drop = FALSE] - p_one[rows, , drop = FALSE])),
                5e-3)
    }
  }
})

test_that("every batch after the first is computed, not carried over", {
  # The failure mode left batch 1 exact and corrupted the rest, so an assertion
  # that only looked at the whole matrix could be satisfied by luck. Compare the
  # Vulkan result against CPU batch by batch.
  skip_if_not("vulkan" %in% pmb_backends(), "No Vulkan GPU available")

  set.seed(42)
  n <- 32L; sl <- 10L; feat <- 4L
  x <- array(rnorm(n * sl * feat), dim = c(n, sl, feat))
  d <- pmb_labels(n)

  build <- function(be) {
    set.seed(104L)
    ggml_model_sequential() |>
      ggml_layer_conv_1d(filters = 7L, kernel_size = 3L, activation = "relu",
                         input_shape = c(sl, feat)) |>
      ggml_layer_batch_norm() |>
      ggml_layer_flatten() |>
      ggml_layer_dense(2L, activation = "softmax") |>
      ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                   backend = be)
  }
  f <- pmb_shared_model(build, x, d$y)
  on.exit(unlink(f), add = TRUE)

  p_cpu <- ggml_predict(ggml_load_model(f, backend = "cpu"), x, batch_size = 8L)
  p_gpu <- ggml_predict(ggml_load_model(f, backend = "vulkan"), x, batch_size = 8L)

  for (ib in seq_len(4L)) {
    rows <- ((ib - 1L) * 8L + 1L):(ib * 8L)
    expect_lt(max(abs(p_gpu[rows, , drop = FALSE] - p_cpu[rows, , drop = FALSE])),
              5e-3)
  }
})

# ── Not specific to one layer type ───────────────────────────

test_that("multi-batch predict is stable for a plain dense model", {
  # No convolution and no batch_norm: the defect was in the predict loop, so the
  # simplest possible model has to be covered too.
  set.seed(43)
  n <- 32L
  x <- matrix(rnorm(n * 6L), n, 6L)
  d <- pmb_labels(n)

  build <- function(be) {
    set.seed(105L)
    ggml_model_sequential() |>
      ggml_layer_dense(12L, activation = "relu", input_shape = 6L) |>
      ggml_layer_dense(2L, activation = "softmax") |>
      ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                   backend = be)
  }
  f <- pmb_shared_model(build, x, d$y)
  on.exit(unlink(f), add = TRUE)

  for (be in pmb_backends()) {
    m <- ggml_load_model(f, backend = be)
    p_one <- ggml_predict(m, x, batch_size = n)
    p_many <- ggml_predict(m, x, batch_size = 8L)
    expect_lt(max(abs(p_many - p_one)), 5e-3)
  }
})

test_that("multi-batch predict is stable for a conv_2d model", {
  set.seed(44)
  n <- 32L; h <- 6L; w <- 6L; ch <- 1L
  x <- array(rnorm(n * h * w * ch), dim = c(n, h, w, ch))
  d <- pmb_labels(n)

  build <- function(be) {
    set.seed(106L)
    ggml_model_sequential() |>
      ggml_layer_conv_2d(filters = 4L, kernel_size = c(3L, 3L),
                         activation = "relu", input_shape = c(h, w, ch)) |>
      ggml_layer_batch_norm() |>
      ggml_layer_flatten() |>
      ggml_layer_dense(2L, activation = "softmax") |>
      ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                   backend = be)
  }
  f <- pmb_shared_model(build, x, d$y)
  on.exit(unlink(f), add = TRUE)

  for (be in pmb_backends()) {
    m <- ggml_load_model(f, backend = be)
    p_one <- ggml_predict(m, x, batch_size = n)
    p_many <- ggml_predict(m, x, batch_size = 8L)
    for (ib in seq_len(4L)) {
      rows <- ((ib - 1L) * 8L + 1L):(ib * 8L)
      expect_lt(max(abs(p_many[rows, , drop = FALSE] - p_one[rows, , drop = FALSE])),
                5e-3)
    }
  }
})

# ── The functional API runs its own predict loop ─────────────

test_that("functional API multi-batch predict is stable", {
  set.seed(45)
  n <- 32L; h <- 6L; w <- 6L; ch <- 2L
  x <- array(rnorm(n * h * w * ch), dim = c(n, h, w, ch))
  d <- pmb_labels(n)

  inp <- ggml_input(shape = c(h, w, ch))
  out <- inp |>
    ggml_layer_conv_2d(filters = 4L, kernel_size = 3L, activation = "relu") |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  on.exit({
    if (!is.null(m$compilation$buffer)) ggml_backend_buffer_free(m$compilation$buffer)
    if (!is.null(m$compilation$ctx_weights)) ggml_free(m$compilation$ctx_weights)
    if (!is.null(m$compilation$sched)) ggml_backend_sched_free(m$compilation$sched)
  }, add = TRUE)

  m <- ggml_fit(m, x, d$y, epochs = 3L, batch_size = 8L, verbose = 0L)

  p_one <- ggml_predict(m, x, batch_size = n)
  p_many <- ggml_predict(m, x, batch_size = 8L)

  for (ib in seq_len(4L)) {
    rows <- ((ib - 1L) * 8L + 1L):(ib * 8L)
    expect_lt(max(abs(p_many[rows, , drop = FALSE] - p_one[rows, , drop = FALSE])),
              5e-3)
  }
})
