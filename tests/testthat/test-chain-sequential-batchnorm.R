# Chain test: Sequential API with BatchNorm training
# Pattern from titanic_classification.R variant 3:
#   dense→BatchNorm→dense→BatchNorm→dense(softmax) + adam + fit→predict
#
# Uses synthetic linearly separable data (no external files).
#
# BatchNorm regression coverage: the layer must normalize over the BATCH axis
# per feature and use the accumulated running estimates at inference time. When
# inference normalizes by the statistics of the current batch instead, a
# prediction depends on which other samples share its batch; on class-sorted
# data every batch holds a single class, normalization removes the shift that
# separates the classes, and a fully trained model scores below an untrained
# one. The batching-invariance tests below are what catch that.

# Two well-separated classes, deliberately sorted by class: with batch_size = 32
# every batch then contains exactly one class -- the worst case for
# batch-statistics-at-inference.
bn_make_data <- function(seed, n = 192L) {
  set.seed(seed)
  list(
    x = rbind(matrix(rnorm(n, -3, 0.5), n / 2, 2),
              matrix(rnorm(n,  3, 0.5), n / 2, 2)),
    y = rbind(matrix(c(1, 0), n / 2, 2, byrow = TRUE),
              matrix(c(0, 1), n / 2, 2, byrow = TRUE))
  )
}

bn_accuracy <- function(p, y) mean(max.col(p) == max.col(y))

bn_build <- function(seed, backend = "cpu") {
  set.seed(seed)
  ggml_model_sequential() |>
    ggml_layer_dense(32L, activation = "relu", input_shape = 2L) |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
                 backend = backend)
}

# ── Sequential + BatchNorm: compile→fit→predict ─────────────

test_that("chain sequential-batchnorm: fit reduces loss", {
  set.seed(123)
  n <- 100L
  # 2-class linearly separable: class 0 centered at (-1,-1), class 1 at (1,1)
  x <- rbind(matrix(rnorm(n, -1, 0.5), n/2, 2),
             matrix(rnorm(n,  1, 0.5), n/2, 2))
  y <- rbind(matrix(c(1,0), n/2, 2, byrow = TRUE),
             matrix(c(0,1), n/2, 2, byrow = TRUE))

  m <- ggml_model_sequential() |>
    ggml_layer_dense(16L, activation = "relu", input_shape = 2L) |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(8L, activation = "relu") |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  h <- ggml_fit(m, x, y, epochs = 100L, batch_size = 10L, verbose = 0L)

  # Loss should decrease
  expect_true(h$history$train_loss[length(h$history$train_loss)] < h$history$train_loss[1])
})

test_that("chain sequential-batchnorm: predict gives valid probabilities", {
  set.seed(123)
  n <- 96L
  x <- rbind(matrix(rnorm(n, -1, 0.5), n/2, 2),
             matrix(rnorm(n,  1, 0.5), n/2, 2))
  y <- rbind(matrix(c(1,0), n/2, 2, byrow = TRUE),
             matrix(c(0,1), n/2, 2, byrow = TRUE))

  m <- ggml_model_sequential() |>
    ggml_layer_dense(16L, activation = "relu", input_shape = 2L) |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  m <- ggml_fit(m, x, y, epochs = 30L, batch_size = 16L, verbose = 0L)
  p <- ggml_predict(m, x, batch_size = 32L)

  expect_equal(nrow(p), nrow(x))
  expect_equal(ncol(p), 2L)
  # Probabilities sum to ~1
  row_sums <- rowSums(p)
  expect_true(all(abs(row_sums - 1.0) < 0.01))
  # All probabilities in [0,1]
  expect_true(all(p >= 0 & p <= 1))
})

test_that("chain sequential-batchnorm: accuracy on separable data", {
  d <- bn_make_data(42L)
  m <- ggml_fit(bn_build(42L), d$x, d$y, epochs = 120L,
                batch_size = 32L, verbose = 0L)
  p <- ggml_predict(m, d$x, batch_size = 32L)
  # Centres at -3/+3 with sd 0.5 are separable with a wide margin; anything
  # short of near-perfect means the layer is destroying the signal rather than
  # the model failing to converge.
  expect_gt(bn_accuracy(p, d$y), 0.95)
})

# ── BatchNorm inference must not depend on batch composition ───

test_that("batchnorm: predictions are invariant to batch size", {
  d <- bn_make_data(42L)
  m <- ggml_fit(bn_build(42L), d$x, d$y, epochs = 120L,
                batch_size = 32L, verbose = 0L)

  # 6 single-class batches vs. one batch holding both classes vs. 3 batches
  p32  <- ggml_predict(m, d$x, batch_size = 32L)
  p192 <- ggml_predict(m, d$x, batch_size = 192L)
  p64  <- ggml_predict(m, d$x, batch_size = 64L)

  expect_lt(max(abs(p32 - p192)), 1e-4)
  expect_lt(max(abs(p32 - p64)),  1e-4)
})

test_that("batchnorm: shuffled batches match class-sorted batches", {
  d <- bn_make_data(42L)
  m <- ggml_fit(bn_build(42L), d$x, d$y, epochs = 120L,
                batch_size = 32L, verbose = 0L)

  p_sorted <- ggml_predict(m, d$x, batch_size = 32L)

  set.seed(99L)
  perm <- sample(nrow(d$x))
  p_shuf <- ggml_predict(m, d$x[perm, , drop = FALSE], batch_size = 32L)

  # Same samples, different batch neighbours -> same probabilities.
  expect_lt(max(abs(p_shuf - p_sorted[perm, , drop = FALSE])), 1e-4)
  expect_equal(bn_accuracy(p_shuf, d$y[perm, , drop = FALSE]),
               bn_accuracy(p_sorted, d$y), tolerance = 1e-6)
})

test_that("batchnorm: single-sample inference works", {
  d <- bn_make_data(42L)
  m <- ggml_fit(bn_build(42L), d$x, d$y, epochs = 120L,
                batch_size = 32L, verbose = 0L)

  p_batched <- ggml_predict(m, d$x, batch_size = 32L)
  # A single sample has no variance of its own; this only works when inference
  # uses the running estimates.
  p_single <- do.call(rbind, lapply(seq_len(16L), function(i)
    ggml_predict(m, d$x[i, , drop = FALSE], batch_size = 1L)))

  expect_equal(nrow(p_single), 16L)
  expect_true(all(is.finite(p_single)))
  expect_lt(max(abs(p_single - p_batched[seq_len(16L), , drop = FALSE])), 1e-4)
})

test_that("batchnorm: training improves on an untrained model", {
  d <- bn_make_data(42L)
  acc_untrained <- bn_accuracy(
    ggml_predict(bn_build(42L), d$x, batch_size = 32L), d$y)

  m <- ggml_fit(bn_build(42L), d$x, d$y, epochs = 120L,
                batch_size = 32L, verbose = 0L)
  acc_trained <- bn_accuracy(ggml_predict(m, d$x, batch_size = 32L), d$y)

  # Guards the specific failure mode where training tunes the net to batch
  # statistics that inference does not have, leaving it worse than random.
  expect_gt(acc_trained, acc_untrained)
})

test_that("batchnorm: training loss does not diverge", {
  d <- bn_make_data(42L)
  m <- ggml_fit(bn_build(42L), d$x, d$y, epochs = 120L,
                batch_size = 32L, verbose = 0L)
  tl <- m$history$train_loss

  expect_lt(tl[length(tl)], tl[1])
  # The loss used to bottom out early and then climb back up.
  expect_lt(tl[length(tl)], min(tl) * 1.15)
})

test_that("batchnorm: running statistics survive save/load", {
  d <- bn_make_data(42L)
  m <- ggml_fit(bn_build(42L), d$x, d$y, epochs = 120L,
                batch_size = 32L, verbose = 0L)
  p_before <- ggml_predict(m, d$x, batch_size = 32L)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  ggml_save_model(m, tmp)
  m2 <- ggml_load_model(tmp)
  p_after <- ggml_predict(m2, d$x, batch_size = 32L)

  # Without serialised running_mean/running_var the reloaded model would
  # normalize with the (0, 1) identity transform and drift.
  expect_lt(max(abs(p_after - p_before)), 1e-4)
})

test_that("batchnorm: Vulkan and CPU agree", {
  skip_if_not(
    tryCatch(ggml_vulkan_available() && ggml_vulkan_device_count() > 0,
             error = function(e) FALSE),
    "No Vulkan GPU available"
  )

  d <- bn_make_data(42L)
  m_cpu <- ggml_fit(bn_build(42L, "cpu"), d$x, d$y, epochs = 120L,
                    batch_size = 32L, verbose = 0L)
  m_gpu <- ggml_fit(bn_build(42L, "vulkan"), d$x, d$y, epochs = 120L,
                    batch_size = 32L, verbose = 0L)

  # The layer used to converge on Vulkan but diverge on CPU.
  expect_gt(bn_accuracy(ggml_predict(m_cpu, d$x, batch_size = 32L), d$y), 0.95)
  expect_gt(bn_accuracy(ggml_predict(m_gpu, d$x, batch_size = 32L), d$y), 0.95)
})
