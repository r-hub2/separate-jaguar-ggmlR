# BatchNorm: train/inference split and running statistics
#
# Demonstrates that ggml_layer_batch_norm() normalizes over the BATCH axis per
# feature, and that inference uses the running estimates accumulated during
# training rather than the statistics of whichever batch a sample lands in.
#
#   - Training uses the current batch's mean/variance
#   - Inference uses running_mean / running_var (EMA, momentum 0.1)
#   - Predictions are therefore independent of batch composition
#   - Running estimates survive ggml_save_model() / ggml_load_model()
#   - Built-in checks at the end of the file
#
# Why this matters: with batch statistics at inference time, a prediction
# depends on which other samples share its batch. On class-sorted data every
# batch then holds a single class, normalization removes the very shift that
# separates the classes, and a fully trained model scores *below* an untrained
# one. The checks below pin that behaviour down.
#
# Usage:
#   Rscript inst/examples/batchnorm_running_stats.R
#
# Tensor layout: [features, batch] — the batch axis is ne[1].

library(ggmlR)

cat("ggmlR version:", ggml_version(), "\n")

# Keep the CPU graph executor modest; this example is about correctness.
ggml_set_n_threads(2L)

BACKEND <- if (isTRUE(tryCatch(ggml_vulkan_available() &&
                               ggml_vulkan_device_count() > 0,
                               error = function(e) FALSE))) "vulkan" else "cpu"
cat("Backend:", BACKEND, "\n\n")

# =============================================================================
# 0.  Synthetic data — two well-separated classes, sorted by class
# =============================================================================
#
# Rows 1..96 are class 1 (centred at -3), rows 97..192 are class 2 (at +3).
# The ordering is deliberate: with batch_size = 32 every batch contains exactly
# one class, which is the worst case for batch-statistics-at-inference.

N <- 192L

make_data <- function(seed) {
  set.seed(seed)
  list(
    x = rbind(matrix(rnorm(N, -3, 0.5), N / 2, 2),
              matrix(rnorm(N,  3, 0.5), N / 2, 2)),
    y = rbind(matrix(c(1, 0), N / 2, 2, byrow = TRUE),
              matrix(c(0, 1), N / 2, 2, byrow = TRUE))
  )
}

accuracy <- function(p, y) mean(max.col(p) == max.col(y))

# =============================================================================
# 1.  Model
# =============================================================================

build_and_train <- function(seed, epochs = 120L) {
  d <- make_data(seed)
  set.seed(seed)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(32L, activation = "relu", input_shape = 2L) |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam",
                 loss      = "categorical_crossentropy",
                 backend   = BACKEND)
  list(model = ggml_fit(m, d$x, d$y, epochs = epochs,
                        batch_size = 32L, verbose = 0L),
       data  = d)
}

cat("Training (seed 42, 120 epochs)...\n")
fit42 <- build_and_train(42L)
model <- fit42$model
dat   <- fit42$data

# =============================================================================
# 2.  The same samples, four different batchings
# =============================================================================
#
# All four must agree: a prediction may not depend on batch composition.

p_sorted <- ggml_predict(model, dat$x, batch_size = 32L)   # 6 single-class batches
p_full   <- ggml_predict(model, dat$x, batch_size = 192L)  # one batch, both classes

set.seed(99L)
perm     <- sample(nrow(dat$x))
p_shuf   <- ggml_predict(model, dat$x[perm, , drop = FALSE], batch_size = 32L)

# One sample at a time — batch statistics would be undefined here (a single
# sample has zero variance), running estimates handle it naturally.
p_single <- do.call(rbind, lapply(seq_len(24L), function(i)
  ggml_predict(model, dat$x[i, , drop = FALSE], batch_size = 1L)))

acc_sorted <- accuracy(p_sorted, dat$y)
acc_full   <- accuracy(p_full,   dat$y)
acc_shuf   <- accuracy(p_shuf,   dat$y[perm, , drop = FALSE])
acc_single <- accuracy(p_single, dat$y[seq_len(24L), , drop = FALSE])

cat("\nAccuracy by batching strategy\n")
cat(sprintf("  class-sorted batches (32) : %.4f\n", acc_sorted))
cat(sprintf("  shuffled batches     (32) : %.4f\n", acc_shuf))
cat(sprintf("  whole set as one batch    : %.4f\n", acc_full))
cat(sprintf("  one sample per call  (1)  : %.4f\n", acc_single))
cat(sprintf("  max |sorted - full| prob  : %.3e\n", max(abs(p_sorted - p_full))))

# =============================================================================
# 3.  A trained model must beat an untrained one
# =============================================================================

set.seed(42L)
untrained <- ggml_model_sequential() |>
  ggml_layer_dense(32L, activation = "relu", input_shape = 2L) |>
  ggml_layer_batch_norm() |>
  ggml_layer_dense(2L, activation = "softmax") |>
  ggml_compile(optimizer = "adam",
               loss      = "categorical_crossentropy",
               backend   = BACKEND)
acc_untrained <- accuracy(ggml_predict(untrained, dat$x, batch_size = 32L), dat$y)

cat(sprintf("\nUntrained accuracy: %.4f  ->  trained: %.4f\n",
            acc_untrained, acc_sorted))

# =============================================================================
# 4.  Loss must not diverge
# =============================================================================
#
# Normalizing by batch statistics at inference time also shows up during
# training: the loss bottoms out early and then climbs back up.

tl <- model$history$train_loss
cat(sprintf("Loss: first=%.4f min=%.4f last=%.4f\n",
            tl[1], min(tl), tl[length(tl)]))

# =============================================================================
# 5.  Running estimates survive a save/load round-trip
# =============================================================================

tmp <- tempfile(fileext = ".rds")
ggml_save_model(model, tmp)
reloaded <- ggml_load_model(tmp)
acc_reloaded <- accuracy(ggml_predict(reloaded, dat$x, batch_size = 32L), dat$y)
unlink(tmp)

cat(sprintf("Reloaded model accuracy: %.4f\n", acc_reloaded))

# =============================================================================
# 6.  Checks
# =============================================================================

cat("\n--- checks ---\n")
ok <- TRUE
check <- function(label, passed) {
  cat(sprintf("  [%s] %s\n", if (passed) "OK" else "FAIL", label))
  if (!passed) ok <<- FALSE
  invisible(passed)
}

check("predictions independent of batch composition",
      max(abs(p_sorted - p_full)) < 1e-4)
check("class-sorted batching matches shuffled batching",
      abs(acc_sorted - acc_shuf) < 1e-6)
check("single-sample inference agrees with batched",
      abs(acc_single - accuracy(p_sorted[seq_len(24L), , drop = FALSE],
                                dat$y[seq_len(24L), , drop = FALSE])) < 1e-6)
check("training beats no training", acc_sorted > acc_untrained)
check("separable data is learned (accuracy > 0.95)", acc_sorted > 0.95)
check("loss did not diverge", tl[length(tl)] <= min(tl) * 1.15)
check("save/load preserves running statistics",
      abs(acc_reloaded - acc_sorted) < 1e-6)

cat(if (ok) "\nAll checks passed.\n" else "\nSome checks FAILED.\n")
