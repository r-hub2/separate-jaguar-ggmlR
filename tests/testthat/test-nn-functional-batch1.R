# Functional API: single-sample inference for conv-shaped inputs.
#
# The functional API builds its own flatten (nn_functional.R) and carried the
# same defect as the sequential one: the batch size was read as
# shape[ggml_n_dims(t)], and because ggml reports a trailing unit dimension as
# absent, a [W, H, C, 1] input looks 3-D and the batch size came off the channel
# axis. Predicting one sample at a time aborted.

fb1_cleanup <- function(model) {
  if (!is.null(model$compilation$buffer)) {
    ggml_backend_buffer_free(model$compilation$buffer)
  }
  if (!is.null(model$compilation$ctx_weights)) {
    ggml_free(model$compilation$ctx_weights)
  }
  if (!is.null(model$compilation$sched)) {
    ggml_backend_sched_free(model$compilation$sched)
  }
}

fb1_labels <- function(n) {
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0
  y
}

test_that("functional conv_2d model predicts a single sample", {
  set.seed(31)
  n <- 16L; h <- 6L; w <- 6L; ch <- 2L
  x <- array(rnorm(n * h * w * ch), dim = c(n, h, w, ch))
  y <- fb1_labels(n)

  inp <- ggml_input(shape = c(h, w, ch))
  out <- inp |>
    ggml_layer_conv_2d(filters = 4L, kernel_size = 3L, activation = "relu") |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  on.exit(fb1_cleanup(m))

  m <- ggml_fit(m, x, y, epochs = 3L, batch_size = 8L, verbose = 0L)

  p_all <- ggml_predict(m, x, batch_size = 8L)
  p_one <- ggml_predict(m, x[1, , , , drop = FALSE], batch_size = 1L)

  expect_equal(nrow(p_one), 1L)
  expect_true(all(is.finite(p_one)))
  expect_lt(max(abs(p_one - p_all[1, , drop = FALSE])), 1e-4)
})

test_that("functional flatten reports the batch size from the element count", {
  # A batch whose size equals neither the channel count nor any spatial extent,
  # so an off-by-one axis read cannot accidentally produce the right answer.
  set.seed(32)
  n <- 12L; h <- 4L; w <- 4L; ch <- 3L
  x <- array(rnorm(n * h * w * ch), dim = c(n, h, w, ch))
  y <- fb1_labels(n)

  inp <- ggml_input(shape = c(h, w, ch))
  out <- inp |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  on.exit(fb1_cleanup(m))

  m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 4L, verbose = 0L)

  for (bs in c(1L, 2L, 4L)) {
    p <- ggml_predict(m, x, batch_size = bs)
    expect_equal(nrow(p), n)
    expect_true(all(is.finite(p)))
  }
})
