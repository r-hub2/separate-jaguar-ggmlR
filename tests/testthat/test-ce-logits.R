# Cross-entropy training feeds logits to the loss node.
#
# ggml_cross_entropy_loss() applies log_softmax to its own input
# (ggml-cpu/ops.cpp, ggml_vec_log_soft_max_f32). A model ending in a softmax
# activation would therefore have softmax applied twice: the reported loss came
# out lower than the true cross-entropy and the gradient was damped. The graph
# built for cross-entropy training drops that final softmax; inference keeps it.

ce_of <- function(y, p) {
  eps <- 1e-7
  -mean(rowSums(y * log(pmax(pmin(p, 1 - eps), eps))))
}

test_that("predict() still returns probabilities after CE training", {
  set.seed(31)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L) |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  set.seed(5)
  n <- 64L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2); y[cbind(1:n, sample(1:2, n, TRUE))] <- 1

  m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 16L, verbose = 0)
  p <- ggml_predict(m, x, batch_size = 16L)

  # Stripping the softmax must not leak into inference.
  expect_equal(unname(rowSums(p)), rep(1, nrow(p)), tolerance = 1e-5)
  expect_true(all(p >= 0 & p <= 1))
})

test_that("reported CE loss matches cross-entropy of the model's outputs", {
  set.seed(31)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L) |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  set.seed(5)
  n_tr <- 64L; n_va <- 32L
  x_tr <- matrix(rnorm(4 * n_tr), n_tr, 4)
  y_tr <- matrix(0, n_tr, 2); y_tr[cbind(1:n_tr, sample(1:2, n_tr, TRUE))] <- 1
  x_va <- matrix(rnorm(4 * n_va, mean = 5), n_va, 4)
  y_va <- matrix(0, n_va, 2); y_va[, 1] <- 1

  m <- ggml_fit(m, x_tr, y_tr, epochs = 2L, batch_size = 16L,
                validation_data = list(x_va, y_va), verbose = 0)

  # Before the fix these differed by a data-dependent factor (~2x here).
  expect_equal(m$history$val_loss[2L], ce_of(y_va, ggml_predict(m, x_va, batch_size = 16L)),
               tolerance = 0.02)
})

test_that("ggml_evaluate() agrees with the reported CE loss", {
  set.seed(31)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L) |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  set.seed(12)
  n <- 64L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2); y[cbind(1:n, sample(1:2, n, TRUE))] <- 1

  m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 16L,
                validation_split = 0.25, verbose = 0, shuffle = FALSE)
  ev <- ggml_evaluate(m, x, y, batch_size = 16L)

  # Same quantity, same scale -- the two are now directly comparable.
  expect_equal(ev$loss, ce_of(y, ggml_predict(m, x, batch_size = 16L)),
               tolerance = 1e-4)
})

test_that("CE training actually learns a separable task", {
  # The damped gradient used to make this crawl; accuracy should now climb.
  set.seed(77)
  n <- 256L
  x <- matrix(rnorm(4 * n), n, 4)
  cls <- as.integer(x[, 1] > 0)
  y <- matrix(0, n, 2); y[cbind(1:n, cls + 1L)] <- 1

  set.seed(31)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  m <- ggml_fit(m, x, y, epochs = 20L, batch_size = 32L, verbose = 0)

  expect_lt(m$history$train_loss[20L], m$history$train_loss[1L])
  expect_gt(m$history$train_accuracy[20L], 0.7)
})

test_that("MSE training is unaffected by the logits change", {
  set.seed(31)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L) |>
    ggml_compile(optimizer = "adam", loss = "mse")

  set.seed(8)
  n <- 64L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2); y[cbind(1:n, sample(1:2, n, TRUE))] <- 1

  # MSE compares against the model's actual output, so the softmax must stay.
  m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 16L, verbose = 0)
  p <- ggml_predict(m, x, batch_size = 16L)
  expect_equal(unname(rowSums(p)), rep(1, nrow(p)), tolerance = 1e-5)
  expect_true(all(is.finite(m$history$train_loss)))
})

test_that("functional models get the same treatment", {
  set.seed(31)
  inp <- ggml_input(shape = 4L)
  out <- inp |> ggml_layer_dense(2L, activation = "softmax")
  m   <- ggml_model(inputs = inp, outputs = out) |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  set.seed(14)
  n <- 64L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2); y[cbind(1:n, sample(1:2, n, TRUE))] <- 1

  m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 32L, verbose = 0)
  p <- ggml_predict(m, x, batch_size = 32L)
  expect_equal(unname(rowSums(p)), rep(1, nrow(p)), tolerance = 1e-5)
  expect_equal(m$history$train_loss[2L], ce_of(y, p), tolerance = 0.15)
})
