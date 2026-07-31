# Tests for Keras-compatible API: compile(), fit(), evaluate(), predict()

test_that("compile() dispatch works for sequential model", {
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 3L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  expect_true(m$compiled)
})

test_that("compile() dispatch works for functional model", {
  x   <- ggml_input(shape = 4L)
  out <- x |> ggml_layer_dense(3L, activation = "softmax")
  m   <- ggml_model(inputs = x, outputs = out)
  m   <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  expect_true(m$compiled)
})

test_that("fit() and evaluate() dispatch for sequential model", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(0, 10, 2)
  y[cbind(1:10, sample(1:2, 10, replace = TRUE))] <- 1

  m <- fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = FALSE)
  result <- evaluate(m, x, y, verbose = FALSE)
  expect_true(is.list(result) || is.numeric(result))
})

test_that("predict() dispatch for sequential model", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(0, 10, 2)
  y[cbind(1:10, sample(1:2, 10, replace = TRUE))] <- 1
  m <- fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = FALSE)

  p <- predict(m, x)
  expect_true(is.matrix(p) || is.numeric(p))
  expect_equal(nrow(p), 10)
})

test_that("fit() and predict() dispatch for functional model", {
  set.seed(42)
  x_in <- ggml_input(shape = 4L)
  out  <- x_in |> ggml_layer_dense(2L, activation = "softmax")
  m    <- ggml_model(inputs = x_in, outputs = out)
  m    <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  # Use batch_size <= n_samples
  n <- 32L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2)
  y[cbind(1:n, sample(1:2, n, replace = TRUE))] <- 1

  m <- fit(m, x, y, epochs = 1L, batch_size = 32L, verbose = FALSE)
  p <- predict(m, x)
  expect_true(is.matrix(p) || is.numeric(p))
})

# ---------------------------------------------------------------------------
# callbacks: both fit paths run the R-side epoch loop (ggml_fit_opt /
# ggml_opt_epoch), so on_epoch_* hooks fire and state$stop truncates training.
# ---------------------------------------------------------------------------

# Counts epochs and stops after `stop_after` of them.
cb_counting <- function(stop_after = NULL) {
  env <- new.env(parent = emptyenv())
  env$begin <- 0L
  env$end   <- 0L
  env$cb <- list(
    on_epoch_begin = function(epoch, logs, state) {
      env$begin <- env$begin + 1L
      invisible(NULL)
    },
    on_epoch_end = function(epoch, logs, state) {
      env$end <- env$end + 1L
      if (!is.null(stop_after) && epoch >= stop_after) state$stop <- TRUE
      invisible(NULL)
    }
  )
  env
}

test_that("callbacks fire per epoch in sequential fit()", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  n <- 32L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2)
  y[cbind(1:n, sample(1:2, n, replace = TRUE))] <- 1

  counter <- cb_counting()
  m <- fit(m, x, y, epochs = 3L, batch_size = 16L, verbose = FALSE,
           callbacks = list(counter$cb))

  expect_equal(counter$begin, 3L)
  expect_equal(counter$end, 3L)
  expect_equal(length(m$history$train_loss), 3L)
})

test_that("callback state$stop truncates sequential training", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  n <- 32L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2)
  y[cbind(1:n, sample(1:2, n, replace = TRUE))] <- 1

  counter <- cb_counting(stop_after = 2L)
  m <- fit(m, x, y, epochs = 10L, batch_size = 16L, verbose = FALSE,
           callbacks = list(counter$cb))

  expect_equal(counter$end, 2L)
  expect_equal(length(m$history$train_loss), 2L)
  expect_equal(m$history$epochs, 1:2)
})

test_that("callbacks fire and stop in functional fit()", {
  set.seed(42)
  x_in <- ggml_input(shape = 4L)
  out  <- x_in |> ggml_layer_dense(2L, activation = "softmax")
  m    <- ggml_model(inputs = x_in, outputs = out)
  m    <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  n <- 32L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2)
  y[cbind(1:n, sample(1:2, n, replace = TRUE))] <- 1

  counter <- cb_counting(stop_after = 2L)
  m <- fit(m, x, y, epochs = 10L, batch_size = 32L, verbose = FALSE,
           callbacks = list(counter$cb))

  expect_equal(counter$end, 2L)
  expect_equal(length(m$history$train_loss), 2L)
})

test_that("early stopping callback is honoured by sequential fit()", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  n <- 32L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2)
  y[cbind(1:n, sample(1:2, n, replace = TRUE))] <- 1

  # On random data train_loss creeps down monotonically (~4e-4 per epoch), so
  # every epoch would count as an improvement and nothing would ever stop.
  # min_delta puts the bar above that drift: the first epoch is recorded as the
  # best, the second fails to beat it by min_delta, and patience = 1 stops.
  es <- ggml_callback_early_stopping(monitor = "train_loss", patience = 1L,
                                     min_delta = 0.1)
  m <- suppressMessages(
    fit(m, x, y, epochs = 20L, batch_size = 16L, verbose = FALSE,
        callbacks = list(es))
  )

  expect_lt(length(m$history$train_loss), 20L)
})

test_that("evaluate() uses keras3 argument order (object, x, y)", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  n <- 32L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2)
  y[cbind(1:n, sample(1:2, n, replace = TRUE))] <- 1
  m <- fit(m, x, y, epochs = 1L, batch_size = 16L, verbose = FALSE)

  # Positional (object, x, y), as in keras3.
  res <- evaluate(m, x, y)
  expect_true(is.list(res) || is.numeric(res))
  expect_named(formals(getS3method("evaluate", "ggml_sequential_model"))[1:3],
               c("object", "x", "y"))

  # Named x= must work too.
  #
  # This previously FAILED with "no applicable method", and an earlier version
  # of this test asserted that failure as expected behaviour. It was a defect,
  # not a constraint: the package re-exported generics::evaluate(x, ...), whose
  # first formal is `x`, so naming the data x= bound the *data* to the dispatch
  # argument and dispatched on a matrix instead of the model. ggmlR now declares
  # its own evaluate(object, ...) generic, which fixes this. Do not restore the
  # old expect_error() form.
  res_named <- evaluate(m, x = x, y = y)
  expect_true(is.list(res_named) || is.numeric(res_named))
})

test_that("predict()/evaluate() accept batch_size = NULL (keras3 default)", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  n <- 32L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2)
  y[cbind(1:n, sample(1:2, n, replace = TRUE))] <- 1
  m <- fit(m, x, y, epochs = 1L, batch_size = 16L, verbose = FALSE)

  p <- predict(m, x, batch_size = NULL)
  expect_equal(nrow(p), n)
  expect_null(formals(getS3method("predict", "ggml_sequential_model"))$batch_size)
})
