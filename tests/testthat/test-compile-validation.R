# ggml_compile() validates optimizer/loss/metrics.
#
# The training code maps these names with switch() statements. Those used to
# fall back to cross-entropy/adamw for anything unrecognised, so a typo or an
# unimplemented loss was silently substituted -- and ggml_evaluate(), which has
# no such fallback, then reported loss = NA for the very same model.

mk_seq <- function() {
  set.seed(1)
  ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L)
}

mk_fun <- function() {
  set.seed(1)
  inp <- ggml_input(shape = 4L)
  ggml_model(inputs = inp, outputs = ggml_layer_dense(inp, 2L, activation = "softmax"))
}

test_that("unsupported loss is rejected, not silently substituted", {
  # binary_crossentropy is not implemented; it used to train as categorical CE.
  expect_error(ggml_compile(mk_seq(), loss = "binary_crossentropy"),
               "Unsupported loss")
  expect_error(ggml_compile(mk_seq(), loss = "binary_crossentropy"),
               "binary_crossentropy is not implemented")
  expect_error(ggml_compile(mk_seq(), loss = "typo_here"), "Unsupported loss")
  expect_error(ggml_compile(mk_fun(), loss = "binary_crossentropy"),
               "Unsupported loss")
})

test_that("unsupported optimizer is rejected", {
  expect_error(ggml_compile(mk_seq(), optimizer = "adamm"), "Unsupported optimizer")
  expect_error(ggml_compile(mk_fun(), optimizer = "rmsprop"), "Unsupported optimizer")
})

test_that("unsupported metrics warn rather than being silently ignored", {
  expect_warning(ggml_compile(mk_seq(), metrics = c("accuracy", "f1")),
                 "Ignoring unsupported metric")
  expect_warning(ggml_compile(mk_fun(), metrics = "auc"),
                 "Ignoring unsupported metric")
  # The supported metric, and no metrics at all, must stay quiet.
  expect_silent(ggml_compile(mk_seq(), metrics = "accuracy"))
  expect_silent(ggml_compile(mk_seq(), metrics = NULL))
})

test_that("every supported optimizer/loss combination still compiles", {
  for (l in c("categorical_crossentropy", "crossentropy", "cross_entropy",
              "mse", "mean_squared_error")) {
    for (o in c("adam", "adamw", "sgd")) {
      m <- ggml_compile(mk_seq(), optimizer = o, loss = l)
      expect_true(m$compiled, info = paste(l, o))
      expect_identical(m$compilation$loss, l)
      expect_identical(m$compilation$optimizer, o)
    }
  }
})

test_that("fit() and evaluate() agree on which losses exist", {
  # Every loss compile() accepts must produce a non-NA loss from evaluate();
  # this is the fit/evaluate divergence that the fallback used to hide.
  set.seed(4)
  n <- 32L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2); y[cbind(1:n, sample(1:2, n, TRUE))] <- 1

  for (l in c("categorical_crossentropy", "crossentropy", "cross_entropy",
              "mse", "mean_squared_error")) {
    m <- ggml_compile(mk_seq(), loss = l)
    m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 16L, verbose = 0)
    ev <- ggml_evaluate(m, x, y, batch_size = 16L)
    expect_false(is.na(ev$loss), info = l)
  }
})

test_that("a hand-modified compilation is caught at fit() time", {
  # Second line of defence: a model whose compilation was not produced by
  # ggml_compile() (loaded, or edited) must fail loudly rather than train as
  # something the user did not ask for.
  set.seed(4)
  n <- 32L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2); y[cbind(1:n, sample(1:2, n, TRUE))] <- 1

  m <- ggml_compile(mk_seq(), loss = "categorical_crossentropy")
  m$compilation$loss <- "binary_crossentropy"
  expect_error(ggml_fit(m, x, y, epochs = 1L, batch_size = 16L, verbose = 0),
               "Unsupported loss")

  m2 <- ggml_compile(mk_seq(), optimizer = "adam")
  m2$compilation$optimizer <- "nope"
  expect_error(ggml_fit(m2, x, y, epochs = 1L, batch_size = 16L, verbose = 0),
               "Unsupported optimizer")
})
