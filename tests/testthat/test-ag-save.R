# Tests for ag_save_model() / ag_load_model() — autograd module state dict.

build_mlp <- function() {
  ag_sequential(
    ag_linear(4L, 8L, activation = "relu"),
    ag_linear(8L, 3L, activation = NULL)
  )
}

build_bn <- function() {
  ag_sequential(
    ag_linear(4L, 8L, activation = "relu"),
    ag_batch_norm(8L),
    ag_linear(8L, 3L, activation = NULL)
  )
}

train_a_bit <- function(model, n_iter = 8L) {
  opt <- optimizer_adam(model$parameters(), lr = 0.05)
  x <- ag_tensor(matrix(stats::runif(4 * 20), 4, 20))
  y <- matrix(0, 3, 20)
  y[cbind(sample.int(3L, 20L, replace = TRUE), seq_len(20L))] <- 1
  ag_train(model)
  for (i in seq_len(n_iter)) {
    with_grad_tape({
      logits <- model$forward(x)
      loss   <- ag_softmax_cross_entropy_loss(logits, y)
    })
    g <- backward(loss)
    opt$step(g)
    opt$zero_grad()
  }
  ag_eval(model)
  model
}

test_that("ag_save_model / ag_load_model roundtrip reproduces forward output", {
  set.seed(1)
  m <- train_a_bit(build_mlp())

  xt <- ag_tensor(matrix(stats::runif(4 * 5), 4, 5))
  out_before <- ggmlR:::.ag_data(m$forward(xt))

  f <- tempfile(fileext = ".rds")
  ag_save_model(m, f, model_fn = build_mlp)
  m2 <- ag_load_model(f)

  out_after <- ggmlR:::.ag_data(m2$forward(xt))
  expect_equal(out_after, out_before, tolerance = 1e-6)
  expect_s3_class(m2, "ag_sequential")
})

test_that("batch-norm running buffers survive the roundtrip", {
  set.seed(2)
  m <- train_a_bit(build_bn())

  rm_before <- m$layers[[2]]$running_mean
  rv_before <- m$layers[[2]]$running_var
  # ensure training actually moved them off the init values
  expect_false(isTRUE(all.equal(rm_before, matrix(0.0, 8L, 1L))))

  f <- tempfile(fileext = ".rds")
  ag_save_model(m, f, model_fn = build_bn)
  m2 <- ag_load_model(f)

  expect_equal(m2$layers[[2]]$running_mean, rm_before)
  expect_equal(m2$layers[[2]]$running_var,  rv_before)
})

test_that("batch-norm buffers survive inside a nested ag_sequential", {
  # Buffer keys must follow the same nested "layer{i}_" scheme that
  # ag_sequential$parameters() uses, otherwise a batch_norm below the top
  # level is silently reset to its init values on load.
  build_nested <- function() {
    ag_sequential(
      ag_linear(4L, 6L),
      ag_sequential(ag_batch_norm(6L), ag_linear(6L, 2L))
    )
  }

  m  <- build_nested()
  bn <- m$layers[[2]]$layers[[1]]
  bn$running_mean <- matrix(seq_len(6) * 1.0, 6L, 1L)
  bn$running_var  <- matrix(seq_len(6) * 3.0, 6L, 1L)

  f <- tempfile(fileext = ".rds")
  ag_save_model(m, f, model_fn = build_nested)

  expect_named(readRDS(f)$buffers,
               c("layer2_layer1_running_mean", "layer2_layer1_running_var"))

  m2  <- ag_load_model(f)
  bn2 <- m2$layers[[2]]$layers[[1]]
  expect_equal(bn2$running_mean, bn$running_mean)
  expect_equal(bn2$running_var,  bn$running_var)
})

test_that("parameter dtype is restored from the file by default", {
  build <- function() ag_sequential(ag_linear(4L, 3L), ag_linear(3L, 2L))
  dtypes_of <- function(m) {
    vapply(m$parameters(),
           function(p) if (is.null(p$dtype)) NA_character_ else p$dtype,
           character(1L))
  }

  old <- ag_dtype("f16")
  on.exit(ag_dtype(old), add = TRUE)
  m <- build()
  f <- tempfile(fileext = ".rds")
  ag_save_model(m, f, model_fn = build)

  # Load in a session whose default differs from the file's.
  ag_dtype("f32")

  expect_true(all(dtypes_of(ag_load_model(f)) == "f16"))
  expect_true(all(dtypes_of(ag_load_model(f, dtype = "f32")) == "f32"))
  expect_true(all(dtypes_of(ag_load_model(f, dtype = "bf16")) == "bf16"))
  expect_error(ag_load_model(f, dtype = "int8"))

  # Weights themselves are unaffected: dtype is an upload precision, and $data
  # is always kept at full R numeric precision.
  expect_equal(
    ggmlR:::.ag_data(ag_load_model(f)$parameters()[[1L]]),
    ggmlR:::.ag_data(m$parameters()[[1L]])
  )
})

test_that("containers without recorded dtypes still load", {
  build <- function() ag_sequential(ag_linear(4L, 3L), ag_linear(3L, 2L))
  f <- tempfile(fileext = ".rds")
  ag_save_model(build(), f, model_fn = build)

  # Files written before dtypes were recorded, and files whose dtype vector
  # covers only some parameters, must not break the load.
  cont <- readRDS(f)
  cont$param_dtypes <- NULL
  f_none <- tempfile(fileext = ".rds")
  saveRDS(cont, f_none)
  expect_s3_class(ag_load_model(f_none), "ag_sequential")

  cont$param_dtypes <- c(layer1_W = "f16")
  f_part <- tempfile(fileext = ".rds")
  saveRDS(cont, f_part)
  m <- ag_load_model(f_part)
  expect_equal(m$parameters()[["layer1_W"]]$dtype, "f16")
})

test_that("buffer collection tolerates layer shapes without buffers", {
  # ag_sequential() with a single list-based layer stores that layer's own
  # fields (W/b/forward/params) directly in $layers, so the recursion must not
  # mistake them for sublayers and descend into the closures.
  expect_length(ggmlR:::.ag_collect_buffers(ag_sequential(ag_linear(4L, 2L))), 0L)
  expect_length(ggmlR:::.ag_collect_buffers(ag_linear(4L, 2L)), 0L)
  expect_length(
    ggmlR:::.ag_collect_buffers(ag_sequential(ag_linear(4L, 2L), ag_dropout(0.1))),
    0L
  )
})

test_that("model_fn can be supplied at load time instead of save time", {
  set.seed(3)
  m <- train_a_bit(build_mlp())
  xt <- ag_tensor(matrix(stats::runif(4 * 5), 4, 5))
  out_before <- ggmlR:::.ag_data(m$forward(xt))

  f <- tempfile(fileext = ".rds")
  ag_save_model(m, f)                       # no model_fn stored
  m2 <- ag_load_model(f, model_fn = build_mlp)

  expect_equal(ggmlR:::.ag_data(m2$forward(xt)), out_before, tolerance = 1e-6)
})

test_that("loading without any model_fn errors clearly", {
  set.seed(4)
  m <- build_mlp()
  f <- tempfile(fileext = ".rds")
  ag_save_model(m, f)                        # no model_fn
  expect_error(ag_load_model(f), "model_fn")
})

test_that("architecture mismatch is rejected", {
  set.seed(5)
  m <- build_mlp()
  f <- tempfile(fileext = ".rds")
  ag_save_model(m, f, model_fn = build_mlp)

  wrong_arch <- function() {
    ag_sequential(ag_linear(4L, 16L), ag_linear(16L, 3L))
  }
  expect_error(ag_load_model(f, model_fn = wrong_arch),
               "mismatch")
})

test_that("ag_save_model rejects non-ag objects", {
  expect_error(ag_save_model(list(1, 2), tempfile()), "ag_sequential")
})

test_that("ag_load_model rejects a non-container file", {
  f <- tempfile(fileext = ".rds")
  saveRDS(list(a = 1), f)
  expect_error(ag_load_model(f, model_fn = build_mlp), "container")
})

test_that("saved container prints a summary", {
  m <- build_mlp()
  f <- tempfile(fileext = ".rds")
  ag_save_model(m, f, model_fn = build_mlp)
  cont <- readRDS(f)
  expect_s3_class(cont, "ggmlR_ag_state")
  expect_output(print(cont), "ag_state")
})
