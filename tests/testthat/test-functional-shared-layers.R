# Tests for ggml_dense() + ggml_apply() shared-layer workflow (C4 gap).
# These functional-API building blocks were exported without direct test
# coverage. Distinct from inline ggml_layer_dense(): a ggml_dense() layer
# object can be applied to multiple inputs sharing the same weights.

test_that("ggml_dense() builds a reusable ggml_layer object", {
  enc <- ggml_dense(8L, activation = "relu", name = "enc")
  expect_s3_class(enc, "ggml_layer")
  expect_equal(enc$node_type, "dense")
  expect_equal(enc$config$units, 8L)
  expect_equal(enc$name, "enc")
  expect_true(enc$trainable)
})

test_that("ggml_apply() requires a tensor node and a layer object", {
  enc <- ggml_dense(4L)
  expect_error(ggml_apply(42, enc), "ggml_tensor_node")

  x <- ggml_input(shape = 4L)
  expect_error(ggml_apply(x, list()), "ggml_layer")
})

test_that("ggml_apply() returns a tensor node carrying the layer sharing key", {
  x   <- ggml_input(shape = 4L)
  enc <- ggml_dense(3L, activation = "relu")
  out <- ggml_apply(x, enc)

  expect_s3_class(out, "ggml_tensor_node")
  expect_equal(out$layer_id, enc$layer_id)   # sharing key == layer identity
  expect_equal(out$node_type, "dense")
  expect_identical(out$parents[[1]], x)
})

test_that("a shared ggml_dense() layer applied to two inputs reuses one layer_id", {
  shared <- ggml_dense(5L, activation = "relu")
  x1 <- ggml_input(shape = 4L)
  x2 <- ggml_input(shape = 4L)

  o1 <- ggml_apply(x1, shared)
  o2 <- ggml_apply(x2, shared)

  # both applications must reference the SAME layer object (weight sharing)
  expect_equal(o1$layer_id, o2$layer_id)
  expect_equal(o1$layer_id, shared$layer_id)
  # but they are distinct graph nodes
  expect_false(identical(o1$id, o2$id))
})

test_that("shared single-input functional model with ggml_apply predicts", {
  set.seed(7)
  shared <- ggml_dense(2L, activation = "softmax")
  x   <- ggml_input(shape = 4L)
  out <- ggml_apply(x, shared)

  m <- ggml_model(inputs = x, outputs = out)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  n  <- 32L
  xa <- matrix(rnorm(4 * n), n, 4)
  p  <- predict(m, xa, batch_size = 32L)
  expect_true(is.matrix(p) || is.numeric(p))
})

test_that("ggml_conv_2d_layer() / ggml_conv_1d_layer() build shareable layer objects", {
  # These are the ggml_apply() factories for convolutions -- the counterpart of
  # ggml_dense(). They used to be named ggml_layer_conv_2d/1d, which collided
  # with the pipe-style layers in nn_layers.R and left them unreachable.
  c2 <- ggml_conv_2d_layer(filters = 4L, kernel_size = 3L, activation = "relu")
  expect_s3_class(c2, "ggml_layer")
  expect_equal(c2$node_type, "conv_2d")
  expect_equal(c2$config$filters, 4L)
  expect_equal(c2$config$kernel_size, c(3L, 3L))   # scalar is expanded

  c1 <- ggml_conv_1d_layer(filters = 2L, kernel_size = 5L)
  expect_s3_class(c1, "ggml_layer")
  expect_equal(c1$node_type, "conv_1d")
  expect_equal(c1$config$kernel_size, 5L)

  # The pipe-style layers keep their own names and stay usable.
  x   <- ggml_input(shape = c(8L, 8L, 1L))
  out <- ggml_layer_conv_2d(x, filters = 4L, kernel_size = 3L)
  expect_s3_class(out, "ggml_tensor_node")
  expect_equal(out$node_type, "conv_2d")
})

test_that("a shared conv layer applied to two inputs reuses one layer_id", {
  shared <- ggml_conv_2d_layer(filters = 4L, kernel_size = 3L)
  x1 <- ggml_input(shape = c(8L, 8L, 1L))
  x2 <- ggml_input(shape = c(8L, 8L, 1L))

  o1 <- ggml_apply(x1, shared)
  o2 <- ggml_apply(x2, shared)

  expect_equal(o1$layer_id, o2$layer_id)
  expect_equal(o1$layer_id, shared$layer_id)
  expect_false(identical(o1$id, o2$id))
  expect_equal(o1$node_type, "conv_2d")
})

test_that("multi-input shared-layer model builds and predicts", {
  set.seed(7)
  shared <- ggml_dense(2L, activation = "softmax")
  x1 <- ggml_input(shape = 4L)
  x2 <- ggml_input(shape = 4L)
  o1 <- ggml_apply(x1, shared)
  o2 <- ggml_apply(x2, shared)

  # Model construction with two inputs sharing one layer must succeed.
  m <- ggml_model(inputs = list(x1, x2), outputs = list(o1, o2))
  expect_s3_class(m, "ggml_functional_model")
  expect_equal(o1$layer_id, o2$layer_id)

  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  # The two outputs sit on INDEPENDENT branches -- output 1 is unreachable from
  # output 2 -- so predict() must expand every output into the compute graph.
  # Expanding only the last one leaves output 1 without a backend buffer.
  n  <- 32L
  xa <- matrix(rnorm(4 * n), n, 4)
  xb <- matrix(rnorm(4 * n), n, 4)
  p  <- predict(m, list(xa, xb), batch_size = 32L)

  expect_type(p, "list")
  expect_length(p, 2L)
  for (po in p) {
    expect_true(is.matrix(po))
    expect_equal(dim(po), c(n, 2L))
    expect_false(anyNA(po))
    expect_equal(rowSums(po), rep(1.0, n), tolerance = 1e-5)  # softmax rows
  }

  # Weight sharing is observable: feeding the SAME data to both inputs must
  # produce the same output from both branches.
  p_same <- predict(m, list(xa, xa), batch_size = 32L)
  expect_equal(p_same[[1L]], p_same[[2L]], tolerance = 1e-6)
})
