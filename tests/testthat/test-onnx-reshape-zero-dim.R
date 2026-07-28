# ONNX Reshape: shape[d] == 0 means "keep the input's ONNX dimension d".
#
# Resolving that needs the input's ONNX rank, and ggml_n_dims() cannot supply it
# -- ggml treats a trailing unit dimension as absent, so an input declared
# [2, 3, 1] reports 2 dims, not 3. The ONNX-dim -> ggml-ne mapping is
# ggml_d = ndims - 1 - d, so an ndims that is one too small shifts every zero
# entry onto the wrong axis. onnx_ops_tensor.c now takes the rank from the tmap
# (the stored ONNX ndims) and falls back to ggml_n_dims() only when unknown.

roz_run <- function(path, inputs) {
  m <- onnx_load(path, device = "cpu")
  onnx_run(m, inputs)[[1]]
}

# Reshape(X, shape) with the shape given as an initializer.
roz_model <- function(in_dims, out_dims, shape_values) {
  inp  <- .onnx_value_info("X", 1L, in_dims)
  outp <- .onnx_value_info("Y", 1L, out_dims)

  shape_raw <- unlist(lapply(shape_values, .int64_bytes))
  shape_t   <- .onnx_tensor("shape", length(shape_values), 7L, shape_raw)
  shape_vi  <- .onnx_value_info("shape", 7L, length(shape_values))

  node  <- .onnx_node("Reshape", c("X", "shape"), "Y")
  graph <- .onnx_graph("reshape_zero", list(node),
                       list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

test_that("onnx Reshape: zero dim is kept when the input has a trailing 1", {
  # Input [2, 3, 1]; shape [0, -1] keeps ONNX dim 0 (== 2) and infers the rest.
  # ggml_n_dims() would report 2 for this tensor, mapping ONNX dim 0 onto the
  # wrong ne[] entry and yielding the wrong kept extent.
  path <- roz_model(c(2L, 3L, 1L), c(2L, 3L), c(0L, -1L))

  x <- c(1, 2, 3, 4, 5, 6)
  r <- as.numeric(roz_run(path, list(X = x)))

  expect_equal(length(r), 6L)
  expect_equal(r, x, tolerance = 1e-5)
})

test_that("onnx Reshape: zero dim with no trailing 1 still works", {
  # Regression guard for the ordinary case, where ggml_n_dims() and the ONNX
  # rank agree.
  path <- roz_model(c(2L, 3L), c(2L, 3L), c(0L, -1L))

  x <- c(1, 2, 3, 4, 5, 6)
  r <- as.numeric(roz_run(path, list(X = x)))

  expect_equal(length(r), 6L)
  expect_equal(r, x, tolerance = 1e-5)
})

test_that("onnx Reshape: two zero dims resolve against the ONNX rank", {
  # [2, 3, 1] -> shape [0, 0, -1]: both kept extents come from the input's own
  # ONNX dims, which is exactly what the trailing unit dimension hides.
  path <- roz_model(c(2L, 3L, 1L), c(2L, 3L, 1L), c(0L, 0L, -1L))

  x <- c(1, 2, 3, 4, 5, 6)
  r <- as.numeric(roz_run(path, list(X = x)))

  expect_equal(length(r), 6L)
  expect_equal(r, x, tolerance = 1e-5)
})
