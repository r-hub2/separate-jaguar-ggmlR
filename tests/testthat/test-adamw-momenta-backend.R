# Regression test for AdamW momenta living on the wrong backend.
#
# GGML_OP_OPT_STEP_ADAMW writes to its src[2]/src[3] (the m and v momenta) even
# though they are passed as sources. ggml_backend_sched treats split inputs as
# read-only: it copies them *into* the executing backend before the split runs
# (ggml_backend_sched_compute_splits) and never copies them back. So when the
# momenta lived in ctx_static -- which is deliberately allocated on CPU, because
# it also holds loss/label tensors whose ops Vulkan does not support -- a Vulkan
# optimizer step updated a scratch copy that was discarded at the end of the
# step. m and v stayed at zero forever, AdamW degenerated into a near-sign step,
# and training on GPU converged visibly worse than on CPU (0.53 vs 0.72 accuracy
# over 10 epochs) while step 1 still matched bit-for-bit, since m = v = 0 there.
#
# The momenta now get their own context/buffer allocated on the backend that
# owns the params, so this compares CPU and Vulkan training directly: with the
# momenta frozen the two backends drift apart from step 2 onwards.
#
# SGD is the control -- it has no momenta, so it was never affected and must
# stay in agreement either way.

skip_no_gpu <- function() {
  skip_if(
    tryCatch({ ggml_vulkan_init(0L); FALSE }, error = function(e) TRUE),
    "No Vulkan GPU available"
  )
}

# Fixed data and starting weights: both backends must start from bit-identical
# state, so nothing here may depend on RNG draw order.
adamw_data <- function(ndata = 32L, ne_in = 4L) {
  X <- matrix(
    seq(-1, 1, length.out = ndata * ne_in) * c(1, -1),
    nrow = ndata, ncol = ne_in
  )
  list(X = X, y = X %*% c(1, 2, 3, 4)[seq_len(ne_in)], ndata = ndata, ne_in = ne_in)
}

# Train a small linear model and return the learned weights.
adamw_train <- function(use_gpu, optimizer, nepoch, d) {
  backend <- if (use_gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  cpu     <- ggml_backend_cpu_init()
  # The CPU backend is kept as a fallback so loss/label ops unsupported on
  # Vulkan still have somewhere to run -- this is the mixed-backend setup in
  # which the momenta placement actually matters.
  sched <- if (use_gpu) {
    ggml_backend_sched_new(list(backend, cpu), parallel = FALSE)
  } else {
    ggml_backend_sched_new(list(cpu), parallel = FALSE)
  }

  ctx <- ggml_init_auto(4 * 1024 * 1024, no_alloc = TRUE)

  batch <- 8L
  x_in <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d$ne_in, batch)
  W    <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d$ne_in, 1L)
  b    <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1L)
  ggml_set_param(x_in); ggml_set_param(W); ggml_set_param(b)

  out <- ggml_add(ctx, ggml_mul_mat(ctx, W, x_in), b)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(W, rep(0.05, d$ne_in))
  ggml_backend_tensor_set_data(b, 0)

  dataset <- ggml_opt_dataset_init(
    type_data = GGML_TYPE_F32, type_label = GGML_TYPE_F32,
    ne_datapoint = d$ne_in, ne_label = 1L,
    ndata = d$ndata, ndata_shard = 1L
  )
  ggml_backend_tensor_set_data(ggml_opt_dataset_data(dataset), as.numeric(t(d$X)))
  ggml_backend_tensor_set_data(ggml_opt_dataset_labels(dataset), as.numeric(d$y))

  on.exit({
    ggml_opt_dataset_free(dataset)
    ggml_backend_buffer_free(buf)
    ggml_free(ctx)
    ggml_backend_sched_free(sched)
    ggml_backend_free(cpu)
    if (use_gpu) ggml_backend_free(backend)
  })

  ggml_fit_opt(sched, ctx, x_in, out, dataset,
               loss_type = ggml_opt_loss_type_mse(),
               optimizer = optimizer,
               nepoch = nepoch, nbatch_logical = batch,
               val_split = 0, silent = TRUE)

  c(ggml_backend_tensor_get_data(W), ggml_backend_tensor_get_data(b))
}

test_that("AdamW momenta persist across steps on Vulkan", {
  skip_no_gpu()

  d <- adamw_data()
  w_cpu <- adamw_train(FALSE, ggml_opt_optimizer_type_adamw(), 10L, d)
  w_gpu <- adamw_train(TRUE,  ggml_opt_optimizer_type_adamw(), 10L, d)

  # Both backends run the same update on the same data from the same start, so
  # only floating-point ordering should separate them. With the momenta pinned
  # at zero the GPU takes a different trajectory entirely and this blows up.
  expect_equal(w_gpu, w_cpu, tolerance = 1e-4)
})

test_that("AdamW moves the weights further than a single step would", {
  skip_no_gpu()

  # Guards the failure mode directly rather than by CPU comparison: frozen
  # momenta make every step roughly +/- alpha, so the weights barely track the
  # data. Ten steps of working AdamW must beat one step by a clear margin.
  d  <- adamw_data()
  w1 <- adamw_train(TRUE, ggml_opt_optimizer_type_adamw(), 1L, d)
  w10 <- adamw_train(TRUE, ggml_opt_optimizer_type_adamw(), 10L, d)

  start <- c(rep(0.05, d$ne_in), 0)
  expect_gt(sum(abs(w10 - start)), sum(abs(w1 - start)))
})

test_that("SGD stays in agreement across backends (control)", {
  skip_no_gpu()

  # SGD has no momenta and so was never affected by the split-input copy; if
  # this ever fails the cause is elsewhere in the optimizer path.
  d <- adamw_data()
  w_cpu <- adamw_train(FALSE, ggml_opt_optimizer_type_sgd(), 10L, d)
  w_gpu <- adamw_train(TRUE,  ggml_opt_optimizer_type_sgd(), 10L, d)

  expect_equal(w_gpu, w_cpu, tolerance = 1e-4)
})
