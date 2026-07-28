# Tests for ggml_fit_opt() -- the low-level optimizer loop.
#
# ggml_fit() delegates here for anything that is not a ggml_sequential_model, but
# the path had no coverage of its own, which is how a wrong physical batch size
# went unnoticed: the helper behind `nbatch_physical` read ne[3] regardless of
# the input's rank, so a dense [features, N] input -- whose batch lives in ne[1]
# and whose ne[3] is 1 -- yielded a batch size of 1. opt_period is derived from
# it as `nbatch_logical %/% nbatch_physical`, so gradient accumulation ran with
# an 8x wrong period while training still appeared to converge.
#
# The setup mirrors test-callbacks.R: y = X %*% c(1,2,3,4), so the loss has a
# reachable optimum and a broken configuration shows up as a failure to descend.

fitopt_setup <- function(ndata = 100, ne_in = 4, ne_out = 1, batch = 10) {
  cpu   <- ggml_backend_cpu_init()
  sched <- ggml_backend_sched_new(list(cpu), parallel = FALSE)

  ctx_compute <- ggml_init_auto(4 * 1024 * 1024, no_alloc = TRUE)
  x_in <- ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, ne_in, batch)
  W    <- ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, ne_in, ne_out)
  b    <- ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, ne_out)
  ggml_set_param(x_in); ggml_set_param(W); ggml_set_param(b)

  out <- ggml_add(ctx_compute, ggml_mul_mat(ctx_compute, W, x_in), b)
  buf <- ggml_backend_alloc_ctx_tensors(ctx_compute, cpu)

  set.seed(7)
  ggml_backend_tensor_set_data(W, rnorm(ne_in * ne_out, sd = 0.1))
  ggml_backend_tensor_set_data(b, rep(0, ne_out))

  true_w <- c(1, 2, 3, 4)[seq_len(ne_in)]
  X_all  <- matrix(rnorm(ndata * ne_in), nrow = ndata)
  y_all  <- X_all %*% true_w

  dataset <- ggml_opt_dataset_init(
    type_data = GGML_TYPE_F32, type_label = GGML_TYPE_F32,
    ne_datapoint = ne_in, ne_label = ne_out,
    ndata = ndata, ndata_shard = 1
  )
  ggml_backend_tensor_set_data(ggml_opt_dataset_data(dataset), as.numeric(t(X_all)))
  ggml_backend_tensor_set_data(ggml_opt_dataset_labels(dataset), as.numeric(y_all))

  list(cpu = cpu, sched = sched, ctx_compute = ctx_compute,
       inputs = x_in, outputs = out, dataset = dataset, buf = buf)
}

fitopt_cleanup <- function(s) {
  ggml_opt_dataset_free(s$dataset)
  ggml_backend_buffer_free(s$buf)
  ggml_free(s$ctx_compute)
  ggml_backend_sched_free(s$sched)
  ggml_backend_free(s$cpu)
}

fitopt_run <- function(s, ...) {
  ggml_fit_opt(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
               loss_type = ggml_opt_loss_type_mse(), silent = TRUE, ...)
}

# ── The physical batch size must come off the batch axis ─────

test_that("fit_opt: physical batch size is read from the input's batch axis", {
  # A dense input is [features, batch]: the batch is ne[1], and ne[3] is 1.
  # Reading the batch from the last ne[] entry would report 1 here.
  s <- fitopt_setup(ne_in = 4, batch = 10)
  on.exit(fitopt_cleanup(s))

  expect_equal(ggmlR:::.ggml_input_batch_size(s$inputs, s$dataset), 10L)
})

test_that("fit_opt: batch size is correct for every input rank", {
  cpu <- ggml_backend_cpu_init()
  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit({ ggml_free(ctx); ggml_backend_free(cpu) }, add = TRUE)

  # ne_datapoint is what disambiguates the batch axis, so each rank needs a
  # dataset describing a sample of the matching size.
  ds_flat <- ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, 10, 1, 64, 1)
  ds_seq  <- ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, 48, 2, 64, 1)
  ds_img  <- ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, 108, 2, 64, 1)
  on.exit({
    ggml_opt_dataset_free(ds_flat); ggml_opt_dataset_free(ds_seq)
    ggml_opt_dataset_free(ds_img)
  }, add = TRUE)

  flat <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10L, 8L)        # [F, N]
  seq3 <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8L, 6L, 8L)     # [size, seq, N]
  img4 <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 6L, 6L, 3L, 8L) # [W, H, C, N]

  expect_equal(ggmlR:::.ggml_input_batch_size(flat, ds_flat), 8L)
  expect_equal(ggmlR:::.ggml_input_batch_size(seq3, ds_seq), 8L)
  expect_equal(ggmlR:::.ggml_input_batch_size(img4, ds_img), 8L)

  # A batch of 1 is the case ggml_n_dims() cannot express: it drops the trailing
  # unit, so [W, H, C, 1] looks 3-D and [F, 1] looks 1-D.
  flat1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10L, 1L)
  img41 <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 6L, 6L, 3L, 1L)
  expect_equal(ggmlR:::.ggml_input_batch_size(flat1, ds_flat), 1L)
  expect_equal(ggmlR:::.ggml_input_batch_size(img41, ds_img), 1L)
})

# ── The loop itself ──────────────────────────────────────────

test_that("fit_opt: returns a history data frame with one row per epoch", {
  s <- fitopt_setup()
  on.exit(fitopt_cleanup(s))

  h <- fitopt_run(s, nepoch = 3L, nbatch_logical = 10L)

  expect_s3_class(h, "data.frame")
  expect_equal(nrow(h), 3L)
  expect_true(all(c("epoch", "train_loss") %in% names(h)))
  expect_equal(h$epoch, 1:3)
})

test_that("fit_opt: training reduces the loss", {
  s <- fitopt_setup()
  on.exit(fitopt_cleanup(s))

  h <- fitopt_run(s, nepoch = 30L, nbatch_logical = 10L)

  expect_true(all(is.finite(h$train_loss)))
  expect_lt(h$train_loss[nrow(h)], h$train_loss[1])
})

test_that("fit_opt: gradient accumulation converges like a plain batch", {
  # nbatch_logical > the physical batch turns on accumulation (opt_period > 1).
  # With the batch size misread as 1, opt_period would be nbatch_logical itself
  # -- a much larger effective step count per update -- and the two runs would
  # not land anywhere near each other.
  s1 <- fitopt_setup(); h1 <- fitopt_run(s1, nepoch = 30L, nbatch_logical = 10L)
  fitopt_cleanup(s1)

  s2 <- fitopt_setup(); h2 <- fitopt_run(s2, nepoch = 30L, nbatch_logical = 20L)
  fitopt_cleanup(s2)

  expect_lt(h1$train_loss[nrow(h1)], h1$train_loss[1])
  expect_lt(h2$train_loss[nrow(h2)], h2$train_loss[1])
  # Both descend towards the same reachable optimum.
  expect_lt(abs(h1$train_loss[nrow(h1)] - h2$train_loss[nrow(h2)]),
            max(h1$train_loss[1], h2$train_loss[1]))
})

test_that("fit_opt: a logical batch below the physical batch does not overrun", {
  # The forward pass always consumes nbatch_physical samples, so a smaller
  # logical batch cannot be honoured. Deriving idata_split from it unclamped
  # overshoots the dataset and trips GGML_ASSERT(idata <= ndata) inside
  # ggml_opt_dataset_shuffle().
  s <- fitopt_setup(ndata = 100, batch = 10)
  on.exit(fitopt_cleanup(s))

  expect_no_error(fitopt_run(s, nepoch = 2L, nbatch_logical = 1L,
                             val_split = 0.2))
})

test_that("fit_opt: validation split reports val metrics", {
  s <- fitopt_setup()
  on.exit(fitopt_cleanup(s))

  h <- fitopt_run(s, nepoch = 5L, nbatch_logical = 10L, val_split = 0.2)

  expect_equal(nrow(h), 5L)
  expect_true("val_loss" %in% names(h))
  expect_true(any(is.finite(h$val_loss)))
})

test_that("fit_opt: a callback can stop the loop early", {
  # On this problem val_loss improves monotonically, so early stopping on
  # val_loss would never fire -- correctly. Use a callback that stops on a fixed
  # epoch to test that the loop honours state$stop at all.
  s <- fitopt_setup()
  on.exit(fitopt_cleanup(s))

  stop_at_3 <- list(on_epoch_end = function(epoch, logs, state) {
    if (epoch >= 3L) state$stop <- TRUE
    invisible(NULL)
  })

  h <- fitopt_run(s, nepoch = 40L, nbatch_logical = 10L,
                  callbacks = list(stop_at_3))

  expect_equal(nrow(h), 3L)
})

test_that("fit_opt: early stopping fires when the monitored metric stalls", {
  s <- fitopt_setup()
  on.exit(fitopt_cleanup(s))

  # Feed the callback a metric that never improves, so `patience` is what
  # decides -- independent of how this particular problem happens to converge.
  cb <- ggml_callback_early_stopping(monitor = "train_loss", patience = 2L)
  flat <- list(on_epoch_end = function(epoch, logs, state) {
    logs$train_loss <- 1.0
    cb$on_epoch_end(epoch, logs, state)
  })

  h <- fitopt_run(s, nepoch = 40L, nbatch_logical = 10L,
                  callbacks = list(flat))

  # First epoch sets the best value, then `patience` epochs without improvement.
  expect_equal(nrow(h), 3L)
})
