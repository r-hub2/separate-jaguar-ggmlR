# Tests for clip_grad_value, check_grad_anomaly and the additional LR
# schedulers (cyclic, onecycle, warmup+cosine, SGDR T_mult).
# Pure R math — no GPU required.

# Build a gradient environment for a single param w.
make_grads <- function(w, x_val = c(1, 1)) {
  x <- ag_tensor(matrix(x_val, 2, 1))
  with_grad_tape({
    out  <- ag_matmul(w, x)
    loss <- ag_mse_loss(out, matrix(0, 2, 1))
  })
  backward(loss)
}

# ============================================================================
# clip_grad_value
# ============================================================================

test_that("clip_grad_value: clamps every element into [-v, v]", {
  w     <- ag_param(matrix(c(1, 2, 3, 4), 2, 2))
  grads <- make_grads(w, c(10, 10))
  clip_grad_value(list(w = w), grads, clip_value = 0.5)

  g <- get0(as.character(w$id), envir = grads)
  expect_true(all(g <= 0.5 + 1e-12))
  expect_true(all(g >= -0.5 - 1e-12))
  # gradients here are large, so clipping must actually bind
  expect_true(any(abs(g) > 0.5 - 1e-12))
})

test_that("clip_grad_value: leaves small gradients untouched", {
  w      <- ag_param(matrix(c(0.01, 0.01, 0.01, 0.01), 2, 2))
  grads  <- make_grads(w)
  before <- get0(as.character(w$id), envir = grads)
  clip_grad_value(list(w = w), grads, clip_value = 1000)
  after  <- get0(as.character(w$id), envir = grads)

  expect_equal(before, after, tolerance = 1e-12)
})

test_that("clip_grad_value: preserves gradient shape", {
  w     <- ag_param(matrix(c(1, 2, 3, 4), 2, 2))
  grads <- make_grads(w, c(10, 10))
  before_dim <- dim(get0(as.character(w$id), envir = grads))
  clip_grad_value(list(w = w), grads, clip_value = 0.1)
  after_dim  <- dim(get0(as.character(w$id), envir = grads))

  expect_equal(after_dim, before_dim)
})

test_that("clip_grad_value: returns pre-clip max abs value", {
  w     <- ag_param(matrix(c(1, 2, 3, 4), 2, 2))
  grads <- make_grads(w, c(10, 10))
  g_before <- get0(as.character(w$id), envir = grads)
  ret <- clip_grad_value(list(w = w), grads, clip_value = 0.5)

  expect_equal(ret, max(abs(g_before)), tolerance = 1e-10)
})

test_that("clip_grad_value: rejects an invalid clip_value", {
  w     <- ag_param(matrix(1, 2, 2))
  grads <- make_grads(w)
  expect_error(clip_grad_value(list(w = w), grads, clip_value = 0), "positive")
  expect_error(clip_grad_value(list(w = w), grads, clip_value = -1), "positive")
})

# ============================================================================
# check_grad_anomaly
# ============================================================================

test_that("check_grad_anomaly: reports ok for healthy gradients", {
  w      <- ag_param(matrix(c(0.1, 0.1, 0.1, 0.1), 2, 2))
  grads  <- make_grads(w)
  report <- check_grad_anomaly(list(w = w), grads)

  expect_s3_class(report, "data.frame")
  expect_equal(nrow(report), 1L)
  expect_equal(report$status, "ok")
  expect_equal(report$n_nan, 0L)
  expect_equal(report$n_inf, 0L)
})

test_that("check_grad_anomaly: detects NaN and warns", {
  w     <- ag_param(matrix(1, 2, 2))
  grads <- make_grads(w)
  key   <- as.character(w$id)
  g     <- get0(key, envir = grads)
  g[1]  <- NaN
  assign(key, g, envir = grads)

  expect_warning(report <- check_grad_anomaly(list(w = w), grads), "anomaly")
  expect_equal(report$status, "nan")
  expect_equal(report$n_nan, 1L)
})

test_that("check_grad_anomaly: detects Inf", {
  w     <- ag_param(matrix(1, 2, 2))
  grads <- make_grads(w)
  key   <- as.character(w$id)
  g     <- get0(key, envir = grads)
  g[2]  <- Inf
  assign(key, g, envir = grads)

  report <- suppressWarnings(check_grad_anomaly(list(w = w), grads))
  expect_equal(report$status, "inf")
  expect_equal(report$n_inf, 1L)
})

test_that("check_grad_anomaly: action='stop' raises, 'silent' stays quiet", {
  w     <- ag_param(matrix(1, 2, 2))
  grads <- make_grads(w)
  key   <- as.character(w$id)
  g     <- get0(key, envir = grads)
  g[1]  <- NaN
  assign(key, g, envir = grads)

  expect_error(check_grad_anomaly(list(w = w), grads, action = "stop"), "anomaly")
  expect_silent(check_grad_anomaly(list(w = w), grads, action = "silent"))
})

test_that("check_grad_anomaly: max_abs threshold flags large gradients", {
  w     <- ag_param(matrix(c(1, 2, 3, 4), 2, 2))
  grads <- make_grads(w, c(10, 10))

  quiet  <- check_grad_anomaly(list(w = w), grads, action = "silent")
  expect_equal(quiet$status, "ok")

  flagged <- check_grad_anomaly(list(w = w), grads, action = "silent",
                                max_abs = 1e-6)
  expect_equal(flagged$status, "large")
})

test_that("check_grad_anomaly: reports params without gradients as missing", {
  w      <- ag_param(matrix(1, 2, 2))
  orphan <- ag_param(matrix(1, 2, 2))
  grads  <- make_grads(w)

  report <- check_grad_anomaly(list(w = w, orphan = orphan), grads,
                               action = "silent")
  expect_equal(nrow(report), 2L)
  expect_equal(report$status[report$param == "orphan"], "missing")
  # a missing gradient is not an anomaly -> no warning
  expect_silent(check_grad_anomaly(list(orphan = orphan), grads))
})

# ============================================================================
# lr_scheduler_cyclic
# ============================================================================

test_that("lr_scheduler_cyclic: triangular cycle stays within bounds", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01)
  sch <- lr_scheduler_cyclic(opt, base_lr = 0.001, max_lr = 0.01,
                             step_size_up = 5L)

  lrs <- vapply(1:20, function(i) sch$step(), numeric(1))
  expect_true(all(lrs >= 0.001 - 1e-12))
  expect_true(all(lrs <= 0.01 + 1e-12))
})

test_that("lr_scheduler_cyclic: peaks at step_size_up and returns to base", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01)
  sch <- lr_scheduler_cyclic(opt, base_lr = 0, max_lr = 1, step_size_up = 4L)

  lrs <- vapply(1:8, function(i) sch$step(), numeric(1))
  expect_equal(lrs[4], 1, tolerance = 1e-10)  # top of the triangle
  expect_equal(lrs[8], 0, tolerance = 1e-10)  # back to base at cycle end
  expect_true(all(diff(lrs[1:4]) > 0))
  expect_true(all(diff(lrs[4:8]) < 0))
})

test_that("lr_scheduler_cyclic: sets optimizer lr to base at construction", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.5)
  lr_scheduler_cyclic(opt, base_lr = 0.001, max_lr = 0.01, step_size_up = 5L)
  expect_equal(opt$lr, 0.001)
})

test_that("lr_scheduler_cyclic: triangular2 halves amplitude each cycle", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01)
  sch <- lr_scheduler_cyclic(opt, base_lr = 0, max_lr = 1, step_size_up = 4L,
                             mode = "triangular2")

  lrs <- vapply(1:16, function(i) sch$step(), numeric(1))
  # peak of cycle 1 (step 4) vs peak of cycle 2 (step 12)
  expect_equal(lrs[4], 1.0, tolerance = 1e-10)
  expect_equal(lrs[12], 0.5, tolerance = 1e-10)
})

test_that("lr_scheduler_cyclic: validates arguments", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01)
  expect_error(lr_scheduler_cyclic(opt, 0.001, 0.01, step_size_up = 0L), "step_size_up")
  expect_error(lr_scheduler_cyclic(opt, base_lr = 1, max_lr = 0.1), "max_lr")
})

# ============================================================================
# lr_scheduler_onecycle
# ============================================================================

test_that("lr_scheduler_onecycle: rises to max_lr then anneals down", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01, momentum = 0.9)
  sch <- lr_scheduler_onecycle(opt, max_lr = 0.1, total_steps = 100L)

  lrs <- vapply(1:100, function(i) sch$step(), numeric(1))
  expect_equal(max(lrs), 0.1, tolerance = 1e-10)
  expect_equal(which.max(lrs), 30L)          # pct_start = 0.3
  expect_true(all(diff(lrs[1:30]) > 0))      # warmup phase
  expect_true(all(diff(lrs[30:100]) < 0))    # annealing phase
  expect_equal(lrs[100], 0.1 / 1e4, tolerance = 1e-10)
})

test_that("lr_scheduler_onecycle: starts at max_lr/div_factor", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01)
  lr_scheduler_onecycle(opt, max_lr = 0.1, total_steps = 50L, div_factor = 25)
  expect_equal(opt$lr, 0.1 / 25, tolerance = 1e-12)
})

test_that("lr_scheduler_onecycle: cycles SGD momentum inversely to lr", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01, momentum = 0.9)
  sch <- lr_scheduler_onecycle(opt, max_lr = 0.1, total_steps = 20L,
                               base_momentum = 0.85, max_momentum = 0.95)
  expect_equal(opt$momentum, 0.95, tolerance = 1e-12)

  moms <- vapply(1:20, function(i) { sch$step(); opt$momentum }, numeric(1))
  # momentum bottoms out where the lr peaks, then climbs back
  expect_equal(min(moms), 0.85, tolerance = 1e-9)
  expect_true(all(moms >= 0.85 - 1e-9 & moms <= 0.95 + 1e-9))
})

test_that("lr_scheduler_onecycle: cycles Adam beta1", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 1e-3)
  sch <- lr_scheduler_onecycle(opt, max_lr = 0.01, total_steps = 20L)
  expect_equal(opt$beta1, 0.95, tolerance = 1e-12)

  for (i in 1:6) sch$step()
  expect_lt(opt$beta1, 0.95)
})

test_that("lr_scheduler_onecycle: cycle_momentum=FALSE leaves momentum alone", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01, momentum = 0.9)
  sch <- lr_scheduler_onecycle(opt, max_lr = 0.1, total_steps = 20L,
                               cycle_momentum = FALSE)
  for (i in 1:10) sch$step()
  expect_equal(opt$momentum, 0.9, tolerance = 1e-12)
})

test_that("lr_scheduler_onecycle: linear strategy also peaks at max_lr", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01)
  sch <- lr_scheduler_onecycle(opt, max_lr = 0.1, total_steps = 50L,
                               anneal_strategy = "linear")
  lrs <- vapply(1:50, function(i) sch$step(), numeric(1))
  expect_equal(max(lrs), 0.1, tolerance = 1e-10)
})

test_that("lr_scheduler_onecycle: extra steps past total_steps are clamped", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01)
  sch <- lr_scheduler_onecycle(opt, max_lr = 0.1, total_steps = 10L)
  for (i in 1:10) sch$step()
  final <- opt$lr
  sch$step(); sch$step()
  expect_equal(opt$lr, final, tolerance = 1e-12)
})

test_that("lr_scheduler_onecycle: validates arguments", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.01)
  expect_error(lr_scheduler_onecycle(opt, 0.1, total_steps = 1L), "total_steps")
  expect_error(lr_scheduler_onecycle(opt, 0.1, 100L, pct_start = 0), "pct_start")
  expect_error(lr_scheduler_onecycle(opt, 0.1, 100L, pct_start = 1), "pct_start")
})

# ============================================================================
# lr_scheduler_warmup_cosine
# ============================================================================

test_that("lr_scheduler_warmup_cosine: warms up then anneals to lr_min", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 1e-3)
  sch <- lr_scheduler_warmup_cosine(opt, warmup_steps = 10L, total_steps = 100L)

  lrs <- vapply(1:100, function(i) sch$step(), numeric(1))
  expect_equal(lrs[10], 1e-3, tolerance = 1e-12)   # peak at end of warmup
  expect_true(all(diff(lrs[1:10]) > 0))            # linear warmup
  expect_true(all(diff(lrs[10:100]) < 0))          # cosine decay
  expect_equal(lrs[100], 0, tolerance = 1e-12)
})

test_that("lr_scheduler_warmup_cosine: starts at warmup_start_lr", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 1e-3)
  lr_scheduler_warmup_cosine(opt, warmup_steps = 5L, total_steps = 50L,
                             warmup_start_lr = 1e-5)
  expect_equal(opt$lr, 1e-5, tolerance = 1e-15)
})

test_that("lr_scheduler_warmup_cosine: respects lr_min", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  sch <- lr_scheduler_warmup_cosine(opt, warmup_steps = 2L, total_steps = 20L,
                                    lr_min = 0.01)
  lrs <- vapply(1:20, function(i) sch$step(), numeric(1))
  expect_equal(lrs[20], 0.01, tolerance = 1e-10)
  expect_true(all(lrs >= 0.01 - 1e-12))
})

test_that("lr_scheduler_warmup_cosine: warmup_steps=0 starts at peak", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  sch <- lr_scheduler_warmup_cosine(opt, warmup_steps = 0L, total_steps = 10L)
  expect_equal(opt$lr, 0.1, tolerance = 1e-12)
  lrs <- vapply(1:10, function(i) sch$step(), numeric(1))
  expect_true(all(diff(lrs) < 0))
})

test_that("lr_scheduler_warmup_cosine: validates arguments", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  expect_error(lr_scheduler_warmup_cosine(opt, -1L, 10L), "warmup_steps")
  expect_error(lr_scheduler_warmup_cosine(opt, 10L, 10L), "total_steps")
})

# ============================================================================
# lr_scheduler_cosine — SGDR T_mult extension
# ============================================================================

test_that("lr_scheduler_cosine: T_mult=1 keeps the old constant period", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  sch <- lr_scheduler_cosine(opt, T_max = 4L, restart = TRUE, T_mult = 1)

  lrs <- vapply(1:12, function(i) sch$step(), numeric(1))
  # cycle repeats exactly every 4 steps
  expect_equal(lrs[1:4], lrs[5:8],  tolerance = 1e-12)
  expect_equal(lrs[1:4], lrs[9:12], tolerance = 1e-12)
})

test_that("lr_scheduler_cosine: T_mult=2 doubles each cycle length", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  sch <- lr_scheduler_cosine(opt, T_max = 4L, restart = TRUE, T_mult = 2)

  lrs <- vapply(1:16, function(i) sch$step(), numeric(1))
  # restarts open cycles of length 4, then 8, then 16 -> steps 1, 5, 13
  expect_equal(lrs[1],  0.1, tolerance = 1e-12)
  expect_equal(lrs[5],  0.1, tolerance = 1e-12)
  expect_equal(lrs[13], 0.1, tolerance = 1e-12)
  # each cycle descends monotonically to its own minimum
  expect_true(all(diff(lrs[1:4])   < 0))
  expect_true(all(diff(lrs[5:12])  < 0))
  expect_true(all(diff(lrs[13:16]) < 0))
})

test_that("lr_scheduler_cosine: restart=FALSE behaviour is unchanged", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  sch <- lr_scheduler_cosine(opt, T_max = 6L, lr_min = 0.01, restart = FALSE)

  lrs <- vapply(1:8, function(i) sch$step(), numeric(1))
  expect_true(all(diff(lrs[1:6]) < 0))
  # holds at the floor once T_max is passed
  expect_equal(lrs[7], lrs[6], tolerance = 1e-12)
  expect_equal(lrs[8], lrs[6], tolerance = 1e-12)
})

test_that("lr_scheduler_cosine: validates T_mult and T_max", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  expect_error(lr_scheduler_cosine(opt, T_max = 0L), "T_max")
  expect_error(lr_scheduler_cosine(opt, T_max = 5L, T_mult = 0.5), "T_mult")
})
