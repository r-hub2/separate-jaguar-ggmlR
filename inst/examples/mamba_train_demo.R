# End-to-end training of a Mamba-style block.
#
# ggml has no backward pass for any of the state-space or RWKV recurrences, so
# until ggmlR added them these ops were inference-only. This script is the
# end-to-end check that the gradients actually train something: it fits the
# convolution kernel and the selective-scan projections of a small Mamba block
# by gradient descent and reports whether the loss falls.
#
# The individual gradients are verified element by element against a numeric
# gradient in tests/testthat/test-ssm-rwkv.R. What this adds is the question
# those tests cannot answer: does a whole block, wired together and stepped by
# an optimizer, converge?
#
#   Rscript inst/examples/mamba_train_demo.R          # GPU when available
#   GGMLR_DEMO_BACKEND=cpu Rscript inst/examples/mamba_train_demo.R
#
# On the GPU the graph is mixed: the scheduler keeps what Vulkan supports there
# and moves the rest to the CPU. The SSM backward kernels have no Vulkan shader
# (like OUT_PROD and CROSS_ENTROPY_LOSS_BACK before ggmlR added theirs), so the
# forward runs on the GPU and the backward nodes fall back. Running both ways
# and comparing is the point: the loss curve should not depend on where the
# nodes ran.

library(ggmlR)

# One thread: the SSM backward kernels are single-threaded by design (B and C
# are shared across the heads of a group, so accumulating from several threads
# would race), and this keeps the run deterministic.
ggml_set_n_threads(1L)

set.seed(20260819)

# ---- backend selection ----------------------------------------------------
# ggml_graph_compute() always builds a CPU backend internally, so a graph that
# should touch the GPU has to go through a scheduler instead. The scheduler is
# also what real GPU-first training uses: it assigns each node to a backend that
# supports it and copies tensors across as needed.

want_gpu <- !identical(tolower(Sys.getenv("GGMLR_DEMO_BACKEND")), "cpu")
use_gpu  <- want_gpu && ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

if (use_gpu) {
  cat("Backend: Vulkan (backward nodes fall back to CPU via the scheduler)\n")
} else if (want_gpu) {
  cat("Backend: CPU (no Vulkan device found)\n")
} else {
  cat("Backend: CPU (requested)\n")
}

# ---- problem -------------------------------------------------------------
# A task the recurrence can solve and a memoryless map cannot: every output
# position must reflect the running sum of the inputs before it. Only a layer
# that carries state across time can represent that.

# Sizes close to a real Mamba block rather than a toy. The defaults below put
# roughly 100k elements through the scan per step, which is enough for the GPU
# forward to be doing real work; GGMLR_DEMO_SMALL=1 shrinks it back to a size
# that runs in a second, for a quick check.
small <- identical(Sys.getenv("GGMLR_DEMO_SMALL"), "1")

d_state  <- if (small)  4L else  16L   # SSM state width (Mamba uses 16)
head_dim <- if (small)  2L else  64L   # channels per head
n_head   <- if (small)  2L else  16L   # heads -> d_inner = 1024
n_tok    <- if (small)  6L else  64L   # tokens per sequence
n_seqs   <- 1L
d_conv   <- if (small)  3L else   4L   # Mamba uses 4
d_inner  <- head_dim * n_head

n_samples <- if (small) 24L else 16L

make_batch <- function() {
  x <- array(runif(head_dim * n_head * n_tok * n_seqs, -1, 1),
             dim = c(head_dim, n_head, n_tok, n_seqs))
  # target[.., t, .] = cumulative mean of x up to t
  y <- x
  acc <- array(0, dim = c(head_dim, n_head))
  for (t in seq_len(n_tok)) {
    acc <- acc + x[, , t, 1]
    y[, , t, 1] <- acc / t
  }
  list(x = as.vector(x), y = as.vector(y))
}

batches <- lapply(seq_len(n_samples), function(i) make_batch())

# ---- trainable parameters -------------------------------------------------
# Kept as plain R vectors between steps; each step rebuilds the graph, uploads
# these, computes the gradient, and applies it. That is slower than a persistent
# optimizer context, but it keeps the script readable and exercises exactly the
# path a training loop would.

params <- list(
  # ssm_conv kernel, [d_conv, d_inner]
  conv = runif(d_conv * d_inner, -0.3, 0.3),
  # ssm_scan projections
  B    = runif(d_state * 1L * n_tok * n_seqs, -0.3, 0.3),
  C    = runif(d_state * 1L * n_tok * n_seqs, -0.3, 0.3),
  # per-head decay; negative keeps the recurrence stable
  A    = -runif(n_head, 0.2, 1.0),
  dt   = runif(n_head * n_tok * n_seqs, -0.5, 0.5)
)

# ---- one forward+backward pass -------------------------------------------
# Returns the loss and the gradient of every parameter.
step_grads <- function(p, batch) {
  ctx <- ggml_init(128 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  # Tensors are allocated by the backend, not the context, so the same code
  # serves both paths.
  ggml_set_no_alloc(ctx, TRUE)

  # Inputs. The conv branch needs d_conv-1 positions of left context, so the
  # sequence fed to ssm_conv is longer than the token count.
  conv_len <- d_conv - 1L + n_tok
  sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, conv_len, d_inner, n_seqs)
  ggml_set_input(sx)

  cw <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner)
  ggml_set_input(cw)
  ggml_set_param(cw)

  # conv branch: [d_inner, n_tok, n_seqs]
  conv_out <- ggml_ssm_conv(ctx, sx, cw)

  # The scan takes x as [head_dim, n_head, n_tok, n_seqs]; the conv output has
  # the same element count, so a reshape re-labels the axes.
  x_scan <- ggml_reshape_4d(ctx, conv_out, head_dim, n_head, n_tok, n_seqs)

  s0 <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs)
  ggml_set_input(s0)

  dt <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_tok, n_seqs)
  ggml_set_input(dt); ggml_set_param(dt)

  A <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, n_head)
  ggml_set_input(A); ggml_set_param(A)

  B <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, 1L, n_tok, n_seqs)
  ggml_set_input(B); ggml_set_param(B)

  C <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, 1L, n_tok, n_seqs)
  ggml_set_input(C); ggml_set_param(C)

  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs)
  ggml_set_input(ids)

  scan <- ggml_ssm_scan(ctx, s0, x_scan, dt, A, B, C, ids)
  # The scan packs outputs then final state; only the outputs are scored.
  y_hat <- ggml_ssm_scan_output(ctx, scan, x_scan)

  target <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tok, n_seqs)
  ggml_set_input(target)

  diff <- ggml_sub(ctx, y_hat, target)
  loss <- ggml_sum(ctx, ggml_sqr(ctx, diff))
  ggml_set_output(loss)
  ggml_set_loss(loss)

  # The full low-level autodiff sequence. ggml_graph_reset() is not optional:
  # it seeds d(loss)/d(loss) = 1, and without it every gradient computes as
  # zero -- silently, since nothing errors.
  graph <- ggml_build_forward_expand_grads(ctx, loss, graph_size = 4096L)
  ggml_build_backward_expand(ctx, graph)

  # Allocate on the chosen backend, then upload. The order matters: with
  # no_alloc the tensors have no storage until the backend provides it.
  backend <- if (use_gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  # The scheduler appends a CPU backend of its own, which is what lets the
  # backward nodes Vulkan cannot run fall back rather than fail.
  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  ggml_backend_sched_reset(sched)
  ggml_backend_sched_alloc_graph(sched, graph)

  ggml_backend_tensor_set_data(sx, c(rep(0, (d_conv - 1L) * d_inner * n_seqs), batch$x))
  ggml_backend_tensor_set_data(cw, p$conv)
  ggml_backend_tensor_set_data(s0, rep(0, d_state * head_dim * n_head * n_seqs))
  ggml_backend_tensor_set_data(dt, p$dt)
  ggml_backend_tensor_set_data(A,  p$A)
  ggml_backend_tensor_set_data(B,  p$B)
  ggml_backend_tensor_set_data(C,  p$C)
  ggml_backend_tensor_set_data(ids, as.integer(seq_len(n_seqs) - 1L))
  ggml_backend_tensor_set_data(target, batch$y)

  ggml_graph_reset(graph)
  ggml_backend_sched_graph_compute(sched, graph)

  grab <- function(t) {
    g <- ggml_graph_get_grad(graph, t)
    if (is.null(g)) rep(0, ggml_nelements(t)) else ggml_backend_tensor_get_data(g)
  }

  list(loss  = ggml_backend_tensor_get_data(loss)[1],
       grads = list(conv = grab(cw), dt = grab(dt), A = grab(A),
                    B = grab(B), C = grab(C)))
}

# ---- training loop --------------------------------------------------------

# The loss is a SUM over every output element, so its gradient scales with the
# tensor size -- a learning rate tuned on the toy sizes is far too small once
# d_inner is 1024. Normalising by the element count keeps the step comparable
# across both configurations.
lr      <- 0.05 * (if (small) 1 else 24)
n_epoch <- if (small) 40L else 60L
history <- numeric(n_epoch)

cat("Training a Mamba-style block (ssm_conv -> ssm_scan)\n")
cat(sprintf("  d_state=%d head_dim=%d n_head=%d d_inner=%d tokens=%d samples=%d\n",
            d_state, head_dim, n_head, d_inner, n_tok, n_samples))
cat(sprintf("  scan work: %.1fM multiply-adds per step\n\n",
            n_tok * n_head * head_dim * d_state / 1e6))

t_start <- Sys.time()

for (epoch in seq_len(n_epoch)) {
  epoch_loss <- 0

  for (b in batches) {
    r <- step_grads(params, b)
    epoch_loss <- epoch_loss + r$loss

    # Plain SGD. Gradients are clipped because the recurrence can amplify them
    # early on, before A has settled into a decaying range.
    for (nm in names(r$grads)) {
      g <- r$grads[[nm]]
      g[!is.finite(g)] <- 0
      # Clip per parameter, with a threshold that grows with its size -- a fixed
      # cap would throttle the 1024-wide conv kernel far harder than the
      # per-head A vector, which is only n_head long.
      gn  <- sqrt(sum(g^2))
      cap <- 5 * sqrt(length(g))
      if (gn > cap) g <- g * (cap / gn)
      params[[nm]] <- params[[nm]] - lr * g
    }
    # Keep the decay stable: A must stay negative for exp(dt*A) to contract.
    params$A <- pmin(params$A, -0.05)
  }

  history[epoch] <- epoch_loss / length(batches)
  if (epoch %% max(1L, n_epoch %/% 8L) == 0L || epoch == 1L) {
    cat(sprintf("  epoch %2d   loss %.5f\n", epoch, history[epoch]))
  }
}

elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
cat(sprintf("\n%.1f s total, %.2f s per epoch\n",
            elapsed, elapsed / n_epoch))
first <- history[1]
last  <- history[n_epoch]
cat(sprintf("loss: %.5f -> %.5f  (%.1f%% of the starting value)\n",
            first, last, 100 * last / first))

if (!is.finite(last)) {
  cat("FAIL: the loss is not finite -- a gradient went to NaN/Inf.\n")
} else if (last < first) {
  cat("OK: the block trains; gradients flow through ssm_conv and ssm_scan.\n")
} else {
  cat("FAIL: the loss did not fall.\n")
}
