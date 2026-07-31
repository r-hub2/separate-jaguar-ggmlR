# ggml_layer_concatenate(): multi-branch feature fusion
#
# Demonstrates concatenating several parallel branches into one feature vector
# inside a single-input functional model, and verifies that the concatenate
# node is fully differentiable — GGML_OP_CONCAT has a backward implementation
# in src/ggml-graph.c, so gradients flow back into EVERY concatenated branch.
#
# Architecture (an "inception-style" block):
#
#   input(8) ─┬─ Dense(8, relu)  ── wide branch  ─┐
#             ├─ Dense(4, relu)  ── narrow branch ├─ concatenate(axis=0) ->
#             └─ Dense(4, tanh)  ── tanh branch  ─┘
#                                                   Dense(8, relu) -> Dense(2, softmax)
#
# Task: binary classification of whether a vector's sum exceeds a threshold.

library(ggmlR)

invisible(ggml_set_n_threads(1L))
set.seed(42)

# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------

N       <- 512L   # divisible by batch_size = 32
N_FEAT  <- 8L

x <- matrix(runif(N * N_FEAT), nrow = N, ncol = N_FEAT)
y <- matrix(0.0, nrow = N, ncol = 2L)
for (i in seq_len(N)) {
  y[i, if (sum(x[i, ]) > N_FEAT / 2) 1L else 2L] <- 1.0
}

cat(sprintf("Dataset: %d samples, %d features, class balance %.2f / %.2f\n\n",
            N, N_FEAT, mean(y[, 1]), mean(y[, 2])))

# ---------------------------------------------------------------------------
# 2. Build the model — three branches fused with ggml_layer_concatenate()
# ---------------------------------------------------------------------------

inp <- ggml_input(shape = N_FEAT, name = "features")

branch_wide   <- inp |> ggml_layer_dense(8L, activation = "relu")
branch_narrow <- inp |> ggml_layer_dense(4L, activation = "relu")
branch_tanh   <- inp |> ggml_layer_dense(4L, activation = "tanh")

# axis = 0 concatenates along the feature dimension: 8 + 4 + 4 = 16 features.
# (axis is 0-based, following the ggml convention; axis = -1 also selects the
#  last non-batch axis.)
fused <- ggml_layer_concatenate(
  list(branch_wide, branch_narrow, branch_tanh),
  axis = 0L,
  name = "fusion"
)

out <- fused |>
  ggml_layer_dense(8L, activation = "relu") |>
  ggml_layer_dense(2L, activation = "softmax")

model <- ggml_model(inputs = inp, outputs = out)
model <- ggml_compile(model, optimizer = "adam",
                      loss = "categorical_crossentropy")

cat("Model compiled. Branch node ids:\n")
cat(sprintf("  wide   = %s\n  narrow = %s\n  tanh   = %s\n\n",
            branch_wide$id, branch_narrow$id, branch_tanh$id))

# ---------------------------------------------------------------------------
# 3. Train, snapshotting weights to prove gradients reach all branches
# ---------------------------------------------------------------------------

# model$node_weights is populated only after a fit, so take the baseline
# snapshot after a single epoch and continue training from there.
model <- ggml_fit(model, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)

snapshot <- function(m) {
  lapply(m$node_weights, function(w) lapply(w, ggml_backend_tensor_get_data))
}
before <- snapshot(model)

model <- ggml_fit(model, x, y, epochs = 15L, batch_size = 32L, verbose = 0L)
after <- snapshot(model)

# ---------------------------------------------------------------------------
# 4. Verify the backward pass through concatenate
# ---------------------------------------------------------------------------

cat("\n--- gradient flow through ggml_layer_concatenate ---\n")

branches <- list(wide   = branch_wide,
                 narrow = branch_narrow,
                 tanh   = branch_tanh)

all_ok <- TRUE
for (nm in names(branches)) {
  nid <- branches[[nm]]$id
  for (k in names(after[[nid]])) {
    delta <- max(abs(after[[nid]][[k]] - before[[nid]][[k]]))
    ok    <- delta > 1e-8
    all_ok <- all_ok && ok
    cat(sprintf("  %-6s $%-6s  max|dW| = %.6g   %s\n",
                nm, k, delta,
                if (ok) "gradients flow" else "*** ZERO GRADIENT ***"))
  }
}

loss <- model$history$train_loss
acc  <- model$history$train_accuracy

cat("\n--- training history ---\n")
for (ep in seq_along(loss)) {
  cat(sprintf("  epoch %2d/%d   loss = %.4f   acc = %.3f\n",
              ep, length(loss), loss[ep], acc[ep]))
}
cat(sprintf("\nloss: %.4f -> %.4f   accuracy: %.3f -> %.3f\n",
            loss[1], loss[length(loss)], acc[1], acc[length(acc)]))

if (all_ok && loss[length(loss)] < loss[1]) {
  cat("\nOK: concatenate is differentiable — every branch trained.\n")
} else {
  cat("\nFAIL: concatenate did not propagate gradients to all branches.\n")
}

# ---------------------------------------------------------------------------
# 5. Predict
# ---------------------------------------------------------------------------

# predict() runs on whole batches, so feed at least batch_size rows.
pred <- ggml_predict(model, x[1:32, , drop = FALSE])
cat("\nFirst 5 predictions (softmax) vs. true labels:\n")
print(data.frame(
  p_class1 = round(pred[1:5, 1], 4),
  p_class2 = round(pred[1:5, 2], 4),
  true     = max.col(y[1:5, , drop = FALSE])
))
