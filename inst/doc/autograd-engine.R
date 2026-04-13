## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(eval = TRUE)

## -----------------------------------------------------------------------------
library(ggmlR)

## -----------------------------------------------------------------------------
# ag_tensor — non-trainable input (e.g. data)
x <- ag_tensor(matrix(1:6 / 6, nrow = 2L))   # [2, 3]

# ag_param — trainable parameter (gradient accumulated)
W <- ag_param(matrix(rnorm(4), nrow = 2L))    # [2, 2]
b <- ag_param(matrix(0.0, 2L, 1L))

## -----------------------------------------------------------------------------
with_grad_tape({
  h    <- ag_relu(ag_add(ag_matmul(W, x), b))   # [2, 3]
  loss <- ag_mean(ag_mul(h, h))                  # scalar MSE-like
})

grads <- backward(loss)   # returns named list of gradients

cat("dL/dW:\n"); print(grads[["W"]])
cat("dL/db:\n"); print(grads[["b"]])

## -----------------------------------------------------------------------------
data(iris)
set.seed(42)

x_all <- t(scale(as.matrix(iris[, 1:4])))        # [4, 150]
y_oh  <- model.matrix(~ Species - 1, iris)
y_all <- t(y_oh)                                  # [3, 150]

idx  <- sample(150L)
x_tr <- x_all[, idx[1:120]];  x_vl <- x_all[, idx[121:150]]
y_tr <- y_all[, idx[1:120]];  y_vl <- y_all[, idx[121:150]]

model <- ag_sequential(
  ag_linear(4L,  64L, activation = "relu"),
  ag_batch_norm(64L),
  ag_dropout(0.3),
  ag_linear(64L, 32L, activation = "relu"),
  ag_linear(32L,  3L)
)

params <- model$parameters()
cat("Parameter tensors:", length(params), "\n")

## -----------------------------------------------------------------------------
# Adam (default lr = 1e-3)
opt <- optimizer_adam(params, lr = 1e-3)

# SGD with momentum
opt_sgd <- optimizer_sgd(params, lr = 0.05, momentum = 0.9)

## -----------------------------------------------------------------------------
BS <- 32L
n  <- ncol(x_tr)

ag_train(model)   # set training mode (enables dropout, batch norm train)
set.seed(1)

for (ep in seq_len(150L)) {
  perm <- sample(n)
  for (b in seq_len(ceiling(n / BS))) {
    idx <- perm[seq((b-1L)*BS + 1L, min(b*BS, n))]
    xb  <- ag_tensor(x_tr[, idx, drop = FALSE])
    yb  <- y_tr[, idx, drop = FALSE]

    with_grad_tape({
      loss <- ag_softmax_cross_entropy_loss(model$forward(xb), yb)
    })
    grads <- backward(loss)
    opt$step(grads)
    opt$zero_grad()
  }

  if (ep %% 50L == 0L)
    cat(sprintf("epoch %3d  loss %.4f\n", ep, loss$data[1]))
}

## -----------------------------------------------------------------------------
opt2 <- optimizer_adam(params, lr = 1e-3)

# Cosine annealing: lr goes from lr_max to lr_min over T_max epochs
sch_cos <- lr_scheduler_cosine(opt2, T_max = 150L, lr_min = 1e-5)

# Step decay: multiply lr by gamma every step_size epochs
sch_step <- lr_scheduler_step(opt2, step_size = 30L, gamma = 0.5)

# Call after each epoch:
# sch_cos$step()

## -----------------------------------------------------------------------------
with_grad_tape({
  loss <- ag_softmax_cross_entropy_loss(model$forward(ag_tensor(x_tr)), y_tr)
})
grads <- backward(loss)

# Clip global gradient norm to max_norm
clip_grad_norm(params, grads, max_norm = 5.0)

opt$step(grads)
opt$zero_grad()

## -----------------------------------------------------------------------------
dl <- ag_dataloader(x_tr, y_tr, batch_size = BS, shuffle = TRUE)

ag_train(model)
for (ep in seq_len(100L)) {
  for (batch in dl$epoch()) {
    with_grad_tape({
      loss <- ag_softmax_cross_entropy_loss(model$forward(batch$x), batch$y$data)
    })
    grads <- backward(loss)
    opt$step(grads);  opt$zero_grad()
  }
}

## -----------------------------------------------------------------------------
ag_eval(model)   # disables dropout, switches batch norm to inference stats

# Forward in chunks to avoid memory pressure
predict_cm <- function(mod, x_cm, chunk = 64L) {
  n   <- ncol(x_cm)
  out <- NULL
  for (s in seq(1L, n, by = chunk)) {
    e  <- min(s + chunk - 1L, n)
    lg <- mod$forward(ag_tensor(x_cm[, s:e, drop = FALSE]))$data
    ev <- exp(lg - apply(lg, 2, max))
    sm <- ev / colSums(ev)
    out <- if (is.null(out)) sm else cbind(out, sm)
  }
  out
}

probs <- predict_cm(model, x_vl)          # [3, 30]
preds <- apply(probs, 2, which.max)
truth <- apply(y_vl, 1, which.max)
cat(sprintf("Val accuracy: %.4f\n", mean(preds == truth)))

## -----------------------------------------------------------------------------
set.seed(7)
W1 <- ag_param(matrix(rnorm(64*4) * sqrt(2/4),  64, 4))
b1 <- ag_param(matrix(0.0, 64, 1))
W2 <- ag_param(matrix(rnorm(3*64) * sqrt(2/64),  3, 64))
b2 <- ag_param(matrix(0.0,  3, 1))

forward <- function(x)
  ag_add(ag_matmul(W2, ag_relu(ag_add(ag_matmul(W1, x), b1))), b2)

opt_raw <- optimizer_adam(list(W1=W1, b1=b1, W2=W2, b2=b2), lr = 1e-3)

for (ep in seq_len(200L)) {
  perm <- sample(n)
  for (b in seq_len(ceiling(n / BS))) {
    idx <- perm[seq((b-1L)*BS+1L, min(b*BS, n))]
    xb  <- ag_tensor(x_tr[, idx, drop = FALSE])
    yb  <- y_tr[, idx, drop = FALSE]
    with_grad_tape({ loss_r <- ag_softmax_cross_entropy_loss(forward(xb), yb) })
    gr <- backward(loss_r)
    opt_raw$step(gr);  opt_raw$zero_grad()
  }
}

## -----------------------------------------------------------------------------
# f16 on GPU, f32 on CPU
device <- tryCatch({ ag_device("gpu"); "gpu" }, error = function(e) "cpu")
ag_dtype(if (device == "gpu") "f16" else "f32")

# All subsequent ag_param / ag_tensor use the selected dtype

## -----------------------------------------------------------------------------
# Full example: inst/examples/dp_train_demo.R

