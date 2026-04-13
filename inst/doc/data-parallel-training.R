## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(eval = TRUE)

## -----------------------------------------------------------------------------
library(ggmlR)

## -----------------------------------------------------------------------------
data(iris)
set.seed(42)

x_cm <- t(scale(as.matrix(iris[, 1:4])))    # [4, 150]
y_oh <- t(model.matrix(~ Species - 1, iris)) # [3, 150]

# Dataset as list of (x, y) pairs — one sample each
dp_data <- lapply(seq_len(ncol(x_cm)), function(i)
  list(x = x_cm[, i, drop = FALSE],
       y = y_oh[, i, drop = FALSE]))

# Model factory — called once per replica
make_model <- function() {
  ag_sequential(
    ag_linear(4L, 32L, activation = "relu"),
    ag_linear(32L, 3L)
  )
}

result <- dp_train(
  make_model = make_model,
  data       = dp_data,
  loss_fn    = function(out, tgt) ag_softmax_cross_entropy_loss(out, tgt),
  forward_fn = function(model, s)  model$forward(ag_tensor(s$x)),
  target_fn  = function(s)         s$y,
  n_gpu      = 1L,         # set to ggml_vulkan_device_count() for multi-GPU
  n_iter     = 2000L,
  lr         = 1e-3,
  verbose    = TRUE
)

cat("Final loss:", result$loss, "\n")
model <- result$model

## -----------------------------------------------------------------------------
n_gpu <- max(1L, ggml_vulkan_device_count())
cat(sprintf("Training on %d GPU(s)\n", n_gpu))

result_mg <- dp_train(
  make_model = make_model,
  data       = dp_data,
  loss_fn    = function(out, tgt) ag_softmax_cross_entropy_loss(out, tgt),
  forward_fn = function(model, s)  model$forward(ag_tensor(s$x)),
  target_fn  = function(s)         s$y,
  n_gpu      = n_gpu,
  n_iter     = 2000L,
  lr         = 1e-3,
  max_norm   = 5.0,      # gradient clipping
  verbose    = FALSE
)

## -----------------------------------------------------------------------------
result <- dp_train(
  make_model = make_model,
  data       = dp_data,
  loss_fn    = function(out, tgt) ag_softmax_cross_entropy_loss(out, tgt),
  forward_fn = function(model, s)  model$forward(ag_tensor(s$x)),
  target_fn  = function(s)         s$y,
  n_gpu      = 1L,
  n_iter     = 2000L,
  lr         = 1e-3,
  max_norm   = 1.0       # clip to unit norm
)

## -----------------------------------------------------------------------------
x_tr <- x_cm[, 1:120];  y_tr <- y_oh[, 1:120]

dl <- ag_dataloader(x_tr, y_tr, batch_size = 32L, shuffle = TRUE)

model2  <- make_model()
params2 <- model2$parameters()
opt2    <- optimizer_adam(params2, lr = 1e-3)

ag_train(model2)
for (ep in seq_len(100L)) {
  for (batch in dl$epoch()) {
    with_grad_tape({
      loss <- ag_softmax_cross_entropy_loss(
        model2$forward(batch$x), batch$y$data)
    })
    grads <- backward(loss)
    opt2$step(grads);  opt2$zero_grad()
  }
}

## -----------------------------------------------------------------------------
# inst/examples/dp_train_demo.R

## -----------------------------------------------------------------------------
# inst/examples/multi_gpu_example.R

