## ----setup, include=FALSE-----------------------------------------------------
# Vignette code is executed locally (NOT_CRAN=true) but not on CRAN, where
# the CPU fallback would multi-thread and trip the "CPU time > elapsed" NOTE.
knitr::opts_chunk$set(eval = identical(Sys.getenv("NOT_CRAN"), "true"))

## -----------------------------------------------------------------------------
# # install.packages("ggmlR")   # once on CRAN
# library(ggmlR)

## -----------------------------------------------------------------------------
# data(iris)
# set.seed(42)
# 
# x <- as.matrix(iris[, 1:4])
# x <- scale(x)                        # standardise
# 
# # one-hot encode species
# y <- model.matrix(~ Species - 1, iris)  # [150 x 3]
# 
# idx   <- sample(nrow(x))
# n_tr  <- 120L
# x_tr  <- x[idx[1:n_tr],  ]
# y_tr  <- y[idx[1:n_tr],  ]
# x_val <- x[idx[(n_tr+1):150], ]
# y_val <- y[idx[(n_tr+1):150], ]

## -----------------------------------------------------------------------------
# model <- ggml_model_sequential() |>
#   ggml_layer_dense(32L, activation = "relu", input_shape = 4L) |>
#   ggml_layer_batch_norm() |>
#   ggml_layer_dropout(0.3, stochastic = TRUE) |>
#   ggml_layer_dense(16L, activation = "relu") |>
#   ggml_layer_dense(3L,  activation = "softmax") |>
#   ggml_compile(
#     optimizer = "adam",
#     loss      = "categorical_crossentropy",
#     metrics   = c("accuracy")
#   )
# 
# print(model)

## -----------------------------------------------------------------------------
# model <- ggml_fit(
#   model,
#   x_tr, y_tr,
#   epochs           = 100L,
#   batch_size       = 32L,
#   validation_split = 0.0,   # we supply our own val set below
#   verbose          = 0L
# )

## -----------------------------------------------------------------------------
# score <- ggml_evaluate(model, x_val, y_val, batch_size = 16L)
# cat(sprintf("Val loss: %.4f   Val accuracy: %.4f\n", score$loss, score$accuracy))
# 
# probs   <- ggml_predict(model, x_val, batch_size = 16L)
# classes <- apply(probs, 1, which.max)
# true    <- apply(y_val, 1, which.max)
# cat("Confusion matrix:\n")
# print(table(true = true, predicted = classes))

## -----------------------------------------------------------------------------
# path <- tempfile(fileext = ".rds")
# ggml_save_model(model, path)
# cat(sprintf("Saved: %s (%.1f KB)\n", path, file.size(path) / 1024))
# 
# model2 <- ggml_load_model(path)
# score2 <- ggml_evaluate(model2, x_val, y_val, batch_size = 16L)
# cat(sprintf("Reloaded model accuracy: %.4f\n", score2$accuracy))

## -----------------------------------------------------------------------------
# inp  <- ggml_input(shape = 4L)
# 
# h    <- inp |> ggml_layer_dense(32L, activation = "relu") |> ggml_layer_batch_norm()
# h    <- h   |> ggml_layer_dense(16L, activation = "relu")
# out  <- h   |> ggml_layer_dense(3L,  activation = "softmax")
# 
# model_fn <- ggml_model(inputs = inp, outputs = out) |>
#   ggml_compile(optimizer = "adam", loss = "categorical_crossentropy",
#                metrics = c("accuracy"))
# 
# model_fn <- ggml_fit(model_fn, x_tr, y_tr,
#                      epochs = 100L, batch_size = 32L, verbose = 0L)
# 
# score_fn <- ggml_evaluate(model_fn, x_val, y_val, batch_size = 16L)
# cat(sprintf("Functional model accuracy: %.4f\n", score_fn$accuracy))

## -----------------------------------------------------------------------------
# data(mtcars)
# set.seed(7)
# 
# # Input group 1: engine (disp, hp, wt)
# # Input group 2: transmission / gearbox (cyl, gear, carb, am)
# engine <- as.matrix(scale(mtcars[, c("disp","hp","wt")]))
# trans  <- as.matrix(scale(mtcars[, c("cyl","gear","carb","am")]))
# y_mpg  <- matrix(scale(mtcars$mpg), ncol = 1L)          # [32 x 1]
# 
# # small dataset — use all for training, evaluate on same data for demo
# x1 <- engine;  x2 <- trans
# 
# inp1 <- ggml_input(shape = 3L, name = "engine")
# inp2 <- ggml_input(shape = 4L, name = "transmission")
# 
# branch1 <- inp1 |> ggml_layer_dense(16L, activation = "relu")
# branch2 <- inp2 |> ggml_layer_dense(16L, activation = "relu")
# 
# merged  <- ggml_layer_add(list(branch1, branch2))        # element-wise add
# out_reg <- merged |>
#   ggml_layer_dense(8L, activation = "relu") |>
#   ggml_layer_dense(1L)
# 
# model_reg <- ggml_model(inputs = list(inp1, inp2), outputs = out_reg) |>
#   ggml_compile(optimizer = "adam", loss = "mse")
# 
# model_reg <- ggml_fit(model_reg,
#                       x = list(x1, x2), y = y_mpg,
#                       epochs = 200L, batch_size = 16L, verbose = 0L)
# 
# preds <- ggml_predict(model_reg, x = list(x1, x2), batch_size = 16L)
# cat(sprintf("Pearson r (scaled mpg): %.4f\n", cor(preds, y_mpg)))

## -----------------------------------------------------------------------------
# br1 <- inp1 |> ggml_layer_dense(8L,  activation = "relu")
# br2 <- inp2 |> ggml_layer_dense(12L, activation = "relu")   # widths differ
# 
# out_cat <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
#   ggml_layer_dense(1L)
# 
# model_cat <- ggml_model(inputs = list(inp1, inp2), outputs = out_cat) |>
#   ggml_compile(optimizer = "adam", loss = "mse")
# model_cat <- ggml_fit(model_cat, x = list(x1, x2), y = y_mpg,
#                       epochs = 200L, batch_size = 16L, verbose = 0L)
# 
# p_cat <- ggml_predict(model_cat, x = list(x1, x2), batch_size = 16L)
# cat(sprintf("Concatenate variant, Pearson r: %.4f\n", cor(p_cat, y_mpg)))

## -----------------------------------------------------------------------------
# set.seed(11)
# 
# # Two targets from the same cars: efficient or not (class), and mpg (value).
# y_cls <- as.integer(mtcars$mpg > median(mtcars$mpg))
# y_cls <- cbind(1 - y_cls, y_cls) * 1.0         # one-hot [32 x 2]
# x_all <- as.matrix(scale(mtcars[, c("disp","hp","wt","cyl")]))
# 
# inp_mo <- ggml_input(shape = 4L, name = "features")
# trunk  <- inp_mo |> ggml_layer_dense(16L, activation = "relu", name = "trunk")
# head_c <- trunk  |> ggml_layer_dense(2L, activation = "softmax", name = "class")
# head_v <- trunk  |> ggml_layer_dense(1L, name = "value")
# 
# model_mo <- ggml_model(inputs = inp_mo, outputs = list(head_c, head_v)) |>
#   ggml_compile(optimizer = "adam",
#                loss         = list(class = "categorical_crossentropy",
#                                    value = "mse"),
#                loss_weights = c(class = 1.0, value = 0.5))
# 
# model_mo <- ggml_fit(model_mo, x_all,
#                      y = list(class = y_cls, value = y_mpg),
#                      epochs = 150L, batch_size = 16L, verbose = 0L)
# 
# # Per-head losses, so a head that stops learning is visible on its own
# # rather than hidden inside the total.
# h <- model_mo$history
# cat(sprintf("class loss: %.4f -> %.4f\n",
#             h$train_class_loss[1], tail(h$train_class_loss, 1)))
# cat(sprintf("value loss: %.4f -> %.4f\n",
#             h$train_value_loss[1], tail(h$train_value_loss, 1)))

## -----------------------------------------------------------------------------
# set.seed(13)
# n_seq <- 128L; S <- 6L; D <- 8L
# 
# # A task that needs mixing across positions: is the sequence sum positive?
# xs   <- array(runif(n_seq * S * D, -1, 1), dim = c(n_seq, S, D))
# lab  <- as.integer(apply(xs, 1, mean) > 0)
# y_sq <- cbind(1 - lab, lab) * 1.0
# 
# inp_t <- ggml_input(shape = c(S, D), name = "seq")
# 
# # Attention sublayer + residual
# att <- inp_t |> ggml_layer_attention(d_model = D, n_heads = 2L, name = "mha")
# h1  <- ggml_layer_add(list(inp_t, att))
# 
# # Position-wise feed-forward sublayer + residual
# ff  <- h1 |> ggml_layer_dense(D * 2L, activation = "relu",
#                               time_distributed = TRUE, name = "ff1")
# ff  <- ff |> ggml_layer_dense(D, time_distributed = TRUE, name = "ff2")
# h2  <- ggml_layer_add(list(h1, ff))
# 
# out_t <- h2 |> ggml_layer_flatten() |>
#   ggml_layer_dense(2L, activation = "softmax", name = "cls")
# 
# model_t <- ggml_model(inputs = inp_t, outputs = out_t) |>
#   ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")
# 
# model_t <- ggml_fit(model_t, xs, y_sq, epochs = 30L, batch_size = 32L,
#                     verbose = 0L)
# cat(sprintf("transformer block accuracy: %.4f\n",
#             tail(model_t$history$train_accuracy, 1)))

## ----eval=FALSE---------------------------------------------------------------
# dec <- inp_t |> ggml_layer_attention(D, n_heads = 2L, causal = TRUE)
# 
# ctx_seq <- ggml_input(shape = c(10L, D))
# x_cross <- ggml_apply(list(dec, ctx_seq), ggml_attention(D, n_heads = 2L))

## -----------------------------------------------------------------------------
# cb_stop <- ggml_callback_early_stopping(
#   monitor   = "val_loss",
#   patience  = 15L,
#   min_delta = 1e-4
# )
# 
# model_cb <- ggml_model_sequential() |>
#   ggml_layer_dense(32L, activation = "relu", input_shape = 4L) |>
#   ggml_layer_dense(3L,  activation = "softmax") |>
#   ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")
# 
# model_cb <- ggml_fit(model_cb, x_tr, y_tr,
#                      epochs           = 300L,
#                      batch_size       = 32L,
#                      validation_split = 0.1,
#                      callbacks        = list(cb_stop),
#                      verbose          = 0L)

## -----------------------------------------------------------------------------
# # Cosine annealing
# cb_cosine <- ggml_schedule_cosine_decay(T_max = 100L, eta_min = 1e-5)
# 
# # Step decay: halve LR every 30 epochs
# cb_step <- ggml_schedule_step_decay(step_size = 30L, gamma = 0.5)
# 
# # Reduce on plateau
# cb_plateau <- ggml_schedule_reduce_on_plateau(
#   monitor  = "val_loss",
#   factor   = 0.5,
#   patience = 10L,
#   min_lr   = 1e-6
# )
# 
# model_lr <- ggml_model_sequential() |>
#   ggml_layer_dense(32L, activation = "relu", input_shape = 4L) |>
#   ggml_layer_dense(3L,  activation = "softmax") |>
#   ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")
# 
# model_lr <- ggml_fit(model_lr, x_tr, y_tr,
#                      epochs           = 150L,
#                      batch_size       = 32L,
#                      validation_split = 0.1,
#                      callbacks        = list(cb_cosine),
#                      verbose          = 0L)

## -----------------------------------------------------------------------------
# # transpose: rows = features, cols = samples
# x_tr_ag  <- t(x_tr)    # [4, 120]
# y_tr_ag  <- t(y_tr)    # [3, 120]
# x_val_ag <- t(x_val)   # [4, 30]
# y_val_ag <- t(y_val)

## -----------------------------------------------------------------------------
# ag_mod <- ag_sequential(
#   ag_linear(4L,  32L, activation = "relu"),
#   ag_batch_norm(32L),
#   ag_dropout(0.3),
#   ag_linear(32L, 16L, activation = "relu"),
#   ag_linear(16L,  3L)
# )
# 
# params <- ag_mod$parameters()
# opt    <- optimizer_adam(params, lr = 1e-3)

## -----------------------------------------------------------------------------
# BS <- 32L
# n  <- ncol(x_tr_ag)
# 
# ag_train(ag_mod)
# set.seed(42)
# 
# for (ep in seq_len(150L)) {
#   perm <- sample(n)
#   for (b in seq_len(ceiling(n / BS))) {
#     idx <- perm[seq((b-1L)*BS + 1L, min(b*BS, n))]
#     xb  <- ag_tensor(x_tr_ag[, idx, drop = FALSE])
#     yb  <- y_tr_ag[, idx, drop = FALSE]
# 
#     with_grad_tape({
#       loss <- ag_softmax_cross_entropy_loss(ag_mod$forward(xb), yb)
#     })
#     grads <- backward(loss)
#     opt$step(grads)
#     opt$zero_grad()
#   }
# 
#   if (ep %% 50L == 0L)
#     cat(sprintf("epoch %d  loss %.4f\n", ep, loss$data[1]))
# }

## -----------------------------------------------------------------------------
# ag_eval(ag_mod)
# 
# # forward in chunks, apply softmax manually
# ag_predict_cm <- function(model, x_cm, chunk = 64L) {
#   n   <- ncol(x_cm)
#   out <- matrix(0.0, nrow(model$forward(ag_tensor(x_cm[,1,drop=FALSE]))$data), n)
#   for (s in seq(1L, n, by = chunk)) {
#     e  <- min(s + chunk - 1L, n)
#     lg <- model$forward(ag_tensor(x_cm[, s:e, drop = FALSE]))$data
#     ev <- exp(lg - apply(lg, 2, max))
#     out[, s:e] <- ev / colSums(ev)
#   }
#   out
# }
# 
# probs_ag <- ag_predict_cm(ag_mod, x_val_ag)          # [3, 30]
# pred_ag  <- apply(probs_ag, 2, which.max)
# true_ag  <- apply(y_val_ag, 1, which.max)            # col-major: rows = classes
# acc_ag   <- mean(pred_ag == true_ag)
# cat(sprintf("Autograd val accuracy: %.4f\n", acc_ag))

## -----------------------------------------------------------------------------
# ag_mod2 <- ag_sequential(
#   ag_linear(4L,  64L, activation = "relu"),
#   ag_batch_norm(64L),
#   ag_dropout(0.3),
#   ag_linear(64L, 32L, activation = "relu"),
#   ag_linear(32L,  3L)
# )
# params2 <- ag_mod2$parameters()
# opt2    <- optimizer_adam(params2, lr = 1e-3)
# sch2    <- lr_scheduler_cosine(opt2, T_max = 150L, lr_min = 1e-5)
# 
# ag_train(ag_mod2)
# set.seed(42)
# 
# for (ep in seq_len(150L)) {
#   perm <- sample(n)
#   for (b in seq_len(ceiling(n / BS))) {
#     idx <- perm[seq((b-1L)*BS + 1L, min(b*BS, n))]
#     xb  <- ag_tensor(x_tr_ag[, idx, drop = FALSE])
#     yb  <- y_tr_ag[, idx, drop = FALSE]
# 
#     with_grad_tape({
#       loss2 <- ag_softmax_cross_entropy_loss(ag_mod2$forward(xb), yb)
#     })
#     grads2 <- backward(loss2)
#     clip_grad_norm(params2, grads2, max_norm = 5.0)
#     opt2$step(grads2)
#     opt2$zero_grad()
#   }
#   sch2$step()
# }
# 
# ag_eval(ag_mod2)
# probs2 <- ag_predict_cm(ag_mod2, x_val_ag)
# acc2   <- mean(apply(probs2, 2, which.max) == true_ag)
# cat(sprintf("ag + cosine + clip  val accuracy: %.4f\n", acc2))

## -----------------------------------------------------------------------------
# make_net <- function(n_in, n_hidden, n_out) {
#   W1 <- ag_param(matrix(rnorm(n_hidden * n_in) * sqrt(2/n_in),  n_hidden, n_in))
#   b1 <- ag_param(matrix(0.0, n_hidden, 1L))
#   W2 <- ag_param(matrix(rnorm(n_out * n_hidden) * sqrt(2/n_hidden), n_out, n_hidden))
#   b2 <- ag_param(matrix(0.0, n_out, 1L))
# 
#   list(
#     forward    = function(x)
#       ag_add(ag_matmul(W2, ag_relu(ag_add(ag_matmul(W1, x), b1))), b2),
#     parameters = function() list(W1=W1, b1=b1, W2=W2, b2=b2)
#   )
# }
# 
# set.seed(1)
# net    <- make_net(4L, 32L, 3L)
# opt_r  <- optimizer_adam(net$parameters(), lr = 1e-3)
# 
# for (ep in seq_len(200L)) {
#   perm <- sample(n)
#   for (b in seq_len(ceiling(n / BS))) {
#     idx <- perm[seq((b-1L)*BS+1L, min(b*BS, n))]
#     xb  <- ag_tensor(x_tr_ag[, idx, drop = FALSE])
#     yb  <- y_tr_ag[, idx, drop = FALSE]
#     with_grad_tape({ loss_r <- ag_softmax_cross_entropy_loss(net$forward(xb), yb) })
#     gr <- backward(loss_r)
#     opt_r$step(gr);  opt_r$zero_grad()
#   }
# }
# 
# probs_r <- ag_predict_cm(net, x_val_ag)
# acc_r   <- mean(apply(probs_r, 2, which.max) == true_ag)
# cat(sprintf("Raw ag_param val accuracy: %.4f\n", acc_r))

## -----------------------------------------------------------------------------
# # Use Vulkan GPU if available, fall back to CPU
# device <- tryCatch({
#   ag_device("gpu")
#   "gpu"
# }, error = function(e) "cpu")
# 
# cat("Running on:", device, "\n")
# 
# # Mixed precision (f16 on GPU, f32 on CPU)
# ag_dtype(if (device == "gpu") "f16" else "f32")

