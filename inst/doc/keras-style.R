## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(eval = FALSE)

## ----sequential---------------------------------------------------------------
# library(ggmlR)
# 
# # Build a simple MLP classifier
# model <- ggml_model_sequential() |>
#   ggml_layer_dense(128L, activation = "relu", input_shape = 64L) |>
#   ggml_layer_dropout(0.3) |>
#   ggml_layer_dense(10L, activation = "softmax")
# 
# # Compile: choose optimizer and loss
# model <- ggml_compile(model,
#                       optimizer = "adam",
#                       loss      = "categorical_crossentropy")
# 
# # Fit on (x, y) — x is a matrix [n_samples × 64], y is [n_samples × 10]
# model <- ggml_fit(model, x, y,
#                   epochs     = 5L,
#                   batch_size = 32L,
#                   verbose    = 0L)
# 
# # Predict on new data
# preds <- ggml_predict(model, x)

## ----functional-residual------------------------------------------------------
# library(ggmlR)
# 
# inp <- ggml_input(shape = 64L)
# 
# # Project and apply non-linearity
# x   <- inp |> ggml_layer_dense(64L, activation = "relu")
# 
# # Skip connection: add the original input to the transformed output
# res <- ggml_layer_add(list(inp, x))
# 
# # Classification head
# out <- res |> ggml_layer_dense(10L, activation = "softmax")
# 
# # Assemble and compile
# m <- ggml_model(inputs = inp, outputs = out)
# m <- ggml_compile(m,
#                   optimizer = "adam",
#                   loss      = "categorical_crossentropy")

## ----embedding----------------------------------------------------------------
# library(ggmlR)
# 
# # Input: integer token sequences of length 20
# inp <- ggml_input(shape = 20L, dtype = "int32")
# 
# out <- inp |>
#   ggml_layer_embedding(vocab_size = 1000L, dim = 64L) |>
#   ggml_layer_flatten() |>
#   ggml_layer_dense(10L, activation = "softmax")
# 
# m <- ggml_model(inputs = inp, outputs = out)

## ----multi-output-------------------------------------------------------------
# library(ggmlR)
# 
# inp    <- ggml_input(shape = 64L)
# hidden <- inp    |> ggml_layer_dense(64L, activation = "relu")
# out    <- hidden |> ggml_layer_dense(10L, activation = "softmax")
# 
# # Expose both the hidden layer and the classifier output
# m <- ggml_model(inputs = inp, outputs = list(hidden, out))
# 
# preds <- ggml_predict(m, x)
# # preds[[1]] — hidden-layer activations, shape [n_samples × 64]
# # preds[[2]] — class probabilities,      shape [n_samples × 10]

## ----siamese------------------------------------------------------------------
# library(ggmlR)
# 
# # Define the shared encoder once
# enc <- ggml_dense(32L, activation = "relu", name = "encoder")
# 
# # Two independent inputs
# x1 <- ggml_input(shape = 16L, name = "left")
# x2 <- ggml_input(shape = 16L, name = "right")
# 
# # Apply the *same* layer (identical weights) to both branches
# h1 <- ggml_apply(x1, enc)
# h2 <- ggml_apply(x2, enc)
# 
# # Merge and classify
# out <- ggml_layer_add(list(h1, h2)) |>
#   ggml_layer_dense(2L, activation = "softmax")
# 
# m <- ggml_model(inputs = list(x1, x2), outputs = out)

