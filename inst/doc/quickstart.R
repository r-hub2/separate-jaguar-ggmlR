## ----setup, include=FALSE-----------------------------------------------------
# Vignette code is executed locally (NOT_CRAN=true) but not on CRAN, where
# the CPU fallback would multi-thread and trip the "CPU time > elapsed" NOTE.
knitr::opts_chunk$set(eval = identical(Sys.getenv("NOT_CRAN"), "true"))

## -----------------------------------------------------------------------------
# library(ggmlR)
# 
# x <- scale(as.matrix(iris[, 1:4]))            # 4 numeric features
# y <- model.matrix(~ Species - 1, iris)        # one-hot, 3 classes
# 
# model <- ggml_model_sequential() |>
#   ggml_layer_dense(16L, activation = "relu", input_shape = 4L) |>
#   ggml_layer_dense(3L,  activation = "softmax") |>
#   ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")
# 
# model <- ggml_fit(model, x, y, epochs = 100L, verbose = 0L)
# 
# pred  <- ggml_predict(model, x)               # [150 x 3] class probabilities
# acc   <- mean(max.col(pred) == as.integer(iris$Species))
# cat(sprintf("accuracy: %.3f\n", acc))

