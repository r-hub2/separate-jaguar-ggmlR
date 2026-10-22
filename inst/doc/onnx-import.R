## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(eval = FALSE, collapse = TRUE, comment = "#>")
library(ggmlR)

## ----eval=FALSE---------------------------------------------------------------
# library(ggmlR)

## ----eval=FALSE---------------------------------------------------------------
# model <- onnx_load("path/to/model.onnx")
# 
# # Model summary (IR and opset version, node count, operators used)
# onnx_summary(model)
# 
# # Input names and shapes (-1 for a dynamic dimension)
# onnx_inputs(model)

## ----eval=FALSE---------------------------------------------------------------
# # Random image batch, [N, C, H, W] as declared — replace with real data
# x <- array(runif(8 * 3 * 224 * 224), dim = c(8L, 3L, 224L, 224L))
# 
# probs <- predict(model, x)   # [8, 1000] for a classifier

## ----eval=FALSE---------------------------------------------------------------
# # A 224 x 224 x 3 image (H, W, C) for a model declared as [1, 3, 224, 224]
# img <- array(runif(224 * 224 * 3), dim = c(224L, 224L, 3L))
# 
# # Storage order of [1, 3, 224, 224] is (W, H, C, N): W fastest
# input <- aperm(img, c(2, 1, 3))
# 
# result <- onnx_run(model, list(input_name = input))

## ----eval=FALSE---------------------------------------------------------------
# cat("Output dim (storage order):", paste(dim(result[[1]]), collapse = " x "), "\n")

## ----eval=FALSE---------------------------------------------------------------
# result <- onnx_run(model, list(
#   input_ids      = as.numeric(tokens),
#   attention_mask = rep(1, length(tokens))
# ))

## ----eval=FALSE---------------------------------------------------------------
# # Check what's available
# if (ggml_vulkan_available()) {
#   cat("Vulkan GPU ready\n")
#   ggml_vulkan_status()
# }
# 
# # Load with explicit device
# model_gpu <- onnx_load("path/to/model.onnx", device = "vulkan")
# model_cpu <- onnx_load("path/to/model.onnx", device = "cpu")

## ----eval=FALSE---------------------------------------------------------------
# model <- onnx_load("path/to/bert.onnx",
#                     input_shapes = list(input_ids = c(1L, 128L)))

## ----eval=FALSE---------------------------------------------------------------
# model_fp16 <- onnx_load("path/to/model.onnx", dtype = "f16")
# result <- onnx_run(model_fp16, list(input = input))

## ----eval=FALSE---------------------------------------------------------------
# # GPU vs CPU benchmark across multiple models
# # inst/examples/benchmark_onnx.R
# 
# # FP16 inference benchmark
# # inst/examples/benchmark_onnx_fp16.R
# 
# # Run all supported ONNX Zoo models
# # inst/examples/test_all_onnx.R
# 
# # BERT sentence similarity
# # inst/examples/bert_similarity.R

