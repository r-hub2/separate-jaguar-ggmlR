## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(eval = FALSE, collapse = TRUE, comment = "#>")
library(ggmlR)

## ----eval=FALSE---------------------------------------------------------------
# library(ggmlR)

## ----eval=FALSE---------------------------------------------------------------
# model <- onnx_load("path/to/model.onnx")
# 
# # Model summary (layers, ops, parameters)
# onnx_summary(model)
# 
# # Input tensor info (name, shape, dtype)
# onnx_inputs(model)

## ----eval=FALSE---------------------------------------------------------------
# # Random image batch — replace with real data
# input <- array(runif(1 * 3 * 224 * 224), dim = c(1L, 3L, 224L, 224L))
# 
# result <- onnx_run(model, list(input_name = input))
# 
# cat("Output shape:", paste(dim(result[[1]]), collapse = " x "), "\n")

## ----eval=FALSE---------------------------------------------------------------
# result <- onnx_run(model, list(
#   input_ids      = array(as.integer(tokens), dim = c(1L, length(tokens))),
#   attention_mask = array(1L, dim = c(1L, length(tokens)))
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

