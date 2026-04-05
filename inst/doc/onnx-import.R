## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(eval = FALSE)

## -----------------------------------------------------------------------------
# library(ggmlR)

## -----------------------------------------------------------------------------
# model <- ggml_onnx_load("path/to/model.onnx")
# 
# # Input / output info
# cat("Inputs:\n");  print(ggml_onnx_inputs(model))
# cat("Outputs:\n"); print(ggml_onnx_outputs(model))

## -----------------------------------------------------------------------------
# # Random image batch — replace with real data
# input <- array(runif(1 * 3 * 224 * 224), dim = c(1L, 3L, 224L, 224L))
# 
# result <- ggml_onnx_run(model, list(input_name = input))
# 
# cat("Output shape:", paste(dim(result[[1]]), collapse = " x "), "\n")

## -----------------------------------------------------------------------------
# result <- ggml_onnx_run(model, list(
#   input_ids      = array(as.integer(tokens), dim = c(1L, length(tokens))),
#   attention_mask = array(1L, dim = c(1L, length(tokens)))
# ))

## -----------------------------------------------------------------------------
# # Check what's available
# if (ggml_vulkan_available()) {
#   cat("Vulkan GPU ready\n")
#   ggml_vulkan_status()
# }
# 
# # Load with explicit backend hint
# model_gpu <- ggml_onnx_load("path/to/model.onnx", backend = "vulkan")
# model_cpu <- ggml_onnx_load("path/to/model.onnx", backend = "cpu")

## -----------------------------------------------------------------------------
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

