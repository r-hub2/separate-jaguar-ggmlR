## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(eval = FALSE)

## -----------------------------------------------------------------------------
# library(ggmlR)

## -----------------------------------------------------------------------------
# if (ggml_vulkan_available()) {
#   cat("Vulkan is available\n")
#   ggml_vulkan_status()              # print device list and properties
# } else {
#   cat("No Vulkan GPU — running on CPU\n")
# }
# 
# n <- ggml_vulkan_device_count()
# cat("Vulkan device count:", n, "\n")

## -----------------------------------------------------------------------------
# # Low-level device registry (all backends including CPU)
# ggml_backend_load_all()
# 
# n_dev <- ggml_backend_dev_count()
# for (i in seq_len(n_dev)) {
#   dev  <- ggml_backend_dev_get(i - 1L)   # 0-based
#   name <- ggml_backend_dev_name(dev)
#   desc <- ggml_backend_dev_description(dev)
#   mem  <- ggml_backend_dev_memory(dev)
#   cat(sprintf("[%d] %s — %s\n", i, name, desc))
#   cat(sprintf("    %.1f GB free / %.1f GB total\n",
#               mem["free"] / 1e9, mem["total"] / 1e9))
# }

## -----------------------------------------------------------------------------
# # Select GPU (falls back to CPU if unavailable)
# device <- tryCatch({
#   ag_device("gpu")
#   "gpu"
# }, error = function(e) {
#   message("GPU not available, using CPU")
#   "cpu"
# })
# 
# cat("Active device:", device, "\n")

## -----------------------------------------------------------------------------
# if (device == "gpu") {
#   ag_dtype("f16")     # half-precision on Vulkan GPU
#   # ag_dtype("bf16") # bfloat16 — falls back to f16 on Vulkan automatically
# } else {
#   ag_dtype("f32")     # full precision on CPU
# }
# 
# cat("Active dtype:", ag_dtype(), "\n")

## -----------------------------------------------------------------------------
# if (ggml_vulkan_available()) {
#   mem <- ggml_vulkan_memory(device_index = 0L)
#   cat(sprintf("GPU memory: %.1f MB free / %.1f MB total\n",
#               mem["free"] / 1e6, mem["total"] / 1e6))
# }

## -----------------------------------------------------------------------------
# n_gpu <- ggml_vulkan_device_count()
# cat(sprintf("Using %d GPU(s)\n", n_gpu))
# 
# # dp_train handles multi-GPU internally — see vignette("data-parallel-training")

## -----------------------------------------------------------------------------
# model <- ggml_model_sequential() |>
#   ggml_layer_dense(64L, activation = "relu", input_shape = 4L) |>
#   ggml_layer_dense(3L,  activation = "softmax") |>
#   ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")
# 
# # Training runs on GPU if Vulkan is available
# model <- ggml_fit(model, x_train, y_train, epochs = 50L, batch_size = 32L)

## -----------------------------------------------------------------------------
# # Weights loaded to GPU once at load time
# model_onnx <- ggml_onnx_load("model.onnx", backend = "vulkan")
# 
# # Repeated inference — no weight re-transfer
# for (i in seq_len(100L)) {
#   out <- ggml_onnx_run(model_onnx, list(input = batch[[i]]))
# }

## -----------------------------------------------------------------------------
# cat(ggml_version(), "\n")
# ggml_vulkan_status()   # shows "Vulkan not available" if not compiled in

