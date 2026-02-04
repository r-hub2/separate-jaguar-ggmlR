## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE
)

## ----check-vulkan-------------------------------------------------------------
# library(ggmlR)
# 
# # Check if Vulkan backend is available
# if (ggml_vulkan_available()) {
#   cat("Vulkan is available!\n")
#   cat("Number of Vulkan devices:", ggml_vulkan_device_count(), "\n")
# } else {
#   cat("Vulkan is not available. Using CPU backend.\n")
# }

## ----list-devices-------------------------------------------------------------
# if (ggml_vulkan_available()) {
#   devices <- ggml_vulkan_list_devices()
#   print(devices)
# 
#   # Get detailed info for each device
#   for (i in seq_len(ggml_vulkan_device_count())) {
#     cat("\nDevice", i - 1, ":\n")
#     cat("  Name:", ggml_vulkan_device_description(i - 1), "\n")
# 
#     mem <- ggml_vulkan_device_memory(i - 1)
#     cat("  Memory:", round(mem / 1024^3, 2), "GB\n")
#   }
# }

## ----init-vulkan--------------------------------------------------------------
# if (ggml_vulkan_available()) {
#   # Initialize Vulkan backend (device 0)
#   vk <- ggml_vulkan_init(device = 0)
# 
#   # Check backend name
#   cat("Backend:", ggml_vulkan_backend_name(vk), "\n")
# 
#   # Verify it's a Vulkan backend
#   stopifnot(ggml_vulkan_is_backend(vk))
# }

## ----gpu-compute--------------------------------------------------------------
# if (ggml_vulkan_available()) {
#   # Initialize Vulkan backend
#   vk <- ggml_vulkan_init(0)
# 
#  # Create scheduler with Vulkan backend
#   sched <- ggml_backend_sched_new(list(vk))
# 
#   # Create context for tensors
#   ctx <- ggml_init(64 * 1024 * 1024)
# 
#   # Create tensors
#   n <- 10000
#   a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
#   b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
# 
#   # Build computation graph
#   c <- ggml_add(ctx, a, b)
#   d <- ggml_mul(ctx, c, c)
#   result <- ggml_sum(ctx, d)
# 
#   graph <- ggml_build_forward_expand(ctx, result)
# 
#   # Reserve memory and allocate
#   ggml_backend_sched_reserve(sched, graph)
#   ggml_backend_sched_alloc_graph(sched, graph)
# 
#   # Set data
#   ggml_set_f32(a, rnorm(n))
#   ggml_set_f32(b, rnorm(n))
# 
#   # Compute on GPU
#   ggml_backend_sched_graph_compute(sched, graph)
# 
#   # Get result
#   cat("Result:", ggml_get_f32(result), "\n")
# 
#   # Cleanup
#   ggml_backend_sched_free(sched)
#   ggml_vulkan_free(vk)
#   ggml_free(ctx)
# }

## ----matmul-gpu---------------------------------------------------------------
# if (ggml_vulkan_available()) {
#   vk <- ggml_vulkan_init(0)
#   sched <- ggml_backend_sched_new(list(vk))
#   ctx <- ggml_init(128 * 1024 * 1024)
# 
#   # Create matrices
#   m <- 1024
#   n <- 1024
#   k <- 1024
# 
#   A <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m)
#   B <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n)
# 
#   # Matrix multiplication: C = A * B^T
#   C <- ggml_mul_mat(ctx, A, B)
# 
#   graph <- ggml_build_forward_expand(ctx, C)
#   ggml_backend_sched_reserve(sched, graph)
#   ggml_backend_sched_alloc_graph(sched, graph)
# 
#   # Initialize with random data
#   ggml_set_f32(A, rnorm(m * k))
#   ggml_set_f32(B, rnorm(n * k))
# 
#   # Time GPU computation
#   start <- Sys.time()
#   ggml_backend_sched_graph_compute(sched, graph)
#   gpu_time <- Sys.time() - start
# 
#   cat("GPU matmul time:", round(as.numeric(gpu_time) * 1000, 2), "ms\n")
#   cat("Result shape:", ggml_tensor_shape(C), "\n")
# 
#   # Cleanup
#   ggml_backend_sched_free(sched)
#   ggml_vulkan_free(vk)
#   ggml_free(ctx)
# }

