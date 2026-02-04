## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE
)

## ----detect-gpus--------------------------------------------------------------
# library(ggmlR)
# 
# if (ggml_vulkan_available()) {
#   n_gpus <- ggml_vulkan_device_count()
#   cat("Available GPUs:", n_gpus, "\n\n")
# 
#   for (i in seq_len(n_gpus)) {
#     cat("GPU", i - 1, ":", ggml_vulkan_device_description(i - 1), "\n")
#     mem_gb <- ggml_vulkan_device_memory(i - 1) / 1024^3
#     cat("  Memory:", round(mem_gb, 2), "GB\n")
#   }
# }

## ----multi-gpu-scheduler------------------------------------------------------
# if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 2) {
#   # Initialize multiple GPU backends
#   gpu0 <- ggml_vulkan_init(0)
#   gpu1 <- ggml_vulkan_init(1)
# 
#   # Create scheduler with multiple backends
#   # Order matters: first backend is preferred for supported operations
#   sched <- ggml_backend_sched_new(list(gpu0, gpu1))
# 
#   cat("Scheduler created with", ggml_backend_sched_get_n_backends(sched),
#       "backends\n")
# 
#   # Check backends
#   for (i in seq_len(ggml_backend_sched_get_n_backends(sched))) {
#     backend <- ggml_backend_sched_get_backend(sched, i - 1)
#     cat("Backend", i - 1, ":", ggml_backend_name(backend), "\n")
#   }
# }

## ----gpu-cpu-fallback---------------------------------------------------------
# if (ggml_vulkan_available()) {
#   # Initialize backends
#   gpu <- ggml_vulkan_init(0)
#   cpu <- ggml_backend_cpu_init()
#   ggml_backend_cpu_set_n_threads(cpu, 4)
# 
#   # GPU first, CPU as fallback
#   sched <- ggml_backend_sched_new(list(gpu, cpu))
# 
#   ctx <- ggml_init(64 * 1024 * 1024)
# 
#   # Create computation
#   a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 1000)
#   b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 1000)
#   c <- ggml_mul_mat(ctx, a, b)
# 
#   graph <- ggml_build_forward_expand(ctx, c)
#   ggml_backend_sched_reserve(sched, graph)
#   ggml_backend_sched_alloc_graph(sched, graph)
# 
#   # Check which backend handles each tensor
#   cat("\nTensor backend assignment:\n")
#   cat("  a:", ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, a)),
#       "\n")
#   cat("  b:", ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, b)),
#       "\n")
#   cat("  c:", ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, c)),
#       "\n")
# 
#   # Cleanup
#   ggml_backend_sched_free(sched)
#   ggml_vulkan_free(gpu)
#   ggml_backend_free(cpu)
#   ggml_free(ctx)
# }

## ----manual-placement---------------------------------------------------------
# if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 2) {
#   gpu0 <- ggml_vulkan_init(0)
#   gpu1 <- ggml_vulkan_init(1)
#   sched <- ggml_backend_sched_new(list(gpu0, gpu1))
# 
#   ctx <- ggml_init(128 * 1024 * 1024)
# 
#   # Create tensors for two parallel computations
#   a1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512)
#   b1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512)
#   c1 <- ggml_mul_mat(ctx, a1, b1)
# 
#   a2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512)
#   b2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512)
#   c2 <- ggml_mul_mat(ctx, a2, b2)
# 
#   # Combine results
#   result <- ggml_add(ctx, c1, c2)
# 
#   graph <- ggml_build_forward_expand(ctx, result)
# 
#   # Manually assign tensors to different GPUs
#   ggml_backend_sched_set_tensor_backend(sched, a1, gpu0)
#   ggml_backend_sched_set_tensor_backend(sched, b1, gpu0)
#   ggml_backend_sched_set_tensor_backend(sched, c1, gpu0)
# 
#   ggml_backend_sched_set_tensor_backend(sched, a2, gpu1)
#   ggml_backend_sched_set_tensor_backend(sched, b2, gpu1)
#   ggml_backend_sched_set_tensor_backend(sched, c2, gpu1)
# 
#   ggml_backend_sched_reserve(sched, graph)
#   ggml_backend_sched_alloc_graph(sched, graph)
# 
#   # Set data and compute
#   ggml_set_f32(a1, rnorm(512 * 512))
#   ggml_set_f32(b1, rnorm(512 * 512))
#   ggml_set_f32(a2, rnorm(512 * 512))
#   ggml_set_f32(b2, rnorm(512 * 512))
# 
#   ggml_backend_sched_graph_compute(sched, graph)
# 
#   cat("Multi-GPU computation completed\n")
#   cat("Result shape:", ggml_tensor_shape(result), "\n")
# 
#   # Cleanup
#   ggml_backend_sched_free(sched)
#   ggml_vulkan_free(gpu0)
#   ggml_vulkan_free(gpu1)
#   ggml_free(ctx)
# }

## ----async-multi-gpu----------------------------------------------------------
# if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 2) {
#   gpu0 <- ggml_vulkan_init(0)
#   gpu1 <- ggml_vulkan_init(1)
#   sched <- ggml_backend_sched_new(list(gpu0, gpu1))
# 
#   ctx <- ggml_init(64 * 1024 * 1024)
# 
#   # Build graph
#   a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100000)
#   b <- ggml_relu(ctx, a)
#   c <- ggml_sum(ctx, b)
# 
#   graph <- ggml_build_forward_expand(ctx, c)
#   ggml_backend_sched_reserve(sched, graph)
#   ggml_backend_sched_alloc_graph(sched, graph)
# 
#   ggml_set_f32(a, rnorm(100000))
# 
#   # Async compute - returns immediately
#   ggml_backend_sched_graph_compute_async(sched, graph)
# 
#   # Do other work here while GPU computes...
#   cat("Computing asynchronously...\n")
# 
#   # Wait for completion
#   ggml_backend_sched_synchronize(sched)
# 
#   cat("Result:", ggml_get_f32(c), "\n")
# 
#   # Cleanup
#   ggml_backend_sched_free(sched)
#   ggml_vulkan_free(gpu0)
#   ggml_vulkan_free(gpu1)
#   ggml_free(ctx)
# }

