# ggmlR 0.5.9

* `ggml_layer_lstm()` — LSTM recurrent layer (unrolled BPTT).
* `ggml_layer_gru()` — GRU recurrent layer (unrolled BPTT).
* `ggml_layer_global_max_pooling_2d()` — reduces `[H,W,C]` to `[C]` via max pooling.
* `ggml_layer_global_average_pooling_2d()` — reduces `[H,W,C]` to `[C]` via average pooling.
* `ggml_save_model()` — saves full model (architecture + weights) to RDS file.
* `ggml_load_model()` — restores a model saved with `ggml_save_model()`.
* `ggml_dense()`, `ggml_conv_2d()`, `ggml_conv_1d()`, `ggml_batch_norm()`, `ggml_embedding()`, `ggml_lstm()`, `ggml_gru()` — layer object constructors returning a reusable `ggml_layer` object.
* `ggml_apply(tensor, layer)` — applies a `ggml_layer` object to a tensor node; shared weights by object identity.

# ggmlR 0.5.7

* `ggml_layer_dropout()` — dropout with deterministic or stochastic (per-epoch Bernoulli mask) mode.
* `ggml_layer_embedding()` — token embedding lookup for integer inputs.
* `ggml_input()` gains `dtype` argument (`"float32"` or `"int32"`).
* Multi-output support in `ggml_model()` and `ggml_predict()`.

# ggmlR 0.5.6

* `ggml_input()` — declare a symbolic input tensor node (Functional API).
* `ggml_model()` — assemble a `ggml_functional_model` from input/output nodes.
* `ggml_layer_add()` — element-wise addition of tensor nodes (residual connections).
* `ggml_layer_concatenate()` — concatenate tensor nodes along an axis.
* All `ggml_layer_*()` functions now accept a `ggml_tensor_node` as first argument (Functional API mode).
* `ggml_compile()`, `ggml_fit()`, `ggml_evaluate()`, `ggml_predict()` are now S3 generics with methods for `ggml_functional_model`.

# ggmlR 0.5.5

* `ggml_fit_opt()` — low-level optimizer loop with callbacks and learning-rate control.
* `ggml_callback_early_stopping()` — stops training when a metric stagnates.
* `ggml_schedule_step_decay()` — step learning-rate decay.
* `ggml_schedule_cosine_decay()` — cosine learning-rate annealing.
* `ggml_schedule_reduce_on_plateau()` — reduces LR when metric stops improving.
* `ggml_opt_init_for_fit()`, `ggml_opt_set_lr()`, `ggml_opt_get_lr()` — learning-rate control without recreating the optimizer context.

# ggmlR 0.5.4

* Vulkan GPU backend support on Windows via `configure.win`.
* Vulkan auto-detected at build time on Linux and Windows.

# ggmlR 0.5.3

* `ggml_layer_conv_1d()` — 1D convolution layer.
* `ggml_layer_batch_norm()` — batch normalization layer.
* `ggml_predict_classes()` — argmax wrapper returning 1-based class indices.
* `summary.ggml_sequential_model()` — detailed model summary with parameter counts.
* `ggml_fit()` now returns `model$history` (class `ggml_history`) with `print` and `plot` methods.
* Sequential API: `ggml_model_sequential()`, `ggml_layer_dense()`, `ggml_layer_conv_2d()`, `ggml_layer_max_pooling_2d()`, `ggml_layer_flatten()`, `ggml_compile()`, `ggml_fit()`, `ggml_evaluate()`, `ggml_predict()`, `ggml_save_weights()`, `ggml_load_weights()`.
* Vulkan GPU backend covering 90%+ of ML operations.

# ggmlR 0.5.2

* `ggml_timestep_embedding()` — sinusoidal timestep embeddings.
* N-D tensor access: `ggml_set_f32_nd()`, `ggml_get_f32_nd()`, `ggml_set_i32_nd()`, `ggml_get_i32_nd()`.
* Tensor utilities: `ggml_tensor_nb()`, `ggml_tensor_num()`, `ggml_tensor_copy()`, `ggml_tensor_set_f32_scalar()`, `ggml_get_first_tensor()`, `ggml_get_next_tensor()`.

# ggmlR 0.5.1

* Static library `libggml.a` exported for linking by dependent packages.
* `gguf.cpp` added for GGUF file format support.
* Headers exported via `inst/include/` for `LinkingTo`.

# ggmlR 0.5.0

* Full optimization/training API: `ggml_opt_init()`, `ggml_opt_free()`, `ggml_opt_fit()`, `ggml_opt_epoch()`, `ggml_opt_eval()`.
* Dataset management: `ggml_opt_dataset_init()`, `ggml_opt_dataset_data()`, `ggml_opt_dataset_labels()`, `ggml_opt_dataset_shuffle()`.
* Training results: `ggml_opt_result_init()`, `ggml_opt_result_loss()`, `ggml_opt_result_accuracy()`, `ggml_opt_result_pred()`.
* Extended backend API: device management, registry, async operations, graph planning, buffer management (~50 new functions).
* Loss functions: MSE, cross-entropy. Optimizers: AdamW, SGD.

# ggmlR 0.4.0

* Multi-GPU backend scheduler API.
* Vulkan GPU backend support.

# ggmlR 0.2.0

* Initial release: R bindings for GGML tensor library.
* Core tensor operations, neural network ops, activation functions, quantization (Q4_0, Q4_1, Q8_0), OpenMP parallelization, computation graph API.
