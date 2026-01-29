# ggmlR 0.4.1

* Fixed spelling notes for CRAN submission
* Updated documentation

# ggmlR 0.4.0

* Added multi-GPU backend scheduler API (14 new functions)
* Added Vulkan GPU backend support (10 new functions)
* Fixed integer overflow for large tensors (>2 GB)
* Improved OpenMP handling for mixed C/C++ code

# ggmlR 0.2.0

* Initial CRAN submission
* R bindings for 'GGML' tensor library
* Core tensor operations: creation, arithmetic, reshaping
* Neural network operations: attention, convolutions, normalization
* Activation functions: GELU, SiLU, ReLU, and variants
* Quantization support (Q4_0, Q4_1, Q8_0)
* OpenMP parallelization for CPU backend
* Computation graph API for building and executing models
