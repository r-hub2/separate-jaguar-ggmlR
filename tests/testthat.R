library(testthat)
library(ggmlR)

if (requireNamespace("mlr3", quietly = TRUE)) library(mlr3)
if (requireNamespace("parsnip", quietly = TRUE)) library(parsnip)

heavy <- c(
  "normalization",
  "sequence-ops",
  "binary-operations",
  "glu",
  "nn-sequential",
  "matrix-ops",
  "callbacks",
  "onnx-quant",
  "cpu-backend",
  "onnx-elementwise",
  "onnx-norm",
  "onnx-misc",
  "onnx-logic",
  "onnx-gemm",
  "onnx-resize",
  "onnx-nn",
  "nn-functional",
  "activations-extended",
  "onnx-ops",
  "transformer-ops",
  "onnx-shape",
  "math-operations",
  "activations",
  "backend-extended",
  "helpers",
  "onnx-reduce",
  "chain-sequential-batchnorm",
  "chain-patterns",
  "onnx-chain-quant",
  "onnx-chain-qlinearconv",
  "onnx-chain-uncovered-ops",
  "onnx-chain-transformer",
  "onnx-chain-broadcast-strict",
  "onnx-chain-maskrcnn-broadcast",
  "onnx-chain-erf-gelu",
  "onnx-chain-resize-broadcast",
  "onnx-chain-constantofshape-int64",
  "onnx-chain-roberta-attn",
  "onnx-chain-batched-matmul",
  "onnx-chain-classify",
  "onnx-chain-xcit-dynamic",
  "onnx-chain-fpn",
  "onnx-chain-pooling",
  "onnx-chain-unet",
  "onnx-chain-audio",
  "onnx-chain-superres",
  "onnx-chain-convtranspose",
  "onnx-chain-eyelike",
  "onnx-chain-decoder",
  "onnx-chain-detect",
  "onnx-chain-powernorm",
  "onnx-chain-bert-mlp",
  "onnx-chain-position",
  "onnx-chain-attn-mask",
  "onnx-chain-postprocess",
  "onnx-chain-cast",
  "onnx-chain-preprocess",
  "model-ops",
  "onnx-boundary",
  "onnx-edge",
  "parsnip",
  "mlr3-learner",
  "keras-api",
  "quants-iq-degenerate",
  "getrows-offload-vulkan"
)

# Tests that allocate large ggml contexts/tensors. They pass fine on a normal
# CRAN machine but blow the memory limit under the valgrind memtest (valgrind's
# shadow memory roughly doubles every allocation), causing a SIGKILL. Skip them
# only when running under valgrind; ordinary CRAN runs are unaffected.
valgrind_skip <- c(
  "q4k-matmul-vulkan",  # ggml_init(1 GiB) context
  "flash-attn-q4k",
  "flash-attn-quants"
)

# Detect valgrind: it injects vgpreload_*.so into the process, visible in
# /proc/self/maps. Fall back to FALSE on platforms without /proc.
under_valgrind <- tryCatch({
  maps <- readLines("/proc/self/maps", warn = FALSE)
  any(grepl("valgrind|vgpreload", maps, ignore.case = TRUE))
}, error = function(e) FALSE)

on_cran <- !identical(Sys.getenv("NOT_CRAN"), "true")

test_dir <- if (dir.exists("testthat")) "testthat" else "tests/testthat"

if (on_cran) {
  message("--- RUNNING LIGHT TESTS ONLY ---")

  all_tests <- list.files(test_dir, pattern = "^test-.*\\.R$")
  all_names <- sub("^test-(.*)\\.R$", "\\1", all_tests)

  skip_set <- heavy
  if (under_valgrind) {
    message("--- VALGRIND DETECTED: skipping memory-heavy tests ---")
    skip_set <- union(heavy, valgrind_skip)
  }

  light_tests <- setdiff(all_names, skip_set)
  message("Tests to run: ", paste(light_tests, collapse = ", "))

  if (length(light_tests) == 0) {
    # fallback
    test_check("ggmlR")
  } else {
    # grepl(filter, names)
    filter_regex <- paste(light_tests, collapse = "|")
    test_check("ggmlR", filter = filter_regex)
  }
} else {
  test_check("ggmlR")
}
