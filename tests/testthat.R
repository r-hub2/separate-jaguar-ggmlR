library(testthat)
library(ggmlR)

heavy <- c(
  "onnx-resize",
  "onnx-elementwise",
  "onnx-logic",
  "onnx-norm",
  "onnx-quant",
  "onnx-nn",
  "onnx-gemm",
  "onnx-misc",
  "onnx-ops",
  "onnx-shape"
)

is_cran_omp2 <- Sys.getenv("OMP_THREAD_LIMIT", "") == "2"

if (is_cran_omp2) {
  # negative lookahead to exclude tests with "test-(heavy)" in their name
  exclude_pattern <- paste0("^(?!.*test-(", paste(heavy, collapse = "|"), "))")
  test_check("ggmlR", filter = exclude_pattern, perl = TRUE)
} else {
  test_check("ggmlR")
}
