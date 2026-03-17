# Restore original thread count after CRAN checks
if (exists(".ggmlr_old_threads")) {
  ggml_set_omp_threads(.ggmlr_old_threads)
}
